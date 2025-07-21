import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import datetime
import paho.mqtt.client as mqtt
from collections import defaultdict
import queue
import threading

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
from hailo_apps_infra.gstreamer_app import disable_qos
from hailo_apps_infra.gstreamer_helper_pipelines import (
    SOURCE_PIPELINE,
    INFERENCE_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    TRACKER_PIPELINE,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    OVERLAY_PIPELINE,
)

# MQTT Configuration
MQTT_TOPIC_STATUS = 'cctv/status'
MQTT_TOPIC_IMAGE = 'cctv/image'
MQTT_TOPIC_STATS = 'cctv/stats'
broker = 'cctv.dewika.id'
port = 1883
mqtt_username = 'pi'
mqtt_password = 'pi'

# Disable GStreamer debug messages
os.environ['GST_DEBUG'] = '0'

class CustomGStreamerDetectionApp(GStreamerDetectionApp):
    def __init__(self, callback, user_data):
        super().__init__(callback, user_data)
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_message)
        
    def on_message(self, bus, message):
        if message.type == Gst.MessageType.QOS:
            # Silently handle QoS messages without printing
            return True
        return False

    def get_pipeline_string(self):
        # Get the base pipeline components with optimized settings
        source_pipeline = (
            f'filesrc location="{self.video_source}" name=source ! '
            f'queue name=source_queue_decode leaky=no max-size-buffers=10 max-size-bytes=0 max-size-time=0 ! '
            f'decodebin name=source_decodebin ! '
            f'queue name=source_scale_q leaky=no max-size-buffers=10 max-size-bytes=0 max-size-time=0 ! '
            f'videoscale name=source_videoscale n-threads=2 qos=false ! '
            f'queue name=source_convert_q leaky=no max-size-buffers=10 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert n-threads=2 name=source_convert qos=false ! '
            f'video/x-raw, pixel-aspect-ratio=1/1, format=RGB, width={self.video_width}, height={self.video_height}'
        )
        
        # Optimize detection pipeline
        detection_pipeline = INFERENCE_PIPELINE(
            hef_path=self.hef_path,
            post_process_so=self.post_process_so,
            post_function_name=self.post_function_name,
            batch_size=1,  # Reduce batch size to 1
            config_json=self.labels_json,
            additional_params=self.thresholds_str)
        
        detection_pipeline_wrapper = INFERENCE_PIPELINE_WRAPPER(detection_pipeline)
        
        # Optimize tracker settings
        tracker_pipeline = TRACKER_PIPELINE(
            class_id=1,
            kalman_dist_thr=0.8,
            iou_thr=0.7,  # Slightly lower IOU threshold
            init_iou_thr=0.5,
            keep_new_frames=1,  # Reduce kept frames
            keep_tracked_frames=10,  # Reduce tracked frames
            keep_lost_frames=1,
            keep_past_metadata=False,
            qos=False)
            
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        
        # Optimize display pipeline
        display_pipeline = (
            f'{OVERLAY_PIPELINE(name="hailo_overlay")} ! '
            f'queue name=hailo_display_videoconvert_q leaky=no max-size-buffers=10 max-size-bytes=0 max-size-time=0 ! '
            f'videoconvert name=hailo_display_videoconvert n-threads=2 qos=false ! '
            f'queue name=hailo_display_q leaky=no max-size-buffers=10 max-size-bytes=0 max-size-time=0 ! '
            f'xvimagesink name=hailo_display sync={self.sync} qos=false'
        )

        # Construct the final pipeline string
        pipeline_string = (
            f'{source_pipeline} ! '
            f'{detection_pipeline_wrapper} ! '
            f'{tracker_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline}'
        )
        
        print("Pipeline string:", pipeline_string)
        return pipeline_string

class ImageSender(threading.Thread):
    def __init__(self, mqtt_client):
        super().__init__()
        self.mqtt_client = mqtt_client
        self.image_queue = queue.Queue(maxsize=5)  # Reduce queue size
        self.running = True
        self.delay_between_sends = 3.0  # Increase delay between sends
        self.daemon = True
        self.batch_size = 2  # Reduce batch size
        self.compression_quality = 80  # Slightly reduce compression quality

    def add_to_queue(self, image_data, tracking_id):
        try:
            # Compress image before adding to queue
            _, img_encoded = cv2.imencode('.jpg', image_data, 
                                        [cv2.IMWRITE_JPEG_QUALITY, self.compression_quality])
            compressed_data = img_encoded.tobytes()
            self.image_queue.put((compressed_data, tracking_id), block=False)
        except queue.Full:
            print(f"Queue full, dropping image for person {tracking_id}")

    def run(self):
        while self.running:
            try:
                # Process multiple images at once
                batch = []
                for _ in range(self.batch_size):
                    try:
                        item = self.image_queue.get(timeout=1.0)
                        batch.append(item)
                    except queue.Empty:
                        break

                if not batch:
                    continue

                # Send all status messages first
                for _, tracking_id in batch:
                    self.mqtt_client.publish(MQTT_TOPIC_STATUS, "0")
                    print(f"Sending status for person {tracking_id}")
                    time.sleep(0.5)  # Small delay between status messages

                # Then send all images
                for image_data, tracking_id in batch:
                    self.mqtt_client.publish(MQTT_TOPIC_IMAGE, image_data)
                    print(f"Published cropped person {tracking_id} to MQTT")
                    time.sleep(self.delay_between_sends)
                    self.image_queue.task_done()

            except Exception as e:
                print(f"Error in image sender thread: {e}")
                continue

    def stop(self):
        self.running = False
        self.join()

class MultiplePersonDetectionCallback(app_callback_class):
    def __init__(self, mqtt_client):
        super().__init__()
        self.mqtt_client = mqtt_client
        self.person_stats = defaultdict(lambda: {
            'total_frames': 0,
            'false_frames': 0,
            'true_frames': 0,
            'last_seen': time.time(),
            'bbox': None,
            'tracking_id': None,
            'missed_frames': 0,
            'size': None,
            'last_alert_time': 0  # Track last alert time
        })
        self.false_frames_array = []
        self.false_detections_array = []
        self.last_reset_time = time.time()
        self.next_tracking_id = 1
        self.max_missed_frames = 20
        self.iou_threshold = 0.4
        self.position_threshold = 0.08
        self.size_threshold = 0.2
        self.alert_cooldown = 30  # Minimum seconds between alerts for same person
        
        # Initialize image sender thread
        self.image_sender = ImageSender(mqtt_client)
        self.image_sender.start()

    def __del__(self):
        if hasattr(self, 'image_sender'):
            self.image_sender.stop()

    def calculate_iou(self, bbox1, bbox2):
        # Calculate Intersection over Union between two bounding boxes
        x1 = max(bbox1.xmin(), bbox2.xmin())
        y1 = max(bbox1.ymin(), bbox2.ymin())
        x2 = min(bbox1.xmax(), bbox2.xmax())
        y2 = min(bbox1.ymax(), bbox2.ymax())
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        
        bbox1_area = (bbox1.xmax() - bbox1.xmin()) * (bbox1.ymax() - bbox1.ymin())
        bbox2_area = (bbox2.xmax() - bbox2.xmin()) * (bbox2.ymax() - bbox2.ymin())
        
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / union if union > 0 else 0

    def calculate_position_change(self, bbox1, bbox2):
        # Calculate center points
        center1_x = (bbox1.xmin() + bbox1.xmax()) / 2
        center1_y = (bbox1.ymin() + bbox1.ymax()) / 2
        center2_x = (bbox2.xmin() + bbox2.xmax()) / 2
        center2_y = (bbox2.ymin() + bbox2.ymax()) / 2
        
        # Calculate distance between centers
        distance = ((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2) ** 0.5
        return distance

    def calculate_size(self, bbox):
        # Calculate the size of the bounding box
        width = bbox.xmax() - bbox.xmin()
        height = bbox.ymax() - bbox.ymin()
        return width * height

    def calculate_size_change(self, size1, size2):
        # Calculate relative size change
        return abs(size1 - size2) / max(size1, size2)

    def find_matching_tracking_id(self, current_bbox):
        best_match = None
        best_score = 0
        current_size = self.calculate_size(current_bbox)
        
        for tracking_id, stats in self.person_stats.items():
            if stats['bbox'] is not None and stats['size'] is not None:
                iou = self.calculate_iou(current_bbox, stats['bbox'])
                position_change = self.calculate_position_change(current_bbox, stats['bbox'])
                size_change = self.calculate_size_change(current_size, stats['size'])
                
                # Calculate a combined score
                if (iou > self.iou_threshold and 
                    position_change < self.position_threshold and
                    size_change < self.size_threshold):
                    score = iou * (1 - position_change) * (1 - size_change)
                    if score > best_score:
                        best_score = score
                        best_match = tracking_id
        
        return best_match

    def assign_tracking_id(self, bbox):
        # Try to find existing tracking ID
        existing_id = self.find_matching_tracking_id(bbox)
        
        if existing_id is not None:
            return existing_id
        
        # If no match found, create new tracking ID
        tracking_id = f"person_{self.next_tracking_id}"
        self.next_tracking_id += 1
        return tracking_id

    def process_detections(self, detections, frame):
        current_time = time.time()
        person_detections = []
        safety_equipment_detections = []
        
        # First, collect all detections
        for detection in detections:
            label = detection.get_label()
            if label == "person":
                person_detections.append(detection)
            elif label in ["buruh", "site-engineer"]:
                safety_equipment_detections.append(detection)

        # Update missed frames for all existing persons
        for tracking_id in list(self.person_stats.keys()):
            self.person_stats[tracking_id]['missed_frames'] += 1

        # Process each person detection
        for person_det in person_detections:
            tracking_id = self.assign_tracking_id(person_det.get_bbox())
            person_stat = self.person_stats[tracking_id]
            
            # Update tracking information
            person_stat['last_seen'] = current_time
            person_stat['total_frames'] += 1
            person_stat['missed_frames'] = 0
            person_stat['bbox'] = person_det.get_bbox()
            person_stat['size'] = self.calculate_size(person_det.get_bbox())

            # Find the closest safety equipment detection
            has_safety_equipment = False
            if safety_equipment_detections:
                person_center_x = (person_det.get_bbox().xmin() + person_det.get_bbox().xmax()) / 2
                person_center_y = (person_det.get_bbox().ymin() + person_det.get_bbox().ymax()) / 2
                
                min_distance = float('inf')
                closest_equipment = None
                
                for equipment in safety_equipment_detections:
                    equipment_center_x = (equipment.get_bbox().xmin() + equipment.get_bbox().xmax()) / 2
                    equipment_center_y = (equipment.get_bbox().ymin() + equipment.get_bbox().ymax()) / 2
                    
                    distance = ((equipment_center_x - person_center_x) ** 2 + 
                              (equipment_center_y - person_center_y) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_equipment = equipment
                
                # If the closest equipment is within a reasonable distance
                if min_distance < 0.2:  # Adjust this threshold as needed
                    has_safety_equipment = True
                    # Remove the used equipment detection
                    safety_equipment_detections.remove(closest_equipment)

            if has_safety_equipment:
                person_stat['true_frames'] += 1
            else:
                person_stat['false_frames'] += 1
                self.false_frames_array.append(frame.copy())
                self.false_detections_array.append({
                    'person_detection': person_det,
                    'tracking_id': tracking_id
                })

        # Clean up old tracking data
        for tracking_id in list(self.person_stats.keys()):
            if (current_time - self.person_stats[tracking_id]['last_seen'] > 5 or 
                self.person_stats[tracking_id]['missed_frames'] > self.max_missed_frames):
                del self.person_stats[tracking_id]

        # Process statistics every 60 seconds
        if current_time - self.last_reset_time >= 60:
            self.process_and_send_frames()
            self.last_reset_time = current_time

    def process_and_send_frames(self):
        if not self.false_frames_array:
            return

        print("\n=== Statistics after 60 seconds ===")
        total_violations = 0
        current_time = time.time()
        
        # Calculate statistics for each person
        for tracking_id, stats in self.person_stats.items():
            total_frames = stats['total_frames']
            if total_frames > 0:
                false_percentage = (stats['false_frames'] / total_frames * 100)
                
                print(f"\nPerson {tracking_id} statistics:")
                print(f"Total frames: {total_frames}")
                print(f"True frames: {stats['true_frames']}")
                print(f"False frames: {stats['false_frames']}")
                print(f"False percentage: {false_percentage:.2f}%")
                
                # Check if enough time has passed since last alert
                if (false_percentage > 50.6 and 
                    current_time - stats['last_alert_time'] >= self.alert_cooldown):
                    total_violations += 1
                    print(f"ALERT: Person {tracking_id} has violated safety rules!")
                    
                    # Find the last frame for this person
                    last_frame = None
                    last_detection = None
                    for frame, detection_info in zip(self.false_frames_array, self.false_detections_array):
                        if detection_info['tracking_id'] == tracking_id:
                            last_frame = frame
                            last_detection = detection_info['person_detection']

                    if last_frame is not None:
                        # Process the frame
                        height, width = last_frame.shape[:2]
                        bbox = last_detection.get_bbox()
                        
                        # Convert relative coordinates to pixel coordinates
                        xmin = int(bbox.xmin() * width)
                        ymin = int(bbox.ymin() * height)
                        xmax = int(bbox.xmax() * width)
                        ymax = int(bbox.ymax() * height)
                        
                        # Add padding and ensure within bounds
                        padding = 10
                        xmin = max(0, xmin - padding)
                        ymin = max(0, ymin - padding)
                        xmax = min(width, xmax + padding)
                        ymax = min(height, ymax + padding)
                        
                        if xmax > xmin and ymax > ymin:
                            cropped_frame = last_frame[ymin:ymax, xmin:xmax]
                            
                            # Add to image sender queue
                            self.image_sender.add_to_queue(cropped_frame, tracking_id)
                            
                            # Update last alert time
                            stats['last_alert_time'] = current_time

        print(f"\n=== Summary ===")
        print(f"Total persons detected: {len(self.person_stats)}")
        print(f"Total safety violations: {total_violations}")
        print("=== End of Report ===\n")

        # Clear arrays after processing
        self.false_frames_array = []
        self.false_detections_array = []

    def publish_mqtt_message(self, topic, message, binary=False):
        try:
            if binary:
                self.mqtt_client.publish(topic, message)
            else:
                self.mqtt_client.publish(topic, message)
                print(f'MQTT published: {topic} -> {message}')
        except Exception as e:
            print(f'Failed to publish MQTT message: {e}')

    def callback(self, pad, info, user_data):
        buffer = info.get_buffer()
        if not buffer:
            return Gst.PadProbeReturn.OK

        # Extract frame data
        format, width, height = get_caps_from_pad(pad)
        frame = get_numpy_from_buffer(buffer, format, width, height)

        # Convert to BGR for OpenCV processing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get detections
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        # Process detections
        self.process_detections(detections, frame)

        return Gst.PadProbeReturn.OK

def setup_mqtt():
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set(mqtt_username, mqtt_password)
    try:
        mqtt_client.connect(broker, port, 60)
        mqtt_client.loop_start()
        print("MQTT connected successfully.")
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
    return mqtt_client

if __name__ == "__main__":
    # Initialize GStreamer
    Gst.init(None)

    # Initialize MQTT
    mqtt_client = setup_mqtt()

    # Initialize detection callback
    detection_callback = MultiplePersonDetectionCallback(mqtt_client)

    # Create and run CustomGStreamerDetectionApp
    try:
        app = CustomGStreamerDetectionApp(detection_callback.callback, detection_callback)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        mqtt_client.loop_stop() 