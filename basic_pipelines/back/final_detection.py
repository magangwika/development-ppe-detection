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

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# MQTT topics
MQTT_TOPIC_STATUS = 'cctv/status'
MQTT_TOPIC_IMAGE = 'cctv/image'
MQTT_TOPIC_STATS = 'cctv/stats'

# MQTT Configuration
broker = 'cctv.dewika.id'
port = 1883
mqtt_username = 'pi'
mqtt_password = 'pi'

class FinalDetectionCallback(app_callback_class):
    def __init__(self, mqtt_client):
        super().__init__()
        self.mqtt_client = mqtt_client
        self.false_frames_array = []  # Array to store false frames
        self.false_detections_array = []  # Array to store detections
        self.total_frames = 0
        self.false_frames = 0
        self.true_frames = 0
        self.last_reset_time = time.time()

    def process_detections(self, detections, frame):
        person_present = False
        buruh_present, pdl_present = False, False
        unknown_present = False
        person_detections = []
        unknown_detections = []
        
        # First check for person detection
        for detection in detections:
            label = detection.get_label()
            if label == "person":
                person_present = True
                person_detections.append(detection)
                break

        # Only process other detections if person is present
        if person_present:
            # Analyze other detections
            for detection in detections:
                label = detection.get_label()
                if label == "buruh":
                    buruh_present = True
                elif label == "site-engineer":
                    pdl_present = True
                elif label == "unknown":
                    unknown_present = True
                    unknown_detections.append(detection)

            # Modify logic to consider a false frame if neither vest nor APD is detected
            no_pdl = unknown_present
            apd_complete = buruh_present or pdl_present

            # Count frames for statistics
            self.total_frames += 1
            if no_pdl:
                self.false_frames += 1
                # Store frame and detections in arrays
                self.false_frames_array.append(frame.copy())
                self.false_detections_array.append({
                    'person_detections': person_detections,
                    'unknown_detections': unknown_detections
                })

            if apd_complete:
                self.true_frames += 1

            # Evaluate statistics every 60 seconds
            current_time = time.time()
            if current_time - self.last_reset_time >= 60:
                self.process_and_send_frames()
                # Reset counters and arrays
                self.total_frames = 0
                self.false_frames = 0
                self.true_frames = 0
                self.false_frames_array = []
                self.false_detections_array = []
                self.last_reset_time = current_time

    def process_and_send_frames(self):
        if not self.false_frames_array:
            return

        # Calculate false percentage
        total_frames = self.false_frames + self.true_frames
        false_percentage = (self.false_frames / total_frames * 100) if total_frames > 0 else 0
        
        # Publish statistics
        stats_message = {
            "total_frames": total_frames,
            "false_frames": self.false_frames,
            "true_frames": self.true_frames,
            "false_percentage": false_percentage
        }
        self.publish_mqtt_message(MQTT_TOPIC_STATS, str(stats_message))
        print(f"Published stats: {stats_message}")

        # If false percentage > 50.6%, process and send the last false frame
        if false_percentage > 50.6 and self.false_frames_array:
            last_frame = self.false_frames_array[-1]
            last_detections = self.false_detections_array[-1]
            
            # Publish status
            self.publish_mqtt_message(MQTT_TOPIC_STATUS, "unknown")
            
            # Get frame dimensions
            height, width = last_frame.shape[:2]
            print(f"Frame dimensions: {width}x{height}")
            
            # Process each person detection that has unknown status
            for i, person_det in enumerate(last_detections['person_detections']):
                # Only process if there are unknown detections
                if last_detections['unknown_detections']:
                    try:
                        # Get person bounding box coordinates (relative coordinates 0-1)
                        bbox = person_det.get_bbox()
                        xmin_rel, ymin_rel, xmax_rel, ymax_rel = bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()
                        
                        # Convert relative coordinates to pixel coordinates
                        xmin = int(xmin_rel * width)
                        ymin = int(ymin_rel * height)
                        xmax = int(xmax_rel * width)
                        ymax = int(ymax_rel * height)
                        
                        print(f"Person {i} bbox: ({xmin}, {ymin}) to ({xmax}, {ymax})")
                        
                        # Ensure coordinates are within frame bounds
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(width, xmax)
                        ymax = min(height, ymax)
                        
                        # Only crop if the coordinates are valid
                        if xmax > xmin and ymax > ymin:
                            # Add some padding to the bounding box
                            padding = 10
                            xmin = max(0, xmin - padding)
                            ymin = max(0, ymin - padding)
                            xmax = min(width, xmax + padding)
                            ymax = min(height, ymax + padding)
                            
                            print(f"Cropping with padding: ({xmin}, {ymin}) to ({xmax}, {ymax})")
                            
                            # Crop the frame to get the entire person
                            cropped_frame = last_frame[ymin:ymax, xmin:xmax]
                            print(f"Cropped image size: {cropped_frame.shape}")
                            
                            # Encode and publish the cropped image
                            _, img_encoded = cv2.imencode('.jpg', cropped_frame)
                            img_bytes = img_encoded.tobytes()
                            print(f"Encoded image size: {len(img_bytes)} bytes")
                            
                            self.publish_mqtt_message(MQTT_TOPIC_IMAGE, img_bytes, binary=True)
                            print(f"Published cropped person {i} to MQTT")
                    except Exception as e:
                        print(f"Error processing person {i}: {str(e)}")

    def publish_mqtt_message(self, topic, message, binary=False):
        try:
            if binary:
                # Untuk gambar, tambahkan logging detail
                if topic == MQTT_TOPIC_IMAGE:
                    print(f'Publishing image to MQTT - Size: {len(message)} bytes')
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
    detection_callback = FinalDetectionCallback(mqtt_client)

    # Create and run GStreamerDetectionApp
    try:
        app = GStreamerDetectionApp(detection_callback.callback, detection_callback)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        mqtt_client.loop_stop() 