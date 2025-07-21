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

# Konfigurasi MQTT
broker = 'cctv.dewika.id'
port = 1883
mqtt_username = 'pi'
mqtt_password = 'pi'

# -----------------------------------------------------------------------------------------------
# User-defined class to handle detections and MQTT
# -----------------------------------------------------------------------------------------------
class UserAppCallback(app_callback_class):
    def __init__(self, mqtt_client):
        super().__init__()
        self.mqtt_client = mqtt_client
        self.last_false_frame = None
        self.last_false_frame_time = time.time()
        self.total_frames = 0
        self.false_frames = 0
        self.true_frames = 0
        self.truefalse_frames = 0
        self.stats_reset_time = time.time()

    def process_detections(self, detections, frame):
        person_present = False
        buruh_present, pdl_present = False, False
        unknown_present = False
        
        # First check for person detection
        for detection in detections:
            label = detection.get_label()
            if label == "person":
                person_present = True
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

            # Modify logic to consider a false frame if neither vest nor APD is detected
            no_pdl = unknown_present
            apd_complete = buruh_present or pdl_present

            # Count frames for statistics
            self.total_frames += 1
            if no_pdl:
                self.false_frames += 1

            if apd_complete:
                self.true_frames += 1

            # Handle no-APD detection
            if no_pdl:
                self.last_false_frame = frame.copy()

            # Evaluate whether to publish or skip MQTT messages
            self.evaluate_and_publish(no_pdl)
        else:
            print("No person detected, skipping frame processing")

    def evaluate_and_publish(self, no_pdl):
        current_time = time.time()
        if current_time - self.last_false_frame_time >= 60:  # Evaluate every 60 seconds
            self.truefalse_frames = self.true_frames + self.false_frames
            false_percentage = (self.false_frames / self.truefalse_frames * 100) if self.truefalse_frames > 0 else 0

            # If false frames exceed 50%, publish MQTT messages
            if false_percentage > 50.6:
                print(f"False frames > 50% ({false_percentage:.2f}%). Sending MQTT messages.")
                self.publish_status_and_frame(no_pdl)
            else:
                print(f"False frames <= 50% ({false_percentage:.2f}%). Skipping MQTT message publishing.")

            # Publish statistics
            stats_message = {
                "total_frames": self.total_frames,
                "true_frames": self.true_frames,
                "false_frames": self.false_frames,
                "true+false": self.truefalse_frames,
                "false_percentage": false_percentage
            }
            self.publish_mqtt_message(MQTT_TOPIC_STATS, str(stats_message))
            print(f"Published stats: {stats_message}")

            # Reset counters
            self.total_frames = 0
            self.false_frames = 0
            self.true_frames = 0
            self.truefalse_frames = 0
            self.last_false_frame_time = current_time

    def publish_status_and_frame(self, no_pdl):
        # Publish status
        self.publish_mqtt_message(MQTT_TOPIC_STATUS, "unknown")

        # Publish the last false frame
        if self.last_false_frame is not None:
            _, img_encoded = cv2.imencode('.jpg', self.last_false_frame)
            img_bytes = img_encoded.tobytes()
            self.publish_mqtt_message(MQTT_TOPIC_IMAGE, img_bytes, binary=True)
            print(f"Published last false frame at {datetime.datetime.now()}")

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

# -----------------------------------------------------------------------------------------------
# MQTT Setup Function
# -----------------------------------------------------------------------------------------------
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

    # Initialize user-defined callback
    user_callback = UserAppCallback(mqtt_client)

    # Create and run GStreamerDetectionApp
    try:
        app = GStreamerDetectionApp(user_callback.callback, user_callback)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        mqtt_client.loop_stop() 