import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import argparse
import json
import time
import threading
import paho.mqtt.client as mqtt
import base64
import uuid
from datetime import datetime
from dotenv import load_dotenv

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

load_dotenv()

# MQTT Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', 'localhost')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = "ppe/detections"
STABLE_DURATION = 3  # seconds

class PPETracker:
    def __init__(self, labels_map):
        self.tracked_persons = {}
        self.lock = threading.Lock()
        self.mqtt_client = self.setup_mqtt()
        self.labels_map = labels_map

    def setup_mqtt(self):
        client = mqtt.Client()
        client.username_pw_set(os.getenv('MQTT_USERNAME'), os.getenv('MQTT_PASSWORD'))
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()
        return client

    # Keep all your tracking logic from previous implementation
    # (calculate_iou, assign_apd_to_person, update_tracked_persons, report_violation)

class user_app_callback_class(app_callback_class):
    def __init__(self, labels_map):
        super().__init__()
        self.tracker = PPETracker(labels_map)
        self.frame_counter = 0
        self.labels_map = labels_map

    def _get_ppe_type(self, label):
        """Convert Hailo label to PPE type"""
        label_lower = label.lower()
        if 'helmet' in label_lower:
            return 'helm'
        if 'vest' in label_lower:
            return 'vest'
        if 'boots' in label_lower:
            return 'boots'
        return None

    def process_detections(self, frame, detections):
        persons = []
        ppe_detections = []
        
        for detection in detections:
            label = self.labels_map.get(str(detection.get_class_id()), "unknown")
            confidence = detection.get_confidence()
            bbox = detection.get_bbox()
            
            # Convert normalized coordinates to pixel values
            height, width = frame.shape[:2]
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            
            if label.lower() == "person":
                persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "apd": {"helm": None, "vest": None, "boots": None}
                })
            else:
                ppe_type = self._get_ppe_type(label)
                if ppe_type:
                    is_compliant = "no-" not in label.lower()
                    ppe_detections.append({
                        "type": ppe_type,
                        "bbox": [x1, y1, x2, y2],
                        "is_compliant": is_compliant
                    })
        
        # Assign PPE to persons
        for ppe in ppe_detections:
            for person in persons:
                iou = self.tracker.calculate_iou(person["bbox"], ppe["bbox"])
                if iou > 0.04:
                    person["apd"][ppe["type"]] = ppe["is_compliant"]
                    break
        
        with self.tracker.lock:
            self.tracker.update_tracked_persons(persons, frame)
        
        return persons

    # Keep your draw_detections and get_status_symbol methods

def main(args):
    # Load labels
    with open(args.labels_json) as f:
        labels_data = json.load(f)
        labels_map = {str(k): v for k, v in labels_data.items()}
    
    # Initialize callback with labels
    user_data = user_app_callback_class(labels_map)
    user_data.use_frame = True
    
    # Configure pipeline
    app = GStreamerDetectionApp(
        app_callback=app_callback_class,
        user_data=user_data,
        hef_path=args.hef_path,
        input_source=args.input
    )
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("Stopping...")
        app.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPE Detection with Hailo-8')
    parser.add_argument('--hef-path', required=True, help='Path to HEF file')
    parser.add_argument('--input', required=True, help='Input source (URL, camera, or file path)')
    parser.add_argument('--labels-json', required=True, help='Path to labels JSON file')
    args = parser.parse_args()
    
    main(args)