import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import threading
import time
import json
import base64
import uuid
from datetime import datetime
from dotenv import load_dotenv
import paho.mqtt.client as mqtt

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

load_dotenv()

# Configuration
MQTT_BROKER = os.getenv('MQTT_BROKER', 'cctv.dewika.id')
# MQTT_BROKER = os.getenv('MQTT_BROKER', '192.168.20.2')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', 'fikih/violations')
CONFIDENCE_THRESHOLD = 0.2
STABLE_DURATION = 3  # seconds

# PPE Categories Mapping for Hailo
HAILO_CLASS_MAP = {
    "Safety Vest": "safety helmet",
    "NO-Safety Helmet": "safety helmet",
    # "Hardhat": "safety helmet",
    # "NO-Hardhat": "safety helmet",
    "Safety Vest": "safety vest",
    "NO-Safety Vest": "safety vest",
    "Safety Boots": "safety boots",
    "NO-Safety Boots": "safety boots"
}

class PPETracker:
    def __init__(self):
        self.tracked_persons = {}
        self.lock = threading.Lock()
        self.mqtt_client = self.setup_mqtt()
        
    def setup_mqtt(self):
        client = mqtt.Client()
        client.username_pw_set(os.getenv('MQTT_USERNAME', 'pi'), os.getenv('MQTT_PASSWORD', 'pi'))
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()
        return client

    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        return intersection / (area1 + area2 - intersection + 1e-6)

    def assign_apd_to_person(self, persons, apd_bbox, apd_type, is_compliant):
        for person in persons:
            iou = self.calculate_iou(person["bbox"], apd_bbox)
            if iou > 0.04:
                person["apd"][apd_type] = is_compliant
                break

    def update_tracked_persons(self, persons, frame):
        current_time = time.time()
        new_tracked = {}
        # print(persons)
        
        for person in persons:
            matched_id = None
            for p_id, p_data in self.tracked_persons.items():
                iou = self.calculate_iou(person["bbox"], p_data["bbox"])
                if iou > 0.5:
                    matched_id = p_id
                    break
            
            person_id = matched_id or str(uuid.uuid4())
            
            new_tracked[person_id] = {
                "bbox": person["bbox"],
                "safety helmet": person["apd"]["safety helmet"],
                "safety vest": person["apd"]["safety vest"],
                "safety boots": person["apd"]["safety boots"],
                "first_seen": self.tracked_persons.get(person_id, {}).get("first_seen", current_time),
                "last_seen": current_time
            }

            # print(new_tracked[person_id]["first_seen"] >= STABLE_DURATION)
            
            if (current_time - new_tracked[person_id]["first_seen"] >= STABLE_DURATION):
                if any([new_tracked[person_id][k] is False for k in ['safety helmet', 'safety vest', 'safety boots']]):
                    print('new report')
                    self.report_violation(person_id, new_tracked[person_id], frame)
                    new_tracked.pop(person_id)
        
        self.tracked_persons = new_tracked

    def report_violation(self, person_id, person_data, frame):
        compliance = [person_data["safety helmet"], person_data["safety vest"], person_data["safety boots"]]
        detected = [c for c in compliance if c is not None]
        percent = int(sum(detected) / len(detected) * 100) if detected else 0
        
        has_violation = any(c is False for c in compliance)
        has_unknown = any(c is None for c in compliance)
        status = 0 if has_violation else 2 if has_unknown else 1
        
        x1, y1, x2, y2 = map(int, person_data["bbox"])
        person_img = frame[y1:y2, x1:x2]
        
        _, buffer = cv2.imencode('.jpg', person_img)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        db_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "image": jpg_as_text,
            "percent": percent,
            "safety helmet": int(person_data["safety helmet"] is True) if person_data["safety helmet"] is not None else 2,
            "safety vest": int(person_data["safety vest"] is True) if person_data["safety vest"] is not None else 2,
            "safety boots": int(person_data["safety boots"] is True) if person_data["safety boots"] is not None else 2,
            "status": status,
            "timestamp": time.time(),
            "person_id": person_id
        }

        self.mqtt_client.publish(MQTT_TOPIC, json.dumps(db_data))
        print(f"Violation reported for person {person_id} - Status: {status}")

class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.tracker = PPETracker()
        self.frame_counter = 0

    def process_detections(self, frame, detections):
        persons = []
        ppe_detections = []
        
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            if confidence < CONFIDENCE_THRESHOLD:
                continue
                
            bbox = detection.get_bbox()
            x1 = int(bbox.xmin() * frame.shape[1])
            y1 = int(bbox.ymin() * frame.shape[0])
            x2 = int(bbox.xmax() * frame.shape[1])
            y2 = int(bbox.ymax() * frame.shape[0])
            
            if label == "Person":
                persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "apd": {"safety helmet": None, "safety vest": None, "safety boots": None}
                })
            elif  label in HAILO_CLASS_MAP:
                ppe_type = label.split('-')[-1].lower()
                is_compliant = "NO-" not in label
                ppe_detections.append({
                    "class": label,
                    "bbox": [x1, y1, x2, y2],
                    "is_compliant": is_compliant,
                    "type": ppe_type
                })
                # print(ppe_detections)
        
        for ppe in ppe_detections:
            for person in persons:
                iou = self.tracker.calculate_iou(person["bbox"], ppe["bbox"])
                if iou > 0.04:
                    person["apd"][ppe["type"]] = ppe["is_compliant"]
                    break
        # print(persons)
        self.tracker.update_tracked_persons(persons, frame)
        return persons

    def draw_detections(self, frame, persons):
        for person in persons:
            x1, y1, x2, y2 = person["bbox"]
            compliance = [person["apd"]["safety helmet"], person["apd"]["safety vest"], person["apd"]["safety boots"]]
            
            if any(c is False for c in compliance):
                color = (0, 0, 255)
            elif any(c is None for c in compliance):
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            status_text = f"H: {self.get_status_symbol(person['apd']['safety helmet'])} V: {self.get_status_symbol(person['apd']['safety vest'])} B: {self.get_status_symbol(person['apd']['safety boots'])}"
            cv2.putText(frame, status_text, (x1, y1 - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame

    def get_status_symbol(self, value):
        return '✓' if value else '✗' if value is False else '?'

    def app_callback(self, pad, info, user_data):
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
        self.process_detections(frame, detections)

        return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    user_data = user_app_callback_class()
    user_data.use_frame = True  # Enable frame processing
    app = GStreamerDetectionApp(user_data.app_callback, user_data)
    app.run()