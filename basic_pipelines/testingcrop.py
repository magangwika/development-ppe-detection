import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import time
import datetime

from hailo_apps_infra.hailo_rpi_common import (
    get_caps_from_pad,
    get_numpy_from_buffer,
    app_callback_class,
)
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# Create output directory if it doesn't exist
OUTPUT_DIR = "ujicoba"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class TestingCallback(app_callback_class):
    def __init__(self):
        super().__init__()
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
                self.process_false_frames()
                # Reset counters and arrays
                self.total_frames = 0
                self.false_frames = 0
                self.true_frames = 0
                self.false_frames_array = []
                self.false_detections_array = []
                self.last_reset_time = current_time

    def process_false_frames(self):
        if not self.false_frames_array:
            return

        # Calculate false percentage
        total_frames = self.false_frames + self.true_frames  # Fix total calculation
        false_percentage = (self.false_frames / total_frames * 100) if total_frames > 0 else 0
        print(f"Statistics - Total: {total_frames}, False: {self.false_frames}, "
              f"True: {self.true_frames}, False %: {false_percentage:.2f}%")

        # If false percentage > 50.6%, process the last false frame
        if false_percentage > 50.6 and self.false_frames_array:
            last_frame = self.false_frames_array[-1]
            last_detections = self.false_detections_array[-1]
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save full frame for reference
            full_frame_path = os.path.join(OUTPUT_DIR, f"full_frame_{timestamp}.jpg")
            cv2.imwrite(full_frame_path, last_frame)
            print(f"Saved full frame to {full_frame_path}")
            
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
                        
                        # Print raw relative coordinates
                        print(f"Raw relative coordinates for person {i}: xmin={xmin_rel}, ymin={ymin_rel}, xmax={xmax_rel}, ymax={ymax_rel}")
                        
                        # Convert relative coordinates to pixel coordinates
                        xmin = int(xmin_rel * width)
                        ymin = int(ymin_rel * height)
                        xmax = int(xmax_rel * width)
                        ymax = int(ymax_rel * height)
                        
                        # Print pixel coordinates
                        print(f"Pixel coordinates for person {i}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
                        
                        # Ensure coordinates are within frame bounds
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(width, xmax)
                        ymax = min(height, ymax)
                        
                        # Print adjusted coordinates
                        print(f"Adjusted coordinates for person {i}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
                        
                        # Only crop if the coordinates are valid
                        if xmax > xmin and ymax > ymin:
                            # Add some padding to the bounding box
                            padding = 10
                            xmin = max(0, xmin - padding)
                            ymin = max(0, ymin - padding)
                            xmax = min(width, xmax + padding)
                            ymax = min(height, ymax + padding)
                            
                            # Crop the frame to get the entire person
                            cropped_frame = last_frame[ymin:ymax, xmin:xmax]
                            
                            # Save the cropped image
                            cropped_path = os.path.join(OUTPUT_DIR, f"person_without_pdl_{i}_{timestamp}.jpg")
                            cv2.imwrite(cropped_path, cropped_frame)
                            print(f"Saved person without PDL {i} to {cropped_path}")
                        else:
                            print(f"Invalid bounding box coordinates for person {i}: xmax={xmax} <= xmin={xmin} or ymax={ymax} <= ymin={ymin}")
                    except Exception as e:
                        print(f"Error processing person {i}: {str(e)}")

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

if __name__ == "__main__":
    # Initialize GStreamer
    Gst.init(None)

    # Initialize testing callback
    testing_callback = TestingCallback()

    # Create and run GStreamerDetectionApp
    try:
        app = GStreamerDetectionApp(testing_callback.callback, testing_callback)
        app.run()
    except Exception as e:
        print(f"An error occurred: {e}") 