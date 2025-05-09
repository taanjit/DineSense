import cv2
import numpy as np
import os
import pandas as pd
import json
from person_detection import detect_persons, initialize_yolo
from collections import defaultdict

# Global dictionary to store calibration data for each camera
calibration_cache = {}

def load_calibration_data(camera_name):
    """
    Load calibration data for a specific camera.
    Caches the data to avoid reloading for each frame.
    
    Args:
        camera_name: Name of the camera folder (e.g., 'camera_1')
        
    Returns:
        DataFrame containing calibration data or None if file not found
    """
    # Check if calibration data is already in cache
    if camera_name in calibration_cache:
        return calibration_cache[camera_name]
    
    # If not in cache, load from file
    calibration_file = f"./app/Backend/camera_calibration_file/{camera_name}_calibration.csv"
    
    if os.path.exists(calibration_file):
        try:
            calibration_data = pd.read_csv(calibration_file)
            print(f"Loaded calibration data for {camera_name}")
            # Store in cache for future use
            calibration_cache[camera_name] = calibration_data
            return calibration_data
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            calibration_cache[camera_name] = None
            return None
    else:
        print(f"Calibration file not found: {calibration_file}")
        calibration_cache[camera_name] = None
        return None

def detect_occupancy(frame, camera_name):
    """
    Process a frame to detect occupancy using calibration data and person detection.
    
    Args:
        frame: The input video frame
        camera_name: Name of the camera folder (e.g., 'camera_1')
        
    Returns:
        tuple: (processed_frame, table_status_json)
            - processed_frame: The frame with bounding boxes drawn
            - table_status_json: JSON string with table occupancy data
    """
    # Create a copy of the frame to avoid modifying the original
    processed_frame = frame.copy()
    
    # Initialize table status dictionary
    table_status = {
        "folder_name": camera_name,
        "tables": []
    }
    
    # Load calibration data for this camera (will use cached data if available)
    calibration_data = load_calibration_data(camera_name)
    
    if calibration_data is not None:
        # Draw bounding boxes for each table in the calibration data
        for _, row in calibration_data.iterrows():
            # Extract bounding box coordinates
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            label = row['label']
            confidence = row['confidence']
            
            # Draw the bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label and confidence
            label_text = f"{label} ({confidence:.2f})"
            cv2.putText(processed_frame, label_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        # Add a message if no calibration data is available
        cv2.putText(processed_frame, f"No calibration data for {camera_name}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Detect persons in the frame and get the processed frame and table counts
    processed_frame, table_counts = detect_persons(processed_frame, calibration_data, return_counts=True)
    
    # Populate the table status data
    if calibration_data is not None:
        for _, row in calibration_data.iterrows():
            label = row['label']
            # Extract table number from label (e.g., "camera_7@table#001" -> "001")
            table_parts = label.split('#')
            if len(table_parts) > 1:
                table_number = table_parts[1]
                # Get count for this table (default to 0 if not found)
                count = table_counts.get(label, 0)
                # Add table data to the list
                table_status["tables"].append({
                    "table_name": table_number,
                    "occupancy": count > 0,
                    "count": count
                })
    
    # Convert to JSON string
    table_status_json = json.dumps(table_status, indent=2)
    
    return processed_frame, table_status_json