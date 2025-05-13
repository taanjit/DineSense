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
    Detect table occupancy in a frame using person detection.
    
    Args:
        frame: The input video frame
        camera_name: The name of the camera
        
    Returns:
        frame: The processed frame with bounding boxes
        json_data: JSON data with table occupancy information
    """
    # Load calibration data for the camera
    calibration_data = load_calibration_data(camera_name)
    
    # Draw table bounding boxes
    if calibration_data is not None:
        for _, row in calibration_data.iterrows():
            table_name = row['label']
            try:
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, table_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except ValueError as e:
                print(f"Warning: Invalid coordinate data for table {table_name}: {e}")
                print(f"Table data: {table_name}")
                continue  # Skip this table and continue with the next one
    
    # Detect persons and get table counts
    processed_frame, table_counts, table_occupancy, table_food_served = detect_persons(frame, calibration_data, return_counts=True)
    
    # Create JSON data
    json_data = {
        "folder_name": camera_name,
        "tables": []
    }
    
    for table_name, count in table_counts.items():
        occupancy = table_occupancy[table_name]
        food_served = table_food_served[table_name]  # Get food served status
        
        table_data = {
            "table_name": table_name,
            "occupancy": occupancy,
            "count": count,
            "Food_served": food_served  # Add Food_served field to JSON
        }
        
        json_data["tables"].append(table_data)
    
    return processed_frame, json_data
 