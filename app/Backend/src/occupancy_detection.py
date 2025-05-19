import cv2
import numpy as np
import os
import pandas as pd
import json
from person_detection import detect_persons, initialize_yolo
from collections import defaultdict
import time
 
 
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
    table_names_from_calibration = []
    if calibration_data is not None:
        for _, row in calibration_data.iterrows():
            table_name = row['label']
            table_names_from_calibration.append(table_name)
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
    
    # Get current timestamp in ISO format
    current_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    
    # Initialize counters for summary
    total_tables = len(table_names_from_calibration) if calibration_data is not None else 0
    occupied_tables = 0
    vacant_tables = 0
    total_customers = 0
    tables_with_food = 0
    
    # Process tables and create table data
    table_data = []
    
    # Ensure all tables from calibration data are included
    if calibration_data is not None:
        for table_name in table_names_from_calibration:
            # Default values for tables not in detection results
            count = table_counts.get(table_name, 0)
            occupancy = table_occupancy.get(table_name, False)
            food_served = table_food_served.get(table_name, False)
            
            # Determine status text
            if occupancy and food_served:
                status = "dining"
            elif occupancy:
                status = "occupied"
            else:
                status = "vacant"
            
            # Update counters for summary
            if occupancy:
                occupied_tables += 1
                total_customers += count
            else:
                vacant_tables += 1
            
            if food_served:
                tables_with_food += 1
            
            # Create table entry
            table_entry = {
                "table_name": table_name,
                "occupancy": occupancy,
                "count": count,
                "food_served": food_served,
                "status": status
            }
            
            table_data.append(table_entry)
    
    # Create summary section
    summary = {
        "total_tables": total_tables,
        "occupied_tables": occupied_tables,
        "vacant_tables": vacant_tables,
        "total_customers": total_customers,
        "tables_with_food": tables_with_food
    }
    
    # Create final JSON structure
    json_data = {
        "folder_name": camera_name,
        "timestamp": current_time,
        "tables": table_data,
        "summary": summary
    }
    
    return processed_frame, json_data