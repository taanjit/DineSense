import cv2
import numpy as np
import os
import pandas as pd
import json
from person_detection import detect_persons, initialize_yolo
from collections import defaultdict
import time
from collections import defaultdict, deque
from table_map import map_table_name
import time
import re
from datetime import datetime
import logging
import json

# Logger setup
logger = logging.getLogger(__name__)
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

# Global state tracker with camera-aware keys
table_state_tracker = defaultdict(lambda: {
    "registry": deque(maxlen=3),
    "active": False,
    "occupied_time": None,
    "unoccupied_time": None,
    "food_served_time": None
})

def detect_occupancy(frame, camera_name):
    calibration_data = load_calibration_data(camera_name)
    table_names_from_calibration = []

    if calibration_data is not None:
        for _, row in calibration_data.iterrows():
            table_name = row['label']
            table_names_from_calibration.append(table_name)
            try:
                x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, table_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except ValueError:
                continue

    processed_frame, table_counts, table_occupancy, table_food_served = detect_persons(
        frame, calibration_data, return_counts=True
    )

    now = datetime.now().isoformat()
    table_data = []
    total_tables = len(table_names_from_calibration)
    occupied_tables = vacant_tables = total_customers = tables_with_food = 0

    for table_name in table_names_from_calibration:
        match = re.search(r'#(\d+)', table_name)
        table_number = match.group(1) if match else None
        mapped_name = map_table_name(camera_name, table_number) if table_number else None
        if not mapped_name:
            continue

        key = f"{camera_name}_{mapped_name}"
        tracker = table_state_tracker[key]

        count = table_counts.get(table_name, 0)
        occupancy = table_occupancy.get(table_name, False)
        food_served = table_food_served.get(table_name, False)

        logger.info(f"model data > {key}, count={count}, occupancy={occupancy}, food_served={food_served}")

        tracker["registry"].append(1 if count > 0 else 0)
        reg_list = list(tracker["registry"])
        reg_str = ''.join(map(str, reg_list))

        # Transition to occupied
        if reg_list == [1, 1, 1] and not tracker["active"]:
            tracker["active"] = True
            tracker["occupied_time"] = now

        # Track food served time
        if food_served and not tracker["food_served_time"]:
            tracker["food_served_time"] = now

        # Transition to vacant
        if reg_list == [0, 0, 0] and tracker["active"]:
            tracker["active"] = False
            tracker["unoccupied_time"] = now

            # Validate duration
            if tracker["occupied_time"] and tracker["unoccupied_time"]:
                try:
                    t_start = datetime.fromisoformat(tracker["occupied_time"])
                    t_end = datetime.fromisoformat(tracker["unoccupied_time"])
                    duration = (t_end - t_start).total_seconds() / 60
                    if duration < 2:
                        logger.info(f"Duration for {key} is {duration:.2f} min < 2 → resetting both timestamps")
                        tracker["occupied_time"] = None
                        tracker["unoccupied_time"] = None
                        tracker["food_served_time"] = None
                    else:
                        logger.info(f"Valid duration ({duration:.2f} min) → keeping occupied_time")

                except Exception as e:
                    logger.warning(f"Failed to compute duration for {key}: {e}")

        # Determine whether to emit JSON
        send_json = tracker["active"] or reg_list == [0, 0, 0]
        logger.info(f"registry details > {key}, registry={reg_str}, sending json={send_json}")
        if not send_json:
            continue

        # Compose output entry
        status = "dining" if occupancy and food_served else "occupied" if occupancy else "vacant"
        if occupancy:
            occupied_tables += 1
            total_customers += count
        else:
            vacant_tables += 1
        if food_served:
            tables_with_food += 1

        table_entry = {
            "table_name": mapped_name,
            "occupancy": occupancy,
            "count": count,
            "food_served": food_served,
            "status": status
        }

        if tracker["occupied_time"] and tracker["active"]:
            table_entry["occupied_time"] = tracker["occupied_time"]
        if tracker["unoccupied_time"] and not tracker["active"]:
            table_entry["unoccupied_time"] = tracker["unoccupied_time"]
        if tracker["food_served_time"] and food_served:
            table_entry["food_served_time"] = tracker["food_served_time"]

        logger.info(f"json data > {json.dumps(table_entry)}")
        table_data.append(table_entry)

    summary = {
        "total_tables": total_tables,
        "occupied_tables": occupied_tables,
        "vacant_tables": vacant_tables,
        "total_customers": total_customers,
        "tables_with_food": tables_with_food
    }

    json_data = {
        "folder_name": camera_name,
        "timestamp": now,
        "tables": table_data,
        "summary": summary
    }

    return processed_frame, json_data