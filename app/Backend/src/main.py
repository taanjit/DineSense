import re
import cv2
import os
import numpy as np
import time
import json
from occupancy_detection import detect_occupancy
from person_detection import initialize_yolo
from table_map import map_table_name
from datetime import datetime
import pymysql
from itertools import cycle

db_config = {
    "host": "192.168.3.82",
    "user": "anjit",
    "password": "12aug@1986",
    "database": "DineSense",
    "port": 3306
}

def get_db_connection():
    return pymysql.connect(
        host=db_config['host'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['database'],
        port=db_config['port']
    )

def update_table_history(conn, cursor, station, occupied_time, unoccupied_time):
    # Calculate duration in minutes
    duration = (unoccupied_time - occupied_time).total_seconds() / 60
    
    # Only insert if duration is at least 2 minutes
    if duration >= 2:
        query = """
        INSERT INTO table_history (station, occupied_time, unoccupied_time, duration)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (station, occupied_time, unoccupied_time, duration))
        print(f"Added table history record for {station} with duration {duration:.2f} minutes")
    else:
        print(f"Skipped table history for {station} - duration of {duration:.2f} minutes is less than threshold (2 minutes)")

def update_seating_layout(conn, station, occupant_count, is_occupied, table):
    try:
        cursor = conn.cursor()
        current_time = datetime.now()
        
        # Debug logging
        print(f"\nDEBUG - Updating {station}:")
        print(f"Raw table data: {table}")
        print(f"Count from JSON: {table['count']}")
        print(f"Status: {table['status']}")
        print(f"Food served: {table.get('food_served', False)}")
        
        cursor.execute("""
            SELECT occupant_count, occupied_time, unoccupied_time, food_served, food_served_time 
            FROM seating_layout 
            WHERE station = %s
        """, (station,))
        result = cursor.fetchone()
        
        if result:
            existing_count, existing_occupied_time, existing_unoccupied_time, food_served, food_served_time = result
            actual_count = table['count']
            current_food_status = table.get('food_served', False)
            
            # Handle food service status changes
            if food_served and not current_food_status:  # Food served changing from TRUE to FALSE
                if table['status'] == 'vacant':
                    # Only set unoccupied_time if table is vacant
                    query = """
                    UPDATE seating_layout 
                    SET food_served = FALSE,
                        unoccupied_time = %s
                    WHERE station = %s
                    """
                    cursor.execute(query, (current_time, station))
                    print(f"Table {station} food service ended and vacant - setting unoccupied time")
                else:
                    # If table is not vacant, set unoccupied_time to NULL
                    query = """
                    UPDATE seating_layout 
                    SET food_served = FALSE,
                        unoccupied_time = NULL
                    WHERE station = %s
                    """
                    cursor.execute(query, (station,))
                    print(f"Table {station} food service ended but still occupied - clearing unoccupied time")
            
            # Check for status transition from vacant to occupied
            if table['status'] == 'occupied' and existing_unoccupied_time is not None:
                # Table was vacant and is now occupied - reset and start new cycle
                query = """
                UPDATE seating_layout 
                SET occupant_count = %s,
                    occupied_time = %s,
                    unoccupied_time = NULL,
                    food_served = FALSE,
                    food_served_time = NULL
                WHERE station = %s
                """
                cursor.execute(query, (actual_count, current_time, station))
                print(f"Table {station} transitioned from vacant to occupied")
            
            # Handle regular count updates
            elif actual_count > 0 and existing_occupied_time is None:
                # First time occupation
                query = """
                UPDATE seating_layout 
                SET occupant_count = %s,
                    occupied_time = %s
                WHERE station = %s
                """
                cursor.execute(query, (actual_count, current_time, station))
            
            elif actual_count > 0:
                # Just update count
                query = """
                UPDATE seating_layout 
                SET occupant_count = %s
                WHERE station = %s
                """
                cursor.execute(query, (actual_count, station))
            
            # Handle food service changes
            if not food_served and current_food_status:
                # Food service started
                query = """
                UPDATE seating_layout 
                SET food_served = TRUE,
                    food_served_time = %s
                WHERE station = %s
                """
                cursor.execute(query, (current_time, station))
            
            # Handle transition to vacant
            elif food_served and not current_food_status and table['status'] == 'vacant':
                if not existing_unoccupied_time:  # Only set if not already set
                    query = """
                    UPDATE seating_layout 
                    SET food_served = FALSE,
                        unoccupied_time = %s
                    WHERE station = %s
                    """
                    cursor.execute(query, (current_time, station))
        
        conn.commit()
        verify_db_update(conn, station)

    except pymysql.Error as err:
        print(f"Database error: {err}")
        conn.rollback()
    finally:
        cursor.close()

def find_video_files(camera_list, base_dir):
    """Find video files from the given camera list."""
    video_files = {}
    for camera in camera_list:
        camera_dir = os.path.join(base_dir, camera)
        if os.path.exists(camera_dir):
            for file in os.listdir(camera_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files[camera] = os.path.join(camera_dir, file)
                    break
    return video_files

def verify_db_update(conn, station):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT station, occupant_count, occupied_time, food_served, food_served_time
            FROM seating_layout 
            WHERE station = %s
        """, (station,))
        result = cursor.fetchone()
        print(f"Verification - Current DB state for {station}: {result}")
        cursor.close()
    except pymysql.Error as err:
        print(f"Verification query failed: {err}")


def main(folder_list=None):
    # Initialize YOLO once
    initialize_yolo()

    # Use provided folder list or default
    if folder_list is None:
        folder_list = ["camera_7"]

    base_dir = "./app/Backend/video_input"
    frame_skip_rate = 10
    output_dir = "./app/Backend/processed_frames"
    os.makedirs(output_dir, exist_ok=True)

    # DB connection once for all cameras
    db_conn = get_db_connection()

    # Reset seating layout at startup
    try:
        cursor = db_conn.cursor()
        cursor.execute("""
                    UPDATE seating_layout 
                    SET occupant_count = 0, 
                        occupied_time = NULL, 
                        unoccupied_time = NULL,
                        food_served = FALSE,
                        food_served_time = NULL
                """)       
        db_conn.commit()
        print("Reset all seating layout data at startup")
        cursor.close()
    except pymysql.Error as err:
        print(f"Error resetting seating layout data at startup: {err}")

    # Initialize video captures for all cameras
    captures = {}
    for camera_name in folder_list:
        camera_folder = os.path.join(base_dir, camera_name)
        video_files = [f for f in os.listdir(camera_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            print(f"No video files found in {camera_folder}")
            continue
                
        video_path = os.path.join(camera_folder, video_files[0])
        print(f"\nInitializing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        captures[camera_name] = cap

    frame_counts = {camera: 0 for camera in folder_list}
    try:
        camera_cycle = cycle(list(captures.keys()))
        while captures:
            camera_name = next(camera_cycle)
            if camera_name not in captures:
                continue

            cap = captures[camera_name]
            ret, frame = cap.read()

            if not ret:
                print(f"End of video from {camera_name}")
                cap.release()
                del captures[camera_name]
                continue

            frame_counts[camera_name] += 1
            # print(f"Camera: {camera_name}, Frame: {frame_counts[camera_name]}")

            # Process every nth frame for each camera independently
            if frame_counts[camera_name] % frame_skip_rate != 0:
                continue

            print(f"Processing frame for {camera_name}")
            processed_frame, table_status_json = detect_occupancy(frame, camera_name)
            print(f"\nTable Status JSON for {camera_name}:")
            print(table_status_json)

            output_path = os.path.join(output_dir, f"{camera_name}.png")
            cv2.imwrite(output_path, processed_frame)

            if table_status_json["tables"]:
                for table in table_status_json['tables']:
                    print(f"Checking table: {table}")
                    match = re.search(r'#(\d+)', table['table_name'])
                    if match:
                        table_number = match.group(1)
                    mapped_station = map_table_name(camera_name, table_number)
                    print(f"Mapped station: {mapped_station}")
                    if mapped_station:
                        # Pass the table object to update_seating_layout
                        update_seating_layout(db_conn, mapped_station, table['count'], 
                                           table['occupancy'], table)  # Added table parameter
                        verify_db_update(db_conn, mapped_station)
                    else:
                        print(f"WARNING: Could not map table '{table['table_name']}'")
            else:
                print(f"No tables detected in {camera_name}")

        # After all videos processed, update table history and reset seating layout
        # if db_conn:
        #     try:
        #         cursor = db_conn.cursor()
        #         cursor.execute("SELECT station, occupied_time FROM seating_layout WHERE occupied_time IS NOT NULL")
        #         occupied_tables = cursor.fetchall()
        #         current_time = datetime.now()
        #         for station, occupied_time in occupied_tables:
        #             update_table_history(db_conn, cursor, station, occupied_time, current_time)

        #         cursor.execute("UPDATE seating_layout SET occupant_count = 0, occupied_time = NULL, unoccupied_time = NULL")
        #         db_conn.commit()
        #         print("Reset all seating layout data")
        #         cursor.close()
        #     except pymysql.Error as err:
        #         print(f"Error resetting seating layout data: {err}")

    finally:
        # Clean up resources
        for cap in captures.values():
            cap.release()
        if db_conn:
            db_conn.close()

if __name__ == "__main__":
    main()