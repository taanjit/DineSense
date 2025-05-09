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
    
    # Only insert if duration is at least 3 minutes
    if duration >= 0.1:
        query = """
        INSERT INTO table_history (station, occupied_time, unoccupied_time, duration)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (station, occupied_time, unoccupied_time, duration))
        print(f"Added table history record for {station} with duration {duration:.2f} minutes")
    else:
        print(f"Skipped table history for {station} - duration of {duration:.2f} minutes is less than threshold (3 minutes)")

def update_seating_layout(conn, station, occupant_count, is_occupied):
    try:
        cursor = conn.cursor()
        current_time = datetime.now()
        
        # Fetch the existing occupant count and times
        cursor.execute("SELECT occupant_count, occupied_time, unoccupied_time FROM seating_layout WHERE station = %s", (station,))
        result = cursor.fetchone()
        
        if result:
            existing_count, existing_occupied_time, existing_unoccupied_time = result
            
            if is_occupied:
                if existing_occupied_time is None:
                    # Table just became occupied
                    print(f"Table {station} just became occupied at {current_time}")
                    query = """
                    UPDATE seating_layout 
                    SET occupant_count = %s, occupied_time = %s, unoccupied_time = NULL
                    WHERE station = %s
                    """
                    cursor.execute(query, (occupant_count, current_time, station))
                else:
                    # Table was already occupied, just update the count if needed
                    if occupant_count != existing_count:
                        query = """
                        UPDATE seating_layout 
                        SET occupant_count = %s
                        WHERE station = %s
                        """
                        cursor.execute(query, (occupant_count, station))
            else:  # Table is not occupied
                if existing_occupied_time is not None:
                    # Table just became unoccupied - record history and reset both time fields to NULL
                    print(f"Table {station} just became unoccupied at {current_time}")
                    update_table_history(conn, cursor, station, existing_occupied_time, current_time)
                    query = """
                    UPDATE seating_layout 
                    SET occupant_count = 0, occupied_time = NULL, unoccupied_time = NULL
                    WHERE station = %s
                    """
                    cursor.execute(query, (station,))
                else:
                    # Table was already unoccupied, no need to update anything
                    pass
        else:
            # Insert new record if it doesn't exist
            print(f"Creating new record for table {station}")
            query = """
            INSERT INTO seating_layout (station, occupant_count, occupied_time, unoccupied_time)
            VALUES (%s, %s, %s, NULL)
            """
            cursor.execute(query, (station, occupant_count, current_time if is_occupied else None))
        
        conn.commit()
    except pymysql.Error as err:
        print(f"Database error: {err}")
        conn.rollback()  # Add rollback on error
    finally:
        cursor.close()
        # Don't close the connection here

def main():
    # Initialize YOLO model for person detection
    initialize_yolo()
    
    # Define folder list and base directory
    folder_list = ["camera_7", "camera_3"]
    base_dir = "./app/Backend/video_input"
    
    # Set frame skip rate (default: 24)
    frame_skip_rate = 240
    
    # Get the path to the camera folder
    camera_folder = os.path.join(base_dir, folder_list[0])
    camera_name = folder_list[0]  # Extract camera name for calibration
    
    # Check if the folder exists
    if not os.path.exists(camera_folder):
        print(f"Error: Folder {camera_folder} does not exist")
        return
    
    # Get all video files in the folder
    video_files = [f for f in os.listdir(camera_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {camera_folder}")
        return
    
    # Use the first video file
    video_path = os.path.join(camera_folder, video_files[0])
    print(f"Processing video: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    output_dir = "./app/Backend/processed_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Variables for frame skipping
    frame_count = 0
    
    # Create DB connection outside the loop
    db_conn = get_db_connection()
    
    try:
        # Play the video
        while cap.isOpened():
            ret, frame = cap.read()
            
            # If frame is read correctly ret is True
            if not ret:
                print("End of video reached. Stopping playback.")
                break  # Exit the loop instead of looping back
            
            # Increment frame counter
            frame_count += 1
            
            # Skip frames based on frame_skip_rate
            if frame_count % frame_skip_rate != 0:
                continue
                
            processed_frame, table_status_json = detect_occupancy(frame, camera_name)
            
            # Print the JSON data to terminal
            print("\nTable Status JSON:")
            print(table_status_json)

            # Display the processed frame
            # cv2.imshow('Occupancy Detection', processed_frame)
            output_path = os.path.join(output_dir, f"{camera_name}.png")
            cv2.imwrite(output_path, processed_frame)
            
            table_status = json.loads(table_status_json)
            for table in table_status['tables']:
                mapped_station = map_table_name(table_status['folder_name'], table['table_name'])
                if mapped_station:
                    update_seating_layout(db_conn, mapped_station, table['count'], table['occupancy'])
    finally:
        # Release resources in finally block
        cap.release()
        print("Video processing ended.")
        
        # Reset all seating_layout data when camera is stopped
        if db_conn:
            try:
                cursor = db_conn.cursor()
                
                # Get all tables that still have occupied_time values
                cursor.execute("SELECT station, occupied_time FROM seating_layout WHERE occupied_time IS NOT NULL")
                occupied_tables = cursor.fetchall()
                
                # Update table_history for any tables still marked as occupied
                current_time = datetime.now()
                for station, occupied_time in occupied_tables:
                    # Add to history if occupied for at least 3 minutes
                    update_table_history(db_conn, cursor, station, occupied_time, current_time)
                
                # Reset all tables to unoccupied state (count=0, times=NULL)
                cursor.execute("""
                    UPDATE seating_layout 
                    SET occupant_count = 0, occupied_time = NULL, unoccupied_time = NULL
                """)
                
                db_conn.commit()
                print("Reset all seating layout data")
                cursor.close()
            except pymysql.Error as err:
                print(f"Error resetting seating layout data: {err}")
            finally:
                # Only close DB connection once at the end
                db_conn.close()

if __name__ == "__main__":
    main()