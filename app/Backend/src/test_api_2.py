import cv2
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Generator, Dict, List
import time
import logging
from occupancy_detection1 import detect_occupancy
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import pymysql
from datetime import datetime
from contextlib import asynccontextmanager
import re
from table_map import map_table_name
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    "host": "192.168.3.82",
    "user": "anjit",
    "password": "12aug@1986",
    "database": "DineSense",
    "port": 3306
}

# Create a thread pool executor for running occupancy detection
occupancy_executor = ThreadPoolExecutor(max_workers=2)

# Global dictionary to store the latest occupancy data for each camera
latest_occupancy_data = {}

output_dir = "./app/Backend/processed_frames"
os.makedirs(output_dir, exist_ok=True)

def get_db_connection():
    """Get a connection to the database"""
    try:
        connection = pymysql.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            port=db_config['port']
        )
        logger.info("Database connection established successfully")
        return connection
    except pymysql.Error as err:
        logger.error(f"Database connection error: {err}")
        raise

def verify_db_update(conn, station):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT station, occupant_count, occupied_time, food_served, food_served_time
            FROM seating_layout 
            WHERE station = %s
        """, (station,))
        result = cursor.fetchone()
        logger.info(f"Verification - Current DB state for {station}: {result}")
        cursor.close()
    except pymysql.Error as err:
        logger.info(f"Verification query failed: {err}")


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
        logger.info(f"Added table history record for {station} with duration {duration:.2f} minutes")
    else:
        logger.info(f"Skipped table history for {station} - duration of {duration:.2f} minutes is less than threshold (2 minutes)")

def update_seating_layout(conn, station, occupant_count, is_occupied, table):
    try:
        cursor = conn.cursor()
        current_time = datetime.now()
        
        cursor.execute("""
            SELECT occupant_count, occupied_time, unoccupied_time, food_served, food_served_time 
            FROM seating_layout 
            WHERE station = %s
        """, (station,))
        result = cursor.fetchone()
        
        if result:
            existing_count, existing_occupied_time, existing_unoccupied_time, food_served, food_served_time = result
            actual_count = table['count']  # Get count from JSON
            current_food_status = table.get('food_served', False)
            
            # Always update count to match JSON value
            query = """
            UPDATE seating_layout 
            SET occupant_count = %s
            WHERE station = %s
            """
            cursor.execute(query, (actual_count, station))
            
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
                    # print(f"Table {station} food service ended and vacant - setting unoccupied time")
                else:
                    # If table is not vacant, set unoccupied_time to NULL
                    query = """
                    UPDATE seating_layout 
                    SET food_served = FALSE,
                        unoccupied_time = NULL
                    WHERE station = %s
                    """
                    cursor.execute(query, (station,))
                    # print(f"Table {station} food service ended but still occupied - clearing unoccupied time")
            
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
                # print(f"Table {station} transitioned from vacant to occupied")
            
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
            
            elif actual_count == 0 and existing_count > 0 and not existing_unoccupied_time:
                # Table has become vacant - set unoccupied time
                query = """
                UPDATE seating_layout 
                SET occupant_count = 0,
                    unoccupied_time = %s
                WHERE station = %s
                """
                cursor.execute(query, (current_time, station))
                logger.info(f"Table {station} has become vacant - setting unoccupied time")
                
                # If there was an occupied time, add an entry to table_history
                if existing_occupied_time:
                    update_table_history(conn, cursor, station, existing_occupied_time, current_time)
                    
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
        # Only call verify_db_update once
        verify_db_update(conn, station)

    except pymysql.Error as err:
        logger.error(f"Database error: {err}")
        conn.rollback()
    finally:
        cursor.close()

from contextlib import asynccontextmanager

running = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global running, db_conn
    running = True
    try:
        # Establish database connection
        db_conn = get_db_connection()
        logger.info("Database connection established at startup")
        
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
            logger.info("Reset all seating layout data at startup")
            cursor.close()
        except pymysql.Error as err:
            logger.error(f"Error resetting seating layout data at startup: {err}")
            
    except Exception as e:
        logger.error(f"Failed to connect to the database at startup: {e}")
    
    yield  # Run app

    # After shutdown
    running = False  # Signal threads to stop
    logger.info("Stopping video streams...")
    if db_conn:
        db_conn.close()
        logger.info("Database connection closed")
    occupancy_executor.shutdown(wait=True)
    logger.info("ThreadPoolExecutor shut down")

app = FastAPI(lifespan=lifespan)


# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# List of camera folders to read videos from
folder_list = ["camera_7", "camera_3"]

# Global database connection
db_conn = None

def get_video_path(camera_folder):
    """
    Get the path to a video file based on the camera folder name
    
    Args:
        camera_folder: Name of the camera folder
    """
    # Path to the video file
    video_path = f"../video_input/{camera_folder}/{camera_folder}.mp4"
    
    # Get the absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_video_path = os.path.join(current_dir, video_path)
    
    # Log the path for debugging
    logger.info(f"Video path for {camera_folder}: {absolute_video_path}")
    
    # Check if file exists
    if not os.path.isfile(absolute_video_path):
        logger.error(f"Video file not found at {absolute_video_path}")
        raise FileNotFoundError(f"Video file not found at {absolute_video_path}")
    
    return absolute_video_path


def run_occupancy_detection(frame, camera_name):
    """Run occupancy detection and update database with results"""
    try:
        processed_frame, json_data = detect_occupancy(frame, camera_name)

        output_path = os.path.join(output_dir, f"{camera_name}.png")
        cv2.imwrite(output_path, processed_frame)

        tables = json_data.get("tables", [])

        if db_conn and tables:
            cursor = db_conn.cursor()

            for table in tables:
                station = table["table_name"]  # This is T1–T23

                # Use get with fallback to None for optional fields
                occupied_time = table.get("occupied_time")
                unoccupied_time = table.get("unoccupied_time")
                food_served_time = table.get("food_served_time")

                cursor.execute("""
                    UPDATE seating_layout
                    SET occupant_count = %s,
                        occupied_time = %s,
                        unoccupied_time = %s,
                        food_served = %s,
                        food_served_time = %s
                    WHERE station = %s
                """, (
                    table["count"],
                    occupied_time,
                    unoccupied_time,
                    table["food_served"],
                    food_served_time,
                    station
                ))

            db_conn.commit()
            cursor.close()

    except Exception as e:
        logger.error(f"Error in occupancy detection for {camera_name}: {str(e)}")

 
def generate_frames(video_path: str, frame_skip: int = 0) -> Generator[bytes, None, None]:
    global running
    """
    Generator function that yields video frames as JPEG bytes
    
    Args:
        video_path: Path to the video file
        frame_skip: Number of frames to skip between each yielded frame (0 = no skip)
    """
    try:
        # Check if the file exists
        if not os.path.isfile(video_path):
            logger.error(f"Error: Video file not found at {video_path}")
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nVideo file not found\r\n'
            return
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            logger.error(f"Error: Could not open video file at {video_path}")
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCould not open video file\r\n'
            return
        
        logger.info(f"Streaming video: {video_path} with frame_skip={frame_skip}")
        
        frame_count = 0
        skip_counter = 0
        
        while running:  # Changed from while True to while running
            # Read a frame
            success, frame = cap.read()
            
            # If frame is read correctly, success is True
            if not success:
                logger.info("End of video - stopping stream")
                break  # Changed from continue to break
            
            # Skip frames if needed
            skip_counter += 1
            if frame_skip > 0 and skip_counter % (frame_skip + 1) != 0:
                continue
            
            # Encode the frame as JPEG
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                # Yield the frame bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
                if frame_count % 100 == 0:
                    camera_name = os.path.basename(os.path.dirname(video_path))
                    logger.info(f"Processing {frame_count} frame for {camera_name} occupancy Detection")
                    
                    # Submit occupancy detection to run in a separate thread
                    # Make a copy of the frame to avoid issues with frame being reused
                    occupancy_executor.submit(run_occupancy_detection, frame.copy(), camera_name)
                
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                logger.error(f"Error encoding frame: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
            logger.info("Video capture released gracefully")

@app.get("/video-feed/{camera_folder}")
async def video_feed(camera_folder: str, skip: int = Query(0, description="Number of frames to skip between each frame")):
    """
    Endpoint that returns a streaming response of video frames for a specific camera
    
    Parameters:
        camera_folder: Name of the camera folder
        skip: Number of frames to skip between each frame (0 = no skip)
    """
    try:
        if camera_folder not in folder_list:
            raise HTTPException(status_code=404, detail=f"Camera {camera_folder} not found in folder list")
            
        video_path = get_video_path(camera_folder)
        return StreamingResponse(
            content=generate_frames(video_path, frame_skip=skip),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in video_feed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-cameras")
async def available_cameras():
    """
    Endpoint that returns the list of available camera folders
    """
    return {"cameras": folder_list}

@app.get("/occupancy-data/{camera_folder}")
async def get_occupancy_data(camera_folder: str):
    """
    Endpoint that returns the latest occupancy data for a specific camera
    
    Parameters:
        camera_folder: Name of the camera folder
    """
    try:
        if camera_folder not in folder_list:
            raise HTTPException(status_code=404, detail=f"Camera {camera_folder} not found in folder list")
            
        # Return the latest data for this camera if available
        if camera_folder in latest_occupancy_data:
            return latest_occupancy_data[camera_folder]
        else:
            # Return empty data structure if no data is available yet
            return {
                "folder_name": camera_folder,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                "tables": [],
                "summary": {
                    "total_tables": 0,
                    "occupied_tables": 0,
                    "vacant_tables": 0,
                    "total_customers": 0,
                    "tables_with_food": 0
                }
            }
    except Exception as e:
        logger.error(f"Error in get_occupancy_data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the FastAPI server
    logger.info("Starting FastAPI server on http://0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)