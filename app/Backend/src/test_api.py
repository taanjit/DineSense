from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import os
from typing import Optional, Dict
from fastapi.middleware.cors import CORSMiddleware
import pymysql
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import logging
from occupancy_detection import detect_occupancy
from main import get_db_connection, update_seating_layout, verify_db_update
from table_map import map_table_name
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

VIDEO_PATHS = {
    'camera_7': "/home/thinkpalm/vs_projects/restaurant_git/DineSense/app/Backend/video_input/camera_7/camera7_modified.mp4",
    'camera_3': "/home/thinkpalm/vs_projects/restaurant_git/DineSense/app/Backend/video_input/camera_3/camera3_modified.mp4"
}


# Create thread pool for processing
process_executor = ThreadPoolExecutor(max_workers=2)

# Database connection
try:
    db_conn = get_db_connection()
    logger.info("Database connection established")
except Exception as e:
    logger.error(f"Database connection failed: {e}")
    db_conn = None

def process_frame(frame, camera_name):
    """Process frame and update database"""
    try:
        processed_frame, table_status_json = detect_occupancy(frame, camera_name)
        logger.info(f"\nTable Status JSON for {camera_name}:")
        logger.info(json.dumps(table_status_json, indent=2))

        if table_status_json["tables"] and db_conn:
            for table in table_status_json['tables']:
                logger.info(f"Processing table: {table}")
                match = re.search(r'#(\d+)', table['table_name'])
                if match:
                    table_number = match.group(1)
                    mapped_station = map_table_name(camera_name, table_number)
                    logger.info(f"Mapped station: {mapped_station}")
                    if mapped_station:
                        update_seating_layout(db_conn, mapped_station, table['count'], 
                                           table['occupancy'], table)
                        verify_db_update(db_conn, mapped_station)
                    else:
                        logger.warning(f"Could not map table '{table['table_name']}'")
        
        return processed_frame

    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return frame

def generate_frames(video_path: str, camera_id: str, skip_frames: int = 0, quality: int = 80):
    """Generator function for video frames with processing"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % (skip_frames + 1) == 0:
                # Process frame in thread pool
                processed_frame = process_executor.submit(
                    process_frame, frame, camera_id
                ).result()
                
                # Encode and yield processed frame
                _, buffer = cv2.imencode('.jpg', processed_frame, [
                    int(cv2.IMWRITE_JPEG_QUALITY), quality
                ])
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + 
                    buffer.tobytes() + b'\r\n'
                )
            
            frame_count += 1
    finally:
        cap.release()

@app.get("/stream/{camera_id}")
async def stream_frames(
    camera_id: str,
    skip_frames: int = 0,
    quality: int = 80
):
    """Stream frames from specified camera"""
    if camera_id not in VIDEO_PATHS:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    
    video_path = VIDEO_PATHS[camera_id]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail=f"Video file for {camera_id} not found")
    
    return StreamingResponse(
        generate_frames(video_path, camera_id, skip_frames, quality),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/available-cameras")
async def list_cameras():
    """List all available camera streams"""
    return {
        "cameras": list(VIDEO_PATHS.keys()),
        "urls": {
            camera_id: f"/stream/{camera_id}" 
            for camera_id in VIDEO_PATHS.keys()
        }
    }

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run("test_api:app", host="0.0.0.0", port=8002, reload=True)
    finally:
        if 'db_conn' in globals() and db_conn:
            db_conn.close()
            logger.info("Database connection closed")
        process_executor.shutdown()
        logger.info("Process executor shutdown")