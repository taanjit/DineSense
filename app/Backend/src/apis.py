import re
import shutil  # Add this import at the top with other imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Change import to use StaticFiles
from pydantic import BaseModel
from typing import List
import os
import cv2
from datetime import datetime
import pymysql
from itertools import cycle

from main import (
    initialize_yolo, 
    detect_occupancy, 
    get_db_connection, 
    update_seating_layout,
    map_table_name,
    db_config,
    main
)

app = FastAPI(title="DineSense API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Ensure fresh processed_frames directory
processed_frames_path = "./app/Backend/processed_frames"
if os.path.exists(processed_frames_path):
    # Remove existing directory and its contents
    shutil.rmtree(processed_frames_path)
    print(f"Removed existing {processed_frames_path} directory")

# Create fresh directory
os.makedirs(processed_frames_path, exist_ok=True)
print(f"Created fresh {processed_frames_path} directory")

# Mount static files AFTER creating directory
app.mount("/frames", StaticFiles(directory=processed_frames_path), name="frames")

class CameraList(BaseModel):
    cameras: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "cameras": ["camera_7", "camera_3"]
            }
        }

@app.get("/available-cameras")
async def list_available_cameras():
    """List all available camera folders with valid video files"""
    try:
        base_dir = "./app/Backend/video_input"
        available_cameras = []
        
        for item in os.listdir(base_dir):
            camera_dir = os.path.join(base_dir, item)
            if os.path.isdir(camera_dir) and item.startswith("camera_"):
                # Check if directory contains video files
                video_files = [f for f in os.listdir(camera_dir) 
                             if f.endswith(('.mp4', '.avi', '.mov'))]
                if video_files:
                    available_cameras.append(item)
        
        return {
            "available_cameras": sorted(available_cameras),
            "total_count": len(available_cameras)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Reset table history
@app.delete("/reset-table-history")
async def reset_table_history():
    """Reset the table_history table and create a backup"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create backup
        backup_query = """
        CREATE TABLE IF NOT EXISTS table_history_backup 
        SELECT * FROM table_history 
        WHERE occupied_time >= DATE_SUB(NOW(), INTERVAL 1 DAY)
        """
        cursor.execute(backup_query)
        
        # Truncate table_history
        cursor.execute("TRUNCATE TABLE table_history")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            "message": "Table history reset successfully",
            "timestamp": datetime.now(),
            "backup": "Last 24 hours data backed up to table_history_backup"
        }
    
    except pymysql.Error as err:
        raise HTTPException(status_code=500, detail=f"Database error: {str(err)}")


# Add an endpoint to get frame URLs
@app.get("/frame-urls")
async def get_frame_urls():
    """Get URLs for all processed frames"""
    try:
        base_url = "http://192.168.1.112:8452/frames"  # Note the /frames path
        frame_urls = {}
        
        if os.path.exists(processed_frames_path):
            for file in os.listdir(processed_frames_path):
                if file.endswith(('.png', '.jpg')):
                    camera_name = file.split('.')[0]  # Remove extension
                    # Format URL to match your desired pattern
                    frame_urls[camera_name] = f"{base_url}/{file}"
        
        return {
            "frame_urls": frame_urls,
            "count": len(frame_urls),
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting frame URLs: {str(e)}")

@app.post("/run-main")
async def run_main(camera_list: CameraList, background_tasks: BackgroundTasks):
    """Run the main function directly with selected cameras"""
    try:
        if not camera_list.cameras:
            raise HTTPException(status_code=400, detail="No cameras selected")

        # Run main function in background task
        background_tasks.add_task(main, camera_list.cameras)

        return {
            "status": "success",
            "message": "Main processing started",
            "selected_cameras": camera_list.cameras,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apis:app", host="0.0.0.0", port=8452, reload=True)