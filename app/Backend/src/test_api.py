from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
import cv2
import os
from typing import Optional
import io
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Hardcoded default video path
DEFAULT_VIDEO_PATH = "/home/thinkpalm/vs_projects/restaurant_git/DineSense/app/Backend/video_input/camera_7/camera7_modified.mp4"  # Change this to your actual path

@app.get("/frame_stream")
async def stream_frames(
    skip_frames: int = 0,
    quality: int = 80
):
    """Stream frames from the default video file"""
    if not os.path.exists(DEFAULT_VIDEO_PATH):
        return {"error": "Default video not found"}, 404
    
    def generate():
        cap = cv2.VideoCapture(DEFAULT_VIDEO_PATH)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % (skip_frames + 1) == 0:
                # Encode as JPEG (or PNG if preferred)
                _, buffer = cv2.imencode('.jpg', frame, [
                    int(cv2.IMWRITE_JPEG_QUALITY), quality
                ])
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + 
                    buffer.tobytes() + b'\r\n'
                )
            
            frame_count += 1
        
        cap.release()
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test_api:app", host="0.0.0.0", port=8002, reload=True)