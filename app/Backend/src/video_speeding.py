import cv2
import os
import glob
import numpy as np
from pathlib import Path
 
def extract_frames_and_create_video(input_path, speed_factor=8):
    """
    Read video files in alphabetical order, extract ALL frames, and create a new video
    with increased playback speed while preserving original resolution and aspect ratio.
    """
    # Handle input path
    if os.path.isfile(input_path):
        video_files = [input_path]
        base_dir = os.path.dirname(input_path)
    else:
        video_files = sorted(glob.glob(os.path.join(input_path, "*.mp4")))
        base_dir = input_path
 
    if not video_files:
        print(f"Error: No video files found at '{input_path}'.")
        return None
 
    # Create frames directory
    frames_dir = os.path.join(base_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
 
    # Clear existing frames
    for existing_frame in glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
        os.remove(existing_frame)
 
    frame_count = 0
    total_frames = 0
    fps, width, height = 30, 640, 480  # Defaults (will be overwritten by first valid video)
 
    # First pass: Count total frames in all videos
    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            total_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
 
    print(f"Total frames to extract: {total_frames}")
    
    # Second pass: Extract all frames
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_file}'.")
            continue
 
        # Read properties from the first video - PRESERVING ORIGINAL RESOLUTION AND ASPECT RATIO
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Original video resolution: {width}x{height}, aspect ratio: {width/height:.2f}")
        print(f"Original video FPS: {fps}")
 
        # Extract ALL frames first
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save every frame with sequential numbering
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            current_frame += 1
            
            # Print progress every 100 frames
            if current_frame % 100 == 0:
                print(f"Extracted {current_frame}/{video_frame_count} frames from current video")
 
        cap.release()
 
    print(f"Total frames extracted: {frame_count}")
    
    # Create output video filename
    file_name = Path(video_files[0]).stem
    output_path = os.path.join(base_dir, f"{file_name}_modified.mp4")
    
    # Create video from frames with speed factor applied
    create_video_from_frames(frames_dir, output_path, fps, (width, height), speed_factor)
    
    return output_path
 
def create_video_from_frames(frames_dir, output_path, fps, frame_size, speed_factor=8):
    """
    Create a video from extracted frames, maintaining original resolution and aspect ratio,
    but applying speed factor by selecting frames.
    """
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    
    if not frame_files:
        print(f"Error: No frames found in '{frames_dir}'.")
        return
    
    # Select frames based on speed factor
    selected_frames = frame_files[::speed_factor]
    
    print(f"Creating output video with resolution: {frame_size[0]}x{frame_size[1]}")
    print(f"Total frames available: {len(frame_files)}")
    print(f"Selected frames for {speed_factor}x speed: {len(selected_frames)}")
    
    # Try different codecs for better compatibility on macOS
    codecs_to_try = [
        ('avc1', 'H.264'),
        ('H264', 'H.264 alternate'),
        ('mp4v', 'MPEG-4')
    ]
    
    out = None
    used_codec = None
    
    for codec, codec_name in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        temp_out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if temp_out.isOpened():
            out = temp_out
            used_codec = codec_name
            print(f"Using codec: {codec} ({codec_name})")
            break
        else:
            temp_out.release()
    
    if out is None:
        print("Error: Could not create video writer with any of the available codecs.")
        return
    
    # Process selected frames
    for i, frame_file in enumerate(selected_frames):
        frame = cv2.imread(frame_file)
        if frame is None:
            print(f"Warning: Could not read frame {frame_file}")
            continue
        
        # Verify frame dimensions match expected output dimensions
        if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
            print(f"Warning: Frame dimensions ({frame.shape[1]}x{frame.shape[0]}) don't match expected dimensions {frame_size}")
        
        # Write frame directly without any modification to maintain resolution
        out.write(frame)
        
        # Print progress every 10% of frames
        if (i + 1) % max(1, len(selected_frames) // 10) == 0:
            print(f"Added {i + 1}/{len(selected_frames)} frames to video ({(i + 1) / len(selected_frames) * 100:.1f}%)")
    
    # Release resources
    out.release()
    
    # Final output check with detailed information
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Video creation complete. Output saved to: {output_path}")
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        print(f"Absolute path: {os.path.abspath(output_path)}")
        
        # Additional verification of output resolution
        verify_cap = cv2.VideoCapture(output_path)
        if verify_cap.isOpened():
            v_width = int(verify_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(verify_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            v_fps = verify_cap.get(cv2.CAP_PROP_FPS)
            verify_cap.release()
            print(f"Video verification: {v_width}x{v_height}, {v_frames} frames, {v_fps} fps")
            print(f"Aspect ratio: {v_width/v_height:.2f}")
            
            # Verify resolution matches original
            if v_width != frame_size[0] or v_height != frame_size[1]:
                print(f"WARNING: Output resolution ({v_width}x{v_height}) doesn't match input ({frame_size[0]}x{frame_size[1]})")
            else:
                print("Resolution and aspect ratio successfully preserved!")
        else:
            print("Warning: Created file exists but cannot be opened as a video")
    else:
        print(f"WARNING: Output file was not created at {output_path}")
 
if __name__ == "__main__":
    input_video_path = "app/Backend/video_input/camera_3/camera_3.mp4"
    speed_factor = 8
    output_path = extract_frames_and_create_video(input_video_path, speed_factor)
    
    if output_path:
        print(f"Successfully created {speed_factor}x speed video at: {output_path}")
        print(f"Absolute path: {os.path.abspath(output_path)}")
        
        # Open the directory containing the output file
        output_dir = os.path.dirname(os.path.abspath(output_path))
        print(f"Output directory: {output_dir}")
        print(f"Files in output directory: {os.listdir(output_dir)}")
    else:
        print("Failed to create modified video.")