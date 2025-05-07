import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from occupancy_detection import process_frame

def find_video_files(camera_list, base_dir):
    """Find video files from the given camera list."""
    video_files = {}
    for camera in camera_list:
        camera_dir = os.path.join(base_dir, camera)
        if os.path.exists(camera_dir):
            for file in os.listdir(camera_dir):
                # Check for common video file extensions
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files[camera] = os.path.join(camera_dir, file)
                    break
    return video_files

def extract_and_process_frames(video_files):
    """Extract frames from multiple videos and display them in a tile grid."""
    # Dictionary to store video captures
    caps = {}
    fps_values = {}
    frame_counts = {}
    
    # Open all video files
    for camera, video_path in video_files.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        caps[camera] = cap
        fps_values[camera] = fps
        frame_counts[camera] = frame_count
        
        print(f"Video loaded: {video_path}")
        print(f"FPS: {fps}, Total frames: {frame_count}")
    
    if not caps:
        print("No videos could be opened.")
        return
    
    # Process frames
    frame_interval = 30  # Process every 30 frames
    current_frames = {camera: 0 for camera in caps.keys()}
    
    while True:
        processed_frames = []
        camera_names = []
        occupancy_data = {}
        
        # Get frames from each camera
        for camera, cap in caps.items():
            if current_frames[camera] >= frame_counts[camera]:
                # Reset to beginning if we've reached the end
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frames[camera] = 0
            
            # Set position to current frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frames[camera])
            ret, frame = cap.read()
            
            if ret:
                # Process frame using occupancy detection
                processed_frame, frame_data = process_frame(frame, camera, return_data=True)
                processed_frames.append(processed_frame)
                camera_names.append(camera)
                
                # Store occupancy data
                if frame_data:
                    table_data = {}
                    for table in frame_data.get("tables", []):
                        table_id = f"{camera}@{table['label']}"
                        table_data[table_id] = {"count": table["occupant_count"]}
                    occupancy_data[camera] = table_data
                
                # Increment frame counter
                current_frames[camera] += frame_interval
        
        # Display frames in a grid
        if processed_frames:
            display_frames_grid(processed_frames, camera_names, occupancy_data)
        
        # Check if user wants to exit
        key = cv2.waitKey(1000)  # Wait for 1 second
        if key == ord('q'):
            break
    
    # Release all video captures
    for cap in caps.values():
        cap.release()
    
    cv2.destroyAllWindows()
    print("Video processing completed")

def display_frames_grid(frames, camera_names, occupancy_data=None):
    """Display frames in a grid with occupancy information below each image."""
    num_frames = len(frames)
    if num_frames == 0:
        return
    
    # Determine grid size based on number of frames
    if num_frames <= 2:
        rows, cols = 1, num_frames
    elif num_frames <= 4:
        rows, cols = 2, 2
    elif num_frames <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3  # Maximum 9 frames
    
    # Create a figure with grid
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(rows, cols, figure=fig)
    
    # Fill the grid with available frames
    for i in range(min(rows*cols, num_frames)):
        row, col = divmod(i, cols)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        
        # Set title as camera name
        ax.set_title(f"{camera_names[i]}", fontsize=12, fontweight='bold')
        
        # Add occupancy information below the image if available
        if occupancy_data and camera_names[i] in occupancy_data:
            camera_data = occupancy_data[camera_names[i]]
            occupancy_text = []
            
            # Extract table information for this camera
            for table_id, data in camera_data.items():
                if table_id.startswith(camera_names[i]):
                    table_label = table_id.split('@')[2]
                    count = data['count']
                    status = "Empty" if count == 0 else f"{count} occupant{'s' if count > 1 else ''}"
                    occupancy_text.append(f"Table {table_label}: {status}")
            
            # Add the text below the image
            if occupancy_text:
                ax.text(0.5, -0.1, '\n'.join(occupancy_text), 
                        horizontalalignment='center',
                        verticalalignment='top',
                        transform=ax.transAxes,
                        fontsize=10,
                        bbox=dict(facecolor='lightblue', alpha=0.7, boxstyle='round,pad=0.5'))
        
        ax.axis('off')
    
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)  # Short pause to update display
    plt.close()

def main():
    # Define camera list and base directory
    camera_list = ["camera_7", "camera_1", "camera_3", "camera_1"]
    base_dir = "/Users/dranjitta/Documents/Projects/Trae Projects/DineSense/DineSense/app/Backend/video_input"
    
    # Find video files
    video_files = find_video_files(camera_list, base_dir)
    
    if video_files:
        print(f"Found {len(video_files)} video files")
        extract_and_process_frames(video_files)
    else:
        print("No video files found in the specified camera directories.")

if __name__ == "__main__":
    main()