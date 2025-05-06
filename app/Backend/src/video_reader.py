import cv2
import os
import csv
from table_detection import process_frame

def read_videos_from_folder(folder_name, frame_skip_rate=1, display_frames=True):
    """
    Read video files from the specified folder within the calibration_videos directory.
    
    Args:
        folder_name (str): Name of the folder containing the video files
        frame_skip_rate (int): Number of frames to skip (1 means process every frame,
                              2 means process every other frame, etc.)
        display_frames (bool): Whether to display frames while processing
    
    Returns:
        list: List of dictionaries containing video information and frames
    """
    # Construct the path to the calibration videos folder
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "calibration_videos", folder_name)
    
    # Check if the folder exists
    if not os.path.exists(base_path):
        print(f"Error: Folder '{folder_name}' does not exist in calibration_videos directory.")
        return []
    
    # Get all video files in the folder
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(base_path) 
                  if os.path.isfile(os.path.join(base_path, f)) and 
                  any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in folder '{folder_name}'.")
        return []
    
    video_data_list = []
    
    # Create a path to the camera calibration folder
    calibration_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "camera_calibration_file")
    
    # Create the calibration folder if it doesn't exist
    if not os.path.exists(calibration_folder):
        os.makedirs(calibration_folder)
        print(f"Created camera calibration folder: {calibration_folder}")
    
    # Create a CSV file to store table detections in the calibration folder
    csv_path = os.path.join(calibration_folder, f"{folder_name}_calibration.csv")
    
    # Dictionary to store unique table detections (to avoid duplicates)
    unique_tables = {}
    
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(base_path, video_file)
        print(f"Reading video: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            continue
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  - Resolution: {frame_width}x{frame_height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {frame_count}")
        print(f"  - Frame skip rate: {frame_skip_rate} (processing 1 out of every {frame_skip_rate} frames)")
        
        # Read frames with skip rate
        frames = []
        detections_list = []
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames based on the skip rate
            if frame_number % frame_skip_rate == 0:
                # Process frame to detect tables - pass the folder_name here
                processed_frame, detections = process_frame(frame, folder_name)
                
                # Store original frame and detections
                frames.append(frame)
                detections_list.append(detections)
                
                # Store unique table detections
                for detection in detections:
                    if 'table_id' in detection:
                        table_id = detection['table_id']
                        bbox = detection['bbox']
                        
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = bbox
                        
                        # Calculate midpoint, width, and height
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Store the detection with highest confidence if it's a new table or has higher confidence
                        if table_id not in unique_tables or detection['confidence'] > unique_tables[table_id]['confidence']:
                            unique_tables[table_id] = {
                                'label': table_id,
                                'mid_x': mid_x,
                                'mid_y': mid_y,
                                'width': width,
                                'height': height,
                                'confidence': detection['confidence'],
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            }
                
                # Display the processed frame if requested
                if display_frames:
                    # Display the frame with table detections
                    cv2.imshow("Table Detection", processed_frame)
                    
                    # Break the loop if 'q' is pressed
                    key = cv2.waitKey(100)  # Display each frame for 100ms
                    if key == ord('q'):
                        break
                
                # Process every 30th frame to avoid flooding the console
                if len(frames) % 30 == 0:
                    print(f"Processed {len(frames)} frames (at video frame {frame_number})")
                    if detections:
                        print(f"  - Detected {len(detections)} tables in the latest frame")
            
            frame_number += 1
        
        # Release the video capture object
        cap.release()
        
        # Close all OpenCV windows if we were displaying frames
        if display_frames:
            cv2.destroyAllWindows()
            
        print(f"Finished processing {video_file} - Extracted {len(frames)} frames")
        
        # Store video information and frames
        video_data = {
            'filename': video_file,
            'path': video_path,
            'width': frame_width,
            'height': frame_height,
            'fps': fps,
            'frame_count': frame_count,
            'frames': frames,
            'detections': detections_list,
            'frame_skip_rate': frame_skip_rate
        }
        
        video_data_list.append(video_data)
    
    # Write unique table detections to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['label', 'mid_x', 'mid_y', 'width', 'height', 'confidence', 'x1', 'y1', 'x2', 'y2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for table_data in unique_tables.values():
            writer.writerow(table_data)
    
    print(f"Saved {len(unique_tables)} unique table detections to {csv_path}")
    
    return video_data_list

def get_video_info(video_path):
    """
    Get basic information about a video file without reading all frames.
    
    Args:
        video_path (str): Path to the video file
    
    Returns:
        dict: Dictionary containing video information or None if video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Release the video capture object
    cap.release()
    
    return {
        'path': video_path,
        'width': frame_width,
        'height': frame_height,
        'fps': fps,
        'frame_count': frame_count
    }