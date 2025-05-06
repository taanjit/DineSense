import cv2
import os
import sys
import csv
import pandas as pd
import numpy as np
import time  # Import time module for timer functionality
import json  # Import JSON module for data serialization
from ultralytics import YOLO  # Import YOLO from ultralytics

def person_detection(frame, model, conf_threshold=0.1, sitting_ratio_threshold=0.5):
    """
    Detect people in a frame using YOLOv8 model and identify sitting persons.
    
    Args:
        frame: The video frame to process
        model: The loaded YOLO model
        conf_threshold: Confidence threshold for detections (default: 0.1)
        sitting_ratio_threshold: Width/height ratio threshold for sitting detection
        
    Returns:
        List of person detections [x1, y1, x2, y2, confidence, is_sitting, mid_x, mid_y]
    """
    # Run the model on the frame
    results = model(frame, verbose=False)
    
    # Extract person detections (class 0 is person in COCO dataset)
    person_detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get class ID
            cls = int(box.cls[0].item())
            
            # Check if it's a person (class 0)
            if cls == 0:
                # Get confidence
                conf = box.conf[0].item()
                
                # Filter by confidence threshold
                if conf >= conf_threshold:
                    # Get bounding box coordinates (x1, y1, x2, y2 format)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate width and height
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Calculate width-to-height ratio
                    ratio = width / height if height > 0 else 0
                    
                    # Determine if the person is sitting based on ratio
                    is_sitting = ratio >= sitting_ratio_threshold
                    
                    # Calculate midpoint
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    
                    # Add to detections with sitting flag and midpoint
                    person_detections.append([x1, y1, x2, y2, conf, is_sitting, mid_x, mid_y])
    
    return person_detections

def find_nearest_table_center(person_mid_x, person_mid_y, calibration_data):
    """
    Find the center of the nearest table to a person.
    
    Args:
        person_mid_x, person_mid_y: Midpoint coordinates of the person
        calibration_data: List of dictionaries containing table calibration data
        
    Returns:
        tuple: (center_x, center_y) of the nearest table, or None if no tables
    """
    if not calibration_data:
        return None
    
    nearest_center = None
    min_distance = float('inf')
    
    for item in calibration_data:
        try:
            x1 = int(float(item['x1']))
            y1 = int(float(item['y1']))
            x2 = int(float(item['x2']))
            y2 = int(float(item['y2']))
            
            # Calculate table center
            table_center_x = (x1 + x2) // 2
            table_center_y = (y1 + y2) // 2
            
            # Calculate Euclidean distance
            distance = ((person_mid_x - table_center_x) ** 2 + 
                        (person_mid_y - table_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_center = (table_center_x, table_center_y)
                
        except (KeyError, ValueError) as e:
            print(f"Error processing calibration item: {e}")
            continue
    
    return nearest_center

def draw_person_detections(frame, detections, calibration_data):
    """
    Draw person detections on the frame with blue color for sitting persons only.
    Also draw pink lines connecting sitting persons to nearest table centers.
    
    Args:
        frame: The video frame
        detections: List of person detections [x1, y1, x2, y2, confidence, is_sitting, mid_x, mid_y]
        calibration_data: List of dictionaries containing table calibration data
        
    Returns:
        frame: The frame with person detections drawn
    """
    for detection in detections:
        x1, y1, x2, y2, confidence, is_sitting, mid_x, mid_y = detection
        
        # Only draw sitting persons
        if is_sitting:
            # Draw rectangle with blue color (BGR format: 255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label with confidence
            label_text = f"Sitting Person ({confidence:.2f})"
            cv2.putText(frame, label_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw midpoint
            cv2.circle(frame, (mid_x, mid_y), 5, (0, 255, 255), -1)
            
            # Find nearest table center and draw pink line
            nearest_center = find_nearest_table_center(mid_x, mid_y, calibration_data)
            if nearest_center:
                # Draw pink line (BGR format: 180, 105, 255)
                cv2.line(frame, (mid_x, mid_y), nearest_center, (180, 105, 255), 2)
    
    return frame

def extract_folder_name(path):
    """
    Extract the folder name from a path.
    
    Args:
        path (str): The full path
        
    Returns:
        str: The folder name
    """
    # Normalize the path to handle different formats
    normalized_path = os.path.normpath(path)
    
    # Split the path into components
    path_components = normalized_path.split(os.sep)
    
    # Return the last component (folder name)
    return path_components[-1] if path_components[-1] else path_components[-2]

def load_calibration_data(camera_name):
    """
    Load calibration data from CSV file based on camera name.
    
    Args:
        camera_name (str): Name of the camera (e.g., 'camera_1')
        
    Returns:
        list: List of dictionaries containing calibration data
    """
    calibration_file = f"/Users/dranjitta/Documents/Projects/Trae Projects/DineSense/DineSense/app/Backend/camera_calibration_file/{camera_name}_calibration.csv"
    
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file not found: {calibration_file}")
        return []
    
    print(f"Loading calibration data from: {calibration_file}")
    
    try:
        df = pd.read_csv(calibration_file)
        calibration_data = df.to_dict('records')
        print(f"Loaded {len(calibration_data)} calibration records")
        return calibration_data
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        
        calibration_data = []
        try:
            with open(calibration_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    calibration_data.append(row)
            print(f"Loaded {len(calibration_data)} calibration records (fallback method)")
            return calibration_data
        except Exception as e2:
            print(f"Error in fallback loading: {e2}")
            return []

def is_point_inside_box(point_x, point_y, x1, y1, x2, y2):
    """
    Check if a point is inside a bounding box.
    
    Args:
        point_x, point_y: Coordinates of the point
        x1, y1, x2, y2: Coordinates of the bounding box
        
    Returns:
        bool: True if the point is inside the box, False otherwise
    """
    return x1 <= point_x <= x2 and y1 <= point_y <= y2

def count_sitting_persons_in_tables(person_detections, calibration_data):
    """
    Count sitting persons inside each table area.
    
    Args:
        person_detections: List of person detections [x1, y1, x2, y2, confidence, is_sitting, mid_x, mid_y]
        calibration_data: List of dictionaries containing table calibration data
        
    Returns:
        dict: Dictionary mapping table labels to sitting person counts
    """
    table_counts = {}
    
    # Initialize counts for each table
    for item in calibration_data:
        table_counts[item['label']] = 0
    
    # Count sitting persons in each table
    for detection in person_detections:
        _, _, _, _, _, is_sitting, mid_x, mid_y = detection
        
        if is_sitting:
            for item in calibration_data:
                try:
                    x1 = int(float(item['x1']))
                    y1 = int(float(item['y1']))
                    x2 = int(float(item['x2']))
                    y2 = int(float(item['y2']))
                    label = item['label']
                    
                    if is_point_inside_box(mid_x, mid_y, x1, y1, x2, y2):
                        table_counts[label] += 1
                except (KeyError, ValueError) as e:
                    print(f"Error processing calibration item: {e}")
                    continue
    
    return table_counts

def draw_circle(frame, x1, y1, x2, y2, color, thickness=2):
    """
    Draw a circle around a table area.
    
    Args:
        frame: The video frame
        x1, y1, x2, y2: Coordinates of the bounding box
        color: Color of the circle (BGR format)
        thickness: Line thickness
        
    Returns:
        frame: The frame with circle drawn
    """
    # Calculate center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Calculate radius based on the bounding box dimensions
    # Using the larger of width/2 or height/2 to ensure the circle encompasses the table
    width = x2 - x1
    height = y2 - y1
    radius = max(width, height) // 2
    
    # Draw the circle
    cv2.circle(frame, (center_x, center_y), radius, color, thickness)
    
    return frame

def draw_bounding_boxes(frame, calibration_data, table_counts):
    """
    Draw bounding boxes and labels on the frame based on calibration data.
    
    Args:
        frame: The video frame
        calibration_data: List of dictionaries containing calibration data
        table_counts: Dictionary mapping table labels to sitting person counts
        
    Returns:
        frame: The frame with bounding boxes drawn
    """
    for item in calibration_data:
        try:
            x1 = int(float(item['x1']))
            y1 = int(float(item['y1']))
            x2 = int(float(item['x2']))
            y2 = int(float(item['y2']))
            label = item['label']
            confidence = float(item['confidence'])
            
            # Get count for this table
            count = table_counts.get(label, 0)
            
            # Draw circle instead of hexagon
            frame = draw_circle(frame, x1, y1, x2, y2, (0, 255, 0), 2)
            
            # Draw label with confidence and count
            label_text = f"{label} ({confidence:.2f}) count:{count:02d}"
            cv2.putText(frame, label_text, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        except (KeyError, ValueError) as e:
            print(f"Error processing calibration item: {e}")
            continue
            
    return frame

def display_video(video_path):
    """
    Read and display a video file from the specified path with bounding boxes.
    Also outputs JSON data for each processed frame with table occupancy counts.
    
    Args:
        video_path (str): Path to the video file or directory containing video files
    """
    # Extract the folder name (camera name)
    camera_name = extract_folder_name(video_path)
    print(f"Video source folder: {camera_name}")
    
    # Load calibration data
    calibration_data = load_calibration_data(camera_name)
    
    if not calibration_data:
        print("No calibration data available. Proceeding without table bounding boxes.")
    
    # Load YOLO model for person detection
    try:
        # Try to load YOLOv8 model
        model = YOLO("yolo12x.pt")  # Using YOLOv8 nano model for faster processing
        yolo_loaded = True
        print("YOLOv12 model loaded successfully for person detection")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Proceeding without person detection.")
        yolo_loaded = False
    
    # Check if the path is a directory
    if os.path.isdir(video_path):
        # List all files in the directory
        files = os.listdir(video_path)
        video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"No video files found in {video_path}")
            return
        
        # Use the first video file found
        video_file = os.path.join(video_path, video_files[0])
        print(f"Playing video: {video_files[0]}")
    else:
        # Use the provided path directly
        video_file = video_path
    
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {frame_width}x{frame_height} at {fps} FPS")
    
    # Create a window
    window_name = f"Video Display - {camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Initialize timer variables
    last_check_time = time.time()
    check_interval = 10  # Check every 10 seconds
    table_counts = {item['label']: 0 for item in calibration_data}
    
    # Variables to store detection results
    person_detections = []
    frame_count = 0
    
    # Read and display video frames
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # If frame is read correctly ret is True
        if not ret:
            print("End of video or error reading frame")
            break
        
        frame_count += 1
        
        # Get current time
        current_time = time.time()
        time_elapsed = current_time - last_check_time
        
        # Process sitting person detection on every frame
        if yolo_loaded:
            person_detections = person_detection(frame, model, conf_threshold=0.1, sitting_ratio_threshold=0.5)
            
            # Update table counts only every check_interval seconds
            if time_elapsed >= check_interval:
                # Count sitting persons
                sitting_count = sum(1 for detection in person_detections if detection[5])
                
                if sitting_count > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Detected {sitting_count} sitting people in frame")
                    
                    # Count sitting persons in each table
                    table_counts = count_sitting_persons_in_tables(person_detections, calibration_data)
                    
                    # Print table occupancy
                    print(f"[{time.strftime('%H:%M:%S')}] Table occupancy updated:")
                    for table, count in table_counts.items():
                        print(f"  {table}: {count} people")
                    
                    # Create JSON data for this frame
                    frame_data = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "frame_number": frame_count,
                        "camera": camera_name,
                        "tables": []
                    }
                    
                    # Add table data
                    for table, count in table_counts.items():
                        table_data = {
                            "label": table,
                            "occupant_count": count
                        }
                        frame_data["tables"].append(table_data)
                    
                    # Output JSON data
                    json_data = json.dumps(frame_data)
                    print(f"JSON Output: {json_data}")
                
                # Reset timer
                last_check_time = current_time
                print(f"[{time.strftime('%H:%M:%S')}] Table count check completed (next check in {check_interval} seconds)")
            
            # Always draw person detections (visual feedback on every frame)
            if any(detection[5] for detection in person_detections):  # If any sitting persons
                frame = draw_person_detections(frame, person_detections, calibration_data)
        
        # Draw table bounding boxes if calibration data is available
        if calibration_data:
            frame = draw_bounding_boxes(frame, calibration_data, table_counts)
        
        # Display the current time on the frame
        time_text = f"Time: {time.strftime('%H:%M:%S')} (Next count in: {max(0, int(check_interval - time_elapsed))}s)"
        cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow(window_name, frame)
        
        # Press 'q' to exit
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame, camera_name):
    """
    Process a video frame for occupancy detection.
    
    Args:
        frame: The video frame to process
        camera_name: Name of the camera (e.g., 'camera_1')
        
    Returns:
        processed_frame: Frame with bounding boxes and occupancy information
    """
    # Load calibration data for this camera
    calibration_data = load_calibration_data(camera_name)
    
    if not calibration_data:
        print(f"No calibration data available for {camera_name}. Proceeding without table bounding boxes.")
    
    # Initialize table counts
    table_counts = {item['label']: 0 for item in calibration_data} if calibration_data else {}
    
    # Try to load YOLO model for person detection if not already loaded
    try:
        if not hasattr(process_frame, 'model'):
            process_frame.model = YOLO("yolo12x.pt")
            process_frame.yolo_loaded = True
            print("YOLOv12 model loaded successfully for person detection")
        
        # Detect persons in the frame
        person_detections = person_detection(frame, process_frame.model, conf_threshold=0.1, sitting_ratio_threshold=0.5)
        
        # Count sitting persons in each table
        if person_detections and calibration_data:
            table_counts = count_sitting_persons_in_tables(person_detections, calibration_data)
            
            # Draw person detections
            if any(detection[5] for detection in person_detections):  # If any sitting persons
                frame = draw_person_detections(frame, person_detections, calibration_data)
    except Exception as e:
        print(f"Error in person detection: {e}")
        process_frame.yolo_loaded = False
    
    # Draw table bounding boxes if calibration data is available
    if calibration_data:
        processed_frame = draw_bounding_boxes(frame, calibration_data, table_counts)
    else:
        processed_frame = frame.copy()
    
    # Add camera name to the frame
    cv2.putText(processed_frame, f"Camera: {camera_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Generate JSON data for this frame
    frame_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera_name,
        "tables": []
    }
    
    # Add table data
    for table_label, count in table_counts.items():
        table_data = {
            "label": table_label,
            "occupant_count": count
        }
        frame_data["tables"].append(table_data)
    
    # Output JSON data in prettified format
    json_data = json.dumps(frame_data, indent=4)
    print("\nTable Occupancy JSON Data:")
    print(json_data)
    
    return processed_frame
    
    return processed_frame

if __name__ == "__main__":
    # Path to the video directory
    video_directory = "/Users/dranjitta/Documents/Projects/Trae Projects/DineSense/DineSense/app/Backend/video_input/camera_7/"
    
    # Display the video
    display_video(video_directory)
    
    print("Video playback completed")