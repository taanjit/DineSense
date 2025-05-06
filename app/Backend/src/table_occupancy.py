import cv2
import os
import csv
import argparse
import numpy as np
import json
import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

def load_calibration_data(folder_name):
    """
    Load table calibration data from the corresponding CSV file.
    
    Args:
        folder_name (str): Name of the folder/camera (e.g., 'video_3' for 'camera_3_calibration.csv')
        
    Returns:
        list: List of dictionaries containing table information
    """
    # Extract camera number from folder name (e.g., 'video_3' -> 'camera_3')
    camera_name = folder_name.replace('video_', 'camera_')
    
    # Construct the path to the calibration file
    calibration_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "camera_calibration_file",
        f"{camera_name}_calibration.csv"
    )
    
    # Check if the calibration file exists
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file '{calibration_file}' not found.")
        return []
    
    # Read the calibration data
    tables = []
    with open(calibration_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert string values to appropriate types
            table = {
                'label': row['label'],
                'mid_x': float(row['mid_x']),
                'mid_y': float(row['mid_y']),
                'width': float(row['width']),
                'height': float(row['height']),
                'confidence': float(row['confidence'])
            }
            
            # If the CSV contains bounding box coordinates, use them
            if 'x1' in row and 'y1' in row and 'x2' in row and 'y2' in row:
                table['x1'] = float(row['x1'])
                table['y1'] = float(row['y1'])
                table['x2'] = float(row['x2'])
                table['y2'] = float(row['y2'])
            else:
                # Calculate bounding box coordinates from midpoint, width, and height
                table['x1'] = table['mid_x'] - (table['width'] / 2)
                table['y1'] = table['mid_y'] - (table['height'] / 2)
                table['x2'] = table['mid_x'] + (table['width'] / 2)
                table['y2'] = table['mid_y'] + (table['height'] / 2)
            
            # Initialize occupancy status
            table['occupied'] = False
            
            tables.append(table)
    
    print(f"Loaded {len(tables)} tables from calibration file.")
    return tables

def is_point_inside_box(point, box):
    """
    Check if a point is inside a bounding box.
    
    Args:
        point (tuple): Point coordinates (x, y)
        box (tuple): Bounding box coordinates (x1, y1, x2, y2)
        
    Returns:
        bool: True if the point is inside the box, False otherwise
    """
    x, y = point
    x1, y1, x2, y2 = box
    
    return x1 <= x <= x2 and y1 <= y <= y2

def detect_occupancy(frame, tables, camera_name, occupancy_threshold=0.15, motion_threshold=10, blur_size=5, table_occupants=None):
    """
    Detect table occupancy based on pixel changes within table bounding boxes.
    Also detect persons using YOLOv8 model and mark them with green bounding boxes.
    
    Args:
        frame (numpy.ndarray): Current video frame (already resized)
        tables (list): List of table dictionaries with bounding box information
        camera_name (str): Name of the camera for labeling
        occupancy_threshold (float): Threshold for determining occupancy based on variance
        motion_threshold (int): Threshold for motion detection
        blur_size (int): Size of Gaussian blur kernel for preprocessing
        table_occupants (dict): Dictionary to track occupants for each table
        
    Returns:
        tuple: (processed_frame, updated_tables, table_occupants)
    """
    # Make a copy of the frame to draw on
    processed_frame = frame.copy()
    
    # Initialize table_occupants if not provided
    if table_occupants is None:
        table_occupants = {}
        for table in tables:
            table_id = f"{camera_name}@table#{table['label'].split('#')[1]}"
            table_occupants[table_id] = {
                "count": 0
            }
    
    # Load YOLOv8 model for person detection
    try:
        # Check if model is already loaded (can be stored as a global variable for efficiency)
        if not hasattr(detect_occupancy, 'model'):
            model_path = "yolo12x.pt"  # Using a standard YOLOv8 model
            # Load the model using ultralytics YOLO
            detect_occupancy.model = YOLO(model_path)
            # Set confidence threshold
            detect_occupancy.conf_threshold = 0.1
            # Person class ID in COCO dataset is 0
            detect_occupancy.person_class_id = 0
            print(f"Loaded YOLO12x model for person detection")
    except Exception as e:
        print(f"Error loading YOLO12x model: {e}")
        detect_occupancy.model = None
    
    # Reset occupant counts for this frame
    for table_id in table_occupants:
        table_occupants[table_id]["count"] = 0
    
    # Detect persons using YOLOv8 if model is loaded
    if hasattr(detect_occupancy, 'model') and detect_occupancy.model is not None:
        # Run inference
        results = detect_occupancy.model(frame, conf=detect_occupancy.conf_threshold)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if the detected object is a person (class_id = 0)
                if int(box.cls) == detect_occupancy.person_class_id:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence
                    conf = float(box.conf)
                    
                    # Only process detections above threshold
                    if conf >= detect_occupancy.conf_threshold:
                        # Calculate height to width ratio
                        height = y2 - y1
                        width = x2 - x1
                        ratio = width / height if height > 0 else 1.0
                        
                        # Calculate midpoint of the person bounding box
                        mid_x = (x1 + x2) // 2
                        mid_y = (y1 + y2) // 2
                        
                        # Determine if person is sitting based on ratio
                        is_sitting = ratio > 0.5
                        
                        # Set color and label based on sitting status
                        if is_sitting:
                            # Blue for sitting persons
                            color = (255, 0, 0)  # BGR format: Blue
                            label = f"Sitting Person: {conf:.2f}"
                            
                            # Check if midpoint is inside any table
                            for table in tables:
                                table_id = f"{camera_name}@table#{table['label'].split('#')[1]}"
                                table_box = (int(table['x1']), int(table['y1']), 
                                            int(table['x2']), int(table['y2']))
                                
                                # Check if midpoint is inside table
                                if is_point_inside_box((mid_x, mid_y), table_box):
                                    # Increment count for this table
                                    table_occupants[table_id]["count"] += 1
                                    
                                    # Add additional label to show this person is counted
                                    label += f" (Counted at {table_id})"
                                    
                                    # Draw a line from midpoint to table center
                                    table_mid_x = int((table_box[0] + table_box[2]) / 2)
                                    table_mid_y = int((table_box[1] + table_box[3]) / 2)
                                    cv2.line(processed_frame, (mid_x, mid_y), 
                                            (table_mid_x, table_mid_y), (0, 255, 255), 2)
                        else:
                            # Green for standing persons
                            color = (0, 255, 0)  # BGR format: Green
                            label = f"Person: {conf:.2f}"
                        
                        # Draw bounding box
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw midpoint
                        cv2.circle(processed_frame, (mid_x, mid_y), 5, (0, 255, 255), -1)
                        
                        # Add label with confidence
                        cv2.putText(
                            processed_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
    
    # Convert frame to grayscale for simpler processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(gray_frame, (blur_size, blur_size), 0)
    
    # Process each table
    for table in tables:
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(table['x1']), int(table['y1']), int(table['x2']), int(table['y2'])
        
        # Ensure coordinates are within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Extract the table region
        table_region = blurred_frame[y1:y2, x1:x2]
        
        # Skip if the region is empty (out of bounds)
        if table_region.size == 0:
            continue
        
        # Calculate features for occupancy detection
        variance = np.var(table_region)
        mean_value = np.mean(table_region)
        
        # Apply edge detection to find objects on the table
        edges = cv2.Canny(table_region, 50, 150)
        edge_count = np.count_nonzero(edges)
        edge_density = edge_count / table_region.size if table_region.size > 0 else 0
        
        # Determine if table is occupied based on multiple features
        # Higher variance and edge density often indicate presence of people/objects
        is_occupied = (variance > occupancy_threshold or 
                       edge_density > 0.05 or 
                       (mean_value < 100 and variance > occupancy_threshold/2))
        
        # Update table occupancy status
        table['occupied'] = is_occupied
        
        # Get table ID
        table_id = f"{camera_name}@table#{table['label'].split('#')[1]}"
        
        # Draw bounding box with color based on occupancy
        color = (0, 0, 255) if table['occupied'] else (0, 255, 0)  # Red if occupied, green if free
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label with table ID, occupancy status, and count
        status = "Occupied" if table['occupied'] else "Free"
        count = table_occupants[table_id]["count"] if table_id in table_occupants else 0
        label = f"{table_id}: {status} (Count: {count})"
        
        # Calculate text position
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(
            processed_frame, 
            (x1, y1 - text_size[1] - 5), 
            (x1 + text_size[0], y1), 
            color, 
            -1
        )
        
        # Draw text
        cv2.putText(
            processed_frame, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            2
        )
    
    return processed_frame, tables, table_occupants

def process_video(folder_name, output_dir=None, display_frames=True, skip_rate=50):
    """
    Process video files from the specified folder to detect table occupancy.
    
    Args:
        folder_name (str): Name of the folder containing the video files
        output_dir (str): Directory to save output video (optional)
        display_frames (bool): Whether to display frames while processing
        skip_rate (int): Process every Nth frame (default: 50)
    """
    # Construct the path to the video input folder
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "video_input", 
        folder_name
    )
    
    # Check if the folder exists
    if not os.path.exists(base_path):
        print(f"Error: Folder '{folder_name}' does not exist in video_input directory.")
        return
    
    # Get all video files in the folder
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(base_path) 
                  if os.path.isfile(os.path.join(base_path, f)) and 
                  any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in folder '{folder_name}'.")
        return
    
    # Load table calibration data
    tables = load_calibration_data(folder_name)
    
    if not tables:
        print("No table calibration data found. Cannot proceed with occupancy detection.")
        return
    
    # Extract camera name from folder name (e.g., 'video_3' -> 'camera_3')
    camera_name = folder_name.replace('video_', 'camera_')
    
    # Initialize table occupants tracking
    table_occupants = {}
    for table in tables:
        table_id = f"{camera_name}@table#{table['label'].split('#')[1]}"
        table_occupants[table_id] = {
            "count": 0
        }
    
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(base_path, video_file)
        print(f"Processing video: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            continue
        
        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Target resolution
        target_width, target_height = 1280, 720
        
        print(f"Video properties:")
        print(f"  - Original resolution: {original_width}x{original_height}")
        print(f"  - Target resolution: {target_width}x{target_height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {frame_count}")
        
        # Calculate scaling factors for table coordinates
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Scale table coordinates to match the target resolution
        scaled_tables = []
        for table in tables:
            scaled_table = table.copy()
            scaled_table['x1'] = table['x1'] * scale_x
            scaled_table['y1'] = table['y1'] * scale_y
            scaled_table['x2'] = table['x2'] * scale_x
            scaled_table['y2'] = table['y2'] * scale_y
            scaled_table['mid_x'] = table['mid_x'] * scale_x
            scaled_table['mid_y'] = table['mid_y'] * scale_y
            scaled_table['width'] = table['width'] * scale_x
            scaled_table['height'] = table['height'] * scale_y
            scaled_tables.append(scaled_table)
        
        # Create video writer if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"processed_{video_file}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        else:
            out = None
        
        # Process frames
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames according to skip_rate (process every Nth frame)
            if frame_number % skip_rate != 0 and frame_number > 0:
                frame_number += 1
                continue
            
            # Resize frame to target resolution
            resized_frame = cv2.resize(frame, (target_width, target_height))
            
            # Detect table occupancy using the scaled tables
            processed_frame, updated_tables, updated_occupants = detect_occupancy(
                resized_frame, scaled_tables, camera_name, 
                table_occupants=table_occupants
            )
            
            # Update table occupants
            table_occupants = updated_occupants
            
            # Write frame to output video if specified
            if out:
                out.write(processed_frame)
            
            # Display the processed frame if requested
            if display_frames:
                cv2.imshow("Table Occupancy Detection", processed_frame)
                
                # Break the loop if 'q' is pressed
                key = cv2.waitKey(1)  # Display each frame for 1ms
                if key == ord('q'):
                    break
            
            # Print progress every 100 frames
            if frame_number % 100 == 0:
                print(f"Processed frame {frame_number}/{frame_count}")
            
            frame_number += 1
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        # Close all OpenCV windows if we were displaying frames
        if display_frames:
            cv2.destroyAllWindows()
            
        print(f"Finished processing {video_file}")
    
    # Output final occupancy counts as JSON
    occupancy_data = {}
    for table_id, data in table_occupants.items():
        occupancy_data[table_id] = {
            "count": data["count"]
        }
    
    # Print JSON data
    print("\nTable Occupancy Counts:")
    print(json.dumps(occupancy_data, indent=2))
    
    # Save JSON data to file if output directory is specified
    if output_dir:
        json_path = os.path.join(output_dir, f"{camera_name}_occupancy_counts.json")
        with open(json_path, 'w') as f:
            json.dump(occupancy_data, f, indent=2)
        print(f"Saved occupancy counts to {json_path}")
    
    return occupancy_data

def process_all_videos(base_dir=None, output_dir=None, display_frames=True, skip_rate=50, max_workers=4):
    """
    Process all videos in all subfolders of the base directory in parallel.
    
    Args:
        base_dir (str): Base directory containing video subfolders
        output_dir (str): Directory to save output videos (optional)
        display_frames (bool): Whether to display frames while processing
        skip_rate (int): Process every Nth frame (default: 50)
        max_workers (int): Maximum number of parallel workers
    """
    # Use default base directory if not specified
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "video_input"
        )
    
    # Create video processor
    processor = VideoProcessor(
        base_dir=base_dir,
        output_dir=output_dir,
        display_frames=display_frames,
        skip_rate=skip_rate,
        max_workers=max_workers
    )
    
    # Process all folders
    return processor.process_all_folders()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Table occupancy detection from video')
    parser.add_argument('--folder', '-f', type=str, help='Name of a specific folder to process (optional)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output directory')
    parser.add_argument('--no-display', action='store_true', help='Do not display frames while processing')
    parser.add_argument('--skip-rate', '-s', type=int, default=50, help='Process every Nth frame (default: 50)')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Maximum number of parallel workers (default: 4)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process videos
    if args.folder:
        # Process a specific folder
        process_video(args.folder, args.output, not args.no_display, args.skip_rate)
    else:
        # Process all folders in parallel
        process_all_videos(
            output_dir=args.output,
            display_frames=not args.no_display,
            skip_rate=args.skip_rate,
            max_workers=args.workers
        )

if __name__ == "__main__":
    main()


class VideoProcessor:
    def __init__(self, base_dir, output_dir=None, display_frames=True, skip_rate=50, max_workers=4):
        """
        Initialize the video processor for parallel processing.
        
        Args:
            base_dir (str): Base directory containing video subfolders
            output_dir (str): Directory to save output videos (optional)
            display_frames (bool): Whether to display frames while processing
            skip_rate (int): Process every Nth frame (default: 50)
            max_workers (int): Maximum number of parallel workers
        """
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.display_frames = display_frames
        self.skip_rate = skip_rate
        self.max_workers = max_workers
        self.frame_queue = queue.Queue(maxsize=100)  # Queue for frames to display
        self.stop_event = threading.Event()
        self.occupancy_data = {}
        
    def find_video_folders(self):
        """Find all subfolders in the base directory that contain video files."""
        video_folders = []
        
        # List all subfolders in the base directory
        for folder_name in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder_name)
            
            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue
            
            # Check if folder contains video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = [f for f in os.listdir(folder_path) 
                          if os.path.isfile(os.path.join(folder_path, f)) and 
                          any(f.lower().endswith(ext) for ext in video_extensions)]
            
            if video_files:
                video_folders.append(folder_name)
        
        return video_folders
    
    def process_video_folder(self, folder_name):
        """Process all videos in a folder and put frames in the queue."""
        print(f"Starting processing for folder: {folder_name}")
        
        # Load table calibration data
        tables = load_calibration_data(folder_name)
        
        if not tables:
            print(f"No table calibration data found for {folder_name}. Skipping.")
            return
        
        # Extract camera name from folder name (e.g., 'video_3' -> 'camera_3')
        camera_name = folder_name.replace('video_', 'camera_')
        
        # Initialize table occupants tracking
        table_occupants = {}
        for table in tables:
            table_id = f"{camera_name}@table#{table['label'].split('#')[1]}"
            table_occupants[table_id] = {
                "count": 0
            }
        
        # Get all video files in the folder
        folder_path = os.path.join(self.base_dir, folder_name)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = [f for f in os.listdir(folder_path) 
                      if os.path.isfile(os.path.join(folder_path, f)) and 
                      any(f.lower().endswith(ext) for ext in video_extensions)]
        
        # Process each video file
        for video_file in video_files:
            if self.stop_event.is_set():
                break
                
            video_path = os.path.join(folder_path, video_file)
            print(f"Processing video: {video_path}")
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Check if video opened successfully
            if not cap.isOpened():
                print(f"Error: Could not open video {video_file}")
                continue
            
            # Get video properties
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Target resolution
            target_width, target_height = 1280, 720
            
            print(f"Video properties for {folder_name}/{video_file}:")
            print(f"  - Original resolution: {original_width}x{original_height}")
            print(f"  - Target resolution: {target_width}x{target_height}")
            print(f"  - FPS: {fps}")
            print(f"  - Total frames: {frame_count}")
            
            # Calculate scaling factors for table coordinates
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            
            # Scale table coordinates to match the target resolution
            scaled_tables = []
            for table in tables:
                scaled_table = table.copy()
                scaled_table['x1'] = table['x1'] * scale_x
                scaled_table['y1'] = table['y1'] * scale_y
                scaled_table['x2'] = table['x2'] * scale_x
                scaled_table['y2'] = table['y2'] * scale_y
                scaled_table['mid_x'] = table['mid_x'] * scale_x
                scaled_table['mid_y'] = table['mid_y'] * scale_y
                scaled_table['width'] = table['width'] * scale_x
                scaled_table['height'] = table['height'] * scale_y
                scaled_tables.append(scaled_table)
            
            # Create video writer if output directory is specified
            out = None
            if self.output_dir:
                camera_output_dir = os.path.join(self.output_dir, camera_name)
                os.makedirs(camera_output_dir, exist_ok=True)
                output_path = os.path.join(camera_output_dir, f"processed_{video_file}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
            
            # Process frames
            frame_number = 0
            
            while cap.isOpened() and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames according to skip_rate (process every Nth frame)
                if frame_number % self.skip_rate != 0 and frame_number > 0:
                    frame_number += 1
                    continue
                
                # Resize frame to target resolution
                resized_frame = cv2.resize(frame, (target_width, target_height))
                
                # Detect table occupancy using the scaled tables
                processed_frame, updated_tables, updated_occupants = detect_occupancy(
                    resized_frame, scaled_tables, camera_name, 
                    table_occupants=table_occupants
                )
                
                # Update table occupants
                table_occupants = updated_occupants
                
                # Write frame to output video if specified
                if out:
                    out.write(processed_frame)
                
                # Add camera name and frame number as overlay
                cv2.putText(
                    processed_frame,
                    f"{camera_name} - Frame {frame_number}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Put frame in queue for display
                if self.display_frames:
                    try:
                        self.frame_queue.put((camera_name, processed_frame), block=True, timeout=1)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                
                # Print progress every 100 frames
                if frame_number % 100 == 0:
                    print(f"{camera_name}: Processed frame {frame_number}/{frame_count}")
                
                frame_number += 1
            
            # Release resources
            cap.release()
            if out:
                out.release()
                
            print(f"Finished processing {folder_name}/{video_file}")
        
        # Store occupancy data
        self.occupancy_data.update(table_occupants)
        
        # Save JSON data to file if output directory is specified
        if self.output_dir:
            camera_output_dir = os.path.join(self.output_dir, camera_name)
            os.makedirs(camera_output_dir, exist_ok=True)
            json_path = os.path.join(camera_output_dir, f"{camera_name}_occupancy_counts.json")
            
            occupancy_output = {}
            for table_id, data in table_occupants.items():
                occupancy_output[table_id] = {
                    "count": data["count"]
                }
                
            with open(json_path, 'w') as f:
                json.dump(occupancy_output, f, indent=2)
            print(f"Saved occupancy counts to {json_path}")
    
    def display_frames_thread(self):
        """Thread to display frames from the queue in alternating manner."""
        last_camera = None
        
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                # Get frame from queue with timeout
                camera_name, frame = self.frame_queue.get(block=True, timeout=0.1)
                
                # Display the frame
                cv2.imshow("Table Occupancy Detection", frame)
                
                # Break the loop if 'q' is pressed
                key = cv2.waitKey(30)  # Display each frame for 30ms
                if key == ord('q'):
                    self.stop_event.set()
                    break
                
                # Mark task as done
                self.frame_queue.task_done()
                
                # Sleep a bit to allow frames from other cameras to be processed
                # This helps with alternating between cameras
                if camera_name == last_camera:
                    time.sleep(0.01)
                
                last_camera = camera_name
                
            except queue.Empty:
                # No frames in queue, just continue
                continue
            except Exception as e:
                print(f"Error in display thread: {e}")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
    
    def process_all_folders(self):
        """Process all video folders in parallel."""
        # Find all folders with videos
        video_folders = self.find_video_folders()
        
        if not video_folders:
            print("No video folders found in the base directory.")
            return {}
        
        print(f"Found {len(video_folders)} video folders: {', '.join(video_folders)}")
        
        # Start display thread if needed
        if self.display_frames:
            display_thread = threading.Thread(target=self.display_frames_thread)
            display_thread.daemon = True
            display_thread.start()
        
        # Process folders in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(video_folders))) as executor:
            futures = [executor.submit(self.process_video_folder, folder) for folder in video_folders]
            
            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing folder: {e}")
        
        # Signal display thread to stop
        self.stop_event.set()
        
        # Wait for display thread to finish if it's running
        if self.display_frames:
            display_thread.join(timeout=5)
        
        # Output final occupancy counts as JSON
        print("\nTable Occupancy Counts:")
        print(json.dumps(self.occupancy_data, indent=2))
        
        return self.occupancy_data

def process_video(folder_name, output_dir=None, display_frames=True, skip_rate=50):
    """
    Process video files from the specified folder to detect table occupancy.
    
    Args:
        folder_name (str): Name of the folder containing the video files
        output_dir (str): Directory to save output video (optional)
        display_frames (bool): Whether to display frames while processing
        skip_rate (int): Process every Nth frame (default: 50)
    """
    # Construct the path to the video input folder
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "video_input", 
        folder_name
    )
    
    # Check if the folder exists
    if not os.path.exists(base_path):
        print(f"Error: Folder '{folder_name}' does not exist in video_input directory.")
        return
    
    # Get all video files in the folder
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = [f for f in os.listdir(base_path) 
                  if os.path.isfile(os.path.join(base_path, f)) and 
                  any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in folder '{folder_name}'.")
        return
    
    # Load table calibration data
    tables = load_calibration_data(folder_name)
    
    if not tables:
        print("No table calibration data found. Cannot proceed with occupancy detection.")
        return
    
    # Extract camera name from folder name (e.g., 'video_3' -> 'camera_3')
    camera_name = folder_name.replace('video_', 'camera_')
    
    # Initialize table occupants tracking
    table_occupants = {}
    for table in tables:
        table_id = f"{camera_name}@table#{table['label'].split('#')[1]}"
        table_occupants[table_id] = {
            "count": 0
        }
    
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(base_path, video_file)
        print(f"Processing video: {video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            continue
        
        # Get video properties
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Target resolution
        target_width, target_height = 1280, 720
        
        print(f"Video properties:")
        print(f"  - Original resolution: {original_width}x{original_height}")
        print(f"  - Target resolution: {target_width}x{target_height}")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {frame_count}")
        
        # Calculate scaling factors for table coordinates
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        
        # Scale table coordinates to match the target resolution
        scaled_tables = []
        for table in tables:
            scaled_table = table.copy()
            scaled_table['x1'] = table['x1'] * scale_x
            scaled_table['y1'] = table['y1'] * scale_y
            scaled_table['x2'] = table['x2'] * scale_x
            scaled_table['y2'] = table['y2'] * scale_y
            scaled_table['mid_x'] = table['mid_x'] * scale_x
            scaled_table['mid_y'] = table['mid_y'] * scale_y
            scaled_table['width'] = table['width'] * scale_x
            scaled_table['height'] = table['height'] * scale_y
            scaled_tables.append(scaled_table)
        
        # Create video writer if output directory is specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"processed_{video_file}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        else:
            out = None
        
        # Process frames
        frame_number = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames according to skip_rate (process every Nth frame)
            if frame_number % skip_rate != 0 and frame_number > 0:
                frame_number += 1
                continue
            
            # Resize frame to target resolution
            resized_frame = cv2.resize(frame, (target_width, target_height))
            
            # Detect table occupancy using the scaled tables
            processed_frame, updated_tables, updated_occupants = detect_occupancy(
                resized_frame, scaled_tables, camera_name, 
                table_occupants=table_occupants
            )
            
            # Update table occupants
            table_occupants = updated_occupants
            
            # Write frame to output video if specified
            if out:
                out.write(processed_frame)
            
            # Display the processed frame if requested
            if display_frames:
                cv2.imshow("Table Occupancy Detection", processed_frame)
                
                # Break the loop if 'q' is pressed
                key = cv2.waitKey(1)  # Display each frame for 1ms
                if key == ord('q'):
                    break
            
            # Print progress every 100 frames
            if frame_number % 100 == 0:
                print(f"Processed frame {frame_number}/{frame_count}")
            
            frame_number += 1
        
        # Release resources
        cap.release()
        if out:
            out.release()
        
        # Close all OpenCV windows if we were displaying frames
        if display_frames:
            cv2.destroyAllWindows()
            
        print(f"Finished processing {video_file}")
    
    # Output final occupancy counts as JSON
    occupancy_data = {}
    for table_id, data in table_occupants.items():
        occupancy_data[table_id] = {
            "count": data["count"]
        }
    
    # Print JSON data
    print("\nTable Occupancy Counts:")
    print(json.dumps(occupancy_data, indent=2))
    
    # Save JSON data to file if output directory is specified
    if output_dir:
        json_path = os.path.join(output_dir, f"{camera_name}_occupancy_counts.json")
        with open(json_path, 'w') as f:
            json.dump(occupancy_data, f, indent=2)
        print(f"Saved occupancy counts to {json_path}")
    
    return occupancy_data

def process_all_videos(base_dir=None, output_dir=None, display_frames=True, skip_rate=50, max_workers=4):
    """
    Process all videos in all subfolders of the base directory in parallel.
    
    Args:
        base_dir (str): Base directory containing video subfolders
        output_dir (str): Directory to save output videos (optional)
        display_frames (bool): Whether to display frames while processing
        skip_rate (int): Process every Nth frame (default: 50)
        max_workers (int): Maximum number of parallel workers
    """
    # Use default base directory if not specified
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "video_input"
        )
    
    # Create video processor
    processor = VideoProcessor(
        base_dir=base_dir,
        output_dir=output_dir,
        display_frames=display_frames,
        skip_rate=skip_rate,
        max_workers=max_workers
    )
    
    # Process all folders
    return processor.process_all_folders()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Table occupancy detection from video')
    parser.add_argument('--folder', '-f', type=str, help='Name of a specific folder to process (optional)')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output directory')
    parser.add_argument('--no-display', action='store_true', help='Do not display frames while processing')
    parser.add_argument('--skip-rate', '-s', type=int, default=50, help='Process every Nth frame (default: 50)')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Maximum number of parallel workers (default: 4)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process videos
    if args.folder:
        # Process a specific folder
        process_video(args.folder, args.output, not args.no_display, args.skip_rate)
    else:
        # Process all folders in parallel
        process_all_videos(
            output_dir=args.output,
            display_frames=not args.no_display,
            skip_rate=args.skip_rate,
            max_workers=args.workers
        )

if __name__ == "__main__":
    main()