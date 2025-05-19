import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import defaultdict
 
# Global variable to store the YOLO models
yolo_model = None
yolo_pose_model = None
custom_plate_model = None  # New global variable for custom plate model
yolo_loaded = False
yolo_pose_loaded = False
custom_plate_loaded = False  # New flag for custom plate model
 
def initialize_models():
    """
    Initialize YOLO models for person detection and pose estimation.
    Also initialize custom model for plate detection.
    This function should be called once at the beginning of the program.
    
    Returns:
        bool: True if models loaded successfully, False otherwise
    """
    global yolo_model, yolo_pose_model, custom_plate_model, yolo_loaded, yolo_pose_loaded, custom_plate_loaded
    
    # Load YOLO model for person detection
    try:
        # Try to load YOLOv8 model
        yolo_model = YOLO("yolo12x.pt")  # Using YOLOv8 nano model for faster processing
        yolo_loaded = True
        print("YOLOv12 model loaded successfully for person detection")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        print("Proceeding without person detection.")
        yolo_loaded = False
    
    # Load YOLO model for pose detection
    try:
        # Try to load YOLO11n-pose model
        yolo_pose_model = YOLO("yolo11n-pose.pt")  # Using YOLOv11n model for pose detection
        yolo_pose_loaded = True
        print("YOLO11n-pose model loaded successfully for pose detection")
    except Exception as e:
        print(f"Error loading YOLO11n-pose model: {e}")
        print("Proceeding without pose detection.")
        yolo_pose_loaded = False
    
    # Load custom model for plate detection
    try:
        # Replace "custom_plate_model.pt" with the actual path to your custom model
        custom_plate_model = YOLO("best.pt")  # Using custom model for plate detection
        custom_plate_loaded = True
        print("Custom plate detection model loaded successfully")
    except Exception as e:
        print(f"Error loading custom plate model: {e}")
        print("Proceeding without custom plate detection.")
        custom_plate_loaded = False
    
    return yolo_loaded
 
def initialize_yolo():
    """
    Initialize the YOLO model for person detection.
    This function should be called once at the beginning of the program.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    return initialize_models()
 
def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: (x, y) coordinates of first point
        point2: (x, y) coordinates of second point
        
    Returns:
        float: Euclidean distance between the points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
 
def detect_persons(frame, calibration_data=None, confidence_threshold=0.05, return_counts=False):
    """
    Detect persons in a frame using YOLO model and mark only sitting persons with blue bounding boxes.
    Use YOLO11n-pose to detect if a person is front-facing based on shoulder positions.
    Map each sitting person to the nearest table from calibration data.
    Also detect food plates using a custom model and mark them with orange bounding boxes.
    
    Args:
        frame: The input video frame
        calibration_data: Optional calibration data for tables
        confidence_threshold: Minimum confidence level for detection (default: 0.05 or 5%)
        return_counts: Whether to return table counts (default: False)
        
    Returns:
        If return_counts is False:
            frame: The processed frame with bounding boxes
        If return_counts is True:
            frame: The processed frame with bounding boxes
            table_counts: Dictionary with table counts
            table_occupancy: Dictionary with table occupancy status
            table_food_served: Dictionary with food served status for each table
    """
    global yolo_model, yolo_pose_model, custom_plate_model, yolo_loaded, yolo_pose_loaded, custom_plate_loaded
    
    # Create copies of the frame for drawing
    processed_frame = frame.copy()
    
    # Initialize dictionaries to store counts and occupancy status
    table_counts = {}
    table_occupancy = {}
    table_food_served = {}  # New dictionary to track food served status
    table_plate_counts = {}  # New dictionary to track plate counts per table
    
    # Dictionary to track if food has been served to a table (persistent across frames)
    # This needs to be a global variable to persist across function calls
    global table_food_served_state
    if 'table_food_served_state' not in globals():
        table_food_served_state = {}
    
    # Initialize all tables with 0 count and False occupancy
    if calibration_data is not None:
        for _, row in calibration_data.iterrows():
            table_name = row['label']
            table_counts[table_name] = 0
            table_occupancy[table_name] = False
            table_food_served[table_name] = False  # Initialize food served status as False
            table_plate_counts[table_name] = 0  # Initialize plate count as 0
            
            # Initialize the persistent state if not already present
            if table_name not in table_food_served_state:
                table_food_served_state[table_name] = False
    
    # Check if YOLO model is loaded
    if not yolo_loaded or yolo_model is None:
        # Try to initialize YOLO if not already loaded
        if not initialize_models():
            # Add a message if YOLO model is not available
            cv2.putText(processed_frame, "YOLO model not available for person detection",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return processed_frame
    
    # Extract table centers from calibration data if available
    table_centers = []
    table_labels = []
    table_boxes = []
    if calibration_data is not None:
        for _, row in calibration_data.iterrows():
            # Extract table bounding box coordinates
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            # Calculate table center
            table_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            table_centers.append(table_center)
            table_labels.append(row['label'])
            table_boxes.append((x1, y1, x2, y2))
    
    # Dictionary to count persons at each table
    table_counts = defaultdict(int)
    
    # Run YOLO11n-pose detection on the entire frame if available
    pose_results = None
    if yolo_pose_loaded and yolo_pose_model is not None:
        pose_results = yolo_pose_model(frame, verbose=False)
    
    try:
        # Run YOLO detection on the frame for persons (class 0)
        person_results = yolo_model(frame, classes=[0], verbose=False)  # Class 0 is 'person' in COCO dataset
        
        # Run detection for plates using custom model instead of YOLO12x
        plate_results = None
        if custom_plate_loaded and custom_plate_model is not None:
            # Use custom model for plate detection - this will detect all classes in the custom model
            plate_results = custom_plate_model(frame, verbose=False)
        else:
            # Fallback to using YOLO12x if custom model is not available
            plate_classes = list(range(39, 41))  # Original classes 39-40
            plate_results = yolo_model(frame, classes=plate_classes, verbose=False)
            cv2.putText(processed_frame, "Using fallback YOLO12x for plate detection",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Count detected plates
        plate_count = 0
        
        # Process plate detection results
        if plate_results and len(plate_results) > 0:
            plate_result = plate_results[0]
            plate_boxes = plate_result.boxes
            
            # Draw bounding boxes for each detected plate with confidence > threshold
            for box in plate_boxes:
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get class ID
                class_id = int(box.cls[0])
                
                # Only process detections above the confidence threshold
                if confidence >= confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate plate center
                    plate_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Draw the plate bounding box in orange
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange color in BGR
                    
                    # Add label with class ID and confidence
                    label_text = f"Plate-{class_id} ({confidence:.2f})"
                    cv2.putText(processed_frame, label_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    
                    plate_count += 1
                    
                    # Map plate to nearest table if calibration data is available
                    if table_centers:
                        nearest_table = None
                        min_distance = float('inf')
                        
                        # First check if plate is inside any table bounding box
                        for i, (tx1, ty1, tx2, ty2) in enumerate(table_boxes):
                            if (tx1 <= plate_center[0] <= tx2 and ty1 <= plate_center[1] <= ty2):
                                nearest_table = table_labels[i]
                                break
                        
                        # If not inside any table, find the nearest table
                        if nearest_table is None:
                            for i, table_center in enumerate(table_centers):
                                distance = calculate_distance(plate_center, table_center)
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_table = table_labels[i]
                        
                        # If a nearest table was found, increment its plate count
                        if nearest_table is not None:
                            table_plate_counts[nearest_table] += 1
                            
                            # Draw a line from plate center to nearest table center
                            nearest_table_idx = table_labels.index(nearest_table)
                            cv2.line(processed_frame, plate_center, table_centers[nearest_table_idx],
                                    (0, 165, 255), 1)  # Orange line
                            
                            # Add table assignment text
                            table_text = f"Table: {nearest_table}"
                            cv2.putText(processed_frame, table_text, (x1, y1-30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        # Process person detection results
        if person_results and len(person_results) > 0:
            # Get the first result (should be only one since we're processing a single frame)
            result = person_results[0]
            
            # Extract detections
            boxes = result.boxes
            
            # Count detected sitting persons
            sitting_count = 0
            front_facing_count = 0
            total_count = 0
            
            # Draw bounding boxes for each detected sitting person with confidence > threshold
            for box in boxes:
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Only process detections above the confidence threshold
                if confidence >= confidence_threshold:
                    total_count += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Calculate width and height
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Calculate width-to-height ratio
                    ratio = width / height if height > 0 else 0
                    
                    # Only mark sitting persons (ratio > 0.51)
                    if ratio > 0.51:
                        # Calculate center of the person
                        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        
                        # Check if this person is front-facing based on shoulder positions from YOLO11n-pose
                        is_front_facing = False
                        right_shoulder_x = None
                        left_shoulder_x = None
                        
                        # Process pose detection results if available
                        if pose_results and len(pose_results) > 0:
                            pose_result = pose_results[0]
                            
                            # Check if keypoints are available
                            if hasattr(pose_result, 'keypoints') and pose_result.keypoints is not None:
                                keypoints = pose_result.keypoints.data
                                
                                # For each detected pose
                                for kp in keypoints:
                                    # Make sure the keypoints array has enough elements
                                    if len(kp) > 6:  # We need at least indices 5 and 6 for shoulders
                                        # Check if this pose is within the current person's bounding box
                                        # Make sure the keypoints at indices 5 and 6 are valid
                                        if kp[5][0] > 0 and kp[5][1] > 0 and kp[6][0] > 0 and kp[6][1] > 0:
                                            pose_center_x = (kp[5][0] + kp[6][0]) / 2  # Average of shoulders
                                            pose_center_y = (kp[5][1] + kp[6][1]) / 2
                                            
                                            # If pose center is within the person bounding box
                                            if (x1 <= pose_center_x <= x2 and y1 <= pose_center_y <= y2):
                                                # Right shoulder is keypoint 6 in YOLO pose
                                                right_shoulder_x = int(kp[6][0])
                                                right_shoulder_y = int(kp[6][1])
                                                cv2.circle(processed_frame, (right_shoulder_x, right_shoulder_y),
                                                          5, (0, 255, 0), -1)
                                                
                                                # Left shoulder is keypoint 5 in YOLO pose
                                                left_shoulder_x = int(kp[5][0])
                                                left_shoulder_y = int(kp[5][1])
                                                cv2.circle(processed_frame, (left_shoulder_x, left_shoulder_y),
                                                          5, (0, 255, 0), -1)
                                                
                                                # Check if right shoulder x is less than left shoulder x
                                                if right_shoulder_x < left_shoulder_x:
                                                    is_front_facing = True
                                                    front_facing_count += 1
                                                
                                                break  # Found the pose for this person
                        
                        # Draw the person bounding box in pink if front-facing, blue otherwise
                        box_color = (255, 0, 255) if is_front_facing else (255, 0, 0)  # Pink or Blue
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Map the center of the person with a circle
                        cv2.circle(processed_frame, person_center, 5, (255, 0, 255), -1)
                        
                        # Find the nearest table if calibration data is available
                        nearest_table = None
                        min_distance = float('inf')
                        
                        if table_centers:
                            # Different mapping logic based on front-facing status
                            if is_front_facing:
                                # For front-facing persons: Map to nearest table with y-value higher than person's midpoint
                                person_mid_y = person_center[1]
                                valid_tables = []
                                valid_table_centers = []
                                valid_table_indices = []
                                
                                for i, table_center in enumerate(table_centers):
                                    table_y = table_center[1]
                                    # Check if table's y is greater than person's y (lower in the image)
                                    if table_y > person_mid_y:
                                        valid_tables.append(table_labels[i])
                                        valid_table_centers.append(table_center)
                                        valid_table_indices.append(i)
                                
                                # Find the nearest valid table
                                if valid_table_centers:
                                    for i, table_center in enumerate(valid_table_centers):
                                        distance = calculate_distance(person_center, table_center)
                                        if distance < min_distance:
                                            min_distance = distance
                                            nearest_table = valid_tables[i]
                                    
                                    # Get the original index of the nearest table
                                    nearest_table_idx = table_labels.index(nearest_table)
                                else:
                                    # No valid tables found (all tables are above the person)
                                    cv2.putText(processed_frame, "No valid tables", (x1, y1-30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            else:
                                # For non-front-facing persons: Map to nearest table without y-value constraint
                                for i, table_center in enumerate(table_centers):
                                    distance = calculate_distance(person_center, table_center)
                                    if distance < min_distance:
                                        min_distance = distance
                                        nearest_table = table_labels[i]
                                
                                # Get the index of the nearest table
                                nearest_table_idx = table_labels.index(nearest_table)
                            
                            # If a nearest table was found, draw the connection and update counts
                            if nearest_table is not None:
                                # Draw a line from person center to nearest table center
                                cv2.line(processed_frame, person_center, table_centers[nearest_table_idx],
                                        (0, 255, 255), 2)
                                
                                # Add label with confidence, front-facing status, and nearest table
                                facing_text = "Front-Facing" if is_front_facing else "Not Front-Facing"
                                label_text = f"Person ({confidence:.2f}, {facing_text})"
                                cv2.putText(processed_frame, label_text, (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                                
                                table_text = f"Table: {nearest_table}"
                                cv2.putText(processed_frame, table_text, (x1, y1-30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                
                                # Increment the count for this table
                                table_counts[nearest_table] += 1
                        else:
                            # Add label with confidence and front-facing status only
                            facing_text = "Front-Facing" if is_front_facing else "Not Front-Facing"
                            label_text = f"Person ({confidence:.2f}, {facing_text})"
                            cv2.putText(processed_frame, label_text, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                        
                        sitting_count += 1
            
            # Add total count to the frame
            cv2.putText(processed_frame, f"Sitting: {sitting_count}, Front-Facing: {front_facing_count}, Plates: {plate_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # # Add model type information
            # model_info = "Using: Custom Model" if custom_plate_loaded else "Using: YOLO12x (fallback)"
            # cv2.putText(processed_frame, f"Plate Detection: {model_info}",
            #             (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Display table counts on the frame
            y_offset = 120
            cv2.putText(processed_frame, "Table Counts:", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw table bounding boxes and display counts
            for i, (table_label, count) in enumerate(sorted(table_counts.items())):
                # Find the index of this table
                if table_label in table_labels:
                    table_idx = table_labels.index(table_label)
                    
                    # Get the table bounding box
                    tx1, ty1, tx2, ty2 = table_boxes[table_idx]
                    
                    # Draw the table bounding box
                    cv2.rectangle(processed_frame, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
                    
                    # Display the table label and count near the table
                    table_center = table_centers[table_idx]
                    count_text = f"{table_label}: {count}"
                    cv2.putText(processed_frame, count_text,
                                (table_center[0] - 20, table_center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                
                # Also display in the table counts section
                y_offset += 25
                cv2.putText(processed_frame, f"{table_label}: {count} persons", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # If no tables have been assigned, show a message
            if not table_counts and table_centers:
                y_offset += 25
                cv2.putText(processed_frame, "No persons mapped to tables", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    except Exception as e:
        print(f"Error during person detection: {e}")
        cv2.putText(processed_frame, f"Error in person detection: {str(e)[:50]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # At the end of the function, modify the return statement:
    # Update food served status based on occupancy and plate count
    for table_name in table_counts.keys():
        # Make sure the table exists in all dictionaries before accessing
        if table_name not in table_occupancy:
            table_occupancy[table_name] = False
        if table_name not in table_plate_counts:
            table_plate_counts[table_name] = 0
        if table_name not in table_food_served:
            table_food_served[table_name] = False
            
        # Set food served to True if table is occupied and has at least one plate
        table_food_served[table_name] = table_occupancy[table_name] and table_plate_counts[table_name] > 0
    
        # After processing all detections, update occupancy status and food served status
        for table_name, count in table_counts.items():
            # Set occupancy to True if at least one person is assigned to the table
            table_occupancy[table_name] = count > 0
            
            # Make sure the table exists in all dictionaries before accessing
            if table_name not in table_plate_counts:
                table_plate_counts[table_name] = 0
            
            # Check if the condition for food served is met (sitting person count > 0 and plate count > 2)
            if count > 0 and table_plate_counts[table_name] >= 2:
                # Set the persistent state to True
                table_food_served_state[table_name] = True
            
            # Reset the persistent state if no persons are at the table
            if count == 0:
                table_food_served_state[table_name] = False
            
            # Set the current food served status based on the persistent state
            table_food_served[table_name] = table_food_served_state[table_name]
    
    if return_counts:
        return processed_frame, table_counts, table_occupancy, table_food_served
    else:
        return processed_frame
 
 