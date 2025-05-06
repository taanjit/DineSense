import cv2
import os
import argparse
import numpy as np
from ultralytics import YOLO

class TableDetectorYOLOv8:
    def __init__(self, model_name="yolo12x.pt", folder_name=None):
        """
        Initialize the table detector with YOLO12 model.
        
        Args:
            model_name (str): Name of the YOLO12 model to use (default: yolo12x.pt)
            folder_name (str): Name of the folder being processed (for table labeling)
        """
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        
        # Set confidence threshold
        self.conf_threshold = 0.05
        
        # Table class ID in COCO dataset is 60
        self.table_class_id = 60
        
        # Store folder name for labeling
        self.folder_name = folder_name if folder_name else "unknown"
        
        print("Table detector initialized successfully")

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1 (list): First bounding box [x1, y1, x2, y2]
            box2 (list): Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Calculate intersection area
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        # Check if boxes overlap
        if x2_min < x1_max or y2_min < y1_max:
            return 0.0
        
        intersection_area = (x2_min - x1_max) * (y2_min - y1_max)
        
        # Calculate areas of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou

    def filter_overlapping_boxes(self, boxes, confidences):
        """
        Filter out overlapping boxes, keeping the one with larger area when overlap > 50%.
        
        Args:
            boxes (list): List of bounding boxes [x1, y1, x2, y2]
            confidences (list): List of confidence scores
            
        Returns:
            tuple: (filtered_boxes, filtered_confidences)
        """
        if len(boxes) <= 1:
            return boxes, confidences
        
        # Calculate areas for all boxes
        areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        
        # Sort boxes by area (largest first)
        indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        
        # Initialize list to keep track of boxes to remove
        to_remove = set()
        
        # Compare each box with others
        for i in range(len(indices)):
            if i in to_remove:
                continue
                
            for j in range(i + 1, len(indices)):
                if j in to_remove:
                    continue
                    
                # Calculate IoU between boxes
                iou = self.calculate_iou(boxes[indices[i]], boxes[indices[j]])
                
                # If overlap is greater than 50%, remove the smaller box
                if iou > 0.1:
                    to_remove.add(j)
        
        # Filter boxes and confidences
        filtered_boxes = [boxes[indices[i]] for i in range(len(indices)) if i not in to_remove]
        filtered_confidences = [confidences[indices[i]] for i in range(len(indices)) if i not in to_remove]
        
        return filtered_boxes, filtered_confidences

    def detect_tables(self, frame):
        """
        Detect tables in the given frame.
        
        Args:
            frame (numpy.ndarray): Input frame (image) for table detection
            
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame: Frame with bounding boxes drawn
                - detections: List of detection results
        """
        # Make a copy of the frame to draw on
        processed_frame = frame.copy()
        
        # Run detection
        results = self.model(frame, conf=self.conf_threshold)
        
        # Process results
        table_boxes = []
        table_confidences = []
        table_classes = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Check if the detected object is a table (class_id = 60)
                if int(box.cls) == self.table_class_id:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get confidence
                    conf = float(box.conf)
                    
                    # Only process detections above threshold
                    if conf >= self.conf_threshold:
                        table_boxes.append([x1, y1, x2, y2])
                        table_confidences.append(conf)
                        table_classes.append('table')
        
        # Filter overlapping boxes
        if table_boxes:
            filtered_boxes, filtered_confidences = self.filter_overlapping_boxes(table_boxes, table_confidences)
        else:
            filtered_boxes, filtered_confidences = [], []
        
        # Create detections list
        detections = []
        
        # Draw bounding boxes for filtered tables
        for i, (box, conf) in enumerate(zip(filtered_boxes, filtered_confidences)):
            x1, y1, x2, y2 = box
            
            # Create table label in the format "folder_name@table#001"
            table_id = f"{self.folder_name}@table#{i+1:03d}"
            
            # Draw green bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with table ID and confidence
            label = f"{table_id}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background rectangle for text
            cv2.rectangle(processed_frame, 
                        (x1, y1 - text_size[1] - 5), 
                        (x1 + text_size[0], y1), 
                        (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(processed_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add detection to list
            detections.append({
                'class': 'table',
                'table_id': table_id,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
        
        return processed_frame, detections

def process_single_frame(image_path, detector, output_dir=None):
    """
    Process a single image file for table detection.
    
    Args:
        image_path (str): Path to the image file
        detector (TableDetectorYOLOv8): Detector instance
        output_dir (str): Directory to save output image (optional)
    """
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Detect tables
    processed_frame, detections = detector.detect_tables(frame)
    
    # Display results
    print(f"Processed {image_path}")
    print(f"Found {len(detections)} tables")
    
    # Display the image
    cv2.imshow("Table Detection", processed_frame)
    cv2.waitKey(0)
    
    # Save the output if requested
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, processed_frame)
        print(f"Saved output to {output_path}")

def process_directory(dir_path, detector, output_dir=None):
    """
    Process all images in a directory for table detection.
    
    Args:
        dir_path (str): Path to the directory containing images
        detector (TableDetectorYOLOv8): Detector instance
        output_dir (str): Directory to save output images (optional)
    """
    # Check if directory exists
    if not os.path.exists(dir_path):
        print(f"Error: Directory {dir_path} does not exist")
        return
    
    # Get all image files in the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(dir_path) 
                  if os.path.isfile(os.path.join(dir_path, f)) and 
                  any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"No image files found in directory {dir_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(dir_path, image_file)
        
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        # Detect tables
        processed_frame, detections = detector.detect_tables(frame)
        
        # Display results
        print(f"Processed {image_file}")
        print(f"Found {len(detections)} tables")
        
        # Display the image (press any key to continue to next image)
        cv2.imshow("Table Detection", processed_frame)
        key = cv2.waitKey(1000)  # Wait for 1 second or key press
        if key == 27:  # ESC key
            break
        
        # Save the output if requested
        if output_dir:
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, processed_frame)
    
    # Close all windows
    cv2.destroyAllWindows()

def process_frame(frame, folder_name=None):
    """
    Process a single frame for table detection.
    
    Args:
        frame (numpy.ndarray): Input frame (image) for table detection
        folder_name (str): Name of the folder being processed (for table labeling)
            
    Returns:
        tuple: (processed_frame, detections)
            - processed_frame: Frame with bounding boxes drawn
            - detections: List of detection results
    """
    # Initialize detector with default settings and folder name
    detector = TableDetectorYOLOv8(folder_name=folder_name)
    
    # Detect tables in the frame
    processed_frame, detections = detector.detect_tables(frame)
    
    return processed_frame, detections

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Table detection using YOLOv8')
    parser.add_argument('input', type=str, help='Path to input image or directory')
    parser.add_argument('--output', '-o', type=str, default=None, help='Path to output directory')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt', help='YOLOv8 model to use')
    parser.add_argument('--conf', '-c', type=float, default=0.1, help='Confidence threshold')
    parser.add_argument('--folder-name', '-f', type=str, default=None, help='Folder name for table labeling')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get folder name from input path if not specified
    folder_name = args.folder_name
    if folder_name is None:
        if os.path.isdir(args.input):
            folder_name = os.path.basename(os.path.normpath(args.input))
        else:
            folder_name = os.path.basename(os.path.dirname(args.input))
    
    # Initialize detector
    detector = TableDetectorYOLOv8(args.model, folder_name)
    detector.conf_threshold = args.conf
    
    # Process input
    if os.path.isfile(args.input):
        process_single_frame(args.input, detector, args.output)
    elif os.path.isdir(args.input):
        process_directory(args.input, detector, args.output)
    else:
        print(f"Error: Input {args.input} is not a valid file or directory")

if __name__ == "__main__":
    main()