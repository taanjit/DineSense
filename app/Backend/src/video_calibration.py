import cv2
import os
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate intersection over union of two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection / float(area1 + area2 - intersection)
    return iou

def draw_hexagon(img, center, size):
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    points = [(int(center[0] + size*np.cos(angle)), 
              int(center[1] + size*np.sin(angle))) for angle in angles]
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], True, (0, 255, 0), 2)
    return points

def add_label(frame, label):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame

def process_frame_calibration(frame, model, folder_name, calibration_data):
    # Run YOLO detection on original resolution
    results = model(frame, classes=60)
    
    # Get all boxes and sort by area (largest first)
    boxes = []
    for r in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = r
        if conf >= 0.3:  # Increased confidence threshold
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            area = (x2 - x1) * (y2 - y1)
            boxes.append((x1, y1, x2, y2, area, conf))
    
    # Sort boxes by area in descending order
    boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Filter overlapping boxes
    filtered_boxes = []
    for i, box1 in enumerate(boxes):
        is_valid = True
        for box2 in filtered_boxes:
            iou = calculate_iou(box1[:4], box2[:4])
            if iou > 0.5:  # 80% overlap threshold
                is_valid = False
                break
        if is_valid:
            filtered_boxes.append(box1)
    
    # Process filtered boxes
    for idx, (x1, y1, x2, y2, area, conf) in enumerate(filtered_boxes):
        # Calculate center in original coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Store original position with table label
        table_label = f"table_{idx+1:03d}"
        calibration_data[folder_name][table_label].add((center_x, center_y))
        
        # Draw visualization
        size = min((x2 - x1), (y2 - y1)) // 2
        draw_hexagon(frame, (center_x, center_y), size)
        
        # Add label with confidence score
        label = f"{folder_name}@{table_label} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def save_calibration_data(calibration_data, folder_name):
    data = []
    if folder_name in calibration_data:
        for table_label, positions in calibration_data[folder_name].items():
            if positions:
                avg_x = sum(x for x, _ in positions) // len(positions)
                avg_y = sum(y for _, y in positions) // len(positions)
                data.append({
                    'folder_name': folder_name,
                    'table_label': table_label,
                    'coordinates': f"({avg_x},{avg_y})"
                })
    
    output_path = f"app/Backend/camera_calibration_{folder_name}.csv"
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Calibration data saved to {output_path}")

def get_display_size(frame):
    target_width = 1920/2
    target_height = 1080/2
    scale = min(target_width/frame.shape[1], target_height/frame.shape[0])
    return (int(frame.shape[1]*scale), int(frame.shape[0]*scale))

def main():
    model = YOLO('yolo12x.pt')
    calibration_data = defaultdict(lambda: defaultdict(set))
    print("YOLO model initialized successfully")
    
    camera_folders = ['camera_1', 'camera_3', 'camera_6', 'camera_7']
    video_base_path = "app/Backend/video_files"
    
    for camera_folder in camera_folders:
        print(f"\nProcessing {camera_folder}")
        folder_path = os.path.join(video_base_path, camera_folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping...")
            continue
            
        videos = sorted([v for v in os.listdir(folder_path) 
                       if v.endswith(('.mp4', '.avi'))])
        
        if not videos:
            print(f"Warning: No videos found in {camera_folder}, skipping...")
            continue
            
        video_path = os.path.join(folder_path, videos[0])
        cap = cv2.VideoCapture(video_path)
        print(f"Processing video: {videos[0]}")
        
        frame_count = 0
        while frame_count < 50:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = process_frame_calibration(frame.copy(), model, 
                                                     camera_folder, calibration_data)
            processed_frame = add_label(processed_frame, 
                                     f"{camera_folder} (Calibrating: {frame_count+1}/100)")
            
            display_size = get_display_size(processed_frame)
            display_frame = cv2.resize(processed_frame, display_size)
            
            cv2.imshow(f'Calibrating {camera_folder}', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        save_calibration_data(calibration_data, camera_folder)
        cap.release()
        cv2.destroyAllWindows()
        print(f"Calibration complete for {camera_folder}")
    
    print("\nAll cameras calibrated successfully")

if __name__ == "__main__":
    main()