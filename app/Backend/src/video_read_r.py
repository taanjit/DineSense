import cv2
import os
import numpy as np
import pandas as pd

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

def load_calibration_data(calibration_path):
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    
    df = pd.read_csv(calibration_path)
    calibration_data = {}
    for _, row in df.iterrows():
        folder = row['folder_name']
        if folder not in calibration_data:
            calibration_data[folder] = {}
        
        # Extract coordinates from string format "(x,y)"
        coords = eval(row['coordinates'])
        calibration_data[folder][row['table_label']] = coords
    return calibration_data

def process_frame(frame, folder_name, calibration_data):
    # Process frame at original resolution
    if folder_name in calibration_data:
        table_positions = calibration_data[folder_name]
        
        for table_label, (center_x, center_y) in table_positions.items():
            # Use proportional size based on frame dimensions
            frame_diagonal = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
            size = int(frame_diagonal * 0.05)  # 5% of diagonal
            
            # Draw hexagon at original coordinates
            draw_hexagon(frame, (center_x, center_y), size)
            
            # Add table label at original coordinates
            label_x = center_x - size
            label_y = center_y - size - 10
            cv2.putText(frame, f"{folder_name}@{table_label}", 
                       (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def get_display_size(frame):
    # Calculate appropriate display size
    target_width = 1920/2  # Full HD width
    target_height = 1080/2  # Full HD height
    
    # Calculate scale to fit in target size
    scale = min(target_width/frame.shape[1], target_height/frame.shape[0])
    
    return (int(frame.shape[1]*scale), int(frame.shape[0]*scale))

def read_video(video_path, calibration_path, camera_name):
    try:
        # Load calibration data
        calibration_data = load_calibration_data(calibration_path)
        print(f"Calibration data loaded successfully from: {calibration_path}")
        
        # Open video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        print(f"Processing video: {os.path.basename(video_path)}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame at original resolution
            processed_frame = process_frame(frame.copy(), camera_name, calibration_data)
            processed_frame = add_label(processed_frame, camera_name)
            
            # Get display size and resize for visualization
            display_size = get_display_size(processed_frame)
            display_frame = cv2.resize(processed_frame, display_size)
            
            # Show video
            cv2.imshow(f'{camera_name} View', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
    
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("Video processing complete")

def main():
    # Base paths
    base_path = "app/Backend"
    video_files_path = os.path.join(base_path, "video_files")
    
    # Camera configuration
    # camera_folders = ['camera_1', 'camera_3', 'camera_6', 'camera_7']
    camera_folders = ['camera_7']
    
    for camera_name in camera_folders:
        # Set up paths
        video_folder = os.path.join(video_files_path, camera_name)
        calibration_file = os.path.join(base_path, f"camera_calibration_{camera_name}.csv")
        
        # Find first video file
        try:
            videos = sorted([v for v in os.listdir(video_folder) 
                           if v.endswith(('.mp4', '.avi'))])
            
            if not videos:
                print(f"No videos found in {video_folder}")
                continue
            
            video_path = os.path.join(video_folder, videos[0])
            
            # Process video
            print(f"\nProcessing {camera_name}")
            read_video(video_path, calibration_file, camera_name)
            
        except Exception as e:
            print(f"Error processing {camera_name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()