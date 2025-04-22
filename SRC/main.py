from SRC import face_attributes
from SRC import station_monitor
from ultralytics import YOLO
import os
import gc

def main():
    video_file_path = "input videos/video_data/video6.mp4"
    csv_file_path = "SRC/calibration_file/updated_station_calibration.csv"
    output_video_path = "SRC/calibration_file/station_data_output.mp4"
    yolo_model_path = "yolo12x.pt"

    # Set environmental variables to limit memory usage
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent TensorFlow from grabbing all GPU memory
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU if available
    
    # Performance optimization settings
    frame_skip = 10  # Process fewer frames (increased from 5 to 10)
    scale = 0.4  # Reduced frame size
    
    # Set global constants in face_attributes module - reduce processing load
    face_attributes.MAX_PEOPLE_PER_FRAME = 6  # Reduced from 10
    face_attributes.MAX_CONCURRENT_THREADS = 2  # Reduced from 4
    
    # Force garbage collection before starting
    gc.collect()
    
    try:
        print("Loading YOLO model...")
        # Load YOLO model with optimized settings
        model = YOLO(yolo_model_path)
        model.conf = 0.5  # Higher confidence threshold
        
        print("Loading station data...")
        # Load station data
        unique_stations = station_monitor.read_unique_stations(csv_file_path)
        
        print("Starting video processing...")
        # Process video
        station_monitor.mark_stations_and_persons_on_video(
            video_file_path, 
            unique_stations, 
            model,
            face_attributes,  # Pass the face_attributes module
            output_video_path,
            frame_skip=frame_skip
        )
        print("Video processing complete.")
    except Exception as e:
        print(f"Process failed with error: {str(e)}")
    finally:
        # Clean up resources
        gc.collect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()