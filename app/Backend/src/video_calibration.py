import os
import sys
import argparse
import cv2
from video_reader import read_videos_from_folder

def process_videos_for_calibration(folder_name, frame_skip_rate):
    """
    Process videos from the specified folder for calibration.
    
    Args:
        folder_name (str): Name of the folder containing the video files
        frame_skip_rate (int): Number of frames to skip during processing
    
    Returns:
        None
    """
    # Get videos from the folder using the imported function
    video_data_list = read_videos_from_folder(folder_name, frame_skip_rate, display_frames=True)
    
    if not video_data_list:
        print("No videos to process for calibration.")
        return
    
    # Process each video for calibration
    for video_data in video_data_list:
        print(f"Calibrating using video: {video_data['filename']}")
        
        # Here you would implement your calibration logic
        # For example:
        # - Detect chessboard corners
        # - Calculate camera matrix
        # - Compute distortion coefficients
        
        # For demonstration, we're just printing some information
        print(f"  - Processed {len(video_data['frames'])} frames")
        print(f"  - Resolution: {video_data['width']}x{video_data['height']}")
        print(f"  - Using 1 out of every {video_data['frame_skip_rate']} frames")
        
        # The frame display code has been moved to the video_reader.py file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Video calibration tool')
    parser.add_argument('folder_name', type=str, help='Name of the folder containing calibration videos')
    parser.add_argument('--frame-skip', '-s', type=int, default=1, 
                        help='Number of frames to skip (1 means process every frame, 2 means every other frame, etc.)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate frame skip rate
    if args.frame_skip < 1:
        print("Error: Frame skip rate must be at least 1")
        return
    
    # Process the videos from the specified folder for calibration
    process_videos_for_calibration(args.folder_name, args.frame_skip)

if __name__ == "__main__":
    main()