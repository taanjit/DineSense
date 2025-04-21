import cv2

def display_video(video_path):
    """
    Reads a video file, resizes its frames to 60% of the original size, and displays them.

    Args:
        video_path (str): Path to the video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    print("Press 'q' to exit the video playback.")

    # Loop to read and display frames
    while True:
        ret, frame = cap.read()
        # print(frame.shape[1], frame.shape[0])

        # Break the loop if no frame is returned (end of video)
        if not ret:
            print("End of video reached.")
            break

        # Resize the frame to 60% of its original size
        frame_resized = cv2.resize(frame, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        print(frame_resized.shape[1], frame_resized.shape[0])

        # Display the resized frame
        cv2.imshow("Video Playback (Resized to 60%)", frame_resized)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Video playback interrupted by user.")
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the video file
    video_file_path = "input videos/calibration_videos/calib_1.mp4"
    
    # Call the function to display the video
    display_video(video_file_path)