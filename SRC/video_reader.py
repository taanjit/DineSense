import cv2

class VideoReader:
    """
    A class to read and display video frames.
    """

    def __init__(self, video_path):
        """
        Initializes the VideoReader with the video file path.

        Args:
            video_path (str): Path to the video file.
        """
        self.video_path = video_path

    def read_and_resize(self, scale=0.6):
        """
        Reads the video file, resizes its frames, and yields them.

        Args:
            scale (float): Scaling factor for resizing the frames.

        Yields:
            frame_resized (numpy.ndarray): Resized video frame.
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of video reached.")
                break

            # Resize the frame
            frame_resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield frame_resized

        cap.release()

    def display_video(self):
        """
        Displays the video frames resized to 60% of their original size.
        """
        print("Press 'q' to exit the video playback.")
        for frame_resized in self.read_and_resize():
            cv2.imshow("Video Playback (Resized to 60%)", frame_resized)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Video playback interrupted by user.")
                break

        cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Path to the video file
#     video_file_path = "Project Cheena_vala/input videos/calibration_videos/calib_1.mp4"

#     # Create a VideoReader instance and display the video
#     video_reader = VideoReader(video_file_path)
#     video_reader.display_video()