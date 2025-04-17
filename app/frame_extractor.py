import cv2
import logging
import time
import os


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def frame_extraction(video_path, frame_skip=20, output_frame_path=None):
    """
    Extract frames from a video file at specified intervals.
    
    Args: 
        video path (str): path of the video file
        frame_skip (int): number of frames to skip between frames
    """

    # opening the video file
    cap =cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file : {video_path}")
        return
    # getting the original frame width and height
    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setting the frame couter to 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info(" End of the video file or cannot fetch the video file")
            break

        if frame_count % frame_skip == 0:
            # extract the frame
            frame_name = f"frame_{frame_count:05d}.jpg"
            if output_frame_path:
                os.makedirs(output_frame_path,exist_ok=True)
                frame_path=os.path.join(output_frame_path,frame_name)
                cv2.imwrite(frame_path,frame)

                logger.info(f"Extracted frame {frame_count} and saved to {frame_path}")
            else:
                logger.info(f"Extracted frame {frame_count} and saved to current directory")
                cv2.imwrite(frame_name,frame)
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Video file closed and all windows destroyed")
    # closing the video file and destroying all windows



if __name__=="__main__":
    # Loading the video file from the source directory
    video_file_path = r"input videos/Video_data/video6.mp4"
    output_frame_path = r"output_frames/"
    frame_extraction(video_path=video_file_path, frame_skip=20,output_frame_path=output_frame_path)


