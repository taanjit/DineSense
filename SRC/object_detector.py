import cv2
import numpy as np
import csv
from video_reader import VideoReader  # Ensure this module contains VideoReader class
from ultralytics import YOLO

class ObjectDetector:
    """
    A class to detect dining tables and nearby chairs in video frames using YOLO.
    """

    def __init__(self, video_path, model_path, confidence_threshold=0.5, proximity_threshold=100):
        """
        Initializes the ObjectDetector with the video file path and YOLO model.

        Args:
            video_path (str): Path of the video file.
            model_path (str): Path of the YOLO model weights (.pt).
            confidence_threshold (float): Minimum confidence score for detection.
            proximity_threshold (int): Maximum distance to group chairs with a table.
        """
        self.video_path = video_path
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.proximity_threshold = proximity_threshold

        # Load YOLO model
        self.model = YOLO(model_path)

        # Define class IDs for chairs and tables
        self.target_classes = {
            "chair": (238, 130, 238),  # Violet
            "dining table": (255, 165, 0)  # Orange
        }

    def detect_and_group_objects(self, frame):
        """
        Detect dining tables and nearby chairs in a single frame and group them into stations.

        Args:
            frame (numpy.ndarray): The video frame to process.

        Returns:
            tuple: Processed frame and a list of stations with their positions.
        """
        results = self.model(frame)  # Run YOLO detection

        chairs = []
        tables = []

        # Process detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  # Get detected object class ID
                class_name = self.model.names[class_id]  # Convert ID to class name

                if class_name in self.target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    confidence = float(box.conf[0])  # Confidence score

                    if confidence > self.confidence_threshold:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if class_name == "chair":
                            chairs.append(((x1, y1, x2, y2), (center_x, center_y)))
                        elif class_name == "dining table":
                            tables.append(((x1, y1, x2, y2), (center_x, center_y)))

        # Group chairs with nearby tables
        stations = []
        station_id = 1
        for table_box, table_center in tables:
            station_chairs = []

            for chair_box, chair_center in chairs:
                distance = np.sqrt((table_center[0] - chair_center[0]) ** 2 +
                                   (table_center[1] - chair_center[1]) ** 2)

                if distance < self.proximity_threshold:
                    station_chairs.append((chair_box, chair_center))  # Append both box and center

            # Draw table and label the station
            cv2.rectangle(frame, (table_box[0], table_box[1]), (table_box[2], table_box[3]),
                          self.target_classes["dining table"], 2)
            cv2.putText(frame, f"Station #{station_id}", (table_box[0], table_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.target_classes["dining table"], 2)

            # Draw chairs associated with the table
            for chair_box, _ in station_chairs:
                cv2.rectangle(frame, (chair_box[0], chair_box[1]), (chair_box[2], chair_box[3]),
                              self.target_classes["chair"], 2)

            # Save station details
            stations.append({
                "station_id": f"Station #{station_id}",
                "table_position": table_center,
                "chairs": [chair_center for _, chair_center in station_chairs]
            })

            station_id += 1

        return frame, stations

    def save_stations_to_csv(self, stations, output_path):
        """
        Save the detected stations and their positions to a CSV file.

        Args:
            stations (list): List of stations with their positions.
            output_path (str): Path to save the CSV file.
        """
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Station ID", "Table Position", "Chair Positions"])

            for station in stations:
                writer.writerow([
                    station["station_id"],
                    station["table_position"],
                    station["chairs"]
                ])

if __name__ == "__main__":
    # Path to the video file
    video_file_path = "input videos/calibration_videos/calib_1.mp4"

    # Path to the YOLO model weights (should be a .pt file)
    yolo_model_path = "yolo12x.pt"  # Ensure this model supports chairs & tables

    # Path to save the CSV file
    csv_output_path = "SRC/calibration_file/station_calibration.csv"

    # Create an instance of ObjectDetector
    detector = ObjectDetector(video_file_path, yolo_model_path, confidence_threshold=0.1, proximity_threshold=150)

    # Create an instance of VideoReader
    video_reader = VideoReader(video_file_path)

    # Variables to store unique stations
    unique_stations = []

    # Process video frames
    frame_count = 0
    for frame in video_reader.read_and_resize():
        print(frame.shape[1], frame.shape[0])   
        frame_count += 1

        # Evaluate every 100 frames
        if frame_count % 100 == 0:
            processed_frame, stations = detector.detect_and_group_objects(frame)

            # Add unique stations
            for station in stations:
                if station not in unique_stations:
                    unique_stations.append(station)

            # Display the processed frame
            cv2.imshow("Chair & Table Detection with Stations", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Save unique stations to CSV
    detector.save_stations_to_csv(unique_stations, csv_output_path)

    # Release resources
    cv2.destroyAllWindows()