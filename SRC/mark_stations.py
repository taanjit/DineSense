import cv2
import csv
import time
from ultralytics import YOLO

threshold = 150  # Distance threshold for detecting persons near stations

def resize_frame(frame, scale=0.6):
    """
    Resizes a video frame to the specified scale.

    Args:
        frame (numpy.ndarray): The input video frame.
        scale (float): The scaling factor for resizing the frame.

    Returns:
        numpy.ndarray: The resized video frame.
    """
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def read_unique_stations(file_path):
    """
    Reads the unique stations from the CSV file.

    Args:
        file_path (str): Path to the CSV file containing unique stations.

    Returns:
        list: List of dictionaries containing station data.
    """
    stations = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            stations.append({
                "Station ID": row["Station ID"],
                "Table Position": eval(row["Table Position"]),  # Convert string to tuple
                "Chair Positions": eval(row["Chair Positions"])  # Convert string to list of tuples
            })
    return stations

def detect_persons(frame, model):
    """
    Detects persons in the frame using the YOLO model.

    Args:
        frame (numpy.ndarray): The input video frame.
        model (YOLO): The YOLO model for detection.

    Returns:
        list: List of bounding boxes for detected persons.
    """
    results = model(frame)  # Run YOLO detection
    person_boxes = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get detected object class ID
            class_name = model.names[class_id]  # Convert ID to class name

            if class_name == "person":  # Filter for persons
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                person_boxes.append((x1, y1, x2, y2))

    return person_boxes

def classify_persons(person_boxes):
    """
    Classifies persons as standing or sitting based on the length-to-width ratio of their bounding boxes.

    Args:
        person_boxes (list): List of bounding boxes for detected persons.

    Returns:
        list: List of tuples containing bounding box and classification label.
    """
    classified_persons = []
    for (x1, y1, x2, y2) in person_boxes:
        width = x2 - x1
        height = y2 - y1
        ratio = height / width

        if ratio > 2:  # Threshold for standing person
            label = "Standing Person"
        else:
            label = "Sitting Person"

        classified_persons.append(((x1, y1, x2, y2), label))
    return classified_persons

def is_person_near_station(classified_persons, station_center, threshold=150):
    """
    Checks if any sitting person is near a station based on a distance threshold.

    Args:
        classified_persons (list): List of tuples containing bounding box and classification label.
        station_center (tuple): The center coordinates of the station.
        threshold (int): The distance threshold.

    Returns:
        bool: True if any sitting person is near the station, False otherwise.
    """
    for ((x1, y1, x2, y2), label) in classified_persons:
        if label == "Sitting Person":  # Consider only sitting persons
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate the center of the person
            distance = ((person_center[0] - station_center[0]) ** 2 + (person_center[1] - station_center[1]) ** 2) ** 0.5
            if distance < threshold:
                return True  # Return True if any sitting person is within the threshold
    return False  # Return False if no sitting person is within the threshold

def mark_stations_and_persons_on_video(video_path, stations, model, output_path=None, frame_skip=10, threshold=150):
    """
    Marks the unique stations and detected persons on the video frames.

    Args:
        video_path (str): Path to the input video file.
        stations (list): List of unique stations.
        model (YOLO): The YOLO model for person detection.
        output_path (str): Path to save the output video (optional).
        frame_skip (int): Number of frames to skip between processing.
        threshold (int): The distance threshold for detecting persons near stations.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object if output_path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0  # Initialize frame counter
    station_timers = {station["Station ID"]: None for station in stations}  # Initialize timers for each station

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames based on the frame_skip value
        if frame_count % frame_skip != 0:
            continue

        # Resize the frame
        frame = resize_frame(frame)

        # Detect persons in the frame
        person_boxes = detect_persons(frame, model)

        # Classify persons as standing or sitting
        classified_persons = classify_persons(person_boxes)

        # Mark each station on the frame
        for station in stations:
            table_position = station["Table Position"]
            chair_positions = station["Chair Positions"]

            # Draw a circle around the station with the threshold as the radius
            cv2.circle(frame, table_position, threshold, (255, 255, 255), 2)  # White circle for threshold area

            # Check if a sitting person is near the station
            station_occupied = False
            if is_person_near_station(classified_persons, table_position, threshold):
                if station_timers[station["Station ID"]] is None:
                    station_timers[station["Station ID"]] = time.time()  # Start the timer
                elif time.time() - station_timers[station["Station ID"]] > 20:  # Check if 20 seconds have passed
                    station_occupied = True
            else:
                station_timers[station["Station ID"]] = None  # Reset the timer if no sitting person is near

            # Draw the station
            if station_occupied:
                color = (0, 0, 255)  # Red for occupied station
                label = station["Station ID"] + " - Occupied"
            else:
                color = (0, 255, 0)  # Green for available station
                label = station["Station ID"]

            cv2.circle(frame, table_position, 10, color, -1)  # Draw the station circle
            cv2.putText(frame, label, (table_position[0] + 15, table_position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw the chair positions
            for chair_position in chair_positions:
                cv2.circle(frame, chair_position, 5, (0, 255, 0), -1)  # Smaller green circles for chairs

        # Mark detected persons with appropriate bounding boxes and labels
        for ((x1, y1, x2, y2), label) in classified_persons:
            if label == "Standing Person":
                color = (0, 165, 255)  # Orange for standing
            else:
                color = (0, 0, 255)  # Red for sitting

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the frame
        cv2.imshow("Marked Stations and Persons", frame)

        # Write the frame to the output video if output_path is provided
        if output_path:
            out.write(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the input video file
    video_file_path = "Project Cheena_vala/input videos/videos/Data 1/video6.mp4"

    # Path to the CSV file containing unique stations
    csv_file_path = "Project Cheena_vala/SRC/calibration_file/updated_station_calibration.csv"

    # Path to save the output video (optional)
    output_video_path = "Project Cheena_vala/SRC/calibration_file/marked_video_with_persons_classified.mp4"

    # Load the YOLO model
    yolo_model_path = "yolo12x.pt"
    model = YOLO(yolo_model_path)

    # Read the unique stations from the CSV file
    unique_stations = read_unique_stations(csv_file_path)

    # Mark the stations and detected persons on the video
    mark_stations_and_persons_on_video(video_file_path, unique_stations, model, output_video_path)

    print(f"Marked video saved to: {output_video_path}")