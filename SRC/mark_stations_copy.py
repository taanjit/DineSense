import cv2
import csv
import time
import numpy as np
from ultralytics import YOLO

threshold = 150  # Distance threshold for detecting persons near stations

def resize_frame(frame, scale=0.6):
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def read_unique_stations(file_path):
    stations = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            stations.append({
                "Station ID": row["Station ID"],
                "Table Position": eval(row["Table Position"]),
                "Chair Positions": eval(row["Chair Positions"])
            })
    return stations

def detect_persons(frame, model):
    results = model(frame)
    person_boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            if class_name == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_boxes.append((x1, y1, x2, y2))
    return person_boxes

def classify_persons(person_boxes):
    classified_persons = []
    for (x1, y1, x2, y2) in person_boxes:
        width = x2 - x1
        height = y2 - y1
        ratio = height / width
        label = "Standing Person" if ratio > 2 else "Sitting Person"
        classified_persons.append(((x1, y1, x2, y2), label))
    return classified_persons

def is_person_near_station(classified_persons, station_center, threshold=150):
    for ((x1, y1, x2, y2), label) in classified_persons:
        if label == "Sitting Person":
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = ((person_center[0] - station_center[0]) ** 2 +
                        (person_center[1] - station_center[1]) ** 2) ** 0.5
            if distance < threshold:
                return True
    return False

def draw_hexagon(frame, center, radius, color, thickness):
    hexagon_points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = np.deg2rad(angle_deg)
        x = int(center[0] + radius * np.cos(angle_rad))
        y = int(center[1] + radius * np.sin(angle_rad))
        hexagon_points.append((x, y))
    hexagon_points = np.array([hexagon_points], np.int32)
    cv2.polylines(frame, [hexagon_points], isClosed=True, color=color, thickness=thickness)

def mark_stations_and_persons_on_video(video_path, stations, model, output_path=None, frame_skip=1, threshold=150):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    scale = 0.6
    frame_width = int(original_frame_width * scale)
    frame_height = int(original_frame_height * scale)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print(f"Error: Could not create output video file {output_path}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            if not out.isOpened():
                print("Error: Failed to create output video with alternative codec")
                return

    frame_count = 0
    station_timers = {station["Station ID"]: None for station in stations}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_skip > 1 and frame_count % frame_skip != 0:
            continue

        frame = resize_frame(frame, scale)
        person_boxes = detect_persons(frame, model)
        classified_persons = classify_persons(person_boxes)

        for station in stations:
            table_position = station["Table Position"]
            chair_positions = station["Chair Positions"]

            # Draw hexagon instead of circle
            draw_hexagon(frame, table_position, threshold, color=(255, 255, 255), thickness=2)

            station_occupied = False
            if is_person_near_station(classified_persons, table_position, threshold):
                if station_timers[station["Station ID"]] is None:
                    station_timers[station["Station ID"]] = time.time()
                elif time.time() - station_timers[station["Station ID"]] > 20:
                    station_occupied = True
            else:
                station_timers[station["Station ID"]] = None

            if station_occupied:
                color = (255, 0, 0)
                label = station["Station ID"] + " - Occupied"
            else:
                color = (0, 255, 0)
                label = station["Station ID"]

            cv2.circle(frame, table_position, 10, color, -1)
            cv2.putText(frame, label, (table_position[0] + 15, table_position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for chair_position in chair_positions:
                cv2.circle(frame, chair_position, 5, (0, 255, 0), -1)

        for ((x1, y1, x2, y2), label) in classified_persons:
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if label == "Sitting Person" and any(
                is_person_near_station([((x1, y1, x2, y2), label)], station["Table Position"], threshold)
                for station in stations
            ):
                color = (0, 0, 255)
            elif label == "Standing Person":
                color = (0, 165, 255)
            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Marked Stations and Persons", frame)

        if output_path and out.isOpened():
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path and out.isOpened():
        out.release()
        print(f"Output video successfully saved to: {output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file_path = "Project Cheena_vala/input videos/videos/Data 1/video6.mp4"
    csv_file_path = "Project Cheena_vala/SRC/calibration_file/updated_station_calibration.csv"
    output_video_path = "Project Cheena_vala/SRC/calibration_file/marked_video_with_persons_classified.mp4"
    yolo_model_path = "yolo12x.pt"

    model = YOLO(yolo_model_path)
    unique_stations = read_unique_stations(csv_file_path)
    mark_stations_and_persons_on_video(video_file_path, unique_stations, model, output_video_path)

    print("Video processing complete.")
