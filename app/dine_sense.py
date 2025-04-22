import cv2
import os
import csv
import time
import json
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Load Yolo model for object detection
yolo_model_path = "yolo12x.pt"

model = YOLO(yolo_model_path)

threshold = 150  # Distance threshold for detecting persons near stations

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

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
    results = model(frame, verbose=False)
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

def detect_face_orientation(frame, box):
    x1, y1, x2, y2 = box
    person_roi = frame[y1:y2, x1:x2]
    rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_roi)
    if results.detections:
        return "Front"
    return "Back"

def is_person_near_station(classified_persons, station_center, threshold=150):
    count = 0
    for ((x1, y1, x2, y2), label) in classified_persons:
        if label.startswith("Sitting Person"):
            person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            distance = ((person_center[0] - station_center[0]) ** 2 +
                        (person_center[1] - station_center[1]) ** 2) ** 0.5
            if distance < threshold:
                count += 1
    return count

def mark_stations_and_persons_on_video(image_file_path, stations,output_path=None, frame_skip=1, threshold=150):
    input_frame = cv2.imread(image_file_path)

    if input_frame is None:
        print("❌ Could not load image.")
        return

    height, width, _ = input_frame.shape
    scale = 1

    
    frame = resize_frame(input_frame, scale=0.6)
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    filename = os.path.basename(image_file_path)
    frame_count = filename.split('_')[1].split('.')[0]
    station_timers = {station["Station ID"]: None for station in stations}
    json_output = {}

    person_boxes = detect_persons(frame, model)
    classified_persons = classify_persons(person_boxes)

    frame_data = []

    for station in stations:
        table_position = station["Table Position"]
        table_position = tuple([int(p * scale) for p in table_position])
        chair_positions = [tuple([int(c * scale) for c in chair]) for chair in station["Chair Positions"]]

        draw_hexagon(frame, table_position, threshold, color=(255, 255, 255), thickness=2)

        sitting_count = is_person_near_station(classified_persons, table_position, threshold)
        station_occupied = False
        if sitting_count > 0:
            if station_timers[station["Station ID"]] is None:
                station_timers[station["Station ID"]] = time.time()
            elif time.time() - station_timers[station["Station ID"]] > 20:
                station_occupied = True
        else:
            station_timers[station["Station ID"]] = None

        color = (255, 0, 0) if station_occupied else (0, 255, 0)
        label = station["Station ID"] + (" - Occupied" if station_occupied else "")
        cv2.circle(frame, table_position, 10, color, -1)
        cv2.putText(frame, label, (table_position[0] + 15, table_position[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for chair_position in chair_positions:
            cv2.circle(frame, chair_position, 5, (0, 255, 0), -1)

        frame_data.append({
            "station": station["Station ID"],
            "occupants": sitting_count,
            "occupation Status": station_occupied
        })

    for ((x1, y1, x2, y2), label) in classified_persons:
        person_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if label == "Sitting Person":
            orientation = detect_face_orientation(frame, (x1, y1, x2, y2))
            label += f" - {orientation}"

            if orientation == "Front":
                eligible_stations = []
                for station in stations:
                    station_pos = tuple([int(p * scale) for p in station["Table Position"]])
                    if station_pos[1] > y2:
                        dist = ((person_center[0] - station_pos[0]) ** 2 +
                                (person_center[1] - station_pos[1]) ** 2) ** 0.5
                        eligible_stations.append((dist, station_pos, station["Station ID"]))

                if eligible_stations:
                    eligible_stations.sort(key=lambda tup: tup[0])
                    nearest_station_pos = eligible_stations[0][1]
                    nearest_station_id = eligible_stations[0][2]
                    cv2.line(frame, person_center, nearest_station_pos, (255, 0, 255), 2)
                    cv2.putText(frame, f"Assigned to {nearest_station_id}",
                                (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                color = (203, 192, 255)  # Pink
            else:
                color = (0, 255, 255)  # Yellow
        elif label == "Standing Person":
            color = (0, 165, 255)
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    json_output[f"frame_{frame_count}"] = frame_data

    # cv2.imshow("Marked Stations and Persons", frame)
    pretty_json = json.dumps(json_output, indent=4)
    print(pretty_json)

    # Save framewise JSON to a file
    with open("framewise_station_data.json", "w") as f:
        json.dump(json_output, f, indent=4)

    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pretty_json


if __name__ == "__main__":
    image_file_path = r"output_frames/frame_03140.jpg"
    csv_file_path = "SRC/calibration_file/updated_station_calibration.csv"
    output_video_path = "SRC/calibration_file/marked_video_with_persons_classified.mp4"
    
    unique_stations = read_unique_stations(csv_file_path)
    json_data=mark_stations_and_persons_on_video(image_file_path, unique_stations,output_video_path)

    print("✅ Image processing complete.")

