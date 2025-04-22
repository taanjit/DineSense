import cv2
import csv
import time
import json
import numpy as np
from ultralytics import YOLO

# Global variables
threshold = 150  # Distance threshold for detecting persons near stations

def resize_frame(frame, scale=0.6):
    """Resize a frame by the given scale factor"""
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def read_unique_stations(file_path):
    """Read station data from CSV file"""
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
    """Detect persons in a frame using YOLO model"""
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
    """Classify persons as sitting or standing based on their aspect ratio"""
    classified_persons = []
    for (x1, y1, x2, y2) in person_boxes:
        width = x2 - x1
        height = y2 - y1
        ratio = height / width
        label = "Standing Person" if ratio > 2 else "Sitting Person"
        classified_persons.append(((x1, y1, x2, y2), label))
    return classified_persons

def draw_hexagon(frame, center, radius, color, thickness):
    """Draw a hexagon on the frame"""
    hexagon_points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = np.deg2rad(angle_deg)
        x = int(center[0] + radius * np.cos(angle_rad))
        y = int(center[1] + radius * np.sin(angle_rad))
        hexagon_points.append((x, y))
    hexagon_points = np.array([hexagon_points], np.int32)
    cv2.polylines(frame, [hexagon_points], isClosed=True, color=color, thickness=thickness)

def monitor_resources():
    """Monitor memory usage and kill process if needed"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024**2  # MB
    
    if mem_usage > 4000:  # 4GB limit
        print(f"Memory usage exceeded limit: {mem_usage} MB")
        os._exit(1)  # Force exit if memory usage is too high

def mark_stations_and_persons_on_video(video_path, stations, model, face_attributes_module, output_path=None, frame_skip=1, threshold=150):
    """Main function to process video and mark stations and persons"""
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
    json_output = {}

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
        
        # Only detect attributes for sitting persons to improve performance
        classified_persons_with_attributes = face_attributes_module.process_detections_with_attributes(classified_persons, frame, sitting_only=True)

        frame_data = []

        for station in stations:
            station_id = station["Station ID"]
            table_position = station["Table Position"]
            chair_positions = station["Chair Positions"]

            draw_hexagon(frame, table_position, threshold, color=(255, 255, 255), thickness=2)

            # Find all persons at this station with details
            station_occupants = []
            male_count = 0
            female_count = 0
            
            for ((x1, y1, x2, y2), label, attributes) in classified_persons_with_attributes:
                if label == "Sitting Person":
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance = ((person_center[0] - table_position[0]) ** 2 +
                                (person_center[1] - table_position[1]) ** 2) ** 0.5
                    if distance < threshold:
                        orientation = face_attributes_module.detect_face_orientation(frame, (x1, y1, x2, y2))
                        
                        # Create occupant data structure
                        occupant_data = {}
                        
                        # Add age if detected
                        if attributes['age'] is not None:
                            occupant_data["age"] = int(attributes['age'])
                        
                        # Add gender if detected
                        if attributes['gender'] is not None:
                            gender = attributes['gender']
                            occupant_data["gender"] = gender
                            
                            # Count genders for station statistics
                            if gender.lower() == 'man':
                                male_count += 1
                            elif gender.lower() == 'woman':
                                female_count += 1
                        
                        station_occupants.append(occupant_data)

            # Update station occupation status
            station_occupied = False
            if len(station_occupants) > 0:
                if station_timers[station_id] is None:
                    station_timers[station_id] = time.time()
                elif time.time() - station_timers[station_id] > 20:
                    station_occupied = True
            else:
                station_timers[station_id] = None

            # Calculate average age for the station
            valid_ages = [occupant.get("age") for occupant in station_occupants if "age" in occupant]
            avg_age = None
            if valid_ages:
                avg_age = sum(valid_ages) / len(valid_ages)

            # Prepare station display
            if station_occupied:
                color = (255, 0, 0)  # Red
                label = station_id + " - Occupied"
            else:
                color = (0, 255, 0)  # Green
                label = station_id
            
            # Add demographic info to the label if available
            if avg_age is not None:
                label += f" - Avg Age: {avg_age:.1f}"
            
            if male_count > 0 or female_count > 0:
                label += f" (M:{male_count}/F:{female_count})"

            cv2.circle(frame, table_position, 10, color, -1)
            cv2.putText(frame, label, (table_position[0] + 15, table_position[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for chair_position in chair_positions:
                cv2.circle(frame, chair_position, 5, (0, 255, 0), -1)

            # Prepare station data for JSON output
            station_data = {
                "station": station_id,
                "occupants": len(station_occupants),
                "occupation_status": station_occupied,
                "male_count": male_count,
                "female_count": female_count,
                "occupant_details": station_occupants
            }
            
            # Add average age to station data if available
            if avg_age is not None:
                station_data["average_age"] = round(avg_age, 1)

            frame_data.append(station_data)

        json_output[f"frame_{frame_count}"] = frame_data

        # Draw person boxes and labels
        for ((x1, y1, x2, y2), label, attributes) in classified_persons_with_attributes:
            if label == "Sitting Person":
                person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                near_station = False
                
                # Check if person is near any station
                for station in stations:
                    station_center = station["Table Position"]
                    distance = ((person_center[0] - station_center[0]) ** 2 +
                                (person_center[1] - station_center[1]) ** 2) ** 0.5
                    if distance < threshold:
                        near_station = True
                        break
                
                if near_station:
                    orientation = face_attributes_module.detect_face_orientation(frame, (x1, y1, x2, y2))
                    if orientation == "Front":
                        color = (203, 192, 255)  # Pink
                        display_label = "Sitting Person - Front"
                    else:
                        color = (0, 255, 255)  # Yellow
                        display_label = "Sitting Person - Back"
                    
                    # Add demographic info to the label if available
                    attribute_info = []
                    if attributes['age'] is not None:
                        attribute_info.append(f"Age: {int(attributes['age'])}")
                    if attributes['gender'] is not None:
                        attribute_info.append(f"Gender: {attributes['gender']}")
                    
                    if attribute_info:
                        display_label += f" - {', '.join(attribute_info)}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # cv2.putText(frame, display_label, (x1, y1 - 10), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            elif label == "Standing Person":
                # Just draw the standing person without additional processing
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(frame, "Standing Person", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("Marked Stations and Persons", frame)

        if output_path and out.isOpened():
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_path and out.isOpened():
        out.release()
        print(f"Output video successfully saved to: {output_path}")

    with open("framewise_station_data.json", "w") as f:
        json.dump(json_output, f, indent=4)

    cv2.destroyAllWindows()