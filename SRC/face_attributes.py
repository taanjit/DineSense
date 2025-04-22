import cv2
import threading
import queue
from deepface import DeepFace
import mediapipe as mp

# Initialize MediaPipe face detector
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Limit the number of concurrent face analysis threads
MAX_CONCURRENT_THREADS = 4

# Process only a subset of people in crowded frames
MAX_PEOPLE_PER_FRAME = 8

def analyze_face_attributes(person_img, results_queue, detection_id=None):
    """Analyze face with better error handling and timeouts"""
    try:
        # Make a copy of the image to avoid reference issues
        person_img_copy = person_img.copy() 
        
        # Apply face detection and attribute analysis with timeout
        analysis = DeepFace.analyze(
            img_path=person_img_copy,
            actions=['age', 'gender'],
            enforce_detection=False,
            silent=True,
            detector_backend='opencv'  # Faster than mtcnn
        )
        
        # Processing results...
        if isinstance(analysis, list) and len(analysis) > 0:
            age = analysis[0].get('age', None)
            gender = analysis[0].get('dominant_gender', None)
        else:
            age = analysis.get('age', None)
            gender = analysis.get('dominant_gender', None)
        
        results_queue.put((detection_id, {'age': age, 'gender': gender}))
    except Exception as e:
        print(f"Face analysis error: {str(e)}")
        results_queue.put((detection_id, {'age': None, 'gender': None}))

def process_detections_with_attributes(detections, frame, sitting_only=True):
    """Modified function with thread limiting and better resource management"""
    attribute_threads = []
    attributes_queue = queue.Queue()
    active_threads = 0
    max_threads = MAX_CONCURRENT_THREADS  # Limit concurrent threads
    
    for i, ((x1, y1, x2, y2), label) in enumerate(detections):
        # Only process sitting persons if sitting_only is True
        if sitting_only and label != "Sitting Person":
            continue
            
        # Limit the number of people processed per frame
        if i >= MAX_PEOPLE_PER_FRAME:
            break
            
        # Extract person ROI for attribute detection
        person_img = frame[y1:y2, x1:x2]
        if person_img.size > 0:  # Make sure the ROI is valid
            # Wait if we've reached max threads
            while active_threads >= max_threads:
                import time
                time.sleep(0.1)
                active_threads = sum(1 for t in attribute_threads if t.is_alive())
                
            thread = threading.Thread(
                target=analyze_face_attributes,
                args=(person_img, attributes_queue, i)
            )
            thread.daemon = True
            thread.start()
            attribute_threads.append(thread)
            active_threads += 1
    
    # Join threads with timeout - use a shorter timeout
    for thread in attribute_threads:
        thread.join(timeout=0.3)  # Reduced timeout
    
    # Collect results from queue and add attributes to detections
    detections_with_attributes = []
    attributes_collected = {}
    
    # First collect all attributes from the queue
    while not attributes_queue.empty():
        detection_id, attributes = attributes_queue.get()
        attributes_collected[detection_id] = attributes
    
    # Then add them to the detections in order
    for i, ((x1, y1, x2, y2), label) in enumerate(detections):
        attributes = attributes_collected.get(i, {'age': None, 'gender': None})
        detections_with_attributes.append(((x1, y1, x2, y2), label, attributes))
    
    return detections_with_attributes

def detect_face_orientation(frame, box):
    """Detect if a face is facing front or back"""
    x1, y1, x2, y2 = box
    person_roi = frame[y1:y2, x1:x2]
    rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_roi)
    if results.detections:
        return "Front"
    return "Back"