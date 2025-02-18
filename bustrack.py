import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Simple Online and Realtime Tracking

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")  # Update with your trained model path

# Initialize the SORT tracker
tracker = Sort()

# Initialize video capture
cap = cv2.VideoCapture('test/KSRTC Bus2.mp4')  # Use '0' for webcam

frame_count = 0
process_every_n_frames = 5  # Adjust as needed
unique_bus_ids = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue

    # Perform detection
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            if box.cls[0] == 0:  # Class 0 for buses
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                detections.append([x1, y1, w, h, box.conf[0]])  # Append with confidence score

    if len(detections) > 0:
        detections = np.array(detections)

    # Update tracker with detections
    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj
        unique_bus_ids.add(obj_id)

        # Draw bounding box and object ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with tracking
    cv2.imshow('Bus Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f'Number of unique buses detected: {len(unique_bus_ids)}')
