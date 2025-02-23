### Explanation:
# 1. **Setup and Initialization**: Initialize YOLO, SORT, Vision API client, and gTTS.
# 2. **Video Capture**: Capture frames from a video file or live camera feed.
# 3. **Frame Processing**: Process every nth frame to reduce computational load.
# 4. **Bus and Board Detection**: Detect buses and boards using the YOLO model.
# 5. **Bus Tracking**: Track buses using SORT and avoid redundant detections.
# 6. **OCR Extraction and TTS**: Crop the board region, extract text using Vision API, and use gTTS to convert text to speech.
# 7. **Display Results**: Draw bounding boxes, IDs, and display processed frames.
# 8. **Play Audio**: Play the generated audio file using `os.system`.

import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from google.cloud import vision
from google.cloud.vision_v1 import types
from gtts import gTTS
import datetime

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/PROJECT/transit-vision-2e6350ac393a.json'  # Update with the new path

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")  # Update with your trained model path

# Initialize the SORT tracker
tracker = Sort()

# Initialize the Vision API client
vision_client = vision.ImageAnnotatorClient()

# Initialize video capture
cap = cv2.VideoCapture("C:/PROJECT/test/VID20250209164511.mp4")  # Use '0' for webcam
# C:/PROJECT/test/VID20250113063317.mp4
# C:/PROJECT/test/VID20250209164359.mp4
# C:/PROJECT/test/VID20250209164417.mp4
# C:/PROJECT/test/VID20250209164426.mp4
# C:/PROJECT/test/VID20250209164511.mp4

frame_count = 0
process_every_n_frames = 5  # Adjust as needed
unique_bus_ids = set()

# Define output directory for audio files
output_dir = "C:/PROJECT/audio/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist


def get_unique_filename(base_name, ext):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
    return f"{base_name}_{timestamp}.{ext}"


def extract_text_from_board(image):
    if image is None or image.size == 0:
        return "", 0

    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        return "", 0

    content = encoded_image.tobytes()
    vision_image = types.Image(content=content)
    response = vision_client.text_detection(image=vision_image)
    texts = response.text_annotations

    if not texts:
        return "", 0

    largest_text = ""
    max_area = 0
    for text in texts[1:]:
        print(text.description)
        vertices = text.bounding_poly.vertices
        x1, y1 = vertices[0].x, vertices[0].y
        x2, y2 = vertices[2].x, vertices[2].y
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area > max_area:
            max_area = area
            largest_text = text.description.strip()
    print("Largest Text:", largest_text)
    return largest_text, max_area


def announce_text(text):
    tts = gTTS(text=text, lang="ml")
    unique_filename = get_unique_filename("announcement", "mp3")
    output_file = os.path.join(output_dir, unique_filename)
    tts.save(output_file)
    print(f"Audio saved to: {output_file}")
    os.system(f"start {output_file}")  # Play the audio file (for Windows; use an alternative command on Linux/macOS)


best_frame_text = ""
best_frame_area = 0

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
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Move tensor to CPU before converting to numpy
                w = x2 - x1
                h = y2 - y1
                detections.append([x1, y1, w, h, box.conf[0].cpu().numpy()])  # Append with confidence score

            elif box.cls[0] == 1:  # Class 1 for boards
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Move tensor to CPU before converting to numpy
                w = x2 - x1
                h = y2 - y1
                cropped_board = frame[y1:y2, x1:x2]
                text, area = extract_text_from_board(cropped_board)

                if area > best_frame_area:
                    best_frame_text = text
                    best_frame_area = area

    if len(detections) > 0:
        detections = np.array(detections)
        # Update tracker with detections
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            if obj_id not in unique_bus_ids:
                unique_bus_ids.add(obj_id)

            # Draw bounding box and object ID
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

    # Display the frame with tracking
    cv2.imshow('Bus Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if best_frame_text:
    announce_text(best_frame_text)

print(f'Number of unique buses detected: {len(unique_bus_ids)}')
