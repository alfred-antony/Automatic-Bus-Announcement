### Explanation:
# 1. **Setup and Initialization**: Initialize YOLO, SORT, Vision API client, and gTTS.
# 2. **Video Capture**: Capture frames from a video file or live camera feed.
# 3. **Frame Processing**: Process every nth frame to reduce computational load.
# 4. **Bus and Board Detection**: Detect buses and boards using the YOLO model.
# 5. **Bus Tracking**: Track buses using SORT and avoid redundant detections.
# 6. **OCR Extraction and TTS**: Crop the board region, extract text using Vision API, and use gTTS to convert text to speech.
# 7. **Display Results**: Draw bounding boxes, IDs, and display processed frames.
# 8. **Play Audio**: Play the generated audio file using `os.system`.
# Cant handle multiple buses

import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from google.cloud import vision
from google.cloud.vision_v1 import types
from gtts import gTTS
import threading
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
cap = cv2.VideoCapture(r"C:\Users\LEGION\Videos\Screen Recordings\Screen Recording 2025-03-11 192037.mp4")  # Use '0' for webcam
# C:/PROJECT/test/VID20250113063317.mp4
# C:/PROJECT/test/VID20250209164359.mp4
# C:/PROJECT/test/VID20250209164417.mp4
# C:/PROJECT/test/VID20250209164426.mp4
# C:/PROJECT/test/VID20250209164511.mp4
# VID20250113063043
# KSRTC Bus3
# "C:\Users\LEGION\Videos\Screen Recordings\Screen Recording 2025-03-11 153003.mp4"
# "C:\Users\LEGION\Videos\Screen Recordings\Screen Recording 2025-03-11 192037.mp4"

frame_count = 0
process_every_n_frames = 5  # Adjust as needed
unique_bus_ids = set()
best_text_per_bus = {}  # Track the best text for each bus

# Define output directory for audio files
output_dir = "C:/PROJECT/audio/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

def get_unique_filename(base_name, ext, text):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
    return f"{text}_{timestamp}.{ext}"

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
    if text:
        print("Announcement final text:", text)
        tts = gTTS(text=text + ' ഭാഗത്തേക്കുള്ള ബസ്സ് എത്തി ചേർന്നിരിക്കുന്നു', lang="ml")
        unique_filename = get_unique_filename("announcement", "mp3", text)
        output_file = os.path.join(output_dir, unique_filename)
        tts.save(output_file)
        print(f"Audio saved to: {output_file}")

        os.system(f"start {output_file}")  # Play the audio file (for Windows; use an alternative command on Linux/macOS)

def process_frame(frame):
    global best_text_per_bus
    results = model(frame)
    detections = []

    for result in results:
        for box in result.boxes:
            if box.cls[0] == 0:  # Class 0 for buses
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                w = x2 - x1
                h = y2 - y1
                detections.append([x1, y1, w, h, int(box.conf[0].cpu().numpy())])  # Ensure obj_id is an integer

            elif box.cls[0] == 1:  # Class 1 for boards
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                w = x2 - x1
                h = y2 - y1
                cropped_board = frame[y1:y2, x1:x2]
                text, area = extract_text_from_board(cropped_board)

                for obj in detections:
                    _, _, _, _, obj_id = obj
                    if obj_id in best_text_per_bus:
                        current_best_text, current_best_area = best_text_per_bus[obj_id]
                        if area > current_best_area:
                            best_text_per_bus[obj_id] = (text, area)
                    else:
                        best_text_per_bus[obj_id] = (text, area)

    if len(detections) > 0:
        detections = np.array(detections)
        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)  # Ensure obj_id is an integer
            if obj_id not in unique_bus_ids:
                unique_bus_ids.add(obj_id)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {int(obj_id)}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % process_every_n_frames != 0:
        continue

    frame = process_frame(frame)

    cv2.imshow('Bus Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Announce best text after processing all frames
for obj_id, (best_text, area) in best_text_per_bus.items():
    announce_text(best_text)

print(f'Number of unique buses detected: {len(unique_bus_ids)}')