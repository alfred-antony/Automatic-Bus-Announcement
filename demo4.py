### Explanation:
# 1. **Setup and Initialization**: Initialize YOLO, SORT, Vision API client, and gTTS.
# 2. **Video Capture**: Capture frames from a video file or live camera feed.
# 3. **Frame Processing**: Process every nth frame to reduce computational load.
# 4. **Bus and Board Detection**: Detect buses and boards using the YOLO model.
# 5. **Bus Tracking**: Track buses using SORT and avoid redundant detections.
# 6. **OCR Extraction and TTS**: Crop the board region, extract text using Vision API, and use gTTS to convert text to speech.
# 7. **Display Results**: Draw bounding boxes, IDs, and display processed frames.
# 8. **Play Audio**: Play the generated audio file using `os.system`.
# CAN handle multiple buses
# But sometimes overwrites old buses with same IDs

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
import queue
import time
import re
from playsound import playsound  # Install with `pip install playsound==1.2.2`


# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/PROJECT/transit-vision-2e6350ac393a.json'  # Update with the new path

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")  # Update with your trained model path

# Initialize the SORT tracker
# tracker = Sort()
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # Adjust as needed

# Initialize the Vision API client
vision_client = vision.ImageAnnotatorClient()

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use '0' for webcam
# C:/PROJECT/test/VID20250113063317.mp4
# C:/PROJECT/test/VID20250209164359.mp4
# C:/PROJECT/test/VID20250209164417.mp4
# C:/PROJECT/test/VID20250209164426.mp4
# C:/PROJECT/test/VID20250209164511.mp4
# VID20250113063043
# KSRTC Bus3
# "C:\Users\LEGION\Videos\Screen Recordings\Screen Recording 2025-03-11 153003.mp4"
# "C:\Users\LEGION\Videos\Screen Recordings\Screen Recording 2025-03-11 192037.mp4"
# "C:\Users\LEGION\Videos\Screen Recordings\Screen Recording 2025-03-11 210715.mp4"

frame_count = 0
process_every_n_frames = 15  # Adjust as needed
# Modify `unique_bus_ids` to include timestamps: {obj_id: last_detected_time}
unique_bus_ids = {}
best_text_per_bus = {}  # Format: {obj_id: {"text": "", "area": 0, "announced": False}}
# Timeout threshold (in seconds) for a bus to be considered "departed"
bus_timeout_seconds = 3

# Define output directory for audio files
output_dir = "C:/PROJECT/audio/"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

def get_unique_filename(base_name, ext):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
    return f"{base_name}_{timestamp}.{ext}"
# def get_unique_filename(base_name, ext, text):
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
#     return f"{text}_{timestamp}.{ext}"

# Initialize the audio queue
audio_queue = queue.Queue()

# Audio playback worker function
def audio_player():
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:  # Exit condition
            break
        try:
            if not os.path.exists(audio_file):
                print(f"File {audio_file} does not exist at playback time. Skipping...")
                continue
            playsound(audio_file)  # Play the audio file
        except Exception as e:
            print(f"Error playing audio file {audio_file}: {e}")

        #Debug log after playback completion
        print(f"Finished playing file: {audio_file}")
        audio_queue.task_done()

# Start the audio player thread
audio_thread = threading.Thread(target=audio_player, daemon=True)
audio_thread.start()

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

    # Define a regex to filter out numbers and English letters
    invalid_text_pattern = re.compile(r'[A-Za-z0-9]')
    # invalid_text_pattern = re.compile(r'^[\u0D00-\u0D7F\s]+$') #Only malayalam characters

    for text in texts[1:]:
        print(text.description)
        detected_text = text.description.strip()
        if invalid_text_pattern.match(detected_text):  # Skip invalid texts
            print(f"Skipping invalid text: {detected_text}")
            continue
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
        unique_filename = get_unique_filename("announcement", "mp3")
        # unique_filename = get_unique_filename("announcement", "mp3", text)
        output_file = os.path.join(output_dir, unique_filename)
        tts.save(output_file)
        print(f"Audio saved to: {output_file}")

        # Wait until the file is fully written
        while not os.path.exists(output_file):
            print(f"Waiting for file {output_file} to be saved...")
            time.sleep(2.0)  # Small delay to ensure the file is saved completely

        # Debug logs for queue addition
        print(f"File being saved: {output_file}")
        print(f"File exists: {os.path.exists(output_file)}")
        print("Adding file to queue...")

        # Add the audio file to the queue
        audio_queue.put(output_file)

        # Additional debug log for queue size
        print(f"Queue size after adding: {audio_queue.qsize()}")


def process_frame(frame):
    global best_text_per_bus, unique_bus_ids
    results = model(frame)
    current_time = time.time()  # Current time in seconds

    # Reset active bus detections for this frame
    active_buses = set()

    # Separate detections for buses and boards
    bus_detections = []
    board_detections = []

    for result in results:
        for box in result.boxes:
            if box.cls[0] == 0:  # Class 0 for buses
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                bus_detections.append({"obj_id": None, "box": [x1, y1, x2, y2]})
            elif box.cls[0] == 1:  # Class 1 for boards
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cropped_board = frame[y1:y2, x1:x2]
                text, area = extract_text_from_board(cropped_board)
                board_detections.append({"box": [x1, y1, x2, y2], "text": text, "area": area})

    # Update tracked buses using SORT
    if len(bus_detections) > 0:
        bus_boxes = np.array([bus["box"] for bus in bus_detections])
        tracked_objects = tracker.update(bus_boxes)

        # Update detection timestamps for active buses
        for i, obj in enumerate(tracked_objects):
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            bus_detections[i]["obj_id"] = obj_id
            active_buses.add(obj_id)  # Mark this bus as active
            unique_bus_ids[obj_id] = current_time  # Update last detected time

            # Link buses to boards inside their bounding boxes
            bus_box = bus_detections[i]["box"]
            for board in board_detections:
                board_box = board["box"]
                if (board_box[0] >= bus_box[0] and board_box[1] >= bus_box[1] and
                        board_box[2] <= bus_box[2] and board_box[3] <= bus_box[3]):
                    if obj_id in best_text_per_bus:
                        if board["area"] > best_text_per_bus[obj_id]["area"]:
                            best_text_per_bus[obj_id].update({"text": board["text"], "area": board["area"]})
                    else:
                        best_text_per_bus[obj_id] = {"text": board["text"], "area": board["area"], "announced": False}

            # Draw bounding box and ID for each bus
            cv2.rectangle(frame, (int(bus_box[0]), int(bus_box[1])), (int(bus_box[2]), int(bus_box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (int(bus_box[0]), int(bus_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Check for buses that are "lost" (no longer detected)
    departed_buses = []
    for obj_id, last_seen_time in unique_bus_ids.items():
        if obj_id not in active_buses and current_time - last_seen_time > bus_timeout_seconds:
            departed_buses.append(obj_id)  # Mark as departed

    # Generate audio for departed buses
    for obj_id in departed_buses:
        if obj_id in best_text_per_bus and not best_text_per_bus[obj_id]["announced"]:
            announce_text(best_text_per_bus[obj_id]["text"])
            best_text_per_bus[obj_id]["announced"] = True

        # Remove departed buses from tracking
        unique_bus_ids.pop(obj_id, None)

    return frame

try:
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
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Releasing resources...")

print()

# Announce best text for each tracked bus
for obj_id, data in best_text_per_bus.items():
    if obj_id is None:  # Skip if the ID is None
        continue

    best_text = data["text"]
    announced = data["announced"]

    # Only announce if the bus hasn't been announced yet and has valid text
    if not announced and best_text:
        print(f"Bus ID: {obj_id}, Best Text: {best_text}, Area: {data['area']}")  # Debugging print
        announce_text(best_text)
        best_text_per_bus[obj_id]["announced"] = True  # Mark as announced

# Stop the audio playback thread
audio_queue.put(None)  # Send exit signal to the thread
audio_thread.join()

print(f'Number of unique buses detected: {len(unique_bus_ids)}')