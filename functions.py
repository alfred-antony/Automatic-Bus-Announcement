import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from google.cloud import vision
from google.cloud.vision_v1 import types
from gtts import gTTS
import threading
from google.cloud import texttospeech
import datetime
import queue
import time
import re
import json
from playsound import playsound

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/PROJECT/transit-vision-2e6350ac393a.json'

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")

# Replace this with your phone's IP
IP_CAMERA_URL = "http://192.168.137.93:8080//video"

# Initialize the SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.2)

# Initialize the Vision API client
vision_client = vision.ImageAnnotatorClient()

# Initialize video capture
cap = cv2.VideoCapture(0)

latest_extracted_text = ""

frame_count = 0
process_every_n_frames = 15
unique_bus_ids = {}
best_text_per_bus = {}
bus_timeout_seconds = 3

bus_history = []
try:
    with open('history.json', 'r') as f:
        bus_history = json.load(f)
except FileNotFoundError:
    bus_history = []

# Define output directory for audio files
output_dir = "C:/PROJECT/audio/"
os.makedirs(output_dir, exist_ok=True)

# Initialize the audio queue
audio_queue = queue.Queue()

def add_to_history(bus_text):
    if bus_text:
        timestamp = datetime.datetime.now().strftime('%H:%M:%S  %d-%m-%Y')
        bus_history.append({'text': bus_text, 'time': timestamp})
        print(f"[HISTORY] Added: {bus_text} at {timestamp}")

        # Save history to file
        with open('history.json', 'w') as f:
            json.dump(bus_history, f)

# Audio playback worker function
def audio_player():
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:
            break
        try:
            if not os.path.exists(audio_file):
                print(f"File {audio_file} does not exist at playback time. Skipping...")
                continue
            playsound(audio_file)  # Play the audio file
        except Exception as e:
            print(f"Error playing audio file {audio_file}: {e}")

        # Debug log after playback completion
        print(f"Finished playing file: {audio_file}")
        audio_queue.task_done()

# Function to get unique file names for audio
def get_unique_filename(base_name, ext):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
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

    # Regex to filter out numbers and English letters
    invalid_text_pattern = re.compile(r'[A-Za-z0-9]')

    for text in texts[1:]:
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

#gTTS
# def announce_text(text, bus_id):
#     print("Inside announce_text")
#     global unique_bus_ids
#     if text:
#         current_time = time.time()
#
#         # Check if the bus has been announced recently
#         if bus_id in unique_bus_ids and current_time - unique_bus_ids[bus_id] < bus_timeout_seconds:
#             print(f"Skipping announcement for bus {bus_id}, too soon after last announcement.")
#             return  # Skip if it's been too soon since the last announcement
#         # if text == NULL:
#
#         print(f"Announcing text for bus {bus_id}: {text}")
#         tts = gTTS(text=text + ' ഭാഗത്തേക്കുള്ള ബസ്സ് എത്തി ചേർന്നിരിക്കുന്നു', lang="ml")
#         unique_filename = get_unique_filename("announcement", "mp3")
#         output_file = os.path.join(output_dir, unique_filename)
#         tts.save(output_file)
#         print(f"Audio saved to: {output_file}")
#
#         # Wait until the file is fully written
#         while not os.path.exists(output_file):
#             print(f"Waiting for file {output_file} to be saved...")
#             time.sleep(0.5)
#
#         # Debug logs for queue addition
#         print(f"File being saved: {output_file}")
#         print(f"File exists: {os.path.exists(output_file)}")
#         print("Adding file to queue...")
#
#         # Add the audio file to the queue
#         audio_queue.put(output_file)
#
#         # Additional debug log for queue size
#         print(f"Queue size after adding: {audio_queue.qsize()}")
#
#         # Update the last announcement time for this bus
#         unique_bus_ids[bus_id] = current_time

#Cloud TTS
def announce_text(text, bus_id):
    global unique_bus_ids
    if not text or text.strip() == "":
        print(f"Skipping announcement for Bus {bus_id}: No valid text.")
        return
    if text:
        current_time = time.time()

        # Check if the bus has been announced recently
        if bus_id in unique_bus_ids and current_time - unique_bus_ids[bus_id] < bus_timeout_seconds:
            print(f"Skipping announcement for bus {bus_id}, too soon after last announcement.")
            return  # Skip if it's been too soon since the last announcement

        print(f"Announcing text for bus {bus_id}: {text}")

        # Initialize the Text-to-Speech client
        tts_client = texttospeech.TextToSpeechClient()

        # Build the text-to-speech input
        synthesis_input = texttospeech.SynthesisInput(text=text + ' ഭാഗത്തേക്കുള്ള ബസ്സ് എത്തി ചേർന്നിരിക്കുന്നു')

        # Select the voice, language, and gender
        voice = texttospeech.VoiceSelectionParams(
            language_code="ml-IN",  # Malayalam language
            name="ml-IN-Chirp3-HD-Kore",  # Specific voice name
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL  # Neutral gender voice
        )

        # Configure audio settings, including speaking rate
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,  # Save the output as MP3
            # speaking_rate=0.9  # Customize speaking rate
        )

        # Perform the text-to-speech request
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save the audio file
        unique_filename = get_unique_filename("announcement", "mp3")
        output_file = os.path.join(output_dir, unique_filename)
        with open(output_file, "wb") as out:
            out.write(response.audio_content)
            print(f"Audio content written to file: {output_file}")

        # Wait until the file is fully written
        while not os.path.exists(output_file):
            print(f"Waiting for file {output_file} to be saved...")
            time.sleep(0.5)

        # Debug logs for queue addition
        print(f"File being saved: {output_file}")
        print(f"File exists: {os.path.exists(output_file)}")
        print("Adding file to queue...")

        # Add the audio file to the queue
        audio_queue.put(output_file)

        # Additional debug log for queue size
        print(f"Queue size after adding: {audio_queue.qsize()}")

        # Update the last announcement time for this bus
        unique_bus_ids[bus_id] = current_time


# # Parameters
N_STABLE_FRAMES = 3  # Number of consecutive frames the text should remain stable
text_history_per_bus = {}  # Stores detected text history for each bus


def process_frame(frame):
    global best_text_per_bus, unique_bus_ids, text_history_per_bus, latest_extracted_text
    results = model(frame)
    current_time = time.time()  # Current time in seconds

    active_buses = set()
    bus_detections = []
    board_detections = []

    for result in results:
        for box in result.boxes:
            # Apply confidence threshold
            if box.conf[0] < 0.6:  # Filter out low-confidence detections
                print(f"Skipping low-confidence detection: {box.conf[0]}")
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            if box.cls[0] == 0:  # Class 0 for buses
                bus_detections.append({"obj_id": None, "box": [x1, y1, x2, y2]})

            elif box.cls[0] == 1:  # Class 1 for boards
                cropped_board = frame[y1:y2, x1:x2]
                text, area = extract_text_from_board(cropped_board)
                board_detections.append({"box": [x1, y1, x2, y2], "text": text, "area": area})

    # Track and match buses
    if len(bus_detections) > 0:
        bus_boxes = np.array([bus["box"] for bus in bus_detections])
        tracked_objects = tracker.update(bus_boxes)

        for i, obj in enumerate(tracked_objects):
            x1, y1, x2, y2, obj_id = obj
            obj_id = int(obj_id)
            bus_detections[i]["obj_id"] = obj_id
            active_buses.add(obj_id)
            unique_bus_ids[obj_id] = current_time

            # Link boards inside bus boxes
            bus_box = bus_detections[i]["box"]
            for board in board_detections:
                board_box = board["box"]
                if (board_box[0] >= bus_box[0] and board_box[1] >= bus_box[1] and
                        board_box[2] <= bus_box[2] and board_box[3] <= bus_box[3]):

                    if obj_id in best_text_per_bus:
                        if board["area"] > best_text_per_bus[obj_id]["area"]:
                            best_text_per_bus[obj_id].update({
                                "text": board["text"],
                                "area": board["area"]
                            })
                    else:
                        best_text_per_bus[obj_id] = {
                            "text": board["text"],
                            "area": board["area"],
                            "announced": False,
                            "last_announced_text": None
                        }

            # Draw bounding box
            cv2.rectangle(frame, (int(bus_box[0]), int(bus_box[1])), (int(bus_box[2]), int(bus_box[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (int(bus_box[0]), int(bus_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Handle departed buses and play audio
    departed_buses = []
    for obj_id, last_seen_time in unique_bus_ids.items():
        if obj_id not in active_buses and current_time - last_seen_time > bus_timeout_seconds:
            departed_buses.append(obj_id)

    for obj_id in departed_buses:
        if obj_id in best_text_per_bus:
            current_text = best_text_per_bus[obj_id]["text"]
            if current_text != best_text_per_bus[obj_id]["last_announced_text"]:
                announce_text(current_text, obj_id)
                best_text_per_bus[obj_id]["last_announced_text"] = current_text
                latest_extracted_text = current_text
                add_to_history(current_text)
                print("TEXT sent LARGEST-->", latest_extracted_text)
        # Clean up
        unique_bus_ids.pop(obj_id, None)
        text_history_per_bus.pop(obj_id, None)
        best_text_per_bus.pop(obj_id, None)

    return frame
