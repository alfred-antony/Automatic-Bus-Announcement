from flask import Flask, render_template, Response, request
from functions import process_frame, audio_player

import threading
import cv2
import queue
import signal

import sys

app = Flask(__name__)
output_frame = None  # Holds the latest processed frame
lock = threading.Lock()  # Ensures thread-safe access to `output_frame`

detection_thread = None  # Global thread for running detection
detection_active = False  # Detection state

# Initialize the audio queue
audio_queue = queue.Queue()
audio_thread = threading.Thread(target=audio_player, daemon=True)
audio_thread.start()

# Your existing code here (YOLO setup, SORT setup, etc.)
# Keep your `process_frame` function and other helper functions unchanged.

# Initialize the video capture (from your camera or video)
# cap = cv2.VideoCapture(0)

@app.route('/start_detection')
def start_detection():
    global detection_thread, detection_active, cap

    if detection_active:
        return "Detection is already running"

    # Open the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open camera"

    detection_active = True
    detection_thread = threading.Thread(target=generate_frames, daemon=True)
    detection_thread.start()
    return "Detection Started"

@app.route('/stop_detection')
def stop_detection():
    global detection_active, cap, audio_queue, audio_thread

    if not detection_active:
        return "Detection is not running"

    detection_active = False

    # Release the camera
    if cap is not None and cap.isOpened():
        cap.release()

    # Stop the audio playback thread
    audio_queue.put(None)  # Send exit signal to the thread
    audio_thread.join()  # Wait for the thread to finish

    return "Detection Stopped"

# Add a signal handler for Ctrl+C
def graceful_shutdown(signal, frame):
    global detection_active, cap, audio_thread, audio_queue

    print("\nCtrl+C detected. Shutting down gracefully...")

    # Stop detection if it's running
    detection_active = False
    if 'cap' in globals() and cap.isOpened():
        cap.release()
        print("Released camera resources.")

    # Stop the audio playback thread
    if audio_queue.qsize() > 0:
        audio_queue.put(None)  # Send the exit signal to the thread
    if audio_thread.is_alive():
        audio_thread.join()
        print("Stopped audio playback thread.")

    # Exit the Flask app explicitly
    print("Exiting the program...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, graceful_shutdown)

def generate_frames():
    global output_frame, lock, detection_active
    frame_count = 0
    while detection_active:
        try:
            ret, frame = cap.read()
            if not ret:
                break

            # Add your processing logic here
            frame_count += 1
            if frame_count % 15 != 0:
                continue

            frame = process_frame(frame)  # Process frame
            with lock:
                output_frame = frame.copy()
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

    print("Stopped generate_frames thread.")


def stream_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                print("No frame available for streaming.")
                continue

            _, encoded_frame = cv2.imencode('.jpg', output_frame)
            frame_bytes = encoded_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stream_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("Shutting down Flask server...")
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()  # Shutdown the server
    return "Server shutting down..."


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
        graceful_shutdown(None, None)
    finally:
        # Stop detection gracefully
        detection_active = False  # Stop the detection thread

        # Release the camera
        if 'cap' in globals() and cap.isOpened():
            cap.release()
            print("Released camera resources.")

        # Stop the audio playback thread
        if audio_queue.qsize() > 0:
            audio_queue.put(None)  # Send the exit signal to the thread
        if audio_thread.is_alive():
            audio_thread.join()
            print("Stopped audio playback thread.")