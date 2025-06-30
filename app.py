import os
import time

from flask import Flask, render_template, Response, request, jsonify
from functions import process_frame, audio_player
import functions

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

# Replace this with your phone's IP
IP_CAMERA_URL = "http://192.168.137.199:8080//video"

# Your existing code here (YOLO setup, SORT setup, etc.)
# Keep your `process_frame` function and other helper functions unchanged.

@app.route('/start_detection')
def start_detection():
    global detection_thread, detection_active, cap

    if detection_active:
        return "Detection is already running"

    # Ensure previous thread is not hanging
    if detection_thread and detection_thread.is_alive():
        return "Previous detection is still shutting down. Please wait."

    try:
        cap = cv2.VideoCapture(IP_CAMERA_URL)
        # cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = None
            return "Error: Could not open IP camera"

        detection_active = True
        detection_thread = threading.Thread(target=generate_frames, daemon=True)
        detection_thread.start()
        print("Detection started.")
        return "Detection Started"

    except Exception as e:
        print(f"Error in start_detection: {e}")
        return f"Error: {str(e)}"


@app.route('/stop_detection')
def stop_detection():
    global detection_active, cap, detection_thread

    if not detection_active:
        return "Detection is not running"

    print("Stopping detection...")

    detection_active = False  # Signal the thread to stop

    # Join the thread
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=5)
        print("Detection thread stopped.")
    detection_thread = None

    # Release the camera
    if cap:
        try:
            cap.release()
            print("Camera released.")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        finally:
            cap = None

    return "Detection Stopped"

@app.route('/latest_text')
def get_latest_text():
    # print("Text RECIEVED:", latest_extracted_text)
    return functions.latest_extracted_text or "Waiting for bus detection..."
    # return "YESSSS working"

@app.route('/bus_history')
def bus_history_route():
    return jsonify(functions.bus_history[-20:][::-1])  # Last 20, most recent first

# Add a signal handler for Ctrl+C
def graceful_shutdown(signal, frame):
    global detection_active, cap, audio_thread, audio_queue

    print("\nCtrl+C detected. Shutting down gracefully...")

    # Stop detection if it's running
    detection_active = False
    if 'cap' in globals() and cap is not None and cap.isOpened():
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
    global output_frame, lock, detection_active, cap
    frame_count = 0

    print("Started generate_frames thread.")

    while detection_active:
        try:
            if cap is None or not cap.isOpened():
                print("Camera lost. Restarting...")
                cap = cv2.VideoCapture(IP_CAMERA_URL)
                if not cap.isOpened():
                    print("Failed to reopen camera. Retrying in 2 seconds...")
                    time.sleep(2)
                    continue

            ret, frame = cap.read()
            if not ret:
                print("Warning: Empty frame received.")
                time.sleep(0.5)
                continue

            frame_count += 1
            if frame_count % 15 != 0:
                continue

            frame = process_frame(frame)
            with lock:
                output_frame = frame.copy()

        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(2)
            continue

    print("Exited generate_frames thread.")


def stream_frames():
    global output_frame, lock
    while True:
        with lock:
            if output_frame is None:
                print("Waiting for first frame...")
                time.sleep(1)  # Small delay before retrying
                continue

            _, encoded_frame = cv2.imencode('.jpg', output_frame)
            frame_bytes = encoded_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def free_flask_port_windows(port=5000):
    try:
        pid = os.popen(f"netstat -ano | findstr :{port}").read()
        if "LISTENING" in pid:
            pid_number = pid.strip().split()[-1]
            os.system(f"taskkill /F /PID {pid_number}")
            print(f"Killed Flask process on port {port}")
    except Exception as e:
        print(f"Error freeing port {port}: {e}")

# Call this before running Flask
free_flask_port_windows(5000)

@app.route('/')
def index():
    return render_template('frontend/index.html')

@app.route('/live')
def live():
    return render_template('frontend/golive.html')  # Serve the live page

@app.route('/history')
def history_page():
    return render_template('frontend/history.html')

# @app.route('/home')
# def live():
#     return render_template('frontend/index.html')  # Serve the live page

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
        if 'cap' in globals() and cap is not None and cap.isOpened():
            cap.release()
            print("Released camera resources.")

        # Stop the audio playback thread
        if audio_queue.qsize() > 0:
            audio_queue.put(None)  # Send the exit signal to the thread
        if audio_thread.is_alive():
            audio_thread.join()
            print("Stopped audio playback thread.")