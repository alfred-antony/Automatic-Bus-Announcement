# 🚌 Automatic Bus Announcement System | Transit Vision 🚏

An AI-powered real-time bus arrival announcement system built using Python, Flask, OpenCV, YOLOv8 Object Detection, OCR (Google Vision API), and SORT Tracking Algorithm.

---

## 📌 Project Overview

Transit Vision is an **Automatic Bus Announcement System** designed for bus stands to assist passengers by announcing incoming buses. The system uses a **live camera feed** to detect KSRTC buses, recognize their **route boards**, and **announce bus arrival information with voice output**.

This is especially useful for **visually impaired passengers** and for **crowded or large bus stands**.

---

## 🚀 Features

- ✅ Real-time **Bus Detection** using YOLOv8
- ✅ **Route Board Text Extraction** using Google Vision OCR API
- ✅ **Unique Bus Tracking** using SORT (Simple Online Realtime Tracking)
- ✅ **Text-to-Speech Bus Announcements**
- ✅ **Bus Arrival History Log** with timestamp
- ✅ Live Web Dashboard built with Flask + HTML/CSS/JavaScript
- ✅ Start/Stop detection from the frontend
- ✅ Scrollable detection history (Bus Log)
- ✅ Graceful shutdown of processes and resources

---

## 🖥️ Technologies Used

- Python
- Flask (Web Framework)
- OpenCV (Video Streaming & Image Processing)
- YOLOv8 (Bus Detection)
- Google Vision API (OCR for Bus Route Text)
- SORT Algorithm (Object Tracking)
- HTML / CSS / JavaScript (Frontend)
- Threading and Queues (for parallel frame processing and audio playback)

---

## ⚙️ How It Works

1. Live camera feed captures frames.
2. YOLOv8 model detects buses and name boards.
3. OCR extracts Malayalam text from detected name boards.
4. SORT assigns a unique ID to each bus for tracking across frames.
5. For each unique bus detected with a new route text, the system:
    - Adds it to arrival history
    - Announces the bus arrival using TTS (Text-to-Speech)
6. Frontend shows:
    - Live video feed
    - Current bus being detected
    - Real-time bus arrival history (log)
    - Detection start/stop control buttons

---

## 📸 Screenshots

*(Add your frontend UI screenshots, sample console logs, and detection sample images here)*

---

## 📂 Project Structure (Key Files/Folders)

app.py # Main Flask app
functions.py # Detection logic, OCR, tracking, history, TTS
templates/ # HTML frontend pages (index.html, golive.html, etc)
static/ # CSS, JS, images
history.json # Persistent Bus Arrival History


---

## 🧠 Algorithms Used

- **YOLOv8** → Bus and Board Detection
- **Google Vision API OCR** → Text extraction from bus boards
- **SORT (Kalman Filter + Hungarian Algorithm)** → Object Tracking across frames
- **TTS (Text-to-Speech)** → Bus arrival audio announcements

---

## 🙋🏻‍♂️ Team Contribution / Customization Done

- ✅ Designed full system flow from live camera to announcement.
- ✅ Customized YOLOv8 model for KSRTC bus detection.
- ✅ Integrated Google Vision OCR for Malayalam text reading.
- ✅ Implemented SORT tracking for unique bus ID assignment.
- ✅ Developed real-time frontend dashboard with history log.
- ✅ Managed server-side threading, API rate control, and graceful shutdown.
- ✅ Implemented history.json file saving for persistent log.

---
