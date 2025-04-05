#testing the detection model
import cv2
import os
import datetime
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")  # Update with your trained model path

# Function to get a unique identifier (timestamp)
def get_unique_filename(base_name, ext):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
    return f"{base_name}_{timestamp}.{ext}"

# Test on an image
def test_image(image_path):
    print(f"Processing image: {image_path}")
    results = model(image_path, show=True)  # Display predictions
    for idx, result in enumerate(results):
        # Generate a unique name with a more precise timestamp
        unique_name = get_unique_filename(f"annotated_image_{idx}", "jpg")
        save_path = os.path.join("C:/PROJECT/output", unique_name)  # Save in the output directory
        annotated_image = result.plot()  # Annotated image
        cv2.imwrite(save_path, annotated_image)
        print(f"Saved annotated image to: {save_path}")


## Test on an image (hidden confidence score)
# def test_image(image_path):
#     print(f"Processing image: {image_path}")
#     results = model(image_path)  # Run inference
#
#     for idx, result in enumerate(results):
#         unique_name = get_unique_filename(f"annotated_image_{idx}", "jpg")
#         save_path = os.path.join("C:/PROJECT/output", unique_name)
#
#         # Get the original image
#         annotated_image = result.orig_img
#
#         # Draw only labels and boxes
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
#             label = model.names[int(box.cls[0])]  # Get class label
#
#             # Draw rectangle
#             cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#             # Put label without confidence score
#             cv2.putText(annotated_image, label, (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         cv2.imwrite(save_path, annotated_image)
#         print(f"Saved annotated image to: {save_path}")

# Test on a video
def test_video(video_path):
    print(f"Processing video: {video_path}")
    # Predict on video and let it save in the default runs/detect/predict folder
    results = model.predict(source=video_path, show=True, save=True)  # Predict on video
    print(f"Annotated video saved in the default YOLO runs directory.")

# Specify input file paths
test_image_paths = [
    "test/rac722-cdl_14345772115_o.jpg"
    # "test/rrc87-full_14227157087_o.jpg",  # Replace with your image path
    # "test/rak158-tvm-mdy_14289621556_o.jpg",
    # "test/KSRTC.jpg",
    # "test/ksrtc2.jpg",
    # "test/test.jpg",
    # "test/test2.jpg",  # Replace with your image path
    # "test/IMG20250113063038.jpg"
]
test_video_paths = [
    # "test/KSRTC Bus Video.mp4",  # Replace with your video path
    # "test/KSRTC Bus2.mp4",
    # "test/KSRTC Bus3.mp4",
    # "test/KSRTC Bus4.mp4",  # Add more video paths here
    # "test/VID20250113062216.mp4",
    # "test/VID20250113063043.mp4",
    # "test/VID20250113063317.mp4"
    # "C:/PROJECT/test/VID20250209164426.mp4"
]

# Run tests on images
for img_path in test_image_paths:
    test_image(img_path)

# Run tests on videos
for vid_path in test_video_paths:
    test_video(vid_path)
