from ultralytics import YOLO
import cv2
import os

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/train3/weights/best.pt")  # Update with your trained model path

# Create output directory if it doesn't exist
output_dir = "C:/PROJECT/output/"
os.makedirs(output_dir, exist_ok=True)

# Test on an image
def test_image(image_path, save_dir):
    print(f"Processing image: {image_path}")
    results = model(image_path, show=True)  # Display predictions
    for idx, result in enumerate(results):
        # Save each result
        annotated_image = result.plot()  # Annotated image
        save_path = os.path.join(save_dir, f"annotated_image_{idx}.jpg")
        cv2.imwrite(save_path, annotated_image)
        print(f"Saved annotated image to: {save_path}")

# Test on a video
def test_video(video_path, save_dir):
    print(f"Processing video: {video_path}")
    results = model.predict(source=video_path, show=True, save=True, save_dir=save_dir)  # Predict on video
    print(f"Annotated video saved in: {save_dir}")

# Specify input file paths
test_image_path = "test/rrc87-full_14227157087_o.jpg"  # Replace with your image path
test_video_path = "test/KSRTC Bus Video.mp4"  # Replace with your video path

# Run tests
test_image(test_image_path, output_dir)
test_video(test_video_path, output_dir)
