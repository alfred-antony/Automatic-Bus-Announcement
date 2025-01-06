from ultralytics import YOLO

# Load a trained YOLOv8 model
model = YOLO("path/to/trained_model.pt")  # Replace with your trained model path
# model = YOLO("yolov8n.pt")

# Perform inference on an image or video
results = model("path/to/image_or_video.jpg", show=True)  # Replace with input file path

# Save the predictions (optional)
results.save("path/to/save_dir")  # Replace with directory to save predictions
