from ultralytics import YOLO

if __name__ == "__main__":
    print("Ultralytics and YOLO imported successfully!")

    # Load a YOLO model (use pre-trained weights for YOLOv8n)
    model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy, 'yolov8n.pt' for faster training

    # Train the model
    model.train(
        data="C:/PROJECT/dataset.yaml",  # Path to dataset.yaml
        epochs=50,            # Reduced epochs for small dataset
        imgsz=640,            # Image size
        batch=16,             # Batch size
        workers=4,            # Number of workers
        device="cuda",        # Ensure GPU is used
        name="bus_and_board"
    )

    # Save results in the 'runs' directory
    print("Training complete. Check the 'runs' folder for results.")
