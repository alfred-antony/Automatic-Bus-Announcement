from ultralytics import YOLO

if __name__ == "__main__":
    print("Ultralytics and YOLOv11 imported successfully!")

    # Load a YOLOv11 model (use pre-trained weights)
    model = YOLO("yolo11n.pt")  # Use 'yolo11s.pt' for better accuracy

    # Train the model
    model.train(
        data="C:/PROJECT/dataset.yaml",  # Path to dataset.yaml
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device="cuda",
        name="bus_board_v11"
    )

    print("Training complete. Check the 'runs' folder for results.")

    # Evaluate the trained model and print metrics
    metrics = model.val()  # Run validation on the trained model
    print("\n=== Model Performance Metrics ===")
    print(f"mAP50: {metrics.box.map:.4f}")
    print(f"mAP50-95: {metrics.box.map50_95:.4f}")
    print(f"Precision: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")
    print(f"Speed (Inference time per image): {metrics.speed:.2f} ms")
