from ultralytics import YOLO

if __name__ == '__main__':
    # Load trained YOLO model
    model = YOLO("C:/PROJECT/runs/detect/bus_board_v11/weights/best.pt")

    # Run validation on the dataset
    metrics = model.val(data="C:/PROJECT/dataset.yaml")

    # Print evaluation metrics
    print(metrics)
