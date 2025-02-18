import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board7/weights/best.pt")  # Update path if needed

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame)
    detections = results[0].boxes  # Get detected objects

    # Check if any objects were detected
    if detections is not None and len(detections) > 0:
        annotated_frame = results[0].plot()  # Annotate the frame

        # Save the frame
        output_path = f"C:/PROJECT/output/frame_{frame_count}.jpg"
        cv2.imwrite(output_path, annotated_frame)
        frame_count += 1

        # Show the annotated frame
        cv2.imshow("Live Inference", annotated_frame)
    else:
        # If nothing detected, show the original frame
        cv2.imshow("Live Inference", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
