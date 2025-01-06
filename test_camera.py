import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("C:/PROJECT/runs/detect/train3/weights/best.pt")  # Update path if needed

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)  # Run inference
    annotated_frame = results[0].plot()  # Annotate the frame

    # Save every frame (or conditionally)
    output_path = f"C:/PROJECT/output/frame_{frame_count}.jpg"
    cv2.imwrite(output_path, annotated_frame)
    frame_count += 1

    # Show annotated frame
    cv2.imshow("Live Inference", annotated_frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
