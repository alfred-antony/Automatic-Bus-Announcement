import os
import cv2
from ultralytics import YOLO
import datetime

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")  # Update with your trained model path


# Function to get a unique identifier (timestamp)
def get_unique_filename(base_name, ext):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
    return f"{base_name}_{timestamp}.{ext}"


# Crop the bus boards from the image
def crop_boards(image, bounding_boxes):
    cropped_images = []
    for (x, y, w, h) in bounding_boxes:
        cropped_image = image[y:y + h, x:x + w]
        cropped_images.append(cropped_image)
    return cropped_images


# Preprocess the cropped board images
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return thresh


# Process an image
def process_image(image_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    results = model(image)

    bounding_boxes = []
    for result in results:
        for box in result.boxes:
            if box.cls[0] == 1:  # Only process the board class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                bounding_boxes.append((x1, y1, w, h))

    cropped_boards = crop_boards(image, bounding_boxes)
    preprocessed_boards = [preprocess_image(board) for board in cropped_boards]

    for idx, preprocessed_image in enumerate(preprocessed_boards):
        # Save the preprocessed image with a unique name
        unique_name = get_unique_filename(f"preprocessed_image_{idx}", "jpg")
        save_path = os.path.join("C:/PROJECT/output", unique_name)
        cv2.imwrite(save_path, preprocessed_image)
        print(f"Saved preprocessed image to: {save_path}")


# Specify input file paths
test_image_paths = [
    "test/rrc87-full_14227157087_o.jpg",
    "test/rak158-tvm-mdy_14289621556_o.jpg",
    "test/KSRTC.jpg",
    "test/ksrtc2.jpg",
    "test/test.jpg",
    "test/test2.jpg",
    "test/IMG20250113063038.jpg"
]

# Run tests on images
for img_path in test_image_paths:
    process_image(img_path)
