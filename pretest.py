#test preprocessing techniques
import os
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")  # Update with your trained model path


def nothing(x):
    pass


# Create trackbars for adjusting preprocessing parameters
cv2.namedWindow('Preprocessing')
cv2.createTrackbar('Gaussian Kernel', 'Preprocessing', 1, 20, nothing)
cv2.createTrackbar('Block Size', 'Preprocessing', 11, 25, nothing)
cv2.createTrackbar('C', 'Preprocessing', 2, 10, nothing)


def preprocess_image(image, k, block_size, c):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
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

    cropped_boards = []
    for (x, y, w, h) in bounding_boxes:
        cropped_boards.append(image[y:y + h, x:x + w])

    while True:
        k = cv2.getTrackbarPos('Gaussian Kernel', 'Preprocessing')
        block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing')
        c = cv2.getTrackbarPos('C', 'Preprocessing')

        if k % 2 == 0:
            k += 1
        if block_size % 2 == 0:
            block_size += 1

        preprocessed_boards = [preprocess_image(board, k, block_size, c) for board in cropped_boards]

        for idx, preprocessed_image in enumerate(preprocessed_boards):
            cv2.imshow(f'Preprocessed Board {idx + 1}', preprocessed_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# Specify input file paths
test_image_paths = [
    # "test/rrc87-full_14227157087_o.jpg",
    # "test/rak158-tvm-mdy_14289621556_o.jpg",
    "test/KSRTC.jpg",
    "test/ksrtc2.jpg",
    "test/test.jpg",
    "test/test2.jpg",
    "test/IMG20250113063038.jpg"
]

# Run tests on images
for img_path in test_image_paths:
    process_image(img_path)
