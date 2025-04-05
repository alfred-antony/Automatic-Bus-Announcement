#for extracting largest text
import os
import cv2
from ultralytics import YOLO
from google.cloud import vision
from google.cloud.vision_v1 import types
import datetime

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/PROJECT/transit-vision-2e6350ac393a.json'  # Update with the new path

# Load the trained YOLO model
model = YOLO("C:/PROJECT/runs/detect/bus_and_board10/weights/best.pt")  # Update with your trained model path

# Initialize a Vision API client
vision_client = vision.ImageAnnotatorClient()


# Function to get a unique identifier (timestamp)
def get_unique_filename(base_name, ext):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Includes milliseconds
    return f"{base_name}_{timestamp}.{ext}"


# Extract text using Google Vision API and find the largest text block
def extract_main_route_name(image):
    success, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()
    vision_image = types.Image(content=content)
    response = vision_client.text_detection(image=vision_image)
    texts = response.text_annotations

    if not texts:
        return "", image

    largest_text = ""
    max_area = 0

    print("All Detected Texts:")
    for text in texts[1:]:
        print(text.description)
        vertices = text.bounding_poly.vertices
        x1, y1 = vertices[0].x, vertices[0].y
        x2, y2 = vertices[2].x, vertices[2].y
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if area > max_area:
            max_area = area
            largest_text = text.description.strip()
            # Draw rectangle around the largest text
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print("Largest Text:", largest_text)  # Print the largest text
    return largest_text, image


# Overlay the extracted text on the original image
def overlay_text(image, text, position):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return image


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

    cropped_boards = [image[y:y + h, x:x + w] for (x, y, w, h) in bounding_boxes]
    for idx, board in enumerate(cropped_boards):
        # Extract the main route name using Vision API
        main_route_name, processed_board = extract_main_route_name(board)

        # Overlay the extracted main route name on the original board image
        processed_board = overlay_text(processed_board, main_route_name, (10, 30))

        # Save the final processed board image with overlaid text
        unique_final_name = get_unique_filename(f"final_board_{idx}", "jpg")
        final_save_path = os.path.join("C:/PROJECT/output", unique_final_name)
        cv2.imwrite(final_save_path, processed_board)
        print(f"Saved final processed image to: {final_save_path}")


# Specify input file paths
test_image_paths = [
    # "test/rrc87-full_14227157087_o.jpg",
    # "test/rak158-tvm-mdy_14289621556_o.jpg",
    # "test/KSRTC.jpg",
    # "test/ksrtc2.jpg",
    # "test/test.jpg",
    # "test/test2.jpg",
    # "test/IMG20250113063038.jpg"
    # "test/Screenshot (201).png",
    # "test/Screenshot (202).png",
    # "test/Screenshot (203).png",
    "test/Screenshot (204).png"
]

# Run tests on images
for img_path in test_image_paths:
    process_image(img_path)