import os
import cv2
from google.cloud import vision
from google.cloud.vision_v1 import types
import datetime

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/PROJECT/transit-vision-2e6350ac393a.json'  # Update with the new path

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

# Process a cropped image
def process_cropped_image(image_path):
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)

    # Extract the main route name using Vision API
    main_route_name, processed_board = extract_main_route_name(image)

    # Overlay the extracted main route name on the original board image
    processed_board = overlay_text(processed_board, main_route_name, (10, 30))

    # Save the final processed board image with overlaid text
    unique_final_name = get_unique_filename("final_board", "jpg")
    final_save_path = os.path.join("C:/PROJECT/output", unique_final_name)
    cv2.imwrite(final_save_path, processed_board)
    print(f"Saved final processed image to: {final_save_path}")

# Specify input file paths
cropped_image_paths = [
    "test/1-256-_jpg.rf.a83300982023ebb888433db94f024313.jpg",  # Replace with your cropped image paths
    "test/1-595-_jpg.rf.7d455937715de9bebf5c7c2173ca9227.jpg",
    "test/1-655-_jpg.rf.2f2cc2e351f378c87150a4e8a8c1ac84.jpg",
    "test/1-671-_jpg.rf.f9706ddd4d968a26ee2676f14ceaa544.jpg",
    "test/crop3.jpg"
]

# Run tests on cropped images
for img_path in cropped_image_paths:
    process_cropped_image(img_path)
