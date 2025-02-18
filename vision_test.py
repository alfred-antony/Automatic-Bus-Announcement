import os
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the environment variable for authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:/PROJECT/transit-vision-2e6350ac393a.json'

# Initialize a Vision API client
client = vision.ImageAnnotatorClient()

# Read the image file
with open('C:/PROJECT/test/KSRTCtemp.jpg', 'rb') as image_file:
    content = image_file.read()
# "C:\Users\LEGION\Pictures\Screenshots\Screenshot 2025-01-21 111240.png"
# "C:\PROJECT\test\crop4.jpg"
# "C:\PROJECT\output\preprocessed_image_1_20250211_003611_056.jpg"
# "C:\PROJECT\output\preprocessed_image_1_20250211_003612_656.jpg"
# "C:\PROJECT\output\annotated_image_0_20250116_014057_885.jpg"
# "C:\PROJECT\test\1-256-_jpg.rf.a83300982023ebb888433db94f024313.jpg"
# "C:\PROJECT\test\1-655-_jpg.rf.2f2cc2e351f378c87150a4e8a8c1ac84.jpg"
# "C:\PROJECT\test\1-671-_jpg.rf.f9706ddd4d968a26ee2676f14ceaa544.jpg"
# "C:\PROJECT\test\crop.jpg"
# Create an image object
image = types.Image(content=content)

# Perform text detection
response = client.text_detection(image=image)
texts = response.text_annotations

# Print the extracted text
for text in texts:
    print(text.description)
