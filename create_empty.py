import os

# Path to your dataset directory containing images
image_directory = "C:/Users/LEGION/Desktop/Negative Samples/images"  # Replace with your actual image directory path
output_directory = "C:/Users/LEGION/Desktop/Negative Samples/labels"  # Replace with your label files directory path

# Ensure the label directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop through all images in the directory
for image_name in os.listdir(image_directory):
    # Check if the file is an image (you can filter based on extensions)
    if image_name.endswith(('.jpg', '.jpeg', '.png')):  # Adjust based on your dataset's image formats
        # Generate the label file name (change extension to .txt)
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_filepath = os.path.join(output_directory, label_filename)

        # Create an empty label file (if it doesn't exist)
        if not os.path.exists(label_filepath):
            with open(label_filepath, 'w') as label_file:
                pass  # Empty file created
            print(f"Created empty label file for: {image_name}")

        # print(f"Created empty label file for: {image_name}")

print("All empty label files created.")
