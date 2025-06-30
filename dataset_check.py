import os

# Define the paths to the images and labels folders
images_path = "C:/PROJECT/datasets/train/images"
labels_path = "C:/PROJECT/datasets/train/labels"

# Get a list of image files and label files
image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]  # Adjust extensions if necessary
label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

# Get the base names (without extensions) of the files
image_basenames = set(os.path.splitext(f)[0] for f in image_files)
label_basenames = set(os.path.splitext(f)[0] for f in label_files)

# Find missing label files
missing_labels = image_basenames - label_basenames

# Find extra label files (labels without corresponding images)
extra_labels = label_basenames - image_basenames

# Output results
if missing_labels:
    print(f"Missing label files for images: {missing_labels}")
else:
    print("All images have corresponding labels.")

if extra_labels:
    print(f"Extra label files without corresponding images: {extra_labels}")
else:
    print("No extra label files found.")
