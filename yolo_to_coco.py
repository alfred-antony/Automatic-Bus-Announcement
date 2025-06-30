import os
import json
import cv2
from glob import glob

# Path to dataset
YOLO_DATASET_PATH = r"C:\Users\LEGION\Pictures\datasets_coco"  # Change this to your actual dataset path
IMAGE_PATHS = {
    "train": os.path.join(YOLO_DATASET_PATH, "train/images"),
    "val": os.path.join(YOLO_DATASET_PATH, "val/images"),
}
ANNOTATION_PATHS = {
    "train": os.path.join(YOLO_DATASET_PATH, "train/labels"),
    "val": os.path.join(YOLO_DATASET_PATH, "val/labels"),
}


# Define COCO structure
def yolo_to_coco(split):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_dict = {}  # Stores class mappings
    annotation_id = 1
    category_id = 1  # Start category IDs from 1

    # Read all YOLO label files
    for image_id, txt_file in enumerate(glob(os.path.join(ANNOTATION_PATHS[split], "*.txt"))):
        img_file = os.path.join(IMAGE_PATHS[split], os.path.basename(txt_file).replace(".txt", ".jpg"))

        # Check if image exists
        if not os.path.exists(img_file):
            print(f"Skipping {txt_file}, image not found.")
            continue

        # Read image dimensions
        img = cv2.imread(img_file)
        height, width, _ = img.shape

        # Add image info to COCO
        coco_format["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_file),
            "width": width,
            "height": height
        })

        with open(txt_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

                # Convert YOLO format (normalized) to COCO format (absolute)
                x_min = int((x_center - bbox_width / 2) * width)
                y_min = int((y_center - bbox_height / 2) * height)
                bbox_width = int(bbox_width * width)
                bbox_height = int(bbox_height * height)

                # If class is not in the dictionary, add it
                if class_id not in category_dict:
                    category_dict[class_id] = category_id
                    coco_format["categories"].append({
                        "id": category_id,
                        "name": f"class_{int(class_id)}",
                        "supercategory": "object"
                    })
                    category_id += 1

                # Add annotation
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "category_id": category_dict[class_id],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

    # Save to JSON file
    output_file = os.path.join(YOLO_DATASET_PATH, f"annotations/{split}_coco.json")
    with open(output_file, "w") as json_file:
        json.dump(coco_format, json_file, indent=4)

    print(f"COCO annotations saved: {output_file}")


# Convert both train and val datasets
yolo_to_coco("train")
yolo_to_coco("val")

print("Conversion Completed!")
