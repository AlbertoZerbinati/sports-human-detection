# Alberto Zerbinati

import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

argparse = argparse.ArgumentParser()
argparse.add_argument("--data_folder", type=str, default="data/images")
argparse.add_argument("--save_folder", type=str, default="data/positive")
argparse.add_argument("--keep_percentage", type=float, default=1)

# Get the args
args = argparse.parse_args()
data_folder = args.data_folder
save_folder = args.save_folder
keep_percentage = args.keep_percentage

# Load YOLOv3 pre-trained model
net = cv2.dnn.readNet(
    "data/data_processing/yolov3.weights", "data/data_processing/yolov3.cfg"
)

# Load COCO class labels (you may need to modify the path)
classes = []
with open("data/data_processing/coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get a list of all image files in the folder
image_files = [
    f for f in os.listdir(data_folder) if f.endswith((".jpg", ".jpeg", ".png", ".JPG"))
]
# Filter out some images
np.random.seed(42)
image_files = np.random.choice(image_files, int(len(image_files) * keep_percentage))

# Loop through each image file
for image_file in tqdm(image_files):
    # Load the image using OpenCV
    image_path = os.path.join(data_folder, image_file)
    image = cv2.imread(image_path)

    # Get the image dimensions
    height, width = image.shape[:2]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run forward pass to get bounding box predictions
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    # Iterate through detections
    for detection in detections:
        for i, obj in enumerate(detection):
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            label = classes[class_id]

            if confidence > 0.9 and label == "person":
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                # save the bounding box crop as a separate image
                crop = image[y : y + h, x : x + w]
                # check if the crop is empty
                if crop.size != 0:
                    # extract the image name without the extension
                    image_name, image_extension = image_file.split(".")

                    file_name = (
                        f"{save_folder}/{image_name}_crop_{i}_.{image_extension}"
                    )
                    cv2.imwrite(file_name, crop)
