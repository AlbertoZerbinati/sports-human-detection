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
argparse.add_argument("--num_crops", type=int, default=6)

# Get the args
args = argparse.parse_args()
data_folder = args.data_folder
save_folder = args.save_folder
keep_percentage = args.keep_percentage
num_crops = args.num_crops

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
    f for f in os.listdir(data_folder) if f.endswith((".jpg", ".jpeg", ".png", ".bmp"))
]
# Filter out some images
np.random.seed(42)
image_files = np.random.choice(image_files, int(len(image_files) * keep_percentage))

print("extracting background crops from dataset...")

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

    # save list of all people detections in a list
    people_detections = []

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

                if h > 0 and w > 0:
                    # add the coordinates of the bounding box to the list
                    people_detections.append([x, y, w, h])

    # extract random patches of random dimensions that do not contain people
    # and save them as separate images
    # retry until num_crops such patches are found
    num_saved = 0
    while num_saved < num_crops:
        # get random width and height
        w = np.random.randint(100, 250)
        h = np.random.randint(100, 250)

        # get random top-left corner coordinates
        x = np.random.randint(0, max(width - w, 1))
        y = np.random.randint(0, max(height - h, 1))

        # check if the crop contains a person, using a threshold of 15 pixels
        contains_person = False
        for person in people_detections:
            if (
                person[0] > x - 15
                and person[1] > y - 15
                and person[0] + person[2] < x + w + 15
                and person[1] + person[3] < y + h + 15
            ):
                contains_person = True
                break

        # if the crop does not contain a person, save it as a separate image
        if not contains_person:
            crop = image[y : y + h, x : x + w]
            # check if the crop is empty
            if crop.size != 0:
                # extract the image name without the extension
                image_name, image_extension = image_file.split(".")

                file_name = (
                    f"{save_folder}/{image_name}_crop_{num_saved}_.{image_extension}"
                )
                cv2.imwrite(file_name, crop)
                num_saved += 1
