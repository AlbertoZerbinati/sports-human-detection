# Alberto Zerbinati

import torch
import torchvision.transforms as transforms
from model import PeopleDetectionCNN
from PIL import Image, ImageDraw

# Load the model
model = PeopleDetectionCNN("cpu")
model.load_state_dict(torch.load("models/people_detection_model_normalized.pth"))
model.eval()

# Load image
image = Image.open("data/Sport_scene_dataset/Images/im1.jpg")

# Preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Sliding window parameters
d = min(image.size[0], image.size[1]) / 5
window_size = (128, 256)
step_size = 20
scales = [1]

input = torch.ones(1, 3, 100, 100)
output = model(input)

print(output.data)

output = model(transform(image).unsqueeze(0))
confidence, predicted = torch.max(output.data, 0)
confidence = confidence.item()
predicted = predicted.item()

print(output.data)

# import sys
# sys.exit()

# Counter for naming files
counter = 0

# List to store the coordinates of detected windows
detected_windows = []

import numpy as np

heatmap = np.zeros((image.height, image.width))

# Sliding window loop
for scale in scales:
    for x in range(0, int(image.width - window_size[0] * scale), step_size):
        for y in range(0, int(image.height - window_size[1] * scale), step_size):
            window = image.crop(
                (x, y, int(x + window_size[0] * scale), int(y + window_size[1] * scale))
            )
            window_tensor = transform(window).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(window_tensor)
                confidence, predicted = torch.max(output.data, 0)
                confidence = confidence.item()
                predicted = predicted.item()

            if predicted == 1 and confidence >= 0.98:
                detected_windows.append(
                    (
                        x,
                        y,
                        int(x + window_size[0] * scale),
                        int(y + window_size[1] * scale),
                        confidence,
                    )
                )

                x1, y1 = x, y
                x2, y2 = int(x + window_size[0] * scale), int(
                    y + window_size[1] * scale
                )
                heatmap[y1:y2, x1:x2] += confidence  # Increment the heatmap grid

                counter += 1
                print(counter, f"-> confidence: {confidence:.2f}, scale: {scale}")

print("Number of detected windows:", len(detected_windows))

# Sliding window parameters
scales = [0.8]

# Sliding window loop
for scale in scales:
    for x in range(0, int(image.width - window_size[0] * scale), step_size):
        for y in range(0, int(image.height - window_size[1] * scale), step_size):
            window = image.crop(
                (x, y, int(x + window_size[0] * scale), int(y + window_size[1] * scale))
            )
            window_tensor = transform(window).unsqueeze(0)  # Add batch dimension

            # Make prediction
            with torch.no_grad():
                output = model(window_tensor)
                confidence, predicted = torch.max(output.data, 0)
                confidence = confidence.item()
                predicted = predicted.item()

            if predicted == 0 and confidence >= 0.98:
                x1, y1 = x, y
                x2, y2 = int(x + window_size[0] * scale), int(
                    y + window_size[1] * scale
                )
                heatmap[y1:y2, x1:x2] -= confidence


# Create a draw object on a copy of the original image
output_image = image.copy()
draw = ImageDraw.Draw(output_image)

# Color the detected windows on the output image
for coords in detected_windows:
    coords, confidence = coords[:-1], coords[-1]
    draw.rectangle(coords, outline="red" if confidence > 0.98 else "lightblue", width=1)

# Save the output image
output_image.save("output/detected_humans.jpg")

import matplotlib.pyplot as plt

# Create the heatmap
plt.imshow(heatmap, cmap="Blues", interpolation="nearest")
plt.colorbar()
plt.title("Human Detection Heatmap")
plt.savefig("output/heatmap.jpg")
