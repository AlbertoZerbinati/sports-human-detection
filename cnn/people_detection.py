import numpy as np
import torch
import torchvision.transforms as transforms
from model import PeopleDetectionCNN
from people_detection_util import (compute_weighted_loss, perform_kmeans,
                                   read_ground_truth_labels, transform)
from PIL import Image, ImageDraw

# Load the model
model = PeopleDetectionCNN("cuda")
model.load_state_dict(torch.load("models/people_detection_model.pth"))
model.eval()

# Read ground truth labels
ground_truth = read_ground_truth_labels("data/annotations.csv")

# Grid search parameters
scales = [0.8, 1.5, 2.5]
alpha = 0.8  # increase to penalize kmeans loss
beta = 0.2  # decrease to penalize low confidence
gamma = 28000  # increase to penalize bigger scales
delta = 300  # increase to penalize more clusters

# Main loop over all images
for img_name, ground_truth_boxes in ground_truth.items():
    image = Image.open(f"data/Sport_scene_dataset/Images/{img_name}")

    print("----------------------")
    print("Working on image:", img_name)
    # print("Image size:", image.size)
    # print()

    # Sliding window parameters
    d = min(image.size[0], image.size[1]) / 5
    window_size = (int(d), int(d * 5 / 3))
    step_size = int(d / 8)

    best_loss = float("inf")
    best_scale = None
    best_detected_windows = None
    best_labels = None
    best_k = None

    # print("Scale:", scale)
    detected_centers = []
    detected_windows = []

    # Sliding window loop
    for scale in scales:
        for x in range(0, int(image.width - window_size[0] * scale), step_size):
            for y in range(0, int(image.height - window_size[1] * scale), step_size):
                window = image.crop(
                    (
                        x,
                        y,
                        int(x + window_size[0] * scale),
                        int(y + window_size[1] * scale),
                    )
                )
                window = transform(window)
                window_tensor = transforms.ToTensor()(window).unsqueeze(0)

                with torch.no_grad():
                    output = model(window_tensor)
                    confidence, predicted = torch.max(output.data, 0)
                    confidence = confidence.item()
                    predicted = predicted.item()

                if predicted == 1 and confidence >= 0.9:
                    center_x = x + window_size[0] * scale / 2
                    center_y = y + window_size[1] * scale / 2
                    detected_centers.append([center_x, center_y])
                    detected_windows.append(
                        (
                            x,
                            y,
                            int(x + window_size[0] * scale),
                            int(y + window_size[1] * scale),
                            confidence,
                        )
                    )

        print("Number of detected windows:", len(detected_windows))

        for k in range(1, 11):  # Assuming max 10 clusters
            if len(detected_centers) < k:
                break
            labels, loss = perform_kmeans(detected_centers, k)

            # compute unsupervised metrics
            confidence_scores = [window[-1] for window in detected_windows]

            if len(confidence_scores) > 0:  # Avoid empty list
                median_confidence = np.median(confidence_scores)
                mean_confidence = np.mean(confidence_scores)
            else:
                median_confidence = 0
                mean_confidence = 0

            weighted_loss = compute_weighted_loss(
                loss,
                median_confidence,
                mean_confidence,
                scale,
                k,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )

            if weighted_loss < best_loss:
                # print(f"New best weighted loss at k {k}: {weighted_loss}")
                best_loss = weighted_loss
                best_scale = scale
                best_detected_windows = detected_windows
                best_labels = labels
                best_k = k

    print("Best scale:", best_scale)
    print("Number of players found:", best_k)

    # Create a draw object on a copy of the original image
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    # Choose best bounding boxes based on clustering
    unique_labels = set(best_labels)
    for label in unique_labels:
        cluster_windows = [
            best_detected_windows[i]
            for i in range(len(best_detected_windows))
            if best_labels[i] == label
        ]
        cluster_windows = sorted(
            cluster_windows, key=lambda x: x[-1], reverse=True
        )  # Sort by confidence
        best_window = cluster_windows[0]  # Choose the box with the highest confidence
        draw.rectangle(best_window[:-1], outline="red", width=2)

        # also draw the ground truth boxes
        for box in ground_truth_boxes:
            draw.rectangle(box, outline="green", width=2)

    predicted_boxes = [
        box[:-1] for box in best_detected_windows
    ]  # Remove confidence scores

    # Save the output image
    output_image.save(f"output/{img_name}")
