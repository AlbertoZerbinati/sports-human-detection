# Alberto Zerbinati

import csv

import torchvision.transforms as transforms
from sklearn.cluster import KMeans

# Preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((100, 100)),
    ]
)


# Function to perform k-means
def perform_kmeans(centers, k):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(centers)
    return kmeans.labels_, kmeans.inertia_


# Function to compute the weighted loss
def compute_weighted_loss(
    kmeans_loss,
    median_confidence,
    mean_confidence,
    scale,
    k,
    alpha=0.8,
    beta=0.2,
    gamma=1000.0,
    delta=50,
):
    scale_penalty = gamma * (scale**4)
    k_penalty = delta * (k**3)
    loss = (
        alpha * kmeans_loss
        - beta * (median_confidence + mean_confidence)
        + scale_penalty
        + k_penalty
    )
    return loss


def read_ground_truth_labels(filename):
    ground_truth = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            image_name, x1, y1, w, h = row
            box = [int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)]
            if image_name not in ground_truth:
                ground_truth[image_name] = []
            ground_truth[image_name].append(box)
    return ground_truth


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Calculate intersection rectangle coordinates
    x1_i = max(x1, x1_gt)
    y1_i = max(y1, y1_gt)
    x2_i = min(x2, x2_gt)
    y2_i = min(y2, y2_gt)

    # Calculate intersection and union areas
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    union = area_box1 + area_box2 - intersection

    # Avoid division by zero
    if union == 0:
        return 0

    return intersection / union


def calculate_total_iou(predicted_boxes, ground_truth_boxes):
    total_iou = 0
    matched_gt_boxes = set()

    for p_box in predicted_boxes:
        highest_iou = 0
        matched_box = None

        for idx, gt_box in enumerate(ground_truth_boxes):
            if idx in matched_gt_boxes:
                continue
            current_iou = iou(p_box, gt_box)
            if current_iou > highest_iou:
                highest_iou = current_iou
                matched_box = idx

        if matched_box is not None and highest_iou >= 0.5:  # 0.5 is the IoU threshold
            matched_gt_boxes.add(matched_box)
            total_iou += highest_iou

    return total_iou, len(matched_gt_boxes), len(predicted_boxes)
