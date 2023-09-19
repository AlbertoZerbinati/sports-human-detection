#ifndef METRICS_HPP
#define METRICS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace Metrics {

struct BoundingBox {
    int x, y, width, height, teamID;
};

// Function to calculate Intersection over Union (IoU), used for evaluating
// player and playing field segmentation
float calculateIoU(const BoundingBox& box1, const BoundingBox& box2);

// Function to calculate Mean Average Precision (mAP), used for evaluating
// player detection
float calculateMAP(const std::vector<std::vector<BoundingBox>>& groundTruths,
                   const std::vector<std::vector<BoundingBox>>& predictions);

}  // namespace Metrics

#endif  // METRICS_HPP
