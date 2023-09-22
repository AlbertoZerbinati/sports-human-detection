#ifndef METRICS_HPP
#define METRICS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/Utils.hpp"

/**
 * MetricsEvaluator class responsible for various types of metrics calculations.
 */
class MetricsEvaluator {
   public:
    /**
     * Calculates Mean Intersection-over-Union (MIoU) for segmentation task.
     * @param predicted Predicted labels in a cv::Mat object.
     * @param groundTruth Ground truth labels in a cv::Mat object.
     * @return The calculated MIoU value as a float.
     */
    float calculateMIoU(const cv::Mat& predicted, const cv::Mat& groundTruth);

    /**
     * Calculates Mean Average Precision (MAP) for object detection task.
     * @param groundTruths Vector of ground truth bounding boxes.
     * @param predictions Vector of predicted bounding boxes.
     * @return The calculated MAP value as a float.
     */
    float calculateMAP(
        const std::vector<Utils::PlayerBoundingBox>& groundTruths,
        const std::vector<Utils::PlayerBoundingBox>& predictions);

   private:
    /**
     * Calculates geometric Intersection-over-Union (IoU) between two bounding
     * boxes.
     * @param bb1 First bounding box.
     * @param bb2 Second bounding box.
     * @return The calculated IoU value as a float.
     */
    float calculateGeometricIoU(const Utils::PlayerBoundingBox& bb1,
                                const Utils::PlayerBoundingBox& bb2);

    /**
     * Calculates IoU for a single class.
     * @param predicted Predicted labels for a single class.
     * @param groundTruth Ground truth labels for a single class.
     * @param label Class label for which to calculate IoU.
     * @return The calculated IoU value for the class as a float.
     */
    float calculateClassIoU(const cv::Mat& predicted,
                            const cv::Mat& groundTruth, int label);

    /**
     * Calculates MAP for specific teams.
     * @param groundTruths Vector of ground truth bounding boxes for specific
     * teams.
     * @param predictions Vector of predicted bounding boxes for specific teams.
     * @param team1 First team ID.
     * @param team2 Second team ID.
     * @return The calculated MAP value for the teams as a float.
     */
    float calculateMAPForTeams(
        const std::vector<Utils::PlayerBoundingBox>& groundTruths,
        const std::vector<Utils::PlayerBoundingBox>& predictions, int team1,
        int team2);

    /**
     * Computes Average Precision (AP) for a single class with a given IoU
     * threshold.
     * @param predictions Vector of predicted bounding boxes for a single class.
     * @param groundTruths Vector of ground truth bounding boxes for a single
     * class.
     * @param iouThreshold IoU threshold for considering a prediction as a true
     * positive.
     * @return The calculated AP value for the class as a float.
     */
    float computeAPSingleClass(
        const std::vector<Utils::PlayerBoundingBox>& predictions,
        std::vector<Utils::PlayerBoundingBox> groundTruths, float iouThreshold);
};

#endif  // METRICS_HPP
