// To-be-splitted

#include "utils/Metrics.hpp"

#include <opencv2/core/core.hpp>
#include <vector>

#include "utils/Utils.hpp"

float MetricsEvaluator::calculateClassIoU(const cv::Mat &predicted,
                                          const cv::Mat &groundTruth,
                                          int label) {
    int intersect = 0;
    int uni = 0;

    for (int i = 0; i < predicted.rows; i++) {
        for (int j = 0; j < predicted.cols; j++) {
            // both true
            if (predicted.at<uchar>(i, j) == label &&
                groundTruth.at<uchar>(i, j) == label) {
                intersect++;
                uni++;
            }
            // only one true
            else if (predicted.at<uchar>(i, j) == label &&
                         groundTruth.at<uchar>(i, j) != label ||
                     predicted.at<uchar>(i, j) != label &&
                         groundTruth.at<uchar>(i, j) == label) {
                uni++;
            }
        }
    }

    if (uni == 0) {
        uni = 1;
    }

    return (float)intersect / uni;
}

float MetricsEvaluator::calculateClassesMIoU(const cv::Mat &predicted,
                                             const cv::Mat &groundTruth) {
    if (!(predicted.size() == groundTruth.size() &&
          predicted.type() == groundTruth.type())) {
        std::cerr
            << "Size or type between predicted and ground truth don't match."
            << std::endl;
        return 0;
    }

    float IoU1 = calculateClassIoU(predicted, groundTruth, 1);
    float IoU2 = calculateClassIoU(predicted, groundTruth, 2);
    float IoU3 = calculateClassIoU(predicted, groundTruth, 3);

    return (IoU1 + IoU2 + IoU3) / 3;
}

float MetricsEvaluator::calculateGeometricIoU(
    const Utils::PlayerBoundingBox &bb1, const Utils::PlayerBoundingBox &bb2) {
    int x_overlap = std::max(
        0, std::min(bb1.x + bb1.w, bb2.x + bb2.w) - std::max(bb1.x, bb2.x));
    int y_overlap = std::max(
        0, std::min(bb1.y + bb1.h, bb2.y + bb2.h) - std::max(bb1.y, bb2.y));

    int overlapArea = x_overlap * y_overlap;
    int unionArea = bb1.w * bb1.h + bb2.w * bb2.h - overlapArea;

    if (unionArea == 0) {
        return 0;
    }

    return (float)overlapArea / unionArea;
}

float MetricsEvaluator::computeAPSingleClass(
    const std::vector<Utils::PlayerBoundingBox> &bb_class,
    std::vector<Utils::PlayerBoundingBox> ground_truths, float iouThreshold) {
    int TP = 0;
    int FP = 0;
    int FN =
        ground_truths
            .size();  // Initialize FN to the total number of ground truth boxes

    std::vector<float> precision_values;
    std::vector<float> recall_values;

    for (Utils::PlayerBoundingBox bb : bb_class) {
        float max_iou = -1;
        int max_iou_index = -1;

        for (int i = 0; i < ground_truths.size(); ++i) {
            Utils::PlayerBoundingBox ground_bb = ground_truths[i];
            float iou_score = calculateGeometricIoU(bb, ground_bb);
            if (iou_score > max_iou) {
                max_iou = iou_score;
                max_iou_index = i;
            }
        }

        if (max_iou > iouThreshold) {
            TP++;
            FN--;  // Decrement FN as we have found a corresponding ground truth
            ground_truths.erase(
                ground_truths.begin() +
                max_iou_index);  // Remove the matched ground truth
        } else {
            FP++;
        }

        float precision = TP / (float)(TP + FP);
        float recall = TP / (float)(TP + FN);
        precision_values.push_back(precision);
        recall_values.push_back(recall);
    }

    // Sort by recall
    std::vector<std::pair<float, float>> rp_pairs;
    for (size_t i = 0; i < precision_values.size(); ++i) {
        rp_pairs.emplace_back(recall_values[i], precision_values[i]);
    }
    std::sort(rp_pairs.begin(), rp_pairs.end());

    // Compute 11-point interpolated AP
    float average_precision = 0;
    for (float r = 0; r <= 1.0; r += 0.1) {
        float maxPrecisionRight = 0;
        for (const auto &pr : rp_pairs) {
            if (pr.first >= r) {
                maxPrecisionRight = std::max(maxPrecisionRight, pr.second);
            }
        }
        average_precision += maxPrecisionRight;
    }
    average_precision /= 11;

    return average_precision;
}

float MetricsEvaluator::calculateMAP(
    const std::vector<Utils::PlayerBoundingBox> &groundTruths,
    const std::vector<Utils::PlayerBoundingBox> &predictions) {
    // Calculate mAP for each team label assignment and return the max
    float mAP1 = calculateMAPForTeams(groundTruths, predictions, 1, 2);
    float mAP2 = calculateMAPForTeams(groundTruths, predictions, 2, 1);

    return std::max(mAP1, mAP2);
}

float MetricsEvaluator::calculateMAPForTeams(
    const std::vector<Utils::PlayerBoundingBox> &groundTruths,
    const std::vector<Utils::PlayerBoundingBox> &predictions, int team1Label,
    int team2Label) {
    std::vector<Utils::PlayerBoundingBox> groundTruthsTeam1;
    std::vector<Utils::PlayerBoundingBox> groundTruthsTeam2;

    std::vector<Utils::PlayerBoundingBox> predictionsTeam1;
    std::vector<Utils::PlayerBoundingBox> predictionsTeam2;

    // for each team, add the ground truths and predictions to the corresponding
    // vector
    for (int i = 0; i < groundTruths.size(); i++) {
        if (groundTruths[i].team == team1Label) {
            groundTruthsTeam1.push_back(groundTruths[i]);
        } else if (groundTruths[i].team == team2Label) {
            groundTruthsTeam2.push_back(groundTruths[i]);
        }
    }
    for (int i = 0; i < predictions.size(); i++) {
        if (predictions[i].team == team1Label) {
            predictionsTeam1.push_back(predictions[i]);
        } else if (predictions[i].team == team2Label) {
            predictionsTeam2.push_back(predictions[i]);
        }
    }

    // Sort detections in descending order of confidence
    // TODO

    // Since confidence is not available, just skip this step
    // std::sort(predictionsTeam1.begin(), predictionsTeam1.end(), [](const
    // Utils::PlayerBoundingBox &a, const
    // Utils::PlayerBoundingBox &b)
    //           { return a.confidence_score > b.confidence_score; });
    // std::sort(predictionsTeam2.begin(), predictionsTeam2.end(), [](const
    // Utils::PlayerBoundingBox &a, const
    // Utils::PlayerBoundingBox &b)
    //           { return a.confidence_score > b.confidence_score; });

    float iouThreshold = 0.5;
    float average_precision_team1 =
        computeAPSingleClass(predictionsTeam1, groundTruthsTeam1, iouThreshold);
    float average_precision_team2 =
        computeAPSingleClass(predictionsTeam2, groundTruthsTeam2, iouThreshold);

    return (average_precision_team1 + average_precision_team2) / 2.0;
}
