// Marco Calì

#include "utils/Metrics.hpp"

#include <opencv2/core/core.hpp>
#include <vector>

#include "utils/Utils.hpp"

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
