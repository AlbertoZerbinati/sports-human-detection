// Alberto Zerbinati

#include "utils/Metrics.hpp"

#include <opencv2/core/core.hpp>
#include <vector>

#include "utils/Utils.hpp"

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

float MetricsEvaluator::calculateMAP(
    const std::vector<Utils::PlayerBoundingBox> &groundTruths,
    const std::vector<Utils::PlayerBoundingBox> &predictions) {
    // Calculate mAP for each team label assignment and return the max
    float mAP1 = calculateMAPForTeams(groundTruths, predictions, 1, 2);
    float mAP2 = calculateMAPForTeams(groundTruths, predictions, 2, 1);

    return std::max(mAP1, mAP2);
}
