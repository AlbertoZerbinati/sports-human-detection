// Alberto Zerbinati

#ifndef METRICS_HPP
#define METRICS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/Utils.hpp"

class MetricsEvaluator {
   public:
    float calculateClassesMIoU(const cv::Mat& predicted,
                               const cv::Mat& groundTruth);
    float calculateMAP(
        const std::vector<Utils::PlayerBoundingBox>& groundTruths,
        const std::vector<Utils::PlayerBoundingBox>& predictions);

   private:
    float calculateGeometricIoU(const Utils::PlayerBoundingBox& bb1,
                                const Utils::PlayerBoundingBox& bb2);
    float calculateClassIoU(const cv::Mat& predicted,
                            const cv::Mat& groundTruth, int label);
    float calculateMAPForTeams(
        const std::vector<Utils::PlayerBoundingBox>& groundTruths,
        const std::vector<Utils::PlayerBoundingBox>& predictions, int team1,
        int team2);
    float computeAPSingleClass(
        const std::vector<Utils::PlayerBoundingBox>& predictions,
        std::vector<Utils::PlayerBoundingBox> groundTruths,
        float iouThreshold);
};

#endif  // METRICS_HPP
