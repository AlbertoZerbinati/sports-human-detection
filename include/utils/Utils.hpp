// Alberto Zerbinati

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <limits>
#include <map>
#include <tuple>

#include "opencv2/core/core.hpp"

namespace Utils {

struct PlayerBoundingBox {
    int x;
    int y;
    int w;
    int h;
    int team;
    cv::Vec3b color;
};

struct ExtendedPlayerBoundingBox : PlayerBoundingBox {
    cv::Mat colorMask;
};

struct Vec3bCompare {
    bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const;
};

cv::Mat reverseColoredMask(const cv::Mat& originalImage, const cv::Mat& mask);
bool areColorsWithinThreshold(const cv::Vec3b& color1, const cv::Vec3b& color2,
                              int threshold);
bool areColorsSame(const cv::Vec3b& color1, const cv::Vec3b& color2);
cv::Vec3b findMostSimilarColor(
    const cv::Vec3b& targetColor,
    const std::map<cv::Vec3b, int, Vec3bCompare> colorMap);

};  // namespace Utils

#endif  // UTILS_HPP
