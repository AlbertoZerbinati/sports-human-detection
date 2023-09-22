// Alberto Zerbinati

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <limits>
#include <map>
#include <tuple>

#include "opencv2/core/core.hpp"

namespace Utils {

// Structure to hold information about a player's bounding box
struct PlayerBoundingBox {
    int x;            // X-coordinate of top-left corner
    int y;            // Y-coordinate of top-left corner
    int w;            // Width of bounding box
    int h;            // Height of bounding box
    int team;         // Team identifier
    cv::Vec3b color;  // Color of bounding box
};

// Structure to hold extended information about a player's bounding box
struct ExtendedPlayerBoundingBox : PlayerBoundingBox {
    cv::Mat colorMask;  // Color mask for the bounding box
};

// Functor to compare two Vec3b objects
struct Vec3bCompare {
    bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const;
};

/**
 * Reverses the colored mask for a given image and mask.
 * @param originalImage Original image matrix.
 * @param mask Mask matrix.
 * @return A new cv::Mat object representing the reversed colored mask.
 */
cv::Mat reverseColoredMask(const cv::Mat& originalImage, const cv::Mat& mask);

/**
 * Checks if two colors are within a certain threshold.
 * @param color1 First color to compare.
 * @param color2 Second color to compare.
 * @param threshold Maximum allowable color difference.
 * @return True if the colors are within the threshold, otherwise false.
 */
bool areColorsWithinThreshold(const cv::Vec3b& color1, const cv::Vec3b& color2,
                              int threshold);

/**
 * Checks if two colors are the same.
 * @param color1 First color to compare.
 * @param color2 Second color to compare.
 * @return True if the colors are the same, otherwise false.
 */
bool areColorsSame(const cv::Vec3b& color1, const cv::Vec3b& color2);

/**
 * Finds the most similar color from a given map of colors.
 * @param targetColor The color to find a match for.
 * @param colorMap A map of possible colors.
 * @return The most similar color from the map.
 */
cv::Vec3b findMostSimilarColor(
    const cv::Vec3b& targetColor,
    const std::map<cv::Vec3b, int, Vec3bCompare> colorMap);

/**
 * Reads bounding boxes from a file.
 * @param filePath Path to the file.
 * @return A vector of PlayerBoundingBox objects.
 */
std::vector<PlayerBoundingBox> readBoundingBoxesFromFile(std::string filePath);

/**
 * Writes bounding boxes to a file.
 * @param boxes Vector of bounding boxes to write.
 * @param filePath Path to the file.
 */
void writeBoundingBoxesToFile(const std::vector<PlayerBoundingBox>& boxes,
                              const std::string& filePath);

/**
 * Saves bounding boxes as an overlay on an image.
 * @param img The original image.
 * @param boundingBoxes Vector of bounding boxes to overlay.
 * @param outputFileName The name of the output file.
 */
void saveBoundingBoxesOnImage(
    const cv::Mat& img, const std::vector<PlayerBoundingBox>& boundingBoxes,
    const std::string& outputFileName);

};  // namespace Utils

#endif  // UTILS_HPP
