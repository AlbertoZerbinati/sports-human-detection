// Alberto Zerbinati

#include "utils/Utils.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace Utils {

bool Vec3bCompare::operator()(const cv::Vec3b& a, const cv::Vec3b& b) const {
    return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
}

cv::Mat reverseColoredMask(const cv::Mat& originalImage, const cv::Mat& mask) {
    // Initialize an output image with the same size and type as the original
    // image
    cv::Mat outputImage =
        cv::Mat::zeros(originalImage.size(), originalImage.type());

    // Loop through each pixel in the mask and original image
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            cv::Vec3b maskPixel = mask.at<cv::Vec3b>(y, x);
            cv::Vec3b originalPixel = originalImage.at<cv::Vec3b>(y, x);

            // If the mask pixel is black (0, 0, 0), use the original image's
            // pixel color
            if (maskPixel[0] == 0 && maskPixel[1] == 0 && maskPixel[2] == 0) {
                outputImage.at<cv::Vec3b>(y, x) = originalPixel;
            } else {
                // Otherwise, set the pixel to black
                outputImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    return outputImage;
}

bool areColorsWithinThreshold(const cv::Vec3b& color1, const cv::Vec3b& color2,
                              int threshold) {
    return std::abs(color1[0] - color2[0]) <= threshold &&
           std::abs(color1[1] - color2[1]) <= threshold &&
           std::abs(color1[2] - color2[2]) <= threshold;
}

bool areColorsSame(const cv::Vec3b& color1, const cv::Vec3b& color2) {
    return areColorsWithinThreshold(color1, color2, 0);
}

cv::Vec3b findMostSimilarColor(
    const cv::Vec3b& targetColor,
    const std::map<cv::Vec3b, int, Vec3bCompare> colorMap) {
    double minDistance = std::numeric_limits<double>::max();
    cv::Vec3b mostSimilarColor;

    for (const auto pair : colorMap) {
        const cv::Vec3b& mapColor = pair.first;
        double distance = std::sqrt(std::pow(targetColor[0] - mapColor[0], 2) +
                                    std::pow(targetColor[1] - mapColor[1], 2) +
                                    std::pow(targetColor[2] - mapColor[2], 2));

        if (distance < minDistance) {
            minDistance = distance;
            mostSimilarColor = mapColor;
        }
    }

    return mostSimilarColor;
}

std::vector<PlayerBoundingBox> readBoundingBoxesFromFile(std::string filePath) {
    std::vector<PlayerBoundingBox> boundingBoxes;
    std::ifstream file(filePath);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filePath << std::endl;
        return boundingBoxes;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        PlayerBoundingBox bbox;
        if (!(iss >> bbox.x >> bbox.y >> bbox.w >> bbox.h >> bbox.team)) {
            std::cerr << "Error reading line: " << line << std::endl;
            continue;
        }
        // You can set the color here if you want, or leave it for later
        boundingBoxes.push_back(bbox);
    }

    file.close();
    return boundingBoxes;
}

void writeBoundingBoxesToFile(const std::vector<PlayerBoundingBox>& boxes,
                              const std::string& filePath) {
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open the output file." << std::endl;
        return;
    }

    for (const auto& box : boxes) {
        outFile << box.x << " " << box.y << " " << box.w << " " << box.h << " "
                << box.team << std::endl;
    }

    outFile.close();
}

};  // namespace Utils
