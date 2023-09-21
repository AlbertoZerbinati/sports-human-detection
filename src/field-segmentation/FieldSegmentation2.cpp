// Alberto Zerbinati

#include "field-detection/FieldSegmentation.hpp"
#include "utils/Utils.hpp"

cv::Mat FieldSegmentation::colorFieldSegmentation(
    const cv::Mat &image, const cv::Vec3b estimatedColor) {
    // Create an empty binary mask with the same size as the input image
    cv::Mat mask = cv::Mat(image.size(), CV_8U);

    // Define a color similarity threshold
    int threshold = 25;

    // Fill the mask with white where the image pixels have color similarity to
    // the estimated color
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (Utils::areColorsWithinThreshold(image.at<cv::Vec3b>(y, x),
                                                estimatedColor, threshold)) {
                mask.at<uchar>(y, x) = 255;  // Set the pixel as white (field)
            }
        }
    }

    // Return a copy of the binary mask
    return mask.clone();
}
