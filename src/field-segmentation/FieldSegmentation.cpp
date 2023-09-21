// Marco Calì

#include "field-detection/FieldSegmentation.hpp"

#include "field-detection/GreenFieldSegmentation.hpp"

cv::Mat FieldSegmentation::segmentField(const cv::Mat &src,
                                        const cv::Vec3b estimatedColor) {
    // Create an empty binary mask of the same size as the input image
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);

    // Extract the estimated color components
    int blue = estimatedColor[0];
    int green = estimatedColor[1];
    int red = estimatedColor[2];

    // Create an instance of the GreenFieldSegmentation class
    GreenFieldSegmentation gfs = GreenFieldSegmentation();

    // Attempt green field segmentation
    mask = gfs.detectGreenField(src);

    // If the resulting mask has too few non-zero pixels, fall back to color
    // segmentation
    if (cv::countNonZero(mask) < 250)
        mask = colorFieldSegmentation(src, estimatedColor);  // Fallback method

    // Return a copy of the final binary mask
    return mask.clone();
}
