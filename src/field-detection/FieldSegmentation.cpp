// Marco Calì

#include "field-detection/FieldSegmentation.hpp"

#include "field-detection/GreenFieldSegmentation.hpp"

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
            if (abs(image.at<cv::Vec3b>(y, x)[0] - estimatedColor[0]) <
                    threshold and
                abs(image.at<cv::Vec3b>(y, x)[1] - estimatedColor[1]) <
                    threshold and
                abs(image.at<cv::Vec3b>(y, x)[2] - estimatedColor[2]) <
                    threshold) {
                mask.at<uchar>(y, x) = 255;  // Set the pixel as white (field)
            }
        }
    }

    // Return a copy of the binary mask
    return mask.clone();
}

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