// Marco Calì

#include "field-detection/FieldSegmentation.hpp"

#include "field-detection/GreenFieldSegmentation.hpp"

cv::Mat FieldSegmentation::colorFieldSegmentation(
    const cv::Mat &image, const cv::Vec3b estimatedColor) {
    cv::Mat mask = cv::Mat(image.size(), CV_8U);
    int threshold = 25;

    // fill the mask with white where the image pixels color is in threshold
    // with the estimated color
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (abs(image.at<cv::Vec3b>(y, x)[0] - estimatedColor[0]) <
                    threshold and
                abs(image.at<cv::Vec3b>(y, x)[1] - estimatedColor[1]) <
                    threshold and
                abs(image.at<cv::Vec3b>(y, x)[2] - estimatedColor[2]) <
                    threshold) {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }
    return mask.clone();
}

cv::Mat FieldSegmentation::segmentField(const cv::Mat &src,
                                        const cv::Vec3b estimatedColor) {
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);

    int blue = estimatedColor[0];
    int green = estimatedColor[1];
    int red = estimatedColor[2];

    GreenFieldSegmentation gfs = GreenFieldSegmentation();
    mask = gfs.detectGreenField(src);

    // if mask is empty or so, then use the color segmentation method
    if (cv::countNonZero(mask) < 250)
        mask = colorFieldSegmentation(src, estimatedColor);  // fallback method

    return mask.clone();
}
