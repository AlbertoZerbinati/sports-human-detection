// Marco Calì

#include "field-segmentation/FieldSegmentation.hpp"
#include "field-segmentation/GreenFieldSegmentation.hpp"

cv::Mat FieldSegmentation::filterRegions(const cv::Mat& mask, double minArea) {
    // Find contours in the binary mask
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // Initialize an empty list to store the filtered contours
    std::vector<std::vector<cv::Point>> filteredContours;

    // Iterate through the detected contours
    for (const auto& contour : contours) {
        // Calculate the area of the current contour
        double area = cv::contourArea(contour);

        // If the area is greater than or equal to the specified minimum area,
        // consider it as a valid region and add it to the filtered list
        if (area >= minArea) filteredContours.push_back(contour);
    }

    // Create an empty mask to draw the filtered regions
    cv::Mat filteredMask = cv::Mat::zeros(mask.size(), CV_8U);

    // Draw the filtered regions on the mask
    cv::drawContours(filteredMask, filteredContours, -1, 255, cv::FILLED);

    return filteredMask;
}

cv::Mat FieldSegmentation::segmentField(const cv::Mat& src,
                                        const cv::Vec3b estimatedColor) {
    // Create an empty binary mask of the same size as the input image
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);

    // Extract the estimated color components
    int blue = estimatedColor[0];
    int green = estimatedColor[1];
    int red = estimatedColor[2];

    float g = static_cast<float>(green) / (green + red + blue);
    if (g > 0.4) {
        // Create an instance of the GreenFieldSegmentation class
        GreenFieldSegmentation gfs = GreenFieldSegmentation();
        std::cout << "Green Field Segmentation" << std::endl;
        // Attempt green field segmentation
        mask = gfs.detectGreenField(src);
    } else
        mask = colorFieldSegmentation(src, estimatedColor);  // Fallback method
    int height = mask.rows;
    int width = mask.cols;
    int diameter =
        static_cast<int>((1.0 / 100.0 * sqrt(height * height + width * width)));

    if (diameter % 2 == 0) diameter++;

    cv::Size size = cv::Size(diameter, diameter);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, size);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, element);

    double filterSize = static_cast<double>(mask.size().area()) / 200.0;
    mask = filterRegions(mask, filterSize);
    // Return a copy of the final binary mask
    return mask.clone();
}

bool FieldSegmentation::isColorCloseToBlack(int b, int g, int r,
                                            int lightnessThreshold) {
    // Create a single-pixel BGR image
    cv::Mat color(1, 1, CV_8UC3, cv::Scalar(b, g, r));

    // Convert BGR to HLS
    cv::Mat colorHLS;
    cv::cvtColor(color, colorHLS, cv::COLOR_BGR2HLS);

    // Extract the lightness (L) component
    int lightness = colorHLS.at<cv::Vec3b>(0, 0)[1];

    // Check if the lightness is below the threshold
    return lightness <= lightnessThreshold;
}

cv::Vec3b FieldSegmentation::estimateFieldColor(const cv::Mat& src) {
    cv::Mat dst;
    int height = src.rows;
    int width = src.cols;
    int diameter =
        static_cast<int>((1.0 / 100.0 * sqrt(height * height + width * width)));

    // Ensure the kernel has odd dimensions by adding 1 if it's even
    if (diameter % 2 == 0) diameter++;

    // Create a circular structuring element with the calculated diameter
    cv::Size size = cv::Size(diameter, diameter);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, size);
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, element);

    cv::Mat data = dst.reshape(1, height * width);
    data.convertTo(data, CV_32F);

    // Set K = 4
    int k = 4;
    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1);
    int flags = cv::KMEANS_RANDOM_CENTERS;

    cv::Mat labels, centers;
    cv::kmeans(data, k, labels, criteria, 10, flags, centers);

    // Initialize variables to keep track of cluster counts
    std::vector<int> clusterCounts(k, 0);

    // Count the number of pixels assigned to each cluster
    for (int i = 0; i < labels.rows; ++i) {
        int clusterIdx = labels.at<int>(i);
        clusterCounts[clusterIdx]++;
    }

    // Find the cluster index with the highest count that is not close to black
    int mostPresentColorIdx = -1;
    for (int i = 0; i < k; ++i) {
        if (clusterCounts[i] > clusterCounts[mostPresentColorIdx]) {
            cv::Vec3f clusterColor = centers.at<cv::Vec3f>(i);
            if (!isColorCloseToBlack(clusterColor[0], clusterColor[1],
                                     clusterColor[2], 50)) {
                mostPresentColorIdx = i;
            }
        }
    }

    if (mostPresentColorIdx == -1) {
        // If no suitable color found, use the color of the cluster with the
        // second highest count
        mostPresentColorIdx =
            0;  // You may change this to another cluster index as needed
    }

    // Extract the most present cluster center (color)
    cv::Vec3f mostPresentColor = centers.at<cv::Vec3f>(mostPresentColorIdx);

    // Convert Vec3f to Vec3b
    cv::Vec3b mostPresentColorBGR(static_cast<uchar>(mostPresentColor[0]),
                                  static_cast<uchar>(mostPresentColor[1]),
                                  static_cast<uchar>(mostPresentColor[2]));

    return mostPresentColorBGR;
}