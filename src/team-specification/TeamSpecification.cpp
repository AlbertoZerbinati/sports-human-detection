// Sedusi Marco

#include "team-specification/TeamSpecification.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

/* findDominantColor
 * function to find k dominant colors in the image.
 * */
cv::Vec3b TeamSpecification::findDominantColor(cv::Mat matrix) {
    //	Each pixel in the image is treated as a separate data point for the
    // clustering
    cv::Mat pixels = matrix.reshape(1, matrix.rows * matrix.cols);

    // Convert the pixel values to floating-point type
    pixels.convertTo(pixels, CV_32F);

    // Number of dominant colors to find (K value)
    int k = 2;

    // Criteria for K-means algorithm
    // TermCriteria is a class in OpenCV used to define termination criteria
    // for iterative algorithms. It allows you to specify when an iterative
    // algorithm should stop based on certain conditions. EPS checks whether
    // the desired accuracy (epsilon) is achieved. In other words, the
    // algorithm will stop if the change in the value being optimized falls
    // below a certain threshold (epsilon). MAX_ITER checks whether the
    // maximum number of iterations is reached. The algorithm will stop if
    // it has performed a specified maximum number of iterations. maxCount:
    // This parameter is set to 100, which means that the algorithm will
    // stop after 100 iterations if the MAX_ITER condition is not met.
    // epsilon is set to 0.2. It defines the desired accuracy for the
    // optimization process.

    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2);

    // Perform K-means clustering
    cv::Mat labels, centers;
    // KMEANS_RANDOM_CENTERS indicates that the initial cluster centers
    // should be chosen randomly.
    kmeans(pixels, k, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);

    // Convert the centers to 8-bit BGR for cv::Mat
    centers.convertTo(centers, CV_8UC1);

    // Create a vector to store the dominant color triplets
    std::vector<cv::Vec3b> dominantColors;

    // Extract the dominant color triplets
    for (int i = 0; i < k; i++) {
        cv::Vec3b triplet(centers.at<uchar>(i, 0), centers.at<uchar>(i, 1),
                          centers.at<uchar>(i, 2));
        dominantColors.push_back(triplet);
    }

    // Remove the black (or gray) from dominants (with threshold)
    for (int i = 0; i < dominantColors.size(); i++) {
        if (dominantColors[i][0] < 25 && dominantColors[i][1] < 25 &&
            dominantColors[i][2] < 25) {
            dominantColors.erase(dominantColors.begin() + i);
            i--;
        }
    }

    return dominantColors[0];
}
