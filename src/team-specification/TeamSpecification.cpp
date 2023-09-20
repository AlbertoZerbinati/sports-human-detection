// Sedusi Marco

#include "team-specification/TeamSpecification.hpp"

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Vec3b TeamSpecification::findDominantColor(
    cv::Mat temp, bool ignoreTeamColors,
    std::map<cv::Vec3b, int, Utils::Vec3bCompare> teamsColors) {
    cv::Mat pixels = temp.reshape(1, temp.rows * temp.cols);

    // Convert the pixel values to floating-point type
    pixels.convertTo(pixels, CV_32F);

    // Number of dominant colors to find (K value)
    int k = 3;  //

    // Criteria for K-means algorithm
    // TermCriteria is a class in OpenCV used to define termination criteria
    // for iterative algorithms. It allows you to specify when an iterative
    // algorithm should stop based on certain conditions. TermCriteria::EPS:
    // This criterion checks whether the desired accuracy (epsilon) is
    // achieved. In other words, the algorithm will stop if the change in
    // the value being optimized falls below a certain threshold (epsilon).
    // TermCriteria::MAX_ITER: This criterion checks whether the maximum
    // number of iterations is reached. The algorithm will stop if it has
    // performed a specified maximum number of iterations. maxCount: This
    // parameter is set to 100, which means that the algorithm will stop
    // after 100 iterations if the TermCriteria::MAX_ITER condition is not
    // met. epsilon: This parameter is set to 0.2. It defines the desired
    // accuracy (epsilon) for the optimization process. If the change in the
    // value being optimized falls below this threshold, the algorithm will
    // stop if the TermCriteria::EPS condition is not met.

    cv::TermCriteria criteria(
        cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 0.2);

    // Perform K-means clustering
    cv::Mat labels, centers;
    kmeans(pixels, k, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);

    // Convert the centers to 8-bit BGR forcv::Mat
    centers.convertTo(centers, CV_8UC1);

    // Create a vector to store the dominant color triplets
    std::vector<cv::Vec3b> dominantColors;

    // Extract the dominant color triplets
    for (int i = 0; i < k; i++) {
        cv::Vec3b triplet(centers.at<uchar>(i, 0), centers.at<uchar>(i, 1),
                          centers.at<uchar>(i, 2));
        dominantColors.push_back(triplet);
    }

    // Output the dominant color triplets
    // cout << "Dominant Colors:" << endl;
    // for (const auto& triplet : dominantColors) {
    //     cout << "RGB: (" << (int)triplet[0] << ", " << (int)triplet[1] <<
    //     ",
    //     "
    //          << (int)triplet[2] << ")" << endl;
    // }

    // remove the black (or gray) cplor from dominants (with threshold)
    for (int i = 0; i < dominantColors.size(); i++) {
        if (dominantColors[i][0] < 5 && dominantColors[i][1] < 5 &&
            dominantColors[i][2] < 5) {
            dominantColors.erase(dominantColors.begin() + i);
            i--;
        }
    }

    if (ignoreTeamColors) {
        return dominantColors[0];  // return dominant color
    }

    cv::Vec3b finalColor;
    bool assigned = false;

    // find the most similar color to the dmoniantcolors[0] in the teamcolors
    // map with the utils function
    cv::Vec3b mostSimilarColor;
    if (teamsColors.size() > 0) {
        mostSimilarColor =
            Utils::findMostSimilarColor(dominantColors[0], teamsColors);
    } else {
        mostSimilarColor = dominantColors[0];
    }

    int threshold = 25;
    // see if it is in threshold with the utils funciton
    if (Utils::areColorsWithinThreshold(dominantColors[0], mostSimilarColor,
                                        threshold)) {
        return mostSimilarColor;
    } else {
        return dominantColors[0];  // return dominant color
    }
}
