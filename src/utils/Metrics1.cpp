// Marco Sedusi

#include <opencv2/core/core.hpp>
#include <vector>

#include "utils/Metrics.hpp"
#include "utils/Utils.hpp"

/* calculateClassIoU
 * Parameters: predicted is the matrix segmented returned by our algorithm
 * groundTruth is reference matrix segmented
 * label is an interger value to indicate the class for which we want to compute
 * the IoU metric */
float MetricsEvaluator::calculateClassIoU(const cv::Mat &predicted,
                                          const cv::Mat &groundTruth,
                                          int label) {
    int intersect = 0;
    int uni = 0;

    for (int i = 0; i < predicted.rows; i++) {
        for (int j = 0; j < predicted.cols; j++) {
            // Both true
            // If the two masks have same value then counter++ for union and
            // intersection
            if (predicted.at<uchar>(i, j) == label &&
                groundTruth.at<uchar>(i, j) == label) {
                intersect++;
                uni++;
            }
            // Only one true
            // If only one of the two masks have the specified value label then
            // counter++ for union
            else if (predicted.at<uchar>(i, j) == label &&
                         groundTruth.at<uchar>(i, j) != label ||
                     predicted.at<uchar>(i, j) != label &&
                         groundTruth.at<uchar>(i, j) == label) {
                uni++;
            }
        }
    }
    // To avoid division by zero
    if (uni == 0) {
        uni = 1;
    }
    // Return the IoU for one class
    return (float)intersect / uni;
}
/* calculateMIoU
 * Parameters: predicted is the matrix segmented returned by our algorithm
 * groundTruth is reference matrix segmented */
float MetricsEvaluator::calculateMIoU(const cv::Mat &predicted,
                                      const cv::Mat &groundTruth) {
    // Two matrices must have same size and type
    if (!(predicted.size() == groundTruth.size() &&
          predicted.type() == groundTruth.type())) {
        std::cerr
            << "Size or type between predicted and ground truth don't match."
            << std::endl;
        return 0;
    }
    // Perform the metric computation for the 3 classes (background, field,
    // team1 and team2)
    float IoU0 = calculateClassIoU(predicted, groundTruth, 0);  // background
    float IoU1 = calculateClassIoU(predicted, groundTruth, 1);  // team1
    float IoU2 = calculateClassIoU(predicted, groundTruth, 2);  // team2
    float IoU3 = calculateClassIoU(predicted, groundTruth, 3);  // field

    // Return the mIoU as float
    return (IoU0 + IoU1 + IoU2 + IoU3) / 4;
}
