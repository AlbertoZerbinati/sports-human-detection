#include "Metrics.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace Metrics {

float calculateIoU(BoundingBox box1, BoundingBox box2) {
    // Calculate the Intersection over Union (IoU) here
    // Return IoU value
    return 0.0;  // Dummy return
}

float calculateMAP(std::vector<std::vector<BoundingBox>> groundTruths,
                   std::vector<std::vector<BoundingBox>> predictions) {
    float totalAP = 0.0;
    int numImages = groundTruths.size();

    for (int i = 0; i < numImages; ++i) {
        std::vector<BoundingBox> groundTruth = groundTruths[i];
        std::vector<BoundingBox> prediction = predictions[i];

        // Calculate True Positives, False Positives, and False Negatives
        int TP = 0, FP = 0, FN = 0;

        for (auto& predBox : prediction) {
            bool isTruePositive = false;

            for (auto& gtBox : groundTruth) {
                if (calculateIoU(predBox, gtBox) > 0.5) {
                    isTruePositive = true;
                    break;
                }
            }

            if (isTruePositive)
                TP++;
            else
                FP++;
        }

        FN = groundTruth.size() - TP;

        // Calculate Precision and Recall
        float precision = static_cast<float>(TP) / (TP + FP);
        float recall = static_cast<float>(TP) / (TP + FN);

        // In a full implementation, we should integrate Precision over
        // different Recall levels
        totalAP += precision;
    }

    // Calculate Mean Average Precision (mAP)
    float mAP = totalAP / numImages;

    return mAP;
}
}  // namespace Metrics
