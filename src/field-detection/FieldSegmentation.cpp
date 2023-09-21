// Marco Calì

#include "field-detection/GreenFieldSegmentation.hpp"

// TODO: refactor into class

cv::Mat greenFieldSegmentation(const cv::Mat &I) {
    // White Lines Removal Through Opening Morphological Operator on the
    // lower-resolution image
    cv::Mat imageOpen;
    double alpha_e = 0.5;

    preprocessing(I, imageOpen, alpha_e);

    // Green Chromaticity Analysis
    cv::Mat imageGCA;
    chromaticityAnalysis(imageOpen, imageGCA);
    imageGCA.convertTo(imageGCA, CV_32F, 1 / 255.0);

    // Adjust the scale factor as for speeding up the training
    double scaleFactor = 0.1;
    cv::Size lowerSizeImage(imageGCA.cols * scaleFactor,
                            imageGCA.rows * scaleFactor);
    cv::Mat reducedImageGCA;
    resize(imageGCA, reducedImageGCA, lowerSizeImage);

    cv::Mat samples =
        reducedImageGCA.reshape(1, reducedImageGCA.rows * reducedImageGCA.cols);

    // Number of Gaussian distributions used for the E-M algorithm.
    int N_G = 6;

    cv::Mat logLikelihoods, labels, probs;
    cv::Ptr<cv::ml::EM> gmm = cv::ml::EM::create();
    trainGMM(gmm, samples, N_G, logLikelihoods, labels, probs);

    // Get covariance of each Gaussian
    std::vector<cv::Mat> covs;
    gmm->getCovs(covs);
    cv::Mat means = gmm->getMeans();
    cv::Mat weights = gmm->getWeights();

    // Compute PDF
    int numPoints = 1000;
    double gMin = 0.0;  // Minimum possible "g" value
    double gMax = 1.0;  // Maximum possible "g" value
    double step = (gMax - gMin) / static_cast<double>(numPoints);

    cv::Mat g = cv::Mat::zeros(1, numPoints, CV_64F);
    for (int i = 0; i < numPoints; ++i) g.at<double>(0, i) = gMin + i * step;

    cv::Mat pdf = computePDF(N_G, numPoints, means, covs, weights, g);
    // Now choose T_G
    double T = 1.0 / 4.0;
    double m0 = findFirstMinimumAfterIndex(
        pdf, g, findFirstMaximumAfterThreshold(pdf, g, T));
    double T_G = std::max(T, m0);
    cv::Mat mask1 = computeMask1(T_G, imageGCA);

    cv::Mat envelope = createEnvelope(covs, means, weights, numPoints, N_G);
    std::vector<LocalMinimum> minima = findLocalMinima(envelope, g, T_G);

    double T_C = 0.15;
    cv::Mat cd_matrix =
        chromaticDistortionMatrix(imageGCA, imageOpen, T_G, minima);
    cv::Mat mask2 = computeMask2(T_C, mask1, cd_matrix);

    // Let's apply the opening to the mask
    preprocessing(mask2, mask2, alpha_e);
    return mask2.clone();
}

cv::Mat colorFieldSegmentation(const cv::Mat &image,
                               const cv::Vec3b estimatedColor) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    int threshold = 25;

    // fill the mask with white  where the image pixels color is in threshold
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

cv::Mat fieldSegmentation(const cv::Mat &src, const cv::Vec3b estimatedColor) {
    cv::Mat mask;
    int blue = estimatedColor[0];
    int green = estimatedColor[1];
    int red = estimatedColor[2];
    mask = greenFieldSegmentation(src);

    // if mask is empty or so, then use the color segmentation method
    if (cv::countNonZero(mask) < 250)
        mask = colorFieldSegmentation(src, estimatedColor);  // fallback method

    return mask.clone();
}
