// Marco Calì

#include "field-segmentation/GreenFieldSegmentation.hpp"

cv::Mat GreenFieldSegmentation::detectGreenField(const cv::Mat &I) {
    // Initialize variables and parameters
    cv::Mat imageOpen;
    double alpha_e = 0.5;  // Opening operation parameter

    // Preprocess the input image using opening operation
    preprocessing(I, imageOpen, alpha_e);

    // Perform Green Chromaticity Analysis on the preprocessed image
    cv::Mat imageGCA;
    chromaticityAnalysis(imageOpen, imageGCA);
    imageGCA.convertTo(imageGCA, CV_32F, 1 / 255.0);

    // Adjust the scale factor to speed up training
    double scaleFactor = 0.1;
    cv::Size lowerSizeImage(imageGCA.cols * scaleFactor,
                            imageGCA.rows * scaleFactor);
    cv::Mat reducedImageGCA;
    resize(imageGCA, reducedImageGCA, lowerSizeImage);

    // Reshape the image for GMM training
    cv::Mat samples =
        reducedImageGCA.reshape(1, reducedImageGCA.rows * reducedImageGCA.cols);

    // Number of Gaussian distributions used for the E-M algorithm
    int N_G = 6;

    // Initialize variables for GMM training
    cv::Mat logLikelihoods, labels, probs;
    cv::Ptr<cv::ml::EM> gmm = cv::ml::EM::create();

    // Train the Gaussian Mixture Model
    trainGMM(gmm, samples, N_G, logLikelihoods, labels, probs);

    // Get covariance matrices, means, and weights from the trained GMM
    std::vector<cv::Mat> covs;
    gmm->getCovs(covs);
    cv::Mat means = gmm->getMeans();
    cv::Mat weights = gmm->getWeights();

    // Compute the PDF over a range of "g" values from 0 to 1
    int numPoints = 1000;
    double gMin = 0.0;  // Minimum possible "g" value
    double gMax = 1.0;  // Maximum possible "g" value
    double step = (gMax - gMin) / static_cast<double>(numPoints);

    cv::Mat g = cv::Mat::zeros(1, numPoints, CV_64F);
    for (int i = 0; i < numPoints; ++i) g.at<double>(0, i) = gMin + i * step;

    cv::Mat pdf = computePDF(N_G, numPoints, means, covs, weights, g);

    // Choose a threshold T_G based on PDF characteristics
    double T = 1.0 / 4.0;
    double m0 = findFirstMinimumAfterIndex(
        pdf, g, findFirstMaximumAfterThreshold(pdf, g, T));
    double T_G = std::max(T, m0);

    // Generate the first mask (mask1) based on the threshold T_G
    cv::Mat mask1 = computeMask1(T_G, imageGCA);

    // Create an envelope representing the GMM distribution
    cv::Mat envelope = createEnvelope(covs, means, weights, numPoints, N_G);

    // Find local minima in the envelope
    std::vector<LocalMinimum> minima = findLocalMinima(envelope, g, T_G);

    // Define a threshold for chromatic distortion
    double T_C = 0.15;

    // Compute the chromatic distortion matrix
    cv::Mat cd_matrix =
        chromaticDistortionMatrix(imageGCA, imageOpen, T_G, minima);

    // Generate the second mask (mask2) based on the chromatic distortion and
    // mask1
    cv::Mat mask2 = computeMask2(T_C, mask1, cd_matrix);

    // Apply opening operation to the mask
    preprocessing(mask2, mask2, alpha_e);

    return mask2.clone();  // Return the final mask2
}

void GreenFieldSegmentation::preprocessing(const cv::Mat &src, cv::Mat &dst,
                                           const double alpha_e) {
    // Calculate the diameter of the opening structure based on image size and
    // alpha_e
    int H = src.rows;
    int W = src.cols;
    int diameter = 2 * int(ceil(alpha_e / 100 * sqrt(pow(H, 2) + pow(W, 2))));

    // Ensure the kernel has odd dimensions by adding 1 if it's even
    if (diameter % 2 == 0) diameter++;

    // Create a circular structuring element with the calculated diameter
    cv::Size size = cv::Size(diameter, diameter);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, size);

    // Apply morphological opening to the input image and store the result in
    // 'dst'
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, element);
}

void GreenFieldSegmentation::chromaticityAnalysis(const cv::Mat &src,
                                                  cv::Mat &dst) {
    // Make sure the destination matrix has the same size as the source
    dst.create(src.size(),
               CV_8UC1);  // CV_8UC1 for single-channel (grayscale) image

    // Iterate over each pixel in the source image
    src.forEach<cv::Vec3b>([&dst](cv::Vec3b &pixel,
                                  const int *position) -> void {
        int B = pixel[0];  // Blue channel value
        int G = pixel[1];  // Green channel value
        int R = pixel[2];  // Red channel value

        // Calculate g(r, c) using the formula (green chromaticity)
        float greenChromaticity = static_cast<float>(G) / (G + R + B);

        // Update the corresponding pixel in the destination matrix
        dst.at<uchar>(position[0], position[1]) =
            static_cast<uchar>(greenChromaticity * 255);  // Scale to [0, 255]
    });
}

void GreenFieldSegmentation::trainGMM(cv::Ptr<cv::ml::EM> &gmm,
                                      const cv::Mat &samples, const int N_G,
                                      cv::Mat &logLikelihoods, cv::Mat &labels,
                                      cv::Mat &probs) {
    // Set the number of Gaussian components in the GMM
    gmm->setClustersNumber(N_G);

    // Set the covariance matrix type to diagonal
    gmm->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);

    // Print a message indicating the training parameters
    std::cout << "Training over " << samples.rows << " samples and "
              << std::to_string(N_G) << " Gaussians" << std::endl;

    // Train the GMM using the provided samples, storing log-likelihoods,
    // labels, and probabilities
    gmm->trainEM(samples, logLikelihoods, labels, probs);
}

double GreenFieldSegmentation::computeGaussian(double x, double mean,
                                               double variance) {
    return 1 / (sqrt(2.0 * CV_PI * variance)) *
           exp(-0.5 * pow(x - mean, 2) / variance);
}

cv::Mat GreenFieldSegmentation::computePDF(const int N_G, const int numPoints,
                                           const cv::Mat &means,
                                           const std::vector<cv::Mat> &covs,
                                           const cv::Mat &weights,
                                           const cv::Mat &g) {
    // Initialize the sum of PDFs to zeros
    cv::Mat sumPDF = cv::Mat::zeros(1, numPoints, CV_64F);

    // Loop through each Gaussian component
    for (int i = 0; i < N_G; ++i) {
        // Extract parameters for the current Gaussian component
        double mean = means.at<double>(0, i);
        double variance = covs[i].at<double>(0, 0);
        double weight = weights.at<double>(0, i);

        // Calculate the PDF for the current Gaussian component
        cv::Mat pdf = cv::Mat::zeros(1, numPoints, CV_64F);
        for (int j = 0; j < numPoints; j++) {
            double gValue = g.at<double>(0, j);
            pdf.at<double>(0, j) =
                weight * computeGaussian(gValue, mean, variance);
        }

        // Add the PDF of the current Gaussian component to the sum
        sumPDF += pdf;
    }

    // Return a copy of the sum of PDFs
    return sumPDF.clone();
}

cv::Mat GreenFieldSegmentation::computeMask1(const double threshold,
                                             const cv::Mat &chromaticity) {
    // Create an empty binary mask with the same size as the chromaticity image
    cv::Mat mask = cv::Mat::zeros(chromaticity.size(), CV_8U);

    // Iterate through each pixel in the chromaticity image
    for (int y = 0; y < chromaticity.rows; ++y) {
        for (int x = 0; x < chromaticity.cols; ++x) {
            // Get the chromaticity value at (x, y)
            double pixel_value =
                static_cast<double>(chromaticity.at<float>(y, x));

            // Compare the chromaticity value with the threshold and set to
            // white if it exceeds the threshold
            if (pixel_value > threshold) {
                mask.at<uchar>(y, x) = 255;  // Set the pixel as white (field)
            }
        }
    }

    // Return a copy of the binary mask
    return mask.clone();
}

int GreenFieldSegmentation::findFirstMaximumAfterThreshold(
    const cv::Mat &pdf, const cv::Mat &g, const double threshold) {
    int numPoints = pdf.cols;
    bool foundThreshold = false;

    // Initialize variables to track the maximum
    double maximumValue = 0.0;
    int maximumIndex = -1;

    // Iterate through the PDF values
    for (int i = 0; i < numPoints; ++i) {
        double currentValue = pdf.at<double>(0, i);

        // Check if we found the threshold (x > threshold)
        if (!foundThreshold && g.at<double>(0, i) > threshold) {
            foundThreshold = true;
        }

        // If we found the threshold, search for the first maximum
        if (foundThreshold) {
            if (currentValue > maximumValue || maximumIndex == -1) {
                maximumValue = currentValue;
                maximumIndex = i;
            } else {
                // Stop searching when the value starts decreasing again
                break;
            }
        }
    }

    // Return the index of the first maximum value found after the threshold
    return maximumIndex;
}

double GreenFieldSegmentation::findFirstMinimumAfterIndex(const cv::Mat &pdf,
                                                          const cv::Mat &g,
                                                          int index) {
    int numPoints = pdf.cols;

    // Initialize variables to track the minimum
    double minimumValue = 0.0;
    int minimumIndex = -1;

    // Iterate through the PDF values starting from the given index
    for (int i = index; i < numPoints; ++i) {
        double currentValue = pdf.at<double>(0, i);

        // Search for the first local minimum
        if (currentValue < minimumValue || minimumIndex == -1) {
            minimumValue = currentValue;
            minimumIndex = i;
        } else {
            // Stop searching when the value starts increasing again
            break;
        }
    }

    // Check if we found the first local minimum after the given index
    if (minimumIndex != -1)
        return g.at<double>(
            0, minimumIndex);  // Return the value of the local minimum
    else
        return -1.0;  // No suitable local minimum found
}

cv::Mat GreenFieldSegmentation::createEnvelope(const std::vector<cv::Mat> &covs,
                                               const cv::Mat &means,
                                               const cv::Mat &weights,
                                               int numPoints, int N_G) {
    // Set up the x-axis range
    cv::Mat envelope = cv::Mat::zeros(1, numPoints, CV_64F);

    double xMin = 0.0;
    double xMax = 1.0;
    double step = (xMax - xMin) / static_cast<double>(numPoints);

    std::vector<cv::Mat> gaussians;
    for (int i = 0; i < covs.size(); ++i) {
        double mean = means.at<double>(0, i);
        double variance = covs[i].at<double>(0, 0);
        double weight = weights.at<double>(0, i);

        // Calculate the PDF for the current Gaussian component
        cv::Mat pdf = cv::Mat::zeros(1, numPoints, CV_64F);
        for (int j = 0; j < numPoints; j++) {
            double x = xMin + j * step;
            pdf.at<double>(0, j) = weight * computeGaussian(x, mean, variance);
        }
        gaussians.push_back(pdf);
    }

    std::vector<double> yValues;
    // For each point
    for (int j = 0; j < numPoints; j++) {
        // For each Gaussian
        for (int i = 0; i < N_G; i++)
            yValues.push_back(gaussians[i].at<double>(
                0, j));  // Create a vector of N_G values
        // Pick the max
        envelope.at<double>(0, j) =
            *max_element(yValues.begin(), yValues.end());

        yValues.clear();
    }

    // Return a copy of the envelope
    return envelope.clone();
}

std::vector<LocalMinimum> GreenFieldSegmentation::findLocalMinima(
    const cv::Mat &envelope, const cv::Mat &g, const double threshold) {
    int numElements = envelope.cols;
    std::vector<LocalMinimum> localMinima;

    double x;
    for (int i = 1; i < numElements - 1; ++i) {
        double currentElement = envelope.at<double>(0, i);
        double prevElement = envelope.at<double>(0, i - 1);
        double nextElement = envelope.at<double>(0, i + 1);
        x = g.at<double>(0, i);

        // Check if the current element is a local minimum
        if (currentElement < prevElement && currentElement < nextElement &&
            x > threshold) {
            LocalMinimum minimum;
            minimum.value = currentElement;
            minimum.x = g.at<double>(0, i);
            localMinima.push_back(minimum);
        }
    }

    return localMinima;
}

float GreenFieldSegmentation::dotProduct(cv::Vec3b v, cv::Vec3b u) {
    float sum = 0;
    for (int i = 0; i < 3; i++) sum += v[i] * u[i];
    return sum;
}

float GreenFieldSegmentation::computeChromaticDistortion(cv::Vec3b v,
                                                         cv::Vec3b u) {
    cv::Vec3b u_v = dotProduct(u, v) / dotProduct(v, v) * v;
    cv::Vec3b u_perp = u - u_v;
    float cd = sqrt(dotProduct(u_perp, u_perp)) / sqrt(dotProduct(u_v, u_v));
    return cd;
}

std::vector<cv::Vec3b> GreenFieldSegmentation::computeMeanColors(
    std::vector<PixelInfo> pixels, std::vector<int> counts) {
    // Vector to store the mean colors
    std::vector<cv::Vec3b> means;
    // Vector to store the sum of pixel values for each group
    std::vector<cv::Vec3f> sums;

    // Initialize sums with zero vectors for each group
    for (int count = 0; count < counts.size() + 1; count++)
        sums.push_back(cv::Vec3f(0, 0, 0));

    // Calculate the sum of pixel values for each group
    // Group 0 doesn't interest us, as it contains values pre-threshold.
    for (PixelInfo pixel : pixels) {
        int group = pixel.group;
        if (group != 0)
            sums[group - 1] +=
                cv::Vec3f(pixel.pixel[0], pixel.pixel[1], pixel.pixel[2]);
    }

    // Calculate the mean colors for each group
    for (int count = 0; count < counts.size(); count++) {
        // Compute the mean color by dividing the sum by the number of pixels in
        // the group
        cv::Vec3f mean = sums[count] / static_cast<float>(counts[count]);
        cv::Vec3b mean_color(static_cast<uchar>(mean_color[0]),
                             static_cast<uchar>(mean_color[1]),
                             static_cast<uchar>(mean_color[2]));
        // Store the mean color in the result vector
        means.push_back(mean_color);
    }
    // Return the vector of mean colors
    return means;
}

cv::Mat GreenFieldSegmentation::chromaticDistortionMatrix(
    const cv::Mat &imageGCA, const cv::Mat &imageOpen, const double T_G,
    const std::vector<LocalMinimum> &minima) {
    std::vector<PixelInfo> pixels;  // Vector to store pixel information
    int numMinima = minima.size();

    if (numMinima == 0) {
        // No minima found, return a matrix of zeros so that mask1 is used.
        return cv::Mat(imageOpen.size(), CV_32F, cv::Scalar(0.0));
    }

    std::vector<int> counts(numMinima + 2,
                            0);  // Initialize counts for each range

    // Assign each pixel to a cluster based on green chromaticity values
    for (int y = 0; y < imageOpen.rows; ++y) {
        for (int x = 0; x < imageOpen.cols; ++x) {
            PixelInfo pixel;
            pixel.pixel = imageOpen.at<cv::Vec3b>(y, x);

            float gValue = imageGCA.at<float>(y, x);

            int range = 0;

            // Determine the range for the pixel based on its green chromaticity
            // value
            if (gValue < T_G && gValue > 0)
                range = 0;
            else if (gValue > T_G && gValue < minima[0].x)
                range = 1;
            else if (gValue > minima[numMinima - 1].x && gValue <= 1)
                range = numMinima + 1;
            else {
                // Search for the range where gValue falls between minima
                for (int i = 0; i < numMinima; ++i) {
                    if (gValue > minima[i].x && gValue <= minima[i + 1].x) {
                        range = i + 2;
                        break;
                    }
                }
            }

            pixel.group = range;
            counts[range]++;
            pixels.push_back(pixel);
        }
    }

    // Compute the mean colors for each cluster
    std::vector<cv::Vec3b> avgs = computeMeanColors(pixels, counts);

    // Create a matrix to store the chromatic distortion values
    cv::Mat cd_matrix(imageOpen.rows, imageOpen.cols, CV_32F);

    // Create a matrix to store the cluster assignments for each pixel
    cv::Mat groupMatrix(imageOpen.rows, imageOpen.cols, CV_8U);

    // Assign each pixel to a cluster based on information in the pixels vector
    for (int y = 0; y < imageOpen.rows; ++y) {
        for (int x = 0; x < imageOpen.cols; ++x) {
            int cluster = pixels[y * imageOpen.cols + x].group;
            groupMatrix.at<uchar>(y, x) = static_cast<uchar>(cluster);
        }
    }

    // Compute the chromatic distortion value for each pixel
    imageOpen.forEach<cv::Vec3b>(
        [&cd_matrix, &groupMatrix, &avgs](cv::Vec3b &pixel,
                                          const int *position) -> void {
            float cd_value = 1.0;
            int group = groupMatrix.at<uchar>(position[0], position[1]);

            if (group != 0)
                cd_value = computeChromaticDistortion(pixel, avgs[group]);

            // Update the corresponding pixel in the destination matrix
            cd_matrix.at<float>(position[0], position[1]) = cd_value;
        });
    // Return the chromatic distortion matrix
    return cd_matrix.clone();
}

cv::Mat GreenFieldSegmentation::computeMask2(const double T_C,
                                             const cv::Mat &mask1,
                                             const cv::Mat &cd_matrix) {
    // Initialize the result mask with zeros
    cv::Mat mask2(mask1.size(), CV_8U, cv::Scalar(0));

    // Check the condition and set values in the mask2 mask
    for (int r = 0; r < mask1.rows; ++r) {
        for (int c = 0; c < mask1.cols; ++c) {
            if (mask1.at<uchar>(r, c) == 255 &&
                cd_matrix.at<float>(r, c) < T_C) {
                // Set pixel to white if condition is met
                mask2.at<uchar>(r, c) = 255;
            }
        }
    }
    // Return the refined mask
    return mask2.clone();
}
