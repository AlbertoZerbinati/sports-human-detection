// Marco Calì

#include "field-detection/FieldSegmentation.hpp"

Mat GreenFieldsSegmentation(const Mat &I /*, String filename*/) {
    // White Lines Removal Through Opening Morphological Operator on the
    // lower-resolution image
    Mat I_open;
    double alpha_e = 0.5;

    preprocessing(I, I_open, alpha_e);

    // Green Chromaticity Analysis
    Mat I_gca;
    green_chromaticity_analysis(
        I_open, I_gca);  // image with values in range 0-255, ready for view

    I_gca.convertTo(I_gca, CV_32F, 1 / 255.0);

    double scale_factor = 0.1;  // Adjust the scale factor as needed
    Size lower_res_size(I_gca.cols * scale_factor, I_gca.rows * scale_factor);
    Mat I_gca_reduced;
    resize(I_gca, I_gca_reduced, lower_res_size);

    Mat samples =
        I_gca_reduced.reshape(1, I_gca_reduced.rows * I_gca_reduced.cols);

    // Number of Gaussian distributions used for the E-M algorithm.
    int N_G = 6;

    Mat log_likelihoods, labels, probs;
    Ptr<ml::EM> gmm =
        gmm_load_trained(samples, N_G, log_likelihoods, labels, probs);

    // Get covariance of each Gaussian
    vector<Mat> covs;
    gmm->getCovs(covs);
    Mat means = gmm->getMeans();
    Mat weights = gmm->getWeights();

    // Compute PDF
    int num_points = 1000;
    double min_g = 0.0;  // Minimum possible "g" value
    double max_g = 1.0;  // Maximum possible "g" value
    double step = (max_g - min_g) / static_cast<double>(num_points);

    Mat g = Mat::zeros(1, num_points, CV_64F);
    for (int i = 0; i < num_points; ++i) g.at<double>(0, i) = min_g + i * step;

    Mat pdf = compute_pdf(N_G, num_points, means, covs, weights, g);
    // Now choose T_G
    double T = 1.0 / 4.0;
    double m0 = findFirstMinimumAfterIndex(
        pdf, g, findFirstMaximumAfterThreshold(pdf, g, T));
    double T_G = std::max(T, m0);
    Mat mask1 = find_M_PF_hat(T_G, I_gca);

    Mat envelope = create_envelope(covs, means, weights, num_points, N_G);
    vector<LocalMinimum> minima = findLocalMinima(envelope, g, T_G);

    double T_C = 0.15;
    Mat cd_matrix = chromatic_distorsion_matrix(I_gca, I_open, T_G, minima);
    Mat mask2 = find_M_PF_tilde(T_C, mask1, cd_matrix);

    // Let's apply the opening to the mask
    preprocessing(mask2, mask2, alpha_e);
    Mat result = apply_mask(mask2, I);

    Mat mask2_inverted;
    bitwise_not(mask2, mask2_inverted);

    Mat result_inverted = apply_mask(mask2_inverted, I);

    // imwrite("../results_images/" + filename + "_field" + ".jpg", result);
    // imwrite("../results_images/" + filename + "_inverted_field" + ".jpg",
    // result_inverted); imwrite("../results_images/" + filename + "_mask" +
    // ".jpg", mask2);
    return mask2.clone();
}

Mat GenericFieldSegmentation(Mat &image, int from_row, int from_column,
                             int row_width, int column_width,
                             double mean_factor, double std_factor) {
    // Convert to RG space
    Mat image_R = Mat::zeros(image.size(), CV_64F);
    Mat image_G = Mat::zeros(image.size(), CV_64F);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            double sum = image.at<Vec3b>(i, j)[0] + image.at<Vec3b>(i, j)[1] +
                         image.at<Vec3b>(i, j)[2];
            image_R.at<double>(i, j) = image.at<Vec3b>(i, j)[0] / sum;
            image_G.at<double>(i, j) = image.at<Vec3b>(i, j)[1] / sum;
        }
    }

    // Set the patch value
    Mat image_patch =
        image(Rect(from_row, from_column, row_width, column_width));

    // imshow("Patch", image_patch);
    // Convert patch to RG space
    Mat patch_R = Mat::zeros(image_patch.size(), CV_64F);
    Mat patch_G = Mat::zeros(image_patch.size(), CV_64F);

    for (int i = 0; i < patch_R.rows; ++i) {
        for (int j = 0; j < patch_R.cols; ++j) {
            double sum = image_patch.at<Vec3b>(i, j)[0] +
                         image_patch.at<Vec3b>(i, j)[1] +
                         image_patch.at<Vec3b>(i, j)[2];
            patch_R.at<double>(i, j) = image_patch.at<Vec3b>(i, j)[0] / sum;
            patch_G.at<double>(i, j) = image_patch.at<Vec3b>(i, j)[1] / sum;
        }
    }

    // Get the mean and std of the patch
    double mean_patch_R = mean(patch_R)[0];
    double mean_patch_G = mean(patch_G)[0];
    double std_patch_R = 0.0;
    double std_patch_G = 0.0;

    for (int i = 0; i < patch_R.rows; ++i) {
        for (int j = 0; j < patch_R.cols; ++j) {
            std_patch_R += pow(patch_R.at<double>(i, j) - mean_patch_R, 2);
            std_patch_G += pow(patch_G.at<double>(i, j) - mean_patch_G, 2);
        }
    }

    std_patch_R = sqrt(std_patch_R / (patch_R.rows * patch_R.cols));
    std_patch_G = sqrt(std_patch_G / (patch_G.rows * patch_G.cols));

    // Get the range of colors of the patch
    Mat prob_R = Mat::zeros(image.size(), CV_64F);
    Mat prob_G = Mat::zeros(image.size(), CV_64F);

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            prob_R.at<double>(i, j) =
                exp(-pow(image_R.at<double>(i, j) - mean_factor * mean_patch_R,
                         2) /
                    (2 * pow(std_factor * std_patch_R, 2))) /
                (std_factor * std_patch_R * sqrt(2 * M_PI));
            prob_G.at<double>(i, j) =
                exp(-pow(image_G.at<double>(i, j) - mean_factor * mean_patch_G,
                         2) /
                    (2 * pow(std_factor * std_patch_G, 2))) /
                (std_factor * std_patch_G * sqrt(2 * M_PI));
        }
    }

    // Get the mask
    Mat prob = prob_R.mul(prob_G);

    // Clean the mask using thresholding
    Mat mask;
    threshold(prob, mask, 0.1, 255, THRESH_BINARY);

    return mask.clone();
}