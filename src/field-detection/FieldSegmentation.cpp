// Marco Calì

#include "field-detection/GreenFieldSegmentation.hpp"

// TODO: refactor into class

cv::Mat GreenFieldsSegmentation(const cv::Mat &I)
{
    // White Lines Removal Through Opening Morphological Operator on the
    // lower-resolution image
    cv::Mat I_open;
    double alpha_e = 0.5;

    preprocessing(I, I_open, alpha_e);

    // Green Chromaticity Analysis
    cv::Mat I_gca;
    green_chromaticity_analysis(
        I_open, I_gca); // image with values in range 0-255, ready for view

    I_gca.convertTo(I_gca, CV_32F, 1 / 255.0);

    // Adjust the scale factor as for speeding up the training
    double scale_factor = 0.1;
    cv::Size lower_res_size(I_gca.cols * scale_factor, I_gca.rows * scale_factor);
    cv::Mat I_gca_reduced;
    resize(I_gca, I_gca_reduced, lower_res_size);

    cv::Mat samples =
        I_gca_reduced.reshape(1, I_gca_reduced.rows * I_gca_reduced.cols);

    // Number of Gaussian distributions used for the E-M algorithm.
    int N_G = 6;

    cv::Mat log_likelihoods, labels, probs;
    cv::Ptr<cv::ml::EM> gmm =
        gmm_load_trained(samples, N_G, log_likelihoods, labels, probs);

    // Get covariance of each Gaussian
    std::vector<cv::Mat> covs;
    gmm->getCovs(covs);
    cv::Mat means = gmm->getMeans();
    cv::Mat weights = gmm->getWeights();

    // Compute PDF
    int num_points = 1000;
    double min_g = 0.0; // Minimum possible "g" value
    double max_g = 1.0; // Maximum possible "g" value
    double step = (max_g - min_g) / static_cast<double>(num_points);

    cv::Mat g = cv::Mat::zeros(1, num_points, CV_64F);
    for (int i = 0; i < num_points; ++i)
        g.at<double>(0, i) = min_g + i * step;

    cv::Mat pdf = compute_pdf(N_G, num_points, means, covs, weights, g);
    // Now choose T_G
    double T = 1.0 / 4.0;
    double m0 = findFirstMinimumAfterIndex(
        pdf, g, findFirstMaximumAfterThreshold(pdf, g, T));
    double T_G = std::max(T, m0);
    cv::Mat mask1 = find_M_PF_hat(T_G, I_gca);

    cv::Mat envelope = create_envelope(covs, means, weights, num_points, N_G);
    std::vector<LocalMinimum> minima = findLocalMinima(envelope, g, T_G);

    double T_C = 0.15;
    cv::Mat cd_matrix = chromatic_distorsion_matrix(I_gca, I_open, T_G, minima);
    cv::Mat mask2 = find_M_PF_tilde(T_C, mask1, cd_matrix);

    // Let's apply the opening to the mask
    preprocessing(mask2, mask2, alpha_e);
    return mask2.clone();
}

cv::Mat ColorFieldSegmentation(const cv::Mat &image, const cv::Vec3b estimated_color)
{
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8U);
    int threshold = 25;

    // fill the mask with white  where the image pixels color is in threshold
    // with the estimated color
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            if (abs(image.at<cv::Vec3b>(y, x)[0] - estimated_color[0]) <
                    threshold and
                abs(image.at<cv::Vec3b>(y, x)[1] - estimated_color[1]) <
                    threshold and
                abs(image.at<cv::Vec3b>(y, x)[2] - estimated_color[2]) <
                    threshold)
            {
                mask.at<uchar>(y, x) = 255;
            }
        }
    }

    return mask.clone();
}

cv::Mat FieldSegmentation(const cv:: Mat &src, const cv::Vec3b estimated_field_color)
{
    cv::Mat mask;
    int blue = estimated_field_color[0];
    int green = estimated_field_color[1];
    int red = estimated_field_color[2];
    mask = GreenFieldsSegmentation(src);

    // if mask is empty or so, then use the color segmentation method
    if (countNonZero(mask) < 250)
        mask = ColorFieldSegmentation(src, estimated_field_color); // fallback method

    return mask.clone();
}
