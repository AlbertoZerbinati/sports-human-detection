// Marco Calì

#include "field-detection/GreenFieldSegmentation.hpp"

void preprocessing(const cv::Mat &src, cv::Mat &dst, const double alpha_e)
{
    // The opening operation is based on the size of the image
    int H = src.rows;
    int W = src.cols;
    int diameter = 2 * int(ceil(alpha_e / 100 * sqrt(pow(H, 2) + pow(W, 2))));

    // Ensure the kernel has odd dimensions
    if (diameter % 2 == 0)
        diameter++;

    // cout << "Diameter of opening structure: " << diameter << endl;
    cv::Size size = cv::Size(diameter, diameter);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, size);
    cv::morphologyEx(src, dst, cv::MORPH_OPEN, element);
}

void green_chromaticity_analysis(const cv::Mat &src, cv::Mat &dst)
{
    // Make sure the destination matrix has the same size as the source
    dst.create(src.size(),
               CV_8UC1); // CV_8UC1 for single-channel (grayscale) image

    src.forEach<cv::Vec3b>(
        [&dst](cv::Vec3b &pixel, const int *position) -> void
        {
            int B = pixel[0]; // Blue channel value
            int G = pixel[1]; // Green channel value
            int R = pixel[2]; // Red channel value

            // Compute g(r,c) using the formula
            float green_chromaticity = static_cast<float>(G) / (G + R + B);

            // Update the corresponding pixel in the destination matrix
            dst.at<uchar>(position[0], position[1]) =
                static_cast<uchar>(green_chromaticity * 255);
        });
}

void train_gmm(cv::Ptr<cv::ml::EM> &gmm, const cv::Mat &samples, const int N_G,
               cv::Mat &log_likelihoods, cv::Mat &labels, cv::Mat &probs)
{
    gmm->setClustersNumber(N_G);
    gmm->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
    std::cout << "Training over " << samples.rows << " samples and "
              << std::to_string(N_G) << " Gaussians" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    gmm->trainEM(samples, log_likelihoods, labels, probs);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the training time duration
    std::chrono::duration<double> duration = end_time - start_time;
    double training_time_seconds = duration.count();

    // Print the training time
    // cout << "Training time: " << training_time_seconds << " seconds." << endl;
}

cv::Ptr<cv::ml::EM> gmm_load_trained(const cv::Mat &samples, const int N_G,
                                     cv::Mat &log_likelihoods, cv::Mat &labels, cv::Mat &probs)
{
    cv::Ptr<cv::ml::EM> gmm = cv::ml::EM::create();
    train_gmm(gmm, samples, N_G, log_likelihoods, labels, probs);
    return gmm;
}

double compute_gaussian(double x, double mean, double variance)
{
    return 1 / (sqrt(2.0 * CV_PI * variance)) *
           exp(-0.5 * pow(x - mean, 2) / variance);
}

cv::Mat compute_pdf(const int N_G, const int num_points, const cv::Mat &means,
                    const std::vector<cv::Mat> &covs, const cv::Mat &weights, const cv::Mat &g)
{
    // Initialize the sum of PDFs to zeros
    cv::Mat sum_pdf = cv::Mat::zeros(1, num_points, CV_64F);

    // Loop through each Gaussian component
    for (int n_gaussian = 0; n_gaussian < N_G; ++n_gaussian)
    {
        double gaussian_mean = means.at<double>(0, n_gaussian);
        double gaussian_variance = covs[n_gaussian].at<double>(0, 0);
        double gaussian_weight = weights.at<double>(0, n_gaussian);

        // Calculate the PDF for the current Gaussian component
        cv::Mat pdf = cv::Mat::zeros(1, num_points, CV_64F);
        for (int j = 0; j < num_points; j++)
        {
            double g_value = g.at<double>(0, j);
            pdf.at<double>(0, j) =
                gaussian_weight *
                compute_gaussian(g_value, gaussian_mean, gaussian_variance);
        }

        // Add the PDF of the current Gaussian component to the sum
        sum_pdf += pdf;
    }

    return sum_pdf.clone();
}

cv::Mat find_M_PF_hat(const double threshold, const cv::Mat &chromaticity)
{
    cv::Mat mask = cv::Mat::zeros(chromaticity.size(), CV_8U);

    // Iterate through each pixel of the images
    for (int y = 0; y < chromaticity.rows; ++y)
    {
        for (int x = 0; x < chromaticity.cols; ++x)
        {
            // Get the pixel intensity at (x, y) (green chromaticity value)
            double pixel_value =
                static_cast<double>(chromaticity.at<float>(y, x));
            // Compare the pixel intensity with m0 and set to white if greater
            // than T_G
            if (pixel_value > threshold)
                mask.at<uchar>(y, x) = 255;
        }
    }
    return mask.clone();
}

cv::Mat apply_mask(const cv::Mat &mask, const cv::Mat &image)
{
    // Create a copy of the original image and apply the mask
    cv::Mat masked_image = cv::Mat::zeros(image.size(), image.type());

    // Copy the original image to the result image where the mask is not zero
    image.copyTo(masked_image, mask);
    return masked_image.clone();
}

int findFirstMaximumAfterThreshold(const cv::Mat &pdf, const cv::Mat &g,
                                   double threshold)
{
    int num_points = pdf.cols;
    bool found_threshold = false;

    // Initialize variables to track maximum
    double maximum_value = 0.0;
    int maximum_index = -1;

    // Iterate through the PDF values
    for (int i = 0; i < num_points; ++i)
    {
        double current_value = pdf.at<double>(0, i);

        // Check if we found the threshold (x > 1/3)
        if (!found_threshold && g.at<double>(0, i) > threshold)
        {
            found_threshold = true;
        }

        // If we found the threshold, search for the first maximum
        if (found_threshold)
        {
            if (current_value > maximum_value || maximum_index == -1)
            {
                maximum_value = current_value;
                maximum_index = i;
            }
            else
            {
                // Stop searching when the value starts decreasing again
                break;
            }
        }
    }

    return maximum_index;
}

double findFirstMinimumAfterIndex(const cv::Mat &pdf, const cv::Mat &g, int index)
{
    int num_points = pdf.cols;

    // Initialize variables to track minimum
    double minimum_value = 0.0;
    int minimum_index = -1;

    // Iterate through the PDF values starting from the given index
    for (int i = index; i < num_points; ++i)
    {
        double current_value = pdf.at<double>(0, i);

        // Search for the first minimum
        if (current_value < minimum_value || minimum_index == -1)
        {
            minimum_value = current_value;
            minimum_index = i;
        }
        else
        {
            // Stop searching when the value starts increasing again
            break;
        }
    }

    // Check if we found the first minimum after the given index
    if (minimum_index != -1)
        return g.at<double>(0, minimum_index);
    else
        return -1.0; // No suitable local minimum found
}

cv::Mat create_envelope(const std::vector<cv::Mat> &covs, const cv::Mat &means,
                        const cv::Mat &weights, int num_points, int N_G)
{
    // Set up the x-axis range
    cv::Mat envelope = cv::Mat::zeros(1, num_points, CV_64F);

    double min_x = 0.0;
    double max_x = 1.0;
    double step = (max_x - min_x) / static_cast<double>(num_points);

    std::vector<cv::Mat> gaussians;
    for (int i = 0; i < covs.size(); ++i)
    {
        double mean = means.at<double>(0, i);
        double variance = covs[i].at<double>(0, 0);
        double weight = weights.at<double>(0, i);

        // Calculate the PDF for the current Gaussian component
        cv::Mat pdf = cv::Mat::zeros(1, num_points, CV_64F);
        for (int j = 0; j < num_points; j++)
        {
            double x = min_x + j * step;
            pdf.at<double>(0, j) = weight * compute_gaussian(x, mean, variance);
        }
        gaussians.push_back(pdf);
    }

    std::vector<double> y_values;
    // For each point
    for (int j = 0; j < num_points; j++)
    {
        // For each Gaussian
        for (int i = 0; i < N_G; i++)
            y_values.push_back(gaussians[i].at<double>(
                0, j)); // Create a vector of N_G values
        // Pick the max
        envelope.at<double>(0, j) =
            *max_element(y_values.begin(), y_values.end());

        y_values.clear();
    }
    return envelope.clone();
}

std::vector<LocalMinimum> findLocalMinima(const cv::Mat &envelope, const cv::Mat &g,
                                          const double threshold)
{
    int num_elements = envelope.cols;
    std::vector<LocalMinimum> local_minima;

    double x;
    for (int i = 1; i < num_elements - 1; ++i)
    {
        double current_element = envelope.at<double>(0, i);
        double prev_element = envelope.at<double>(0, i - 1);
        double next_element = envelope.at<double>(0, i + 1);
        x = g.at<double>(0, i);
        if (current_element < prev_element && current_element < next_element &&
            x > threshold)
        {
            LocalMinimum minimum;
            minimum.value = current_element;
            minimum.x = g.at<double>(0, i);
            local_minima.push_back(minimum);
        }
    }

    return local_minima;
}

float dot_product(cv::Vec3b v, cv::Vec3b u)
{
    float sum = 0;
    for (int i = 0; i < 3; i++)
        sum += v[i] * u[i];
    return sum;
}

float compute_chromatic_distortion(cv::Vec3b v, cv::Vec3b u)
{
    cv::Vec3b u_v = dot_product(u, v) / dot_product(v, v) * v;
    cv::Vec3b u_perp = u - u_v;
    float cd = sqrt(dot_product(u_perp, u_perp)) / sqrt(dot_product(u_v, u_v));
    return cd;
}

std::vector<cv::Vec3b> compute_mean_colors(std::vector<PixelInfo> pixels,
                                           std::vector<int> counts)
{
    std::vector<cv::Vec3b> means;
    std::vector<cv::Vec3f> sums; // sum 1 sum2 sum3 sum4

    for (int count = 0; count < counts.size() + 1; count++)
        sums.push_back(cv::Vec3f(0, 0, 0));

    for (PixelInfo pixel : pixels)
    {
        int group = pixel.group;
        if (group != 0)
            sums[group - 1] +=
                cv::Vec3f(pixel.pixel[0], pixel.pixel[1], pixel.pixel[2]);
    }

    for (int count = 0; count < counts.size(); count++)
    {
        cv::Vec3f mean = sums[count] / static_cast<float>(counts[count]);
        cv::Vec3b mean_color(static_cast<uchar>(mean_color[0]),
                             static_cast<uchar>(mean_color[1]),
                             static_cast<uchar>(mean_color[2]));
        means.push_back(mean_color);
    }
    return means;
}

cv::Mat chromatic_distorsion_matrix(const cv::Mat &I_gca, const cv::Mat &I_open,
                                    const double T_G,
                                    const std::vector<LocalMinimum> &minima)
{
    std::vector<PixelInfo> pixels;
    int num_minima = minima.size();
    if (num_minima == 0)
    {
        // cout << "No minima found, returning 0.0 matrix" << endl;
        return cv::Mat(I_open.size(), CV_32F, cv::Scalar(0.0));
    }

    std::vector<int> counts(num_minima + 2, 0); // Initialize counts for each range

    // Assign each pixel to a cluster
    for (int y = 0; y < I_open.rows; ++y)
    {
        for (int x = 0; x < I_open.cols; ++x)
        {
            PixelInfo pixel;
            pixel.pixel = I_open.at<cv::Vec3b>(y, x);

            float g_value = I_gca.at<float>(y, x);

            // Find the appropriate range for the pixel
            int range = 0;
            if (g_value < T_G && g_value > 0)
                range = 0;
            else if (g_value > T_G && g_value < minima[0].x)
                range = 1;
            else if (g_value > minima[num_minima - 1].x && g_value <= 1)
                range = num_minima + 1;
            else
            {
                // Search for the range where g_value falls between minima
                for (int i = 0; i < num_minima; ++i)
                {
                    if (g_value > minima[i].x && g_value <= minima[i + 1].x)
                    {
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

    // Compute the mean for each cluster
    std::vector<cv::Vec3b> avgs = compute_mean_colors(pixels, counts);

    // Compute the distorsion value for each pixel depending on the cluster
    cv::Mat cd_matrix(I_open.rows, I_open.cols, CV_32F);

    int height = I_open.rows;
    int width = I_open.cols;

    cv::Mat group_matrix(I_open.rows, I_open.cols, CV_8U);

    // Assign each pixel to a cluster based on the information in the pixels
    // vector
    for (int y = 0; y < I_open.rows; ++y)
    {
        for (int x = 0; x < I_open.cols; ++x)
        {
            // Get the cluster assignment from the corresponding pixel in the
            // pixels vector
            int cluster = pixels[y * I_open.cols + x].group;

            // Assign the cluster value to the corresponding location in the
            // group_matrix
            group_matrix.at<uchar>(y, x) = static_cast<uchar>(cluster);
        }
    }

    I_open.forEach<cv::Vec3b>([&cd_matrix, &group_matrix, &avgs](
                                  cv::Vec3b &pixel, const int *position) -> void
                              {
        float cd_value = 1.0;
        int group = group_matrix.at<uchar>(position[0], position[1]);

        if (group == 1)
            cd_value = compute_chromatic_distortion(pixel, avgs[0]);
        else if (group == 2)
            cd_value = compute_chromatic_distortion(pixel, avgs[1]);
        else if (group == 3)
            cd_value = compute_chromatic_distortion(pixel, avgs[2]);
        else if (group == 4)
            cd_value = compute_chromatic_distortion(pixel, avgs[3]);

        // Update the corresponding pixel in the destination matrix
        cd_matrix.at<float>(position[0], position[1]) = cd_value; });

    return cd_matrix.clone();
}

cv::Mat find_M_PF_tilde(const double T_C, const cv::Mat &mask1, const cv::Mat &cd_matrix)
{
    // Initialize the result mask with zeros
    cv::Mat mask2(mask1.size(), CV_8U, cv::Scalar(0));

    // Check the condition and set values in the mask2 mask
    for (int r = 0; r < mask1.rows; ++r)
        for (int c = 0; c < mask1.cols; ++c)
            if (mask1.at<uchar>(r, c) == 255 && cd_matrix.at<float>(r, c) < T_C)
                mask2.at<uchar>(r, c) = 255;
    return mask2.clone();
}
