// Marco Calì

#include "../../include/field-detection/GreenFieldSegmentation.hpp"

String get_image_path(String filename) {
    String dataset_path = "../assets/dataset/Images/";
    String image_filename = dataset_path + filename + ".jpg";
    return image_filename;
}

void preprocessing(const Mat &src, Mat &dst, const double alpha_e) {
    // Get some important properties such as width and height
    int H = src.rows;
    int W = src.cols;
    cout << "Height: " << H << ", width: " << W << endl;
    // Define the kernel size and shape (circular), which is dependent on the
    // image size double alpha_e = 0.5; // suggested 0.5
    int diameter = 2 * int(ceil(alpha_e / 100 * sqrt(pow(H, 2) + pow(W, 2))));

    // Ensure the kernel has odd dimensions
    if (diameter % 2 == 0) diameter++;

    // cout << "Diameter of opening structure: " << diameter << endl;
    Size size = Size(diameter, diameter);
    Mat element = getStructuringElement(MORPH_ELLIPSE, size);
    morphologyEx(src, dst, MORPH_OPEN, element);
}

void green_chromaticity_analysis(const cv::Mat &src, cv::Mat &dst) {
    // Make sure the destination matrix has the same size as the source
    dst.create(src.size(),
               CV_8UC1);  // CV_8UC1 for single-channel (grayscale) image

    src.forEach<cv::Vec3b>(
        [&dst](cv::Vec3b &pixel, const int *position) -> void {
            int B = pixel[0];  // Blue channel value
            int G = pixel[1];  // Green channel value
            int R = pixel[2];  // Red channel value

            // Compute g(r,c) using the formula
            float green_chromaticity = static_cast<float>(G) / (G + R + B);

            // Update the corresponding pixel in the destination matrix
            dst.at<uchar>(position[0], position[1]) =
                static_cast<uchar>(green_chromaticity * 255);
        });
}

Mat compute_histogram(const Mat &src) {
    // Calculate the histogram
    int histSize = 256;  // Number of bins
    float range[] = {0, 256};
    const float *histRange = {range};
    Mat hist;
    calcHist(&src, 1, nullptr, Mat(), hist, 1, &histSize, &histRange);

    // Create an image to display the histogram
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound(static_cast<double>(histWidth) / histSize);
    Mat histImage(histHeight, histWidth, CV_8UC3, Scalar(0, 0, 0));

    // Normalize the histogram values to fit in the image height
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Draw the histogram
    for (int i = 1; i < histSize; i++) {
        line(histImage,
             Point(binWidth * (i - 1),
                   histHeight - cvRound(hist.at<float>(i - 1))),
             Point(binWidth * (i), histHeight - cvRound(hist.at<float>(i))),
             Scalar(255, 0, 0), 2, 8, 0);
    }

    return histImage.clone();
}

void save_model(const Ptr<ml::EM> &gmm, String filename) {
    String filepath = "../model/trained_gmm_" + filename + "_NG_" +
                      to_string(gmm->getClustersNumber()) + ".yml";
    FileStorage fs(filepath, FileStorage::WRITE);
    if (fs.isOpened()) {
        gmm->write(fs);
        fs.release();
    } else
        cerr << "Error: Could not open the file for writing." << endl;
}

void train_gmm(Ptr<ml::EM> &gmm, const Mat &samples, const int N_G,
               Mat &log_likelihoods, Mat &labels, Mat &probs) {
    gmm->setClustersNumber(N_G);
    gmm->setCovarianceMatrixType(ml::EM::COV_MAT_DIAGONAL);
    cout << "Training over " << samples.rows << " samples and "
         << to_string(N_G) << " Gaussians" << endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    gmm->trainEM(samples, log_likelihoods, labels, probs);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate the training time duration
    std::chrono::duration<double> duration = end_time - start_time;
    double training_time_seconds = duration.count();

    // Print the training time
    cout << "Training time: " << training_time_seconds << " seconds." << endl;
}

Ptr<ml::EM> gmm_load_trained(const Mat &samples, const int N_G,
                             Mat &log_likelihoods, Mat &labels, Mat &probs) {
    Ptr<ml::EM> gmm = ml::EM::create();
    train_gmm(gmm, samples, N_G, log_likelihoods, labels, probs);
    return gmm;
}

double compute_gaussian(double x, double mean, double variance) {
    return 1 / (sqrt(2.0 * CV_PI * variance)) *
           exp(-0.5 * pow(x - mean, 2) / variance);
}

Mat compute_pdf(const int N_G, const int num_points, const Mat &means,
                const vector<Mat> &covs, const Mat &weights, const Mat &g) {
    // Initialize the sum of PDFs to zeros
    Mat sum_pdf = Mat::zeros(1, num_points, CV_64F);

    // Loop through each Gaussian component
    for (int n_gaussian = 0; n_gaussian < N_G; ++n_gaussian) {
        double gaussian_mean = means.at<double>(0, n_gaussian);
        double gaussian_variance = covs[n_gaussian].at<double>(0, 0);
        double gaussian_weight = weights.at<double>(0, n_gaussian);

        // Calculate the PDF for the current Gaussian component
        Mat pdf = Mat::zeros(1, num_points, CV_64F);
        for (int j = 0; j < num_points; j++) {
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

Mat find_M_PF_hat(const double threshold, const Mat &chromaticity) {
    Mat mask = Mat::zeros(chromaticity.size(), CV_8U);

    // Iterate through each pixel of the images
    for (int y = 0; y < chromaticity.rows; ++y) {
        for (int x = 0; x < chromaticity.cols; ++x) {
            // Get the pixel intensity at (x, y) (green chromaticity value)
            double pixel_value =
                static_cast<double>(chromaticity.at<float>(y, x));
            // cout << pixel_value << endl;
            // Compare the pixel intensity with m0 and set to white if greater
            // than T_G
            if (pixel_value > threshold) mask.at<uchar>(y, x) = 255;
        }
    }
    return mask.clone();
}

Mat apply_mask(const Mat &mask, const Mat &image) {
    // Create a copy of the original image and apply the mask
    Mat masked_image = Mat::zeros(image.size(), image.type());

    // Copy the original image to the result image where the mask is not zero
    image.copyTo(masked_image, mask);
    return masked_image.clone();
}

int findFirstMaximumAfterThreshold(const Mat &pdf, const Mat &g,
                                   double threshold) {
    int num_points = pdf.cols;
    bool found_threshold = false;

    // Initialize variables to track maximum
    double maximum_value = 0.0;
    int maximum_index = -1;

    // Iterate through the PDF values
    for (int i = 0; i < num_points; ++i) {
        double current_value = pdf.at<double>(0, i);

        // Check if we found the threshold (x > 1/3)
        if (!found_threshold && g.at<double>(0, i) > threshold) {
            found_threshold = true;
        }

        // If we found the threshold, search for the first maximum
        if (found_threshold) {
            if (current_value > maximum_value || maximum_index == -1) {
                maximum_value = current_value;
                maximum_index = i;
            } else {
                // Stop searching when the value starts decreasing again
                break;
            }
        }
    }

    return maximum_index;
}

double findFirstMinimumAfterIndex(const Mat &pdf, const Mat &g, int index) {
    int num_points = pdf.cols;

    // Initialize variables to track minimum
    double minimum_value = 0.0;
    int minimum_index = -1;

    // Iterate through the PDF values starting from the given index
    for (int i = index; i < num_points; ++i) {
        double current_value = pdf.at<double>(0, i);

        // Search for the first minimum
        if (current_value < minimum_value || minimum_index == -1) {
            minimum_value = current_value;
            minimum_index = i;
        } else {
            // Stop searching when the value starts increasing again
            break;
        }
    }

    // Check if we found the first minimum after the given index
    if (minimum_index != -1)
        return g.at<double>(0, minimum_index);
    else
        return -1.0;  // No suitable local minimum found
}

void visualize_gaussians(const vector<Mat> &covs, const Mat &means,
                         const Mat &weights, int num_points, double T_G) {
    // Set up the x-axis range
    double min_x = 0.0;
    double max_x = 1.0;
    double step = (max_x - min_x) / static_cast<double>(num_points);

    // Find the maximum PDF value among all Gaussian components
    double max_pdf = 0.0;
    for (int i = 0; i < covs.size(); ++i) {
        double gaussian_mean = means.at<double>(0, i);
        double gaussian_variance = covs[i].at<double>(0, 0);
        double gaussian_weight = weights.at<double>(0, i);

        // Calculate the PDF for the current Gaussian component
        Mat pdf = Mat::zeros(1, num_points, CV_64F);
        for (int j = 0; j < num_points; j++) {
            double x = min_x + j * step;
            pdf.at<double>(0, j) =
                gaussian_weight *
                compute_gaussian(x, gaussian_mean, gaussian_variance);
        }

        // Find the maximum PDF value for this component
        double component_max_pdf =
            *max_element(pdf.begin<double>(), pdf.end<double>());
        if (component_max_pdf > max_pdf) max_pdf = component_max_pdf;
    }

    // Create a canvas for plotting
    int plotWidth = 800;
    int plotHeight = 400;
    Mat plot = Mat::zeros(plotHeight, plotWidth, CV_8UC3);

    // Iterate through Gaussian components and plot them
    for (int i = 0; i < covs.size(); ++i) {
        double gaussian_mean = means.at<double>(0, i);
        double gaussian_variance = covs[i].at<double>(0, 0);
        double gaussian_weight = weights.at<double>(0, i);

        // Calculate the PDF for the current Gaussian component
        Mat pdf = Mat::zeros(1, num_points, CV_64F);
        for (int j = 0; j < num_points; j++) {
            double x = min_x + j * step;
            pdf.at<double>(0, j) =
                gaussian_weight *
                compute_gaussian(x, gaussian_mean, gaussian_variance);
        }

        // Normalize the PDF based on the maximum PDF value
        pdf /= max_pdf;

        // Plot the PDF of the current Gaussian component
        for (int j = 0; j < num_points - 1; ++j) {
            int x1 = static_cast<int>(j * (plotWidth - 1) / (num_points - 1));
            int x2 =
                static_cast<int>((j + 1) * (plotWidth - 1) / (num_points - 1));
            int y1 = plotHeight -
                     static_cast<int>(pdf.at<double>(0, j) * (plotHeight - 1));
            int y2 = plotHeight - static_cast<int>(pdf.at<double>(0, j + 1) *
                                                   (plotHeight - 1));
            line(plot, Point(x1, y1), Point(x2, y2), Scalar(255, 255, 0), 2);
        }
    }
    // Draw x-axis ticks (adjust as needed)
    for (int i = 0; i <= 10; ++i) {
        int x = i * (plotWidth - 1) / 10;
        line(plot, Point(x, plotHeight - 5), Point(x, plotHeight + 5),
             Scalar(255, 255, 255), 2);
        putText(plot, to_string(i * 0.1), Point(x - 10, plotHeight + 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    // Draw a vertical line at x = 1/3
    int x_vertical = static_cast<int>((1.0 / 3.0) * (plotWidth - 1));
    line(plot, Point(x_vertical, 0), Point(x_vertical, plotHeight),
         Scalar(0, 255, 0), 2);

    int Tg_vertical = static_cast<int>(T_G * (plotWidth - 1));
    line(plot, Point(Tg_vertical, 0), Point(Tg_vertical, plotHeight),
         Scalar(0, 0, 255), 2);
    // Show the plot
    imshow("PDF of Gaussian Components", plot);
    waitKey(0);
}

void show_pdf(const Mat &pdf, const int num_points, const Mat &g,
              const double threshold) {
    Mat normalised_pdf = pdf.clone();
    // Normalize the sum for visualization
    double maxSumPDF = *max_element(pdf.begin<double>(), pdf.end<double>());
    for (int i = 0; i < num_points; ++i)
        normalised_pdf.at<double>(0, i) = pdf.at<double>(0, i) / maxSumPDF;

    // Visualize the sum of PDFs for all Gaussian components
    int plotWidth = 800;
    int plotHeight = 400;
    Mat plot = Mat::zeros(plotHeight, plotWidth, CV_8UC3);
    for (int i = 0; i < num_points - 1; ++i) {
        int x1 = static_cast<int>(g.at<double>(0, i) * (plotWidth - 1));
        int x2 = static_cast<int>(g.at<double>(0, i + 1) * (plotWidth - 1));
        int y1 = plotHeight -
                 static_cast<int>(normalised_pdf.at<double>(0, i) * plotHeight);
        int y2 =
            plotHeight -
            static_cast<int>(normalised_pdf.at<double>(0, i + 1) * plotHeight);
        line(plot, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 2);
    }

    // Draw x-axis ticks (adjust as needed)
    for (int i = 0; i <= 10; ++i) {
        int x = i * (plotWidth - 1) / 10;
        line(plot, Point(x, plotHeight - 5), Point(x, plotHeight + 5),
             Scalar(255, 255, 255), 2);
        putText(plot, to_string(i * 0.1), Point(x - 10, plotHeight + 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    // Draw a vertical line at x = 1/3
    int x_vertical = static_cast<int>((1.0 / 3.0) * (plotWidth - 1));
    line(plot, Point(x_vertical, 0), Point(x_vertical, plotHeight),
         Scalar(0, 255, 0), 2);

    int Tg_vertical = static_cast<int>(threshold * (plotWidth - 1));
    line(plot, Point(Tg_vertical, 0), Point(Tg_vertical, plotHeight),
         Scalar(0, 255, 255), 2);
    imshow("Probability Density Function of g", plot);
    waitKey(0);
}

void print_gaussians_data(const int N_G, const Mat &means,
                          const vector<Mat> &covariances, const Mat &weights) {
    std::cout.precision(3);
    for (int i = 0; i < N_G; i++)
        cout << "Gaussian #" << i + 1 << " has prior "
             << weights.at<double>(0, i) << ", \tvariance "
             << covariances[i].at<double>(0, 0) << ", \tmean "
             << means.at<double>(0, i) << endl;
}

Mat create_envelope(const vector<Mat> &covs, const Mat &means,
                    const Mat &weights, int num_points, int N_G) {
    // Set up the x-axis range
    Mat envelope = Mat::zeros(1, num_points, CV_64F);

    double min_x = 0.0;
    double max_x = 1.0;
    double step = (max_x - min_x) / static_cast<double>(num_points);

    vector<Mat> gaussians;
    for (int i = 0; i < covs.size(); ++i) {
        double mean = means.at<double>(0, i);
        double variance = covs[i].at<double>(0, 0);
        double weight = weights.at<double>(0, i);

        // Calculate the PDF for the current Gaussian component
        Mat pdf = Mat::zeros(1, num_points, CV_64F);
        for (int j = 0; j < num_points; j++) {
            double x = min_x + j * step;
            pdf.at<double>(0, j) = weight * compute_gaussian(x, mean, variance);
        }
        gaussians.push_back(pdf);
    }

    vector<double> y_values;
    // For each point
    for (int j = 0; j < num_points; j++) {
        // For each Gaussian
        for (int i = 0; i < N_G; i++)
            y_values.push_back(gaussians[i].at<double>(
                0, j));  // Create a vector of N_G values
        // Pick the max
        envelope.at<double>(0, j) =
            *max_element(y_values.begin(), y_values.end());

        y_values.clear();
    }
    return envelope.clone();
}

vector<LocalMinimum> findLocalMinima(const Mat &envelope, const Mat &g,
                                     const double threshold) {
    int num_elements = envelope.cols;
    vector<LocalMinimum> local_minima;

    double x;
    for (int i = 1; i < num_elements - 1; ++i) {
        double current_element = envelope.at<double>(0, i);
        double prev_element = envelope.at<double>(0, i - 1);
        double next_element = envelope.at<double>(0, i + 1);
        x = g.at<double>(0, i);
        if (current_element < prev_element && current_element < next_element &&
            x > threshold) {
            LocalMinimum minimum;
            minimum.value = current_element;
            minimum.x = g.at<double>(0, i);
            cout << "Found local minimum at (x,y)=(" << minimum.x << " ,"
                 << minimum.value << ")" << endl;
            local_minima.push_back(minimum);
        }
    }

    return local_minima;
}

float dot_product(Vec3b v, Vec3b u) {
    float sum = 0;
    for (int i = 0; i < 3; i++) sum += v[i] * u[i];
    return sum;
}

float compute_chromatic_distortion(Vec3b v, Vec3b u) {
    Vec3b u_v = dot_product(u, v) / dot_product(v, v) * v;
    Vec3b u_perp = u - u_v;
    float cd = sqrt(dot_product(u_perp, u_perp)) / sqrt(dot_product(u_v, u_v));
    return cd;
}

vector<Vec3b> compute_mean_colors(vector<PixelInfo> pixels,
                                  vector<int> counts) {
    vector<Vec3b> means;
    vector<Vec3f> sums;  // sum 1 sum2 sum3 sum4

    for (int count = 0; count < counts.size() + 1; count++)
        sums.push_back(Vec3f(0, 0, 0));

    for (PixelInfo pixel : pixels) {
        int group = pixel.group;
        if (group != 0)
            sums[group - 1] +=
                Vec3f(pixel.pixel[0], pixel.pixel[1], pixel.pixel[2]);
    }

    for (int count = 0; count < counts.size(); count++) {
        Vec3f mean = sums[count] / static_cast<float>(counts[count]);
        Vec3b mean_color(static_cast<uchar>(mean_color[0]),
                         static_cast<uchar>(mean_color[1]),
                         static_cast<uchar>(mean_color[2]));
        means.push_back(mean_color);
    }
    return means;
}

void plotEnvelope(const Mat &x_values, const Mat &envelope) {
    int plotWidth = 800;
    int plotHeight = 400;

    Mat plot = Mat::zeros(plotHeight, plotWidth, CV_8UC3);

    for (int i = 0; i < x_values.cols - 1; ++i) {
        int x1 = static_cast<int>(x_values.at<double>(0, i) * (plotWidth - 1));
        int x2 =
            static_cast<int>(x_values.at<double>(0, i + 1) * (plotWidth - 1));
        int y1 = plotHeight -
                 static_cast<int>(envelope.at<double>(0, i) * plotHeight);
        int y2 = plotHeight -
                 static_cast<int>(envelope.at<double>(0, i + 1) * plotHeight);

        line(plot, Point(x1, y1), Point(x2, y2), Scalar(0, 0, 255), 2);
    }

    imshow("Envelope", plot);
    waitKey(0);
}

Mat chromatic_distorsion_matrix(const Mat &I_gca, const Mat &I_open,
                                const double T_G,
                                const vector<LocalMinimum> &minima) {
    vector<PixelInfo> pixels;
    int num_minima = minima.size();
    if (num_minima == 0) {
        cout << "No minima found, returning 0.0 matrix" << endl;
        return Mat(I_open.size(), CV_32F, Scalar(0.0));
    }

    vector<int> counts(num_minima + 2, 0);  // Initialize counts for each range

    // Assign each pixel to a cluster
    for (int y = 0; y < I_open.rows; ++y) {
        for (int x = 0; x < I_open.cols; ++x) {
            PixelInfo pixel;
            pixel.pixel = I_open.at<Vec3b>(y, x);

            float g_value = I_gca.at<float>(y, x);

            // Find the appropriate range for the pixel
            int range = 0;
            if (g_value < T_G && g_value > 0)
                range = 0;
            else if (g_value > T_G && g_value < minima[0].x)
                range = 1;
            else if (g_value > minima[num_minima - 1].x && g_value <= 1)
                range = num_minima + 1;
            else {
                // Search for the range where g_value falls between minima
                for (int i = 0; i < num_minima; ++i) {
                    if (g_value > minima[i].x && g_value <= minima[i + 1].x) {
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
    vector<Vec3b> avgs = compute_mean_colors(pixels, counts);

    // Compute the distorsion value for each pixel depending on the cluster
    Mat cd_matrix(I_open.rows, I_open.cols, CV_32F);

    int height = I_open.rows;
    int width = I_open.cols;

    Mat group_matrix(I_open.rows, I_open.cols, CV_8U);

    // Assign each pixel to a cluster based on the information in the pixels
    // vector
    for (int y = 0; y < I_open.rows; ++y) {
        for (int x = 0; x < I_open.cols; ++x) {
            // Get the cluster assignment from the corresponding pixel in the
            // pixels vector
            int cluster = pixels[y * I_open.cols + x].group;

            // Assign the cluster value to the corresponding location in the
            // group_matrix
            group_matrix.at<uchar>(y, x) = static_cast<uchar>(cluster);
        }
    }

    I_open.forEach<Vec3b>([&cd_matrix, &group_matrix, &avgs](
                              Vec3b &pixel, const int *position) -> void {
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
        cd_matrix.at<float>(position[0], position[1]) = cd_value;
    });

    return cd_matrix.clone();
}

Mat find_M_PF_tilde(const double T_C, const Mat &mask1, const Mat &cd_matrix) {
    // Initialize the result mask with zeros
    Mat mask2(mask1.size(), CV_8U, Scalar(0));

    // Check the condition and set values in the mask2 mask
    for (int r = 0; r < mask1.rows; ++r)
        for (int c = 0; c < mask1.cols; ++c)
            if (mask1.at<uchar>(r, c) == 255 && cd_matrix.at<float>(r, c) < T_C)
                mask2.at<uchar>(r, c) = 255;
    return mask2.clone();
}