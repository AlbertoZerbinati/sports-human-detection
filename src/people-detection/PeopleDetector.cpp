// Alberto Zerbinati

#include "people-detection/PeopleDetector.hpp"

#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

PeopleDetector::PeopleDetector(const std::string& model_path) {
    loadModel(model_path);
}

void PeopleDetector::loadModel(const std::string& model_path) {
    try {
        model_ = torch::jit::load(model_path);
        model_.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        std::cerr << e.what();
        exit(EXIT_FAILURE);
    }
}

std::vector<DetectedWindow> PeopleDetector::detectPeople(const cv::Mat& image) {
    // Define hyperparameters
    const std::vector<float>& scales{0.8, 1.2};  // 0.4, 0.7, 1
    const float gamma{2};
    const float delta{1.3};

    std::vector<DetectedWindow> best_windows;             // Best windows so far
    float best_loss = std::numeric_limits<float>::max();  // Best loss so far
    std::vector<int> best_labels;  // Best labels so far (a label tells us which
                                   // cluster a window belongs to)
    float best_scale;              // Best scale so far

    // Loop through each scale
    for (const float& scale : scales) {
        // Perform sliding window on the image at this scale
        std::vector<DetectedWindow> detected_windows =
            performSlidingWindow(image, scale);

        // Get the number of detected windows
        int num_windows = detected_windows.size();
        // std::cout << "Number of detected windows at scale " << scale << ": "
        //           << num_windows << std::endl;

        int max_classes = std::min(num_windows, 9);

        // best_windows = detected_windows;

        // Perform K-means clustering on detected windows to remove duplicates
        for (int k = 1; k <= max_classes; ++k) {
            // Extract the centers of the detected windows
            std::vector<cv::Point2f> centers;
            for (const auto& win : detected_windows) {
                cv::Point2f center((win.x + win.w) / 2.0,
                                   (win.y + win.h) / 2.0);
                centers.push_back(center);
            }

            // Perform K-means clustering
            std::pair<std::vector<int>, float> labelsAndLoss;
            labelsAndLoss = performKMeans(centers, k);

            std::vector<int> labels = labelsAndLoss.first;
            float loss = labelsAndLoss.second;

            // Compute weighted loss for each cluster and select the best ones
            float weighted_loss =
                computeWeightedLoss(loss, scale, k, gamma, delta);

            if (weighted_loss < best_loss) {
                best_loss = weighted_loss;
                best_labels = labels;
                best_windows = detected_windows;
                best_scale = scale;

                // std::cout << "New best loss for k " << k << " :" << best_loss
                //           << std::endl;
            }
        }
    }

    // Finally: for each cluster in the best clustering, select the best
    // window of that cluster.
    std::vector<DetectedWindow> predicted_boxes;
    std::set<int> unique_labels(best_labels.begin(), best_labels.end());

    for (const auto& label : unique_labels) {
        std::vector<DetectedWindow> cluster_windows;

        // Filter windows that have the current label
        for (size_t i = 0; i < best_windows.size(); ++i) {
            if (best_labels[i] == label) {
                cluster_windows.push_back(best_windows[i]);
            }
        }

        // Sort by confidence
        std::sort(cluster_windows.begin(), cluster_windows.end(),
                  [](const DetectedWindow& a, const DetectedWindow& b) {
                      return a.confidence > b.confidence;
                  });

        // Choose the box with the highest confidence
        if (!cluster_windows.empty()) {
            DetectedWindow best_window = cluster_windows[0];
            predicted_boxes.push_back(best_window);
        }
    }

    return predicted_boxes;
}

std::vector<DetectedWindow> PeopleDetector::performSlidingWindow(
    const cv::Mat& image, float scale) {
    std::vector<DetectedWindow> detected_windows;

    // Resize the image based on the scale
    // cv::Mat resized_image;
    // cv::resize(image, resized_image, cv::Size(), scale, scale);

    // Define window size and step size based on image w and h
    float base_window_width = float(image.cols) / 5.0;
    float base_window_height = float(image.rows) / 3.0;
    int window_width = int(base_window_width * scale);
    int window_height = int(base_window_height * scale);
    int step_size = 50;

    // save the resized image with a rectangle of the size for inspection
    // cv::rectangle(resized_image, cv::Point(0, 0),
    //               cv::Point(window_width, window_height), cv::Scalar(0, 255,
    //               0), 2);
    // // assign a name based on the scale
    // std::string name = "resized_image_" + std::to_string(scale) + ".jpg";
    // cv::imwrite(name, resized_image);

    float confidence_threshold = 0.85;  // Set your confidence threshold here

    for (int x = 0; x <= image.cols - window_width; x += step_size) {
        for (int y = 0; y <= image.rows - window_height; y += step_size) {
            cv::Mat image_copy = image.clone();
            cv::Rect window(x, y, window_width, window_height);

            cv::Mat crop = image_copy(window);

            torch::Tensor crop_tensor;
            if (!imageToTensor(crop, crop_tensor)) {
                std::cerr << "Error converting image to tensor\n";
                continue;  // Skip this window if conversion fails
            }

            // Create a vector of inputs
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(crop_tensor);

            // Execute the model
            at::Tensor output = model_.forward(inputs).toTensor();

            // Process the output
            std::vector<double> probs;
            for (int i = 0; i < output.size(0); i++) {
                double value = output[i].item<double>();
                probs.push_back(value);
            }

            // Get the maximum probability and its index
            auto max_prob = std::max_element(probs.begin(), probs.end());
            int max_prob_index = std::distance(probs.begin(), max_prob);

            if (max_prob_index == 1 && *max_prob >= confidence_threshold) {
                DetectedWindow win;
                win.x = x;
                win.y = y;
                win.w = window_width;
                win.h = window_height;
                win.confidence = *max_prob;

                detected_windows.push_back(win);
                // std::cout << "Detected window: " << win.x << " " << win.y <<
                // " "
                //           << win.w << " " << win.h << " " << win.confidence
                //           << std::endl;
            }
        }
    }

    return detected_windows;
}

bool PeopleDetector::imageToTensor(cv::Mat& image, torch::Tensor& tensor) {
    try {
        cv::Mat image_copy = image.clone();
        cv::cvtColor(image_copy, image_copy, cv::COLOR_BGR2RGB);
        cv::resize(image_copy, image_copy, cv::Size(100, 100));

        // mean and std are specified in the PyTorch documentation
        image_copy.convertTo(image_copy, CV_32F, 1.0 / 255.0);
        cv::Scalar mean_val(0.485, 0.456, 0.406);
        cv::Scalar std_val(0.229, 0.224, 0.225);
        cv::subtract(image_copy, mean_val, image_copy);
        cv::divide(image_copy, std_val, image_copy);

        // Transpose the image to match the PyTorch tensor format (C, H, W)
        cv::transpose(image_copy, image_copy);

        tensor = torch::from_blob(image_copy.data, {1, 100, 100, 3},
                                  torch::kFloat32);

        tensor = tensor.permute({0, 3, 1, 2});

        return true;
    } catch (...) {
        return false;
    }
}

std::pair<std::vector<int>, float> PeopleDetector::performKMeans(
    const std::vector<cv::Point2f>& centers, int k) {
    std::vector<int> labels(centers.size(), 0);
    int n = centers.size();
    float inertia = 0.0;

    if (k > n) {
        throw std::invalid_argument("k is greater than points size.");
    }

    std::vector<cv::Point2f> centroids;
    std::srand(std::time(0));
    centroids.push_back(centers[std::rand() % n]);

    while (centroids.size() < k) {
        std::vector<float> distances;
        for (const auto& p : centers) {
            float minDistanceFromCentroids = std::numeric_limits<float>::max();
            for (const auto& c : centroids) {
                float dist = cv::norm(p - c);
                minDistanceFromCentroids =
                    std::min(minDistanceFromCentroids, dist);
            }
            distances.push_back(minDistanceFromCentroids);
        }
        int newCentroidIndex =
            std::distance(distances.begin(),
                          std::max_element(distances.begin(), distances.end()));
        centroids.push_back(centers[newCentroidIndex]);
    }

    // K-means loop
    for (int iter = 0; iter < 100; ++iter) {  // Assuming max 100 iterations
        // Assign each point to the nearest centroid and update labels
        for (size_t i = 0; i < centers.size(); ++i) {
            float minDist = std::numeric_limits<float>::max();
            int minIndex = 0;
            for (size_t j = 0; j < centroids.size(); ++j) {
                float dist = cv::norm(centers[i] - centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    minIndex = j;
                }
            }
            labels[i] = minIndex;
        }

        // Recalculate centroids as the mean of all points in the cluster
        std::vector<cv::Point2f> newCentroids(k, cv::Point2f(0, 0));
        std::vector<int> counts(k, 0);
        for (size_t i = 0; i < labels.size(); ++i) {
            newCentroids[labels[i]] += centers[i];
            counts[labels[i]]++;
        }
        for (size_t i = 0; i < centroids.size(); ++i) {
            if (counts[i] > 0) {
                newCentroids[i] *= (1.0 / counts[i]);
            }
        }
        centroids = newCentroids;
    }

    // Compute inertia (sum of squared distances to nearest centroid)
    for (size_t i = 0; i < centers.size(); ++i) {
        inertia += cv::norm(centers[i] - centroids[labels[i]]);
    }

    return {labels, inertia};
}

float PeopleDetector::computeWeightedLoss(float kmeansLoss, float scale, int k,
                                          float gamma, float delta) {
    double scalePenalty = kmeansLoss / 10 * gamma / std::pow(scale, 1);
    double kPenalty = kmeansLoss / 10 * delta * std::pow(k, 1);
    double loss = kmeansLoss + scalePenalty + kPenalty;
    // std::cout << "kmeansLoss: " << kmeansLoss
    //           << " scalePenalty: " << scalePenalty << " kPenalty: " << kPenalty
    //           << " loss: " << loss << std::endl;
    return loss;
}
