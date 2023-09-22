// Alberto Zerbinati

#ifndef PEOPLE_DETECTOR_HPP
#define PEOPLE_DETECTOR_HPP

#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * Structure to store the information about a detected window.
 */
struct DetectedWindow {
    int x, y, w, h;
    float confidence;
};

/**
 * PeopleDetector class responsible for detecting people in images.
 */
class PeopleDetector {
   public:
    /**
     * Constructs a PeopleDetector object.
     * @param model_path Path to the trained model.
     */
    PeopleDetector(const std::string& model_path);

    /**
     * Detects people in the given image.
     * @param image The input image in cv::Mat format.
     * @return A vector of DetectedWindow structs containing the detected
     * regions.
     */
    std::vector<DetectedWindow> detectPeople(const cv::Mat& image);

   private:
    torch::jit::script::Module model_;  // Loaded PyTorch model for detection.

    /**
     * Loads the PyTorch model from the given path.
     * @param model_path Path to the model file.
     */
    void loadModel(const std::string& model_path);

    /**
     * Converts an OpenCV image to a PyTorch tensor. Applies also the necessary
     * preprocessing needed for the image to be compatible with the model (e.g.
     * resize, normalization, etc).
     * @param image The input OpenCV image.
     * @param tensor The output PyTorch tensor.
     * @return True if successful, false otherwise.
     */
    bool imageToTensor(cv::Mat& image, torch::Tensor& tensor);

    /**
     * Performs sliding window detection on an image.
     * @param image The input image.
     * @param scale The scale factor for the sliding window.
     * @return A vector of DetectedWindow structs.
     */
    std::vector<DetectedWindow> performSlidingWindow(const cv::Mat& image,
                                                     float scale);

    /**
     * Applies k-means clustering to a set of centers.
     * @param centers The set of centers (points).
     * @param k The number of clusters.
     * @return A pair containing the cluster indices and the final loss.
     */
    std::pair<std::vector<int>, float> performKMeans(
        const std::vector<cv::Point2f>& centers, int k);

    /**
     * Computes the weighted loss from a k-means clustering and additional info.
     * @param kmeansLoss The k-means loss.
     * @param scale The scale factor.
     * @param k The number of clusters.
     * @param gamma The gamma parameter for loss weighting.
     * @param delta The delta parameter for loss weighting.
     * @return The weighted loss.
     */
    float computeWeightedLoss(float kmeansLoss, float scale, int k, float gamma,
                              float delta);
};

#endif  // PEOPLE_DETECTOR_HPP
