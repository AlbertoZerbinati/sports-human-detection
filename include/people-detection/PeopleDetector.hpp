// Alberto Zerbinati

#ifndef PEOPLE_DETECTOR_HPP
#define PEOPLE_DETECTOR_HPP

#include <torch/torch.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct DetectedWindow {
    int x, y, w, h;
    float confidence;
};

class PeopleDetector {
   public:
    PeopleDetector(const std::string& model_path);
    std::vector<DetectedWindow> detectPeople(const cv::Mat& image);

   private:
    torch::jit::script::Module model_;

    void loadModel(const std::string& model_path);
    bool ImageToTensor(cv::Mat& image, torch::Tensor& tensor);
    std::vector<DetectedWindow> performSlidingWindow(const cv::Mat& image,
                                                     float scale);
    std::pair<std::vector<int>, float> performKMeans(
        const std::vector<cv::Point2f>& centers, int k);
    float computeWeightedLoss(float kmeansLoss, float scale, int k, float gamma,
                              float delta);
};

#endif  // PEOPLE_DETECTOR_HPP
