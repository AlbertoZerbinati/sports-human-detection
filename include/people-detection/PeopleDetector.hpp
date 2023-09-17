#ifndef PEOPLE_DETECTOR_HPP
#define PEOPLE_DETECTOR_HPP

#include <torch/script.h>  // One-stop header for LibTorch

#include <opencv2/opencv.hpp>

class PeopleDetector {
   public:
    PeopleDetector(const std::string& model_path);
    ~PeopleDetector();

    void LoadModel();
    void ReadImage(const std::string& image_path);
    void SlidingWindowApproach(int window_size, int step_size);

   private:
    std::string model_path_;
    torch::jit::script::Module model_;
    cv::Mat input_image_;
};

#endif  // PEOPLE_DETECTOR_HPP
