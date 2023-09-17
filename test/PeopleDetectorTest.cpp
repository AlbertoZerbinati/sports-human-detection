#include <iostream>
#include <opencv2/opencv.hpp>

#include "people-detection/PeopleDetector.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>"
                  << std::endl;
        return 1;
    }

    // Initialize PeopleDetector with a model path
    PeopleDetector detector(argv[1]);

    // Read a test image
    cv::Mat test_image = cv::imread(argv[2]);
    if (test_image.empty()) {
        std::cerr << "Error: Could not read the test image." << std::endl;
        return 1;
    }

    // Perform people detection
    std::vector<DetectedWindow> detected_windows =
        detector.detectPeople(test_image);

    // Display the results
    for (const auto& window : detected_windows) {
        cv::rectangle(test_image,
                      cv::Rect(window.x, window.y, window.w, window.h),
                      cv::Scalar(0, 255, 0), 2);
        std::cout << "Detected window at (" << window.x << ", " << window.y
                  << " of size (" << window.w << ", " << window.h
                  << ") with confidence: " << window.confidence << std::endl;
    }

    // Save the image with detected windows
    cv::imwrite("Detected_People.jpg", test_image);

    return 0;
}

// #include <torch/script.h>
// #include <torch/torch.h>

// #include <iostream>
// #include <opencv2/opencv.hpp>

// #include "people-detection/PeopleDetector.hpp"

// int main(int argc, char** argv) {
//     if (argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>"
//                   << std::endl;
//         return 1;
//     }

//     torch::jit::script::Module model_ = torch::jit::load(argv[1]);
//     // model_.to(at::kCPU, torch::k);
//     // model_.eval();

//     // create a ones tensor of size 1,3,100,100
//     torch::Tensor crop_tensor_ = torch::ones({1, 3, 100, 100});

//     // run the model and print the result
//     std::vector<torch::jit::IValue> inputs_;
//     inputs_.push_back(crop_tensor_);

//     at::Tensor output_ = model_.forward(inputs_).toTensor();

//     std::cout << output_ << std::endl;

//     cv::Mat test_image = cv::imread(argv[2]);
//     cv::Mat image = test_image.clone();

//     cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//     cv::resize(image, image, cv::Size(100, 100));

//     image.convertTo(image, CV_32F, 1.0 / 255.0);

//     // mean and std are specified in the PyTorch documentation
//     cv::Scalar mean_val(0.485, 0.456, 0.406);
//     cv::Scalar std_val(0.229, 0.224, 0.225);
//     cv::subtract(image, mean_val, image);
//     cv::divide(image, std_val, image);

//     // Transpose the image to match the PyTorch tensor format (C, H, W)
//     cv::transpose(image, image);

//     // save image for inspection
//     // cv::imwrite("test_image.jpg", image);

//     torch::Tensor crop_tensor = torch::from_blob(
//         image.data, {1, image.rows, image.cols, 3}, torch::kFloat32);

//     crop_tensor = crop_tensor.permute({0, 3, 1, 2});

//     // Create a vector of inputs
//     std::vector<torch::jit::IValue> inputs;
//     inputs.push_back(crop_tensor);

//     // // Execute the model
//     at::Tensor output = model_.forward(inputs).toTensor();

//     // // print output shape
//     // std::cout << "Output shape: " << output.sizes() << std::endl;

//     std::cout << output << std::endl;

//     // // Get the maximum probability and its index
//     // auto max_prob = std::max_element(probs.begin(), probs.end());
//     // int max_prob_index = std::distance(probs.begin(), max_prob);

//     // if (max_prob_index == 1 && *max_prob >= 0.9) {
//     //     std::cout << "Person detected with confidence: " << *max_prob
//     //               << std::endl;
//     // } else {
//     //     std::cout << "No person detected." << std::endl;
//     // }

//     return 0;
// }
