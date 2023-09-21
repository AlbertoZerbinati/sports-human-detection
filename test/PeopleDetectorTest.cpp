// // Alberto Zerbinati

// #include <iostream>
// #include <opencv2/opencv.hpp>

// #include "people-detection/PeopleDetector.hpp"

// int main(int argc, char** argv) {
//     if (argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>"
//                   << std::endl;
//         return 1;
//     }

//     // Initialize PeopleDetector with a model path
//     PeopleDetector detector(argv[1]);

//     // Read a test image
//     cv::Mat test_image = cv::imread(argv[2]);
//     if (test_image.empty()) {
//         std::cerr << "Error: Could not read the test image." << std::endl;
//         return 1;
//     }

//     // Perform people detection
//     std::vector<DetectedWindow> detected_windows =
//         detector.detectPeople(test_image);

//     // Display the results
//     for (const auto& window : detected_windows) {
//         cv::rectangle(test_image,
//                       cv::Rect(window.x, window.y, window.w, window.h),
//                       cv::Scalar(0, 255, 0), 2);
//         std::cout << "Detected window at (" << window.x << ", " << window.y
//                   << " of size (" << window.w << ", " << window.h
//                   << ") with confidence: " << window.confidence << std::endl;
//     }

//     // Save the image with detected windows
//     std::string output_image_path = argv[2];
//     output_image_path =
//         output_image_path.substr(output_image_path.find_last_of("/") + 1);

//     cv::imwrite(output_image_path, test_image);

//     return 0;
// }
