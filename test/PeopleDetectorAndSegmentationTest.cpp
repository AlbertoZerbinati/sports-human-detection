// Alberto Zerbinati

#include <iostream>
#include <opencv2/opencv.hpp>

#include "people-detection/PeopleDetector.hpp"
#include "people-segmentation/PeopleSegmentation.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>"
                  << std::endl;
        return 1;
    }

    // Initialize PeopleDetector with a model path
    PeopleDetector detector(argv[1]);

    // Create a PeopleSegmentation object
    PeopleSegmentation people_segmentation;

    // Read a test image
    cv::Mat test_image = cv::imread(argv[2]);
    if (test_image.empty()) {
        std::cerr << "Error: Could not read the test image." << std::endl;
        return 1;
    }

    // Perform people detection
    std::vector<DetectedWindow> detected_windows =
        detector.detectPeople(test_image);

    std::cout << "Detected " << detected_windows.size() << " people."
              << std::endl;

    cv::Mat copy_image = test_image.clone();
    // Display the results
    for (const auto& window : detected_windows) {
        cv::rectangle(copy_image,
                      cv::Rect(window.x, window.y, window.w, window.h),
                      cv::Scalar(0, 255, 0), 2);
        std::cout << "Detected window at (" << window.x << ", " << window.y
                  << " of size (" << window.w << ", " << window.h
                  << ") with confidence: " << window.confidence << std::endl;
    }

    // Save the image with detected windows
    std::string output_image_path_0 = argv[2];
    output_image_path_0 =
        output_image_path_0.substr(output_image_path_0.find_last_of("/") + 1);

    cv::imwrite(output_image_path_0, copy_image);

    // mat for the full segmentation
    cv::Mat full_size_segmentation(test_image.size(), CV_8UC3,
                                   cv::Scalar(0, 0, 0));

    int i = 0;
    // for each detected window
    for (const auto& window : detected_windows) {
        // extract the mat representing the rectangle with coordinate of the
        // window
        cv::Rect rect(window.x, window.y, window.w, window.h);

        cv::Mat window_mat = test_image(rect);

        // perform people segmentation
        cv::Mat people_segmentation_mat =
            people_segmentation.merger(window_mat);

        // save the image with detected people
        // the output name is just the name of the image (substring after the
        // last /) stripped of the extension (after .) plus a number (i)
        std::string output_image_path = argv[2];
        output_image_path =
            output_image_path.substr(output_image_path.find_last_of("/") + 1);
        output_image_path =
            output_image_path.substr(0, output_image_path.find_last_of("."));
        output_image_path += "_" + std::to_string(i) + ".jpg";

        printf("Saving image %s\n", output_image_path.c_str());

        cv::imwrite(output_image_path, people_segmentation_mat);

        i++;

        // Paint the segmented areas red in the full-size image
        for (int y = 0; y < people_segmentation_mat.rows; ++y) {
            for (int x = 0; x < people_segmentation_mat.cols; ++x) {
                cv::Vec3b pixel = people_segmentation_mat.at<cv::Vec3b>(y, x);
                // Assuming white (or near white) indicates segmentation
                if (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0) {
                    full_size_segmentation.at<cv::Vec3b>(window.y + y,
                                                         window.x + x) =
                        cv::Vec3b(0, 0, 255);  // Red
                }
            }
        }
    }

    // Save the full-size segmentation image
    cv::imwrite("full_size_segmentation.jpg", full_size_segmentation);

    // Save the image with detected windows
    // std::string output_image_path = argv[2];
    // output_image_path =
    //     output_image_path.substr(output_image_path.find_last_of("/") + 1);

    // cv::imwrite(output_image_path, test_image);

    return 0;
}
