// Alberto Zerbinati

#include <iostream>
#include <opencv2/opencv.hpp>

#include "pipeline/Pipeline.hpp"
#include "utils/Utils.hpp"

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_path> <model_path> <groundTruthBBoxesFilePath> "
                     "<groundTruthSegmentationMaskPath>"
                  << std::endl;
        return 1;
    }

    // Load the test image
    cv::Mat img = cv::imread(argv[1]);
    if (img.empty()) {
        std::cerr << "Error: Could not read the test image." << std::endl;
        return 1;
    }

    // Initialize the Pipeline
    Pipeline pipeline(img, argv[2], argv[3], argv[4]);

    // Run the pipeline
    PipelineRunOutput runOutput = pipeline.run();

    std::cout << "\nRun completed! Evaluating results..." << std::endl;

    // extract the file name from the path
    std::string img_name = argv[1];
    img_name = img_name.substr(img_name.find_last_of("/") + 1);
    img_name = img_name.substr(0, img_name.find_last_of("."));

    // Evaluate the pipeline output
    PipelineEvaluateOutput evalOutput = pipeline.evaluate(runOutput);

    // Print the evaluation metrics
    std::cout << "\n  - mIoU: " << evalOutput.mIoU << std::endl;
    std::cout << "  - mAP: " << evalOutput.mAP << std::endl;

    std::cout << "\nSaving outputs to files: " << img_name + "_bin_mask.png"
              << ", " << img_name + "_color_mask.png"
              << ", " << img_name + "_bboxes.txt"
              << ", " << img_name + "_bboxes.png" << std::endl;

    // Save the outputs
    cv::imwrite(img_name + "_bin_mask.png", runOutput.segmentationBinMask);
    cv::imwrite(img_name + "_color_mask.png", runOutput.segmentationColorMask);
    Utils::writeBoundingBoxesToFile(
        runOutput.boundingBoxes,
        img_name +
            "_bboxes.txt");  // TODO: here there are some players returned by
                             // the pipeline that are assigned to team -1 ...
                             // manage this throughout the pipeline and here
    Utils::saveBoundingBoxesOnImage(img, runOutput.boundingBoxes,
                                    img_name + "_bboxes.png");

    return 0;
}
