// Alberto Zerbinati

#include <iostream>
#include <opencv2/opencv.hpp>

#include "pipeline/Pipeline.hpp"

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

    // extract the file name from the path
    std::string img_name = argv[1];
    img_name = img_name.substr(img_name.find_last_of("/") + 1);
    img_name = img_name.substr(0, img_name.find_last_of("."));

    // Save the outputs
    cv::imwrite(img_name + "_bin_mask.png", runOutput.segmentationBinMask);
    cv::imwrite(img_name + "_color_mask.png", runOutput.segmentationColorMask);

    // TODO: write the bounding boxes to file

    // Evaluate the pipeline (if needed)
    PipelineEvaluateOutput evalOutput = pipeline.evaluate(runOutput);

    // Output results (for demonstration)
    std::cout << "\nNumber of detected players: "
              << runOutput.boundingBoxes.size() << std::endl;

    // Add your code to display or save results
    // ...

    return 0;
}
