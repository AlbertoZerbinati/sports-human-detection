// Alberto Zerbinati

#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <opencv2/core/core.hpp>
#include <vector>

#include "people-detection/PeopleDetector.hpp"
#include "people-segmentation/PeopleSegmentation.hpp"
#include "team-specification/TeamSpecification.hpp"

struct Player {
    int x;
    int y;
    int w;
    int h;
    int team;
    cv::Vec3b color;
};

struct PipelineRunOutput {
    std::vector<Player> boundingBoxes;
    cv::Mat segmentationBinMask;
    cv::Mat segmentationColorMask;
};

struct PipelineEvaluateOutput {
    float mIoU;  // for segmentation task
    float mAP;   // for detection task
};

class Pipeline {
   public:
    // Constructor
    Pipeline(const cv::Mat& image, std::string groundTruthBBoxesFilePath,
             std::string groundTruthSegmentationMaskPath);

    // Destructor
    ~Pipeline();

    // Runs the pipeline on the given image, returning the output
    PipelineRunOutput run();

    PipelineEvaluateOutput evaluate(PipelineRunOutput detections);

   private:
    PeopleDetector peopleDetector;
    PeopleSegmentation peopleSegmentation;
};

#endif  // PIPELINE_HPP
