// Alberto Zerbinati

#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <map>
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
    Pipeline(const cv::Mat& image, const std::string model_path,
             std::string groundTruthBBoxesFilePath,
             std::string groundTruthSegmentationMaskPath);

    // Destructor
    ~Pipeline();

    // Runs the pipeline on the given image, returning the output
    PipelineRunOutput run();

    // Evaluates the pipeline output, returning the evaluation metrics
    PipelineEvaluateOutput evaluate(PipelineRunOutput detections);

   private:
    cv::Mat image_;
    std::string model_path_;
    PeopleDetector peopleDetector_;
    PeopleSegmentation peopleSegmentation_;

    cv::Vec3b extractFieldColor(const cv::Mat& originalWindow,
                                const cv::Mat& personMask);

    cv::Vec3b extractTeamColor(
        const cv::Mat& mask,
        std::map<cv::Vec3b, int, Utils::Vec3bCompare> teamsColors = {});
};

#endif  // PIPELINE_HPP
