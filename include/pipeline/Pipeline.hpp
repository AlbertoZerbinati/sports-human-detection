// Alberto Zerbinati

#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <map>
#include <opencv2/core/core.hpp>
#include <vector>

#include "people-detection/PeopleDetector.hpp"
#include "people-segmentation/PeopleSegmentation.hpp"
#include "team-specification/TeamSpecification.hpp"
#include "utils/Metrics.hpp"

/**
 * Structure to store the output of a pipeline run.
 */
struct PipelineRunOutput {
    std::vector<Utils::PlayerBoundingBox> boundingBoxes;
    cv::Mat segmentationBinMask;
    cv::Mat segmentationColorMask;
};

/**
 * Structure to store the evaluation metrics of a pipeline run.
 */
struct PipelineEvaluateOutput {
    float mIoU;  // for segmentation task
    float mAP;   // for detection task
};

/**
 * Pipeline class responsible for coordinating detection, segmentation, and
 * evaluation.
 */
class Pipeline {
   public:
    /**
     * Constructs a Pipeline object.
     * @param image Input image in cv::Mat format.
     * @param model_path Path to the model to use for detection/segmentation.
     * @param groundTruthBBoxesFilePath Path to the ground truth bounding boxes
     * file.
     * @param groundTruthSegmentationMaskPath Path to the ground truth
     * segmentation mask file.
     */
    Pipeline(const cv::Mat& image, const std::string model_path,
             std::string groundTruthBBoxesFilePath,
             std::string groundTruthSegmentationMaskPath);

    /**
     * Runs the pipeline on the given image and returns the output.
     * @return The PipelineRunOutput structure containing the results.
     */
    PipelineRunOutput run();

    /**
     * Evaluates the output of a pipeline run.
     * @param detections Output of the pipeline run to evaluate.
     * @return The PipelineEvaluateOutput structure containing the evaluation
     * metrics.
     */
    PipelineEvaluateOutput evaluate(PipelineRunOutput detections);

   private:
    cv::Mat image_;           // The image on which the pipeline operates.
    std::string model_path_;  // Path to the model used for detection and
                              // segmentation.
    PeopleDetector peopleDetector_;  // Object for performing people detection.
    PeopleSegmentation
        peopleSegmentation_;  // Object for performing people segmentation.
    MetricsEvaluator metricsEvaluator_;      // Object for evaluating detection
                                             // and segmentation metrics.
    std::string groundTruthBBoxesFilePath_;  // Path to the ground truth
                                             // bounding boxes file.
    std::string groundTruthSegmentationMaskPath_;  // Path to the ground truth
                                                   // segmentation mask file.

    /**
     * Extracts the dominant field color from an image window and a mask.
     * @param originalWindow Original image window.
     * @param personMask Mask for the person.
     * @return Dominant field color as a cv::Vec3b object.
     */
    cv::Vec3b extractFieldColor(const cv::Mat& originalWindow,
                                const cv::Mat& personMask);

    /**
     * Extracts the dominant team color from a mask.
     * @param mask Mask in cv::Mat format.
     * @param teamsColors Optional map of team colors to ignore.
     * @return Dominant team color as a cv::Vec3b object.
     */
    cv::Vec3b extractTeamColor(
        const cv::Mat& mask,
        std::map<cv::Vec3b, int, Utils::Vec3bCompare> teamsColors = {});
};

#endif  // PIPELINE_HPP
