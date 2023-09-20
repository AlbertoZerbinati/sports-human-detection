// Marco Calì

#ifndef FIELD_SEGMENTATION_HPP
#define FIELD_SEGMENTATION_HPP

// TODO: This causes make to fail...
// I don't know why but anyway this code seriously needs refactoring
// #include "field-detection/GreenFieldSegmentation.hpp"

#include <opencv2/core/core.hpp>

// TODO: make a class...

// TODO: rename to lowercase...
cv::Mat GreenFieldsSegmentation(const cv::Mat &I);
cv::Mat GenericFieldSegmentation(const cv::Mat &image,
                                 const cv::Vec3b estimated_field_color,
                                 double mean_factor = 1, double std_factor = 1);
cv::Mat ColorFieldSegmentation(const cv::Mat &src,
                               const cv::Vec3b estimated_field_color);
cv::Mat FieldSegmentation(const cv::Mat &src,
                          const cv::Vec3b estimated_field_color);

#endif  // FIELD_SEGMENTATION_HPP
