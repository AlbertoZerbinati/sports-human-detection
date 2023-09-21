// Marco Calì

#ifndef FIELD_SEGMENTATION_HPP
#define FIELD_SEGMENTATION_HPP

#include <opencv2/core/core.hpp>

// TODO: make a class...

// TODO: rename to lowercase...
cv::Mat greenFieldSegmentation(const cv::Mat &I);
cv::Mat colorFieldSegmentation(const cv::Mat &src,
                               const cv::Vec3b estimated_field_color);
cv::Mat fieldSegmentation(const cv::Mat &src,
                          const cv::Vec3b estimated_field_color);

#endif  // FIELD_SEGMENTATION_HPP
