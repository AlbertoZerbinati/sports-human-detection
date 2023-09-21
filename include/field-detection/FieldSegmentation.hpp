// Marco Calì

#ifndef FIELD_SEGMENTATION_HPP
#define FIELD_SEGMENTATION_HPP

#include <opencv2/core/core.hpp>

class FieldSegmentation {
   public:
    cv::Mat segmentField(const cv::Mat &src,
                         const cv::Vec3b estimated_field_color);

   private:
    cv::Mat colorFieldSegmentation(const cv::Mat &src,
                                   const cv::Vec3b estimated_field_color);
};

#endif  // FIELD_SEGMENTATION_HPP
