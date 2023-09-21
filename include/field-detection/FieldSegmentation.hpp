// Marco Calì

#ifndef FIELD_SEGMENTATION_HPP
#define FIELD_SEGMENTATION_HPP

#include <opencv2/core/core.hpp>

class FieldSegmentation {
   public:
    void segmentField(const cv::Mat &src, cv::Mat &dst,
                      const cv::Vec3b estimated_field_color);

   private:
    void colorFieldSegmentation(const cv::Mat &src, cv::Mat &dst,
                                const cv::Vec3b estimated_field_color);
};

#endif  // FIELD_SEGMENTATION_HPP
