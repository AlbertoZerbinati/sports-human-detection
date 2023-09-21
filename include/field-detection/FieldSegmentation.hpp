// FieldSegmentation.hpp
// Author: Marco Calì

#ifndef FIELD_SEGMENTATION_HPP
#define FIELD_SEGMENTATION_HPP

#include <opencv2/core/core.hpp>

/**
 * @brief A class for segmenting the sport field in an image.
 *
 * This class provides methods for segmenting a field from an input image
 * based on the estimated field color.
 */
class FieldSegmentation {
   public:
    /**
     * @brief Segment the field in an image using a two-step approach.
     *
     * This function aims to segment the field in the input image. It first
     * attempts to perform green field segmentation using the
     * `GreenFieldSegmentation` class. If the resulting mask contains too few
     * non-zero pixels (indicating a weak segmentation), it falls back to
     * color-based segmentation using the `colorFieldSegmentation` method.
     *
     * @param src The input image to be segmented.
     * @param estimatedColor The estimated color of the field.
     * @return A binary mask where white pixels indicate the segmented field.
     */
    cv::Mat segmentField(const cv::Mat &src,
                         const cv::Vec3b estimated_field_color);

   private:
    /**
     * @brief Segment the field based on color similarity to an estimated color.
     *
     * This function performs color-based segmentation on an input image to
     * identify regions that closely match the estimated color. It creates a
     * binary mask where pixels with color similarity to the estimated color are
     * marked as foreground (white).
     *
     * @param image The input image to be segmented.
     * @param estimatedColor The estimated color used as a reference for
     * segmentation.
     * @return A binary mask where white pixels indicate regions similar to the
     * estimated color.
     */
    cv::Mat colorFieldSegmentation(const cv::Mat &image,
                                   const cv::Vec3b estimated_field_color);
};

#endif  // FIELD_SEGMENTATION_HPP
