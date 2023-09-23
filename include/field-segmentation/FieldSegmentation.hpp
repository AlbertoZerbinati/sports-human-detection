// Marco Calì

#ifndef FIELD_SEGMENTATION_HPP
#define FIELD_SEGMENTATION_HPP

#include <opencv2/core/core.hpp>

/**
 * A class for segmenting the sport field in an image.
 *
 * This class provides methods for segmenting a field from an input image
 * based on the estimated field color.
 */
class FieldSegmentation {
   public:
    /**
     * Segment the field in an image using a two-step approach.
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

    /**
     * @brief Estimate the most dominant color within an input image region.
     *
     * This function estimates the most dominant color within an input image
     * region by performing the following steps:
     * 1. Apply morphological opening to reduce noise and smoothen the region.
     * 2. Perform K-Means clustering to segment the region into color clusters.
     * 3. Identify the cluster with the highest pixel count, excluding colors
     * close to black, as it is assumed that a field cannot be black.
     * 4. Return the estimated dominant color in BGR format (cv::Vec3b).
     *
     * @param src Input image region (BGR color space).
     * @return The estimated dominant color as a 3-channel BGR vector
     * (cv::Vec3b).
     */
    cv::Vec3b estimateFieldColor(const cv::Mat &src);

   private:
    /**
     * @brief Filter and retain regions in a binary mask based on their area.
     *
     * This function takes a binary mask, identifies individual regions within
     * it using contours, and retains regions with an area greater than or equal
     * to a specified minimum area.
     *
     * The goal is to remove small areas due to the threshold being too low.
     *
     * @param mask Binary mask representing regions of interest.
     * @param minArea Minimum area required for a region to be retained.
     * @return A binary mask where regions meeting the area criteria are filled
     * (CV_8U).
     */
    cv::Mat filterRegions(const cv::Mat &mask, const double minArea);

    /**
     * @brief Check if a given color is close to black in the HLS color space.
     *
     * This function checks if a given color in the BGR color space is close to
     * black when converted to the HLS color space, based on the lightness (L)
     * component.
     *
     * @param b Blue channel value (0-255).
     * @param g Green channel value (0-255).
     * @param r Red channel value (0-255).
     * @param lightnessThreshold Threshold for considering a color close to
     * black.
     * @return True if the color is close to black, otherwise false.
     */
    bool isColorCloseToBlack(int b, int g, int r, int lightnessThreshold);

    /**
     * Segment the field based on color similarity to an estimated color.
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
