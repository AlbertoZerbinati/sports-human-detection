// Alberto Zerbinati

#ifndef TEAM_SPECIFICATION_HPP
#define TEAM_SPECIFICATION_HPP

#include <map>
#include <opencv2/core/core.hpp>

#include "utils/Utils.hpp"

/**
 * Namespace containing functions for team specification.
 */
namespace TeamSpecification {

/**
 * Finds the dominant color in an image.
 * @param matrix Input image in cv::Mat format.
 * @return The dominant color as a cv::Vec3b object.
 */
cv::Vec3b findDominantColor(cv::Mat matrix);

};  // namespace TeamSpecification

#endif  // TEAM_SPECIFICATION_HPP
