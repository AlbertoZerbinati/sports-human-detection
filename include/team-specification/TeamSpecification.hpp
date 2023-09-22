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
 * Finds the dominant color in an image, with an option to ignore team colors.
 * If team colors are not ignored, the function uses an internal threshold to
 * check if the calculated dominant color is similar to one of the teamColors.
 * @param matrix Input image in cv::Mat format.
 * @param ignoreTeamColors Flag to ignore predefined team colors.
 * @param teamsColors Map of team colors/count to be optionally ignored.
 * @return The dominant color as a cv::Vec3b object.
 */
cv::Vec3b findDominantColor(
    cv::Mat matrix, bool ignoreTeamColors,
    std::map<cv::Vec3b, int, Utils::Vec3bCompare> teamsColors);

};  // namespace TeamSpecification

#endif  // TEAM_SPECIFICATION_HPP
