// Alberto Zerbinati

#ifndef TEAM_SPECIFICATION_HPP
#define TEAM_SPECIFICATION_HPP

#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "utils/Utils.hpp"

namespace TeamSpecification {

cv::Vec3b findDominantColor(
    cv::Mat matrix, bool ignoreTeamColors,
    std::map<cv::Vec3b, int, Utils::Vec3bCompare> teamsColors);

};  // namespace TeamSpecification

#endif  // TEAM_SPECIFICATION_HPP
