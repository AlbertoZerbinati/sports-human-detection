// Alberto Zerbinati

#ifndef TEAM_SPECIFICATION_HPP
#define TEAM_SPECIFICATION_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace TeamSpecification {

cv::Vec3b findDominantColor(cv::Mat temp, cv::Vec3b team1Color,
                            cv::Vec3b team2Color, cv::Vec3b extraColor);

};  // namespace TeamSpecification

#endif  // TEAM_SPECIFICATION_HPP