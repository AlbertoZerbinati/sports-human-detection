// Alberto Zerbinati

#include "pipeline/Pipeline.hpp"

#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <tuple>

#include "field-segmentation/FieldSegmentation.hpp"
#include "utils/Utils.hpp"

Pipeline::Pipeline(const cv::Mat& image, std::string model_path,
                   std::string groundTruthBBoxesFilePath,
                   std::string groundTruthSegmentationMaskPath)
    : image_(image.clone()),
      model_path_(model_path),
      peopleDetector_(model_path),
      peopleSegmentation_(),
      groundTruthBBoxesFilePath_(groundTruthBBoxesFilePath),
      groundTruthSegmentationMaskPath_(groundTruthSegmentationMaskPath) {}

PipelineRunOutput Pipeline::run() {
    // create the output variables
    PipelineRunOutput output;
    std::vector<Utils::ExtendedPlayerBoundingBox> extendedPlayers;
    cv::Mat segmentationBinMask;
    cv::Mat segmentationColorMask;

    // Clone the image
    cv::Mat image_clone = image_.clone();

    // Perform field detection
    std::cout << "\nPerforming field segmentation..." << std::endl;

    FieldSegmentation fieldSegmentation = FieldSegmentation();
    Vec3b estimatedFieldColor =
        fieldSegmentation.estimateFieldColor(image_clone);

    cv::Mat fieldSegmentationMat =
        fieldSegmentation.segmentField(image_clone, estimatedFieldColor);

    std::cout << "Field color (BGR): " << (int)estimatedFieldColor[0] << ", "
              << (int)estimatedFieldColor[1] << ", "
              << (int)estimatedFieldColor[2] << std::endl;

    // Perform people detection
    std::cout << "\nPerforming players detection..." << std::endl;
    std::vector<DetectedWindow> detected_windows =
        peopleDetector_.detectPeople(image_clone);

    // Teams colors map. Each bounding box can vote for a color, and the two
    // most voted colors will be the team colors
    std::map<cv::Vec3b, int, Utils::Vec3bCompare> teamsColors;

    std::cout << "\nIterating over " << detected_windows.size()
              << " detected windows to segment players and extract teams and "
                 "field colors..."
              << std::endl;
    // Main logic loop for segmentation
    int i = 0;
    for (const auto& window : detected_windows) {
        // Extract the bounding box
        cv::Rect rect(window.x, window.y, window.w, window.h);
        cv::Mat windowMat = image_clone(rect).clone();

        // Perform people segmentation
        cv::Mat peopleSegmentationMat =
            cv::Mat::zeros(image_clone.cols, image_clone.rows, CV_8UC3);
        peopleSegmentation_.segmentPeople(windowMat, peopleSegmentationMat);

        // Extract team color
        cv::Vec3b teamColor =
            extractTeamColor(peopleSegmentationMat, teamsColors);

        // Create a Player object and populate its fields (not the team yet!)
        Utils::ExtendedPlayerBoundingBox player;
        player.x = window.x;
        player.y = window.y;
        player.w = window.w;
        player.h = window.h;
        player.dominantColor = teamColor;
        player.colorMask = peopleSegmentationMat;
        player.team = -1;  // not assigned yet

        // Add the Player object to the vector
        extendedPlayers.push_back(player);

        i++;
    }

    // find team 1 color
    int max = 0;
    cv::Vec3b team1Color;
    for (auto& pair : teamsColors) {
        if (pair.second > max) {
            max = pair.second;
            team1Color = pair.first;
        }
    }

    // find team 2 color
    max = 0;
    cv::Vec3b team2Color;
    for (auto& pair : teamsColors) {
        if (pair.second > max &&
            !Utils::areColorsSame(pair.first, team1Color)) {
            max = pair.second;
            team2Color = pair.first;
        }
    }

    std::cout << "\nTeam 1 color (BGR): " << (int)team1Color[0] << ", "
              << (int)team1Color[1] << ", " << (int)team1Color[2] << std::endl;
    std::cout << "Team 2 color (BGR): " << (int)team2Color[0] << ", "
              << (int)team2Color[1] << ", " << (int)team2Color[2] << std::endl;

    std::cout << "\nAssigning teams to players..." << std::endl;
    // now we have the team colors and the field color, and all the masks
    // we can iterate through the extendedPlayers and assign them to teams
    for (auto& player : extendedPlayers) {
        // if the player color is within threshold of team1 color, assign it to
        // team 1
        if (Utils::areColorsWithinThreshold(player.dominantColor, team1Color,
                                            25)) {
            player.team = 1;
            player.teamConfidence = Utils::colorSimilarityConfidence(
                player.dominantColor, team1Color);
        } else if (Utils::areColorsWithinThreshold(player.dominantColor,
                                                   team2Color, 25)) {
            player.team = 2;
            player.teamConfidence = Utils::colorSimilarityConfidence(
                player.dominantColor, team2Color);
        } else {
            std::cout << "Found a person which is not a team player!"
                      << std::endl;
        }
    }

    std::cout << "\nCreating final segmentation masks..." << std::endl;

    // we can also create the final segmentation masks by combining the field
    // segmentation mask with all extendedPlayers segmentation masks. we apply
    // the field mask as last, as it is the most reliable usually
    segmentationBinMask = cv::Mat::zeros(image_clone.size(), CV_8UC1);
    segmentationColorMask = cv::Mat::zeros(image_clone.size(), CV_8UC3);

    for (auto& player : extendedPlayers) {
        for (int y = player.y; y < (player.y + player.h); ++y) {
            for (int x = player.x; x < (player.x + player.w); ++x) {
                cv::Vec3b maskValue =
                    player.colorMask.at<cv::Vec3b>(y - player.y, x - player.x);

                // If the mask value is not black
                if (Utils::areColorsSame(maskValue, cv::Vec3b(0, 0, 0))) {
                    continue;
                }

                // Possible TODO: check sovrappositions and improve
                // segmentation/detection.

                // Update segmentationBinMask based on exact team check
                if (player.team == 1) {
                    segmentationBinMask.at<uchar>(y, x) = 1;
                } else if (player.team == 2) {
                    segmentationBinMask.at<uchar>(y, x) = 2;
                }

                // Update segmentationColorMask based on exact team check
                cv::Vec3b color;
                if (player.team == 1) {
                    color = cv::Vec3b(0, 0, 255);
                } else if (player.team == 2) {
                    color = cv::Vec3b(255, 0, 0);
                }
                segmentationColorMask.at<cv::Vec3b>(y, x) = color;
            }
        }
    }

    // apply the field mask to the segmentation masks
    for (int y = 0; y < fieldSegmentationMat.rows; ++y) {
        for (int x = 0; x < fieldSegmentationMat.cols; ++x) {
            uchar fieldMaskValue = fieldSegmentationMat.at<uchar>(y, x);

            // Skip if the field mask value is 0
            if (fieldMaskValue == 0) {
                continue;
            }

            // Update segmentationBinMask
            segmentationBinMask.at<uchar>(y, x) = 3;

            // Update segmentationColorMask
            segmentationColorMask.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        }
    }

    // cast extendedplayers to extendedPlayers because we no longer need the
    // masks
    std::vector<Utils::PlayerBoundingBox> players;
    for (auto& player : extendedPlayers) {
        player.colorMask.release();
        players.push_back(player);
    }

    // populate the output object
    output.boundingBoxes = players;
    output.segmentationBinMask = segmentationBinMask;
    output.segmentationColorMask = segmentationColorMask;

    return output;
}

PipelineEvaluateOutput Pipeline::evaluate(PipelineRunOutput detections) {
    PipelineEvaluateOutput evalOutput;
    float mIoU;
    float mAP;

    // Calculate mIoU
    // read the ground truth segmentation mask
    cv::Mat groundTruthSegmentationMask =
        cv::imread(groundTruthSegmentationMaskPath_, cv::IMREAD_GRAYSCALE);

    if (groundTruthSegmentationMask.empty()) {
        std::cerr << "Error reading ground truth segmentation mask."
                  << std::endl;
        mIoU = 0;
    } else {
        mIoU = metricsEvaluator_.calculateMIoU(detections.segmentationBinMask,
                                               groundTruthSegmentationMask);
    }

    // Calculate mAP
    std::vector<Utils::PlayerBoundingBox> groundTruths =
        Utils::readBoundingBoxesFromFile(groundTruthBBoxesFilePath_);

    mAP =
        metricsEvaluator_.calculateMAP(groundTruths, detections.boundingBoxes);

    // populate the output object
    evalOutput.mIoU = mIoU;
    evalOutput.mAP = mAP;

    return evalOutput;
}

cv::Vec3b Pipeline::extractTeamColor(
    const cv::Mat& mask,
    std::map<cv::Vec3b, int, Utils::Vec3bCompare>& teamsColors) {
    // extract the dominant color of the team from the inverted mask
    cv::Vec3b teamColor = TeamSpecification::findDominantColor(mask);

    // Update the map
    bool found = false;
    for (auto& pair : teamsColors) {
        if (Utils::areColorsWithinThreshold(pair.first, teamColor, 25)) {
            pair.second++;
            found = true;
            break;
        }
    }
    if (!found) {
        teamsColors.insert(std::pair<cv::Vec3b, int>(teamColor, 1));
    }

    return teamColor;
}
