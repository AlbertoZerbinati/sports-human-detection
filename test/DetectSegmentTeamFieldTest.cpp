// // Alberto Zerbinati

// #include <iostream>
// #include <opencv2/opencv.hpp>

// #include "field-detection/FieldSegmentation.hpp"
// #include "people-detection/PeopleDetector.hpp"
// #include "people-segmentation/PeopleSegmentation.hpp"
// #include "team-specification/TeamSpecification.hpp"

// // struct to hold player coordinates, team, color
// struct Player {
//     int x;
//     int y;
//     int w;
//     int h;
//     int team;
//     cv::Vec3b color;
// };

// struct Vec3bCompare {
//     bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const {
//         return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
//     }
// };

// int main(int argc, char** argv) {
//     if (argc != 3) {
//         std::cerr << "Usage: " << argv[0] << " <model_path> <image_path>"
//                   << std::endl;
//         return 1;
//     }

//     // Initialize PeopleDetector with a model path
//     PeopleDetector detector(argv[1]);

//     // Create a PeopleSegmentation object
//     PeopleSegmentation people_segmentation;

//     // Read a test image
//     cv::Mat test_image = cv::imread(argv[2]);
//     if (test_image.empty()) {
//         std::cerr << "Error: Could not read the test image." << std::endl;
//         return 1;
//     }

//     // Perform people detection
//     std::vector<DetectedWindow> detected_windows =
//         detector.detectPeople(test_image);

//     std::cout << "Detected " << detected_windows.size() << " people."
//               << std::endl;

//     cv::Mat copy_image = test_image.clone();
//     // Display the results
//     for (const auto& window : detected_windows) {
//         cv::rectangle(copy_image,
//                       cv::Rect(window.x, window.y, window.w, window.h),
//                       cv::Scalar(0, 255, 0), 2);
//         std::cout << "Detected window at (" << window.x << ", " << window.y
//                   << " of size (" << window.w << ", " << window.h
//                   << ") with confidence: " << window.confidence << std::endl;
//     }

//     // Save the image with detected windows
//     std::string output_image_path_0 = argv[2];
//     output_image_path_0 =
//         output_image_path_0.substr(output_image_path_0.find_last_of("/") + 1);

//     cv::imwrite(output_image_path_0, copy_image);

//     // mat for the full segmentation
//     cv::Mat full_size_segmentation(test_image.size(), CV_8UC3,
//                                    cv::Scalar(0, 0, 0));

//     // team shirt color
//     cv::Vec3b team1Color = cv::Vec3b(0, 0, 0);
//     int countColor1 = 0;
//     cv::Vec3b team2Color = cv::Vec3b(0, 0, 0);
//     int countColor2 = 0;
//     cv::Vec3b extraColor = cv::Vec3b(0, 0, 0);
//     int countExtraColor = 0;

//     // create a vector of players to hold the coordinates, team and color
//     std::vector<Player> players;

//     // global color of field as a map which at each color assigns a counter
//     // to count the number of times that color is found
//     std::map<cv::Vec3b, int, Vec3bCompare> fieldColors;

//     int i = 0;
//     // for each detected window
//     for (const auto& window : detected_windows) {
//         // extract the mat representing the rectangle with coordinate of the
//         // window
//         cv::Rect rect(window.x, window.y, window.w, window.h);

//         cv::Mat window_mat = test_image(rect);

//         // perform people segmentation
//         cv::Mat people_segmentation_mat =
//             people_segmentation.segmentPeople(window_mat);

//         // extract field color from the window by reverting the
//         // people_segmentation_mat (consider only the pixels which are black in
//         // the people_segmentation_mat) from the window_mat
//         cv::Mat field_mat = cv::Mat::zeros(window_mat.size(), CV_8UC3);
//         for (int y = 0; y < people_segmentation_mat.rows; ++y) {
//             for (int x = 0; x < people_segmentation_mat.cols; ++x) {
//                 cv::Vec3b pixel = people_segmentation_mat.at<cv::Vec3b>(y, x);
//                 if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
//                     field_mat.at<cv::Vec3b>(y, x) =
//                         window_mat.at<cv::Vec3b>(y, x);
//                 }
//             }
//         }

//         // extract the dominant color of the field from the field_mat
//         cv::Vec3b fieldColor = TeamSpecification::findDominantColor(
//             field_mat, true, team1Color, team2Color, extraColor);

//         // check if any of the colors in the fieldColors map is similar to the
//         // fieldColor (with threshold 15) if so increment the counter of that
//         // color, otherwise add the fieldColor to the map with counter 1
//         bool found = false;
//         for (auto& pair : fieldColors) {
//             if (abs(pair.first[0] - fieldColor[0]) < 20 &&
//                 abs(pair.first[1] - fieldColor[1]) < 20 &&
//                 abs(pair.first[2] - fieldColor[2]) < 20) {
//                 pair.second++;
//                 found = true;
//                 break;
//             }
//         }
//         if (!found) {
//             fieldColors.insert(std::pair<cv::Vec3b, int>(fieldColor, 1));
//         }

//         // save the image with detected people
//         // the output name is just the name of the image (substring after the
//         // last /) stripped of the extension (after .) plus a number (i)
//         std::string output_image_path = argv[2];
//         output_image_path =
//             output_image_path.substr(output_image_path.find_last_of("/") + 1);
//         output_image_path =
//             output_image_path.substr(0, output_image_path.find_last_of("."));
//         output_image_path += "_" + std::to_string(i) + ".jpg";

//         // printf("Saving image %s\n", output_image_path.c_str());

//         cv::imwrite(output_image_path, people_segmentation_mat);

//         Vec3b dominantColor = TeamSpecification::findDominantColor(
//             people_segmentation_mat, false, team1Color, team2Color, extraColor);

//         // cout the colors
//         cout << "Dominant Color:" << endl;
//         cout << "BGR: (" << (int)dominantColor[0] << ", "
//              << (int)dominantColor[1] << ", " << (int)dominantColor[2] << ")"
//              << endl;

//         // check if team color are still black and also if the dominant color
//         // was not assigned to a team already
//         if ((team1Color[0] == 0 && team1Color[1] == 0 && team1Color[2] == 0) &&
//             (abs(dominantColor[0] - team2Color[0]) != 0 &&
//              abs(dominantColor[1] - team2Color[1]) != 0 &&
//              abs(dominantColor[2] - team2Color[2]) != 0) &&
//             (abs(dominantColor[0] - extraColor[0]) != 0 &&
//              abs(dominantColor[1] - extraColor[1]) != 0 &&
//              abs(dominantColor[2] - extraColor[2]) != 0)) {
//             team1Color = dominantColor;
//             countColor1++;
//             std::cout << "assigned to team 1" << std::endl;
//         } else if ((team2Color[0] == 0 && team2Color[1] == 0 &&
//                     team2Color[2] == 0) &&
//                    (abs(dominantColor[0] - team1Color[0]) != 0 &&
//                     abs(dominantColor[1] - team1Color[1]) != 0 &&
//                     abs(dominantColor[2] - team1Color[2]) != 0) &&
//                    (abs(dominantColor[0] - extraColor[0]) != 0 &&
//                     abs(dominantColor[1] - extraColor[1]) != 0 &&
//                     abs(dominantColor[2] - extraColor[2]) != 0)) {
//             team2Color = dominantColor;
//             countColor2++;
//             std::cout << "assigned to team 2" << std::endl;
//         } else if ((extraColor[0] == 0 && extraColor[1] == 0 &&
//                     extraColor[2] == 0) &&
//                    (abs(dominantColor[0] - team1Color[0]) != 0 &&
//                     abs(dominantColor[1] - team1Color[1]) != 0 &&
//                     abs(dominantColor[2] - team1Color[2]) != 0) &&
//                    (abs(dominantColor[0] - team2Color[0]) != 0 &&
//                     abs(dominantColor[1] - team2Color[1]) != 0 &&
//                     abs(dominantColor[2] - team2Color[2]) != 0)) {
//             extraColor = dominantColor;
//             countExtraColor++;
//             std::cout << "assigned to team extra" << std::endl;
//         }

//         // swap extra color with one of the teams if it has greater count
//         if (countColor1 < countColor2 && countColor1 < countExtraColor) {
//             Vec3b temp = team1Color;
//             team1Color = extraColor;
//             extraColor = temp;
//             std::cout << "swapped team 1 with extra" << std::endl;
//         } else if (countColor2 < countColor1 && countColor2 < countExtraColor) {
//             Vec3b temp = team2Color;
//             team2Color = extraColor;
//             extraColor = temp;
//             std::cout << "swapped team 2 with extra" << std::endl;
//         }

//         // create a player and add it to the vector
//         Player player;
//         player.x = window.x;
//         player.y = window.y;
//         player.w = window.w;
//         player.h = window.h;
//         player.color = dominantColor;
//         if (dominantColor[0] == team1Color[0] &&
//             dominantColor[1] == team1Color[1] &&
//             dominantColor[2] == team1Color[2])
//             player.team = 1;
//         else if (dominantColor[0] == team2Color[0] &&
//                  dominantColor[1] == team2Color[1] &&
//                  dominantColor[2] == team2Color[2])
//             player.team = 2;
//         else if (dominantColor[0] == extraColor[0] &&
//                  dominantColor[1] == extraColor[1] &&
//                  dominantColor[2] == extraColor[2])
//             player.team = 0;
//         else
//             player.team = -1;
//         players.push_back(player);

//         // Paint the segmented areas red in the full-size image
//         for (int y = 0; y < people_segmentation_mat.rows; ++y) {
//             for (int x = 0; x < people_segmentation_mat.cols; ++x) {
//                 cv::Vec3b pixel = people_segmentation_mat.at<cv::Vec3b>(y, x);
//                 // Assuming white (or near white) indicates segmentation
//                 if (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0) {
//                     full_size_segmentation.at<cv::Vec3b>(
//                         window.y + y, window.x + x) = dominantColor;
//                 }
//             }
//         }
//         i++;
//         std::cout << std::endl;
//     }

//     // now extract the most common color from the fieldColors map
//     int max = 0;
//     cv::Vec3b fieldColor;
//     for (auto& pair : fieldColors) {
//         if (pair.second > max) {
//             max = pair.second;
//             fieldColor = pair.first;
//         }
//     }

//     // create an image with the color to inspect it
//     cv::Mat fieldColorMat = cv::Mat::zeros(100, 100, CV_8UC3);
//     for (int y = 0; y < fieldColorMat.rows; ++y) {
//         for (int x = 0; x < fieldColorMat.cols; ++x) {
//             fieldColorMat.at<cv::Vec3b>(y, x) = fieldColor;
//         }
//     }
//     cv::imwrite("field_color.jpg", fieldColorMat);

//     // print the color
//     cout << "Field Color:" << endl;
//     cout << "BGR: (" << (int)fieldColor[0] << ", " << (int)fieldColor[1] << ", "
//          << (int)fieldColor[2] << ")" << endl;

//     // perform field segmentation
//     cv::Mat field_segmentation_mat = FieldSegmentation(test_image, fieldColor);

//     // Save the field segmentation image
//     cv::imwrite("field_segmentation.jpg", field_segmentation_mat);

//     // Save the full-size segmentation image
//     cv::imwrite("full_size_segmentation.jpg", full_size_segmentation);

//     // cout all the players
//     cout << "Players:" << endl;
//     for (int i = 0; i < players.size(); i++) {
//         cout << "Player " << i << ": (" << players[i].x << ", " << players[i].y
//              << ") of size (" << players[i].w << ", " << players[i].h
//              << ") with color: (" << (int)players[i].color[0] << ", "
//              << (int)players[i].color[1] << ", " << (int)players[i].color[2]
//              << ") and team: " << players[i].team << endl;
//     }

//     // Save the image with detected windows
//     // std::string output_image_path = argv[2];
//     // output_image_path =
//     //     output_image_path.substr(output_image_path.find_last_of("/") + 1);

//     // cv::imwrite(output_image_path, test_image);

//     return 0;
// }
