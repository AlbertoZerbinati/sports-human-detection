// Alberto Zerbinati

// #include "field-detection/GreenFieldSegmentation.hpp"

// using namespace cv;

// TODO...

// Mat ColorFieldSegmentation(const Mat &image, const Vec3b estimated_color) {
//     Mat mask = Mat::zeros(image.size(), CV_8U);
//     int threshold = 25;

//     // fill the mask with white  where the image pixels color is in threshold
//     // with the estimated color
//     for (int y = 0; y < image.rows; y++) {
//         for (int x = 0; x < image.cols; x++) {
//             if (abs(image.at<Vec3b>(y, x)[0] - estimated_color[0]) <
//                     threshold and
//                 abs(image.at<Vec3b>(y, x)[1] - estimated_color[1]) <
//                     threshold and
//                 abs(image.at<Vec3b>(y, x)[2] - estimated_color[2]) <
//                     threshold) {
//                 mask.at<uchar>(y, x) = 255;
//             }
//         }
//     }

//     return mask.clone();
// }
