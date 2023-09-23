// Marco Sedusi

#include "people-segmentation/PeopleSegmentation.hpp"

#include "team-specification/TeamSpecification.hpp"

// Packages
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

// Skin-Detection function
/* Parameters: original image
 * destination image is the original image with only the skin detected */
void PeopleSegmentation::skinDetect(const Mat& original, Mat& dest) {
    // Support Mat var
    // Convert the image from BGR to the HSV color space
    cv::Mat img_HSV;
    cv::cvtColor(original, img_HSV, cv::COLOR_BGR2HSV);

    // Lower and upper bounds for skin color in HSV
    // Colors are described by Hue, Saturation and Value
    cv::Scalar lowerBound(0, 30, 70);
    cv::Scalar upperBound(27, 150, 220);

    // Create a mask for skin color within the specified range described above
    cv::Mat skin;
    cv::inRange(img_HSV, lowerBound, upperBound, skin);

    // Apply the mask to the original image
    // Define a kernel
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    // Perform Morphological operations in order to remove small noise with the
    // Opening operation, Closing operation is used to fill small gaps in the
    // detected regions
    cv::morphologyEx(skin, skin, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(skin, skin, cv::MORPH_CLOSE, kernel);

    // We have used the bitwise_and function in order to combine the original
    // image with the skin mask obtained This operation keeps only the parts of
    // the original image that correspond to the white regions of the skin mask,
    // effectively highlighting the detected skin regions while blacking out
    // other areas
    cv::bitwise_and(original, original, dest, skin);
}

// People-Segmentation function
/* Parameters: original image
 * destination image is the original image with only person segmentation */
void PeopleSegmentation::peopleSegm(const Mat& original, Mat& dest) {
    // Define the support images
    cv::Mat mask = Mat::zeros(original.cols, original.rows, CV_8UC1);
    cv::Mat bgdModel;
    cv::Mat fgdModel;

    int x = original.cols;
    int y = original.rows;

    // Aspect ratio consideration
    // Different rect size for grabcut algorithm are taken into account, in
    // order to recognize the subject in the bounding box

    /* Grabcut parameters:
     * original is the input image
     * mask serves as an initial mask for the algorithm. The mask is used to
     * specify which parts of the image are known foreground, known background,
     * probable foreground, and probable background. It is modified by the
     * grabCut algorithm during its execution to refine the segmentation. rect
     * defines a rectangular region in the original image. bgdModel and fdgModel
     * are matrices used by the grabCut algorithm to model the foreground and
     * background. 10 is the number of iterations the grabCut algorithm should
     * run for. GC_INIT_WITH_RECT is a flag that is used to specify the
     * initialization method for the algorithm. In this case, it indicates that
     * the algorithm should be initialized using the rectangular region defined
     * by the rect parameter.
     * */

    // Case 4:3
    if (y > x + (x / 3)) {
        if (y < 200) {
            cv::Rect rect = Rect(x / 8, y / 20, x - (x / 8), y - (y / 18));
            cv::grabCut(original, mask, rect, bgdModel, fgdModel, 10,
                        cv::GC_INIT_WITH_RECT);
        } else {
            cv::Rect rect = Rect(x / 8, y / 20, x - (x / 5), y - (y / 9));
            cv::grabCut(original, mask, rect, bgdModel, fgdModel, 10,
                        cv::GC_INIT_WITH_RECT);
        }

    }
    // Case 16:9
    else if (y + (y / 3) < x) {
        cv::Rect rect = Rect(x / 4, y / 6, x - (x / 4), y - (y / 6));
        cv::grabCut(original, mask, rect, bgdModel, fgdModel, 10,
                    cv::GC_INIT_WITH_RECT);
    }
    // Case 1:1
    else {
        cv::Rect rect = Rect(x / 6, y / 9, x - (x / 4) - (x / 7), y - (y / 4));
        cv::grabCut(original, mask, rect, bgdModel, fgdModel, 10,
                    cv::GC_INIT_WITH_RECT);
    }

    // Take only pixels segmented as foreground
    cv::Mat mask2 = (mask == 1) + (mask == 3);
    original.copyTo(dest, mask2);
}

/*  SegmentPeople function merging images obtained from skin-detection and
 * people-segmentation into an unique image*/
/* Parameters: original image
 * destination image is the original image with only person segmentation and
 * skin detection */
void PeopleSegmentation::segmentPeople(const Mat& original, Mat& dest) {
    // Support vars
    cv::Mat skinResult = Mat::zeros(original.cols, original.rows, CV_8UC3);
    cv::Mat peopleResult = Mat::zeros(original.cols, original.rows, CV_8UC3);

    // Call function
    skinDetect(original, skinResult);
    peopleSegm(original, peopleResult);

    // Merge the two mask
    dest = skinResult + peopleResult;

    // Clean the image after processing
    for (int i = 0; i < original.rows; i++) {
        for (int j = 0; j < original.cols; j++) {
            if (!(dest.at<Vec3b>(i, j)[0] == 0 &&
                  dest.at<Vec3b>(i, j)[1] == 0 &&
                  dest.at<Vec3b>(i, j)[2] == 0)) {
                dest.at<Vec3b>(i, j)[0] = original.at<Vec3b>(i, j)[0];
                dest.at<Vec3b>(i, j)[1] = original.at<Vec3b>(i, j)[1];
                dest.at<Vec3b>(i, j)[2] = original.at<Vec3b>(i, j)[2];
            }
        }
    }
}
