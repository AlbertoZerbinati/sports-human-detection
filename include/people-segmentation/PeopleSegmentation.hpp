// Marco Sedusi

#ifndef PEOPLE_SEGMENTATION_HPP
#define PEOPLE_SEGMENTATION_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

/**
 * This class is used to segment people from the background.
 */
class PeopleSegmentation {
   public:
    /**
     * This method is used to segment people from the background.
     * @param original is the original image
     * @param dest is the original image with only person segmentation
     */
    void segmentPeople(const Mat& original, Mat& dest);

   private:
    /**
     * This method is used to detect skin from the original image.
     * @param original is the original image
     * @param dest is the original image with only the skin detected
     */
    void skinDetect(const Mat& original, Mat& dest);

    /**
     * This method is used to segment people from the background.
     * @param original is the original image
     * @param dest is the original image with only person segmentation
     */
    void peopleSegm(const Mat& original, Mat& dest);
};

#endif  // PEOPLE_SEGMENTATION_HPP
