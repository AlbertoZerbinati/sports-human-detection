// Sedusi Marco

#ifndef PEOPLE_SEGMENTATION_HPP
#define PEOPLE_SEGMENTATION_HPP

// packages
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

class PeopleSegmentation {
   public:
    void segmentPeople(const Mat& original, Mat& dest);

   private:
    void skinDetect(const Mat& original, Mat& dest);
    void peopleSegm(const Mat& original, Mat& dest);
};

#endif  // PEOPLE_SEGMENTATION_HPP
