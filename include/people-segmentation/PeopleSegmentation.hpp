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
    Mat segmentPeople(const Mat& original);

   private:
    Mat skinDetect(const Mat& original);
    Mat peopleSegm(const Mat& original);
};

#endif  // PEOPLE_SEGMENTATION_HPP
