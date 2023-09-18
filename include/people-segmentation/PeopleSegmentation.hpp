/*
@Author Sedusi Marco
@Date 18-09-2023
Project-Name:sport-human-detectin
Task:people-segmentation
*/
#ifndef PEOPLE_SEGMENTATION_HPP
#define PEOPLE_SEGMENTATION_HPP

//packages
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

class PeopleSegmentation {
	public:
			Mat skinDetect(Mat origin);
			Mat peopleSegm(Mat original);
};

#endif  // PEOPLE_SEGMENTATION_HPP
