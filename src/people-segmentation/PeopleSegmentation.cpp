/*
@Author Sedusi Marco
@Date 18-09-2023
Project-Name:sport-human-detectin
Task:people-segmentation
*/

#include "people-segmentation/PeopleSegmentation.hpp"

//packages
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/core.hpp"

//namespace
using namespace cv;
using namespace std;


//skin-detection-function
/**/
Mat skinDetect(Mat original) {
    // support Mat var
    // convert the image to the HSV color space
    Mat img_HSV;
    cvtColor(original, img_HSV, COLOR_BGR2HSV);

    // lower and upper bounds for skin color in HSV
    // colors are described by yours Hue, Saturation and Value in the range [0,255]
    Scalar lowerBound(0, 30, 70);  
    Scalar upperBound(27, 150, 220); 

    // create a mask for skin color within the specified range described above
    Mat skin;
    inRange(img_HSV, lowerBound, upperBound, skin);

    // apply the mask to the originalal image
    // define a kernel
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));

    // apply morphological operations
    morphologyEx(skin, skin, MORPH_OPEN, kernel);
    morphologyEx(skin, skin, MORPH_CLOSE, kernel);
    Mat final;
    bitwise_and(original, original, final, skin);

    // return the result (skin detection)
    return final;

}



Mat PeopleSegmentation::peopleSegm(Mat original) {

    //define the support images
    Mat mask = Mat::zeros(original.cols, original.rows, CV_8UC1);
    Mat bgdModel;
    Mat fgdModel;

    int x = original.cols;
    int y = original.rows;

    // different rect size for grabcut algorithm are taken into account, in order to recognize the subject in the bounding box
    
    
    // case 4:3
    if (y > x + (x / 3)) {
        //img too small
        if (y < 200) {
            Rect rect = Rect(x / 8, y / 20, x - (x / 8), y - (y / 18));
            grabCut(original, mask, rect, bgdModel, fgdModel, 10, GC_INIT_WITH_RECT);
        }
        else {
            Rect rect = Rect(x / 8, y / 20, x - (x / 5), y - (y / 9));
            grabCut(original, mask, rect, bgdModel, fgdModel, 10, GC_INIT_WITH_RECT);
        }
        //Rect rect = Rect(x/8, y/20,x-(x/5), y-(y/20));

    }
    // case 16:9
    else if (y + (y / 3) < x) {
        Rect rect = Rect(x / 4, y / 6, x - (x / 4), y - (y / 6));
        grabCut(original, mask, rect, bgdModel, fgdModel, 10, GC_INIT_WITH_RECT);
    }
    // case 1:1
    else {
        Rect rect = Rect(x / 6, y / 9, x - (x / 4) - (x / 7), y - (y / 4));
        grabCut(original, mask, rect, bgdModel, fgdModel, 10, GC_INIT_WITH_RECT);
        cout << "i";
    }
    // Rect rect=Rect(5, 5, src.cols - 6, src.rows - 6);
    // grabCut(src, mask, rect, bgdModel, fgdModel, 10, GC_INIT_WITH_RECT);
    // support mask in order to take only the pixels marked are foreground by grabcut
    Mat mask2 = (mask == 1) + (mask == 3);

    Mat result(original.cols, original.rows, CV_8UC3);
    original.copyTo(result, mask2);

    return result;
}



/*Function to merge results image obtained from skin-detection and people-segmentation into an unique image*/
Mat PeopleSegmentation::merger(Mat original) {
    //temp vars
    Mat temp_1 = Mat::zeros(original.cols, original.rows, CV_8UC3);
    Mat temp_2 = Mat::zeros(original.cols, original.rows, CV_8UC3);

    //call function
    temp_1 = skinDetect(src);
    temp_2 = peopleSegm(src);

    
    //merge the two images
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            temp_2.at<Vec3b>(i, j)[0] += temp_1.at<Vec3b>(i, j)[0];
            temp_2.at<Vec3b>(i, j)[1] += temp_1.at<Vec3b>(i, j)[1];
            temp_2.at<Vec3b>(i, j)[2] += temp_1.at<Vec3b>(i, j)[2];
        }
    }
        
     //clean the image after processing
    for (int i = 0; i < original.rows; i++) {
        for (int j = 0; j < original.cols; j++) {
            if (!(temp_2.at<Vec3b>(i, j)[0] == 0 && temp_2.at<Vec3b>(i, j)[1] == 0 && temp_2.at<Vec3b>(i, j)[2] == 0)) {
                temp_2.at<Vec3b>(i, j)[0] = original.at<Vec3b>(i, j)[0];
                temp_2.at<Vec3b>(i, j)[1] = original.at<Vec3b>(i, j)[1];
                temp_2.at<Vec3b>(i, j)[2] = original.at<Vec3b>(i, j)[2];
            }
            
        }
    }
    return temp_2;
}
