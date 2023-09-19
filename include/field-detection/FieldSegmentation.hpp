// Marco Calì

#include "field-detection/GreenFieldSegmentation.hpp"

Mat GreenFieldsSegmentation(const Mat &I);
Mat GenericFieldSegmentation(Mat &image, int from_row, int from_column,
                             int row_width, int column_width,
                             double mean_factor = 1, double std_factor = 1);