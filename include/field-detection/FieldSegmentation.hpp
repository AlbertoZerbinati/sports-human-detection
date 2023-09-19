// @author: Marco Calì

#include "field-detection/GreenFieldSegmentation.hpp"

Mat GreenFieldsSegmentation(const Mat &I);
Mat GenericFieldSegmentation(const Mat &image,
                             const Vec3b estimated_field_color,
                             double mean_factor = 1, double std_factor = 1);
Mat FieldSegmentation(const Mat &src, const Vec3b estimated_field_color);
