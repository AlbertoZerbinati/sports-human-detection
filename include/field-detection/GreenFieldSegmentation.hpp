// Marco Calì

#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <vector>

using namespace std;  // TODO: remove this
using namespace cv;   // TODO: remove this

// TODO: make a class...

struct LocalMinimum {
    double value;
    double x;
};

struct PixelInfo {
    Vec3b pixel;  // The pixel values (RGB)
    int group;    // The group the pixel belongs to (0, 1, or 2)
};

String get_image_path(String filename);
void preprocessing(const Mat &src, Mat &dst, const double alpha_e);
void green_chromaticity_analysis(const cv::Mat &src, cv::Mat &dst);
Mat compute_histogram(const Mat &src);
void save_model(const Ptr<ml::EM> &gmm, String filename);
void train_gmm(Ptr<ml::EM> &gmm, const Mat &samples, const int N_G,
               Mat &log_likelihoods, Mat &labels, Mat &probs);
Ptr<ml::EM> gmm_load_trained(const Mat &samples, const int N_G,
                             Mat &log_likelihoods, Mat &labels, Mat &probs);
double compute_gaussian(double x, double mean, double variance);
void print_gaussians_data(const int N_G, const Mat &means,
                          const vector<Mat> &covariances, const Mat &weights);
Mat compute_pdf(const int N_G, const int num_points, const Mat &means,
                const vector<Mat> &covs, const Mat &weights, const Mat &g);
void show_pdf(const Mat &pdf, const int num_points, const Mat &g,
              const double threshold);
Mat find_M_PF_hat(const double threshold, const Mat &chromaticity);
Mat apply_mask(const Mat &mask, const Mat &image);

void visualize_gaussians(const vector<Mat> &covs, const Mat &means,
                         const Mat &weights, int num_points, double T_G);
int findFirstMaximumAfterThreshold(const Mat &pdf, const Mat &g,
                                   double threshold);
double findFirstMinimumAfterIndex(const Mat &pdf, const Mat &g, int index);
Mat create_envelope(const vector<Mat> &covs, const Mat &means,
                    const Mat &weights, int num_points, int N_G);

vector<LocalMinimum> findLocalMinima(const Mat &envelope, const Mat &g,
                                     const double T_G);

float compute_chromatic_distortion(Vec3b v, Vec3b u);

Mat find_M_PF_tilde(const double T_C, const Mat &mask_M_PF_hat,
                    const Mat &cd_matrix);
Mat chromatic_distorsion_matrix(const Mat &I_gca, const Mat &I_open,
                                const double T_G,
                                const vector<LocalMinimum> &minima);
void plotEnvelope(const Mat &x_values, const Mat &envelope);
