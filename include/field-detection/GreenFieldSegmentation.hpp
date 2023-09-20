// Marco Calì

#include <chrono>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <vector>

// TODO: make a class...

struct LocalMinimum
{
    double value;
    double x;
};

struct PixelInfo
{
    cv::Vec3b pixel; // The pixel values (RGB)
    int group;       // The group the pixel belongs to (0, 1, or 2)
};

void preprocessing(const cv::Mat &src, cv::Mat &dst, const double alpha_e);
void green_chromaticity_analysis(const cv::Mat &src, cv::Mat &dst);
void train_gmm(cv::Ptr<cv::ml::EM> &gmm, const cv::Mat &samples, const int N_G,
               cv::Mat &log_likelihoods, cv::Mat &labels, cv::Mat &probs);
cv::Ptr<cv::ml::EM> gmm_load_trained(const cv::Mat &samples, const int N_G,
                                     cv::Mat &log_likelihoods, cv::Mat &labels, cv::Mat &probs);
double compute_gaussian(double x, double mean, double variance);
cv::Mat compute_pdf(const int N_G, const int num_points, const cv::Mat &means,
                    const std::vector<cv::Mat> &covs, const cv::Mat &weights, const cv::Mat &g);
cv::Mat find_M_PF_hat(const double threshold, const cv::Mat &chromaticity);
cv::Mat apply_mask(const cv::Mat &mask, const cv::Mat &image);
int findFirstMaximumAfterThreshold(const cv::Mat &pdf, const cv::Mat &g,
                                   double threshold);
double findFirstMinimumAfterIndex(const cv::Mat &pdf, const cv::Mat &g, int index);
cv::Mat create_envelope(const std::vector<cv::Mat> &covs, const cv::Mat &means,
                        const cv::Mat &weights, int num_points, int N_G);

std::vector<LocalMinimum> findLocalMinima(const cv::Mat &envelope, const cv::Mat &g,
                                          const double T_G);
float compute_chromatic_distortion(cv::Vec3b v, cv::Vec3b u);
cv::Mat find_M_PF_tilde(const double T_C, const cv::Mat &mask_M_PF_hat,
                        const cv::Mat &cd_matrix);
cv::Mat chromatic_distorsion_matrix(const cv::Mat &I_gca, const cv::Mat &I_open,
                                    const double T_G,
                                    const std::vector<LocalMinimum> &minima);
