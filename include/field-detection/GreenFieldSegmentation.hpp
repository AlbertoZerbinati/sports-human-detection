// Marco Calì
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <vector>

struct LocalMinimum {
    double value;
    double x;
};

struct PixelInfo {
    cv::Vec3b pixel;  // The pixel values (RGB)
    int group;        // The group the pixel belongs to (0, 1, or 2)
};

class GreenFieldSegmentation {
   public:
    cv::Mat detectGreenField(const cv::Mat &I);

   private:
    void preprocessing(const cv::Mat &src, cv::Mat &dst, const double alpha_e);
    void chromaticityAnalysis(const cv::Mat &src, cv::Mat &dst);
    void trainGMM(cv::Ptr<cv::ml::EM> &gmm, const cv::Mat &samples,
                  const int N_G, cv::Mat &logLikelihoods, cv::Mat &labels,
                  cv::Mat &probs);
    double computeGaussian(double x, double mean, double variance);
    cv::Mat computePDF(const int N_G, const int numPoints, const cv::Mat &means,
                       const std::vector<cv::Mat> &covs, const cv::Mat &weights,
                       const cv::Mat &g);
    int findFirstMaximumAfterThreshold(const cv::Mat &pdf, const cv::Mat &g,
                                       double threshold);
    std::vector<cv::Vec3b> computeMeanColors(std::vector<PixelInfo> pixels,
                                             std::vector<int> counts);
    double findFirstMinimumAfterIndex(const cv::Mat &pdf, const cv::Mat &g,
                                      int index);
    cv::Mat createEnvelope(const std::vector<cv::Mat> &covs,
                           const cv::Mat &means, const cv::Mat &weights,
                           int numPoints, int N_G);
    std::vector<LocalMinimum> findLocalMinima(const cv::Mat &envelope,
                                              const cv::Mat &g,
                                              const double T_G);
    float computeChromaticDistortion(cv::Vec3b v, cv::Vec3b u);
    cv::Mat chromaticDistortionMatrix(const cv::Mat &imageGCA,
                                      const cv::Mat &imageOpen,
                                      const double T_G,
                                      const std::vector<LocalMinimum> &minima);
    cv::Mat computeMask1(const double threshold, const cv::Mat &chromaticity);
    cv::Mat computeMask2(const double T_C, const cv::Mat &mask_M_PF_hat,
                         const cv::Mat &cd_matrix);
    float dotProduct(cv::Vec3b v, cv::Vec3b u);
};