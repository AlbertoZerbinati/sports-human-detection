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
    /**
     * Detect green fields in an input image.
     *
     * This function performs green field segmentation on the input image
     * and returns a mask highlighting the green fields.
     *
     * @param I The input image.
     * @return A binary mask where green fields are highlighted.
     */
    cv::Mat detectGreenField(const cv::Mat &I);

   private:
    /**
     * Preprocess the input image by performing an opening morphological
     * operator based on the image size.
     *
     * @param src The input image.
     * @param dst The preprocessed image.
     * @param alpha_e A parameter to manage the structuring element size.
     */
    void preprocessing(const cv::Mat &src, cv::Mat &dst, const double alpha_e);

    /**
     * Analyze (green) chromaticity of the input image.
     *
     * @param src The input image.
     * @param dst The chromaticity matrix in the interval 0-255.
     */
    void chromaticityAnalysis(const cv::Mat &src, cv::Mat &dst);

    /**
     * Train a Gaussian Mixture Model (GMM) on the input data.
     *
     * This function trains a Gaussian Mixture Model (GMM) on the provided input
     * samples. The goal is that of estimating the gaussian distributions that
     * generate the green chromaticity profile.It initializes the GMM model
     * using the input `gmm` parameter and populates the model with information
     * obtained from the samples.
     *
     * @param gmm A pointer to the GMM model that will be trained (output).
     * @param samples The input data samples to be used for training.
     * @param N_G The number of Gaussian components in the model.
     * @param logLikelihoods Log likelihood values for each sample (output).
     * @param labels Cluster labels for each sample (output).
     * @param probs Probabilities for each sample to belong to each cluster
     * (output).
     */
    void trainGMM(cv::Ptr<cv::ml::EM> &gmm, const cv::Mat &samples,
                  const int N_G, cv::Mat &logLikelihoods, cv::Mat &labels,
                  cv::Mat &probs);

    /**
     * Compute the Gaussian probability density for a given value.
     *
     * This function calculates the Gaussian probability density for a given
     * value `x` using the specified mean and variance parameters.
     *
     * @param x The value for which the Gaussian probability density is
     * calculated.
     * @param mean The mean (average) of the Gaussian distribution.
     * @param variance The variance (spread) of the Gaussian distribution.
     * @return The Gaussian probability density for the input `x`.
     */
    double computeGaussian(double x, double mean, double variance);

    /**
     * Compute the Probability Density Function (PDF) of a Gaussian
     * Mixture Model.
     *
     * This function calculates the PDF of a Gaussian Mixture Model (GMM) for a
     * collection of input values specified in the parameter `g`. The PDF is
     * computed based on the GMM's means, covariances, and mixture weights.
     *
     * @param N_G The number of Gaussian components in the GMM.
     * @param numPoints The number of input points in the linear space `g`.
     * @param means A matrix of GMM means, one mean vector per component.
     * @param covs A vector of covariance matrices, one covariance matrix per
     * component.
     * @param weights The mixture weights of the GMM components.
     * @param g A collection of input values for which the PDF is computed. It's
     * similar to a linspace of input values from 0 to 1.
     *  @return A row vector containing the computed PDF values corresponding
     * to the input values in `g`. Each position in the output corresponds to
     * the PDF value for the respective input value.
     */
    cv::Mat computePDF(const int N_G, const int numPoints, const cv::Mat &means,
                       const std::vector<cv::Mat> &covs, const cv::Mat &weights,
                       const cv::Mat &g);

    /**
     * Find the first maximum value in the Probability Density Function
     * (PDF) after a specified threshold on the input values.
     *
     * This function searches for the first local maximum value in the PDF
     * (Probability Density Function) that occurs after a specified threshold on
     * the input values `g`. It helps identify significant peaks in the PDF that
     * occur beyond what in the paper is indicated by T_G.
     *
     * @param pdf The PDF values to search for maxima.
     * @param g The corresponding input values (linspace).
     * @param threshold The threshold on the input values, beyond which maxima
     * are considered.
     * @return The index of the first maximum in the PDF after the specified
     * threshold.
     */
    int findFirstMaximumAfterThreshold(const cv::Mat &pdf, const cv::Mat &g,
                                       const double threshold);

    /**
     * Compute the mean colors for each range between minima as specified
     * by the paper.
     *
     * This function calculates the mean colors from a collection of pixels and
     * their associated counts. It is useful for determining the average color
     * values of a group of pixels, where each pixel is represented as a
     * `PixelInfo` structure and its count reflects the number of occurrences.
     *
     * @param pixels A vector of `PixelInfo` structures, each containing pixel
     * values and group information needed for computing the group mean.
     * @param counts A vector of counts, where each count corresponds to the
     * number of occurrences for each range between minima.
     * @return A vector of `cv::Vec3b` representing the computed mean colors.
     */
    std::vector<cv::Vec3b> computeMeanColors(std::vector<PixelInfo> pixels,
                                             std::vector<int> counts);

    /**
     * Find the first minimum value in the Probability Density Function
     * (PDF) after a specified index.
     *
     * This function searches for the first local minimum value in the PDF
     * (Probability Density Function) that occurs after the specified index in
     * the `g` parameter. It helps identify significant valleys or dips in the
     * PDF that appear after a particular position.
     *
     * @param pdf The PDF values to search for minima.
     * @param g The corresponding input values (linspace).
     * @param index The index in the `g` parameter after which to search for a
     * minimum.
     * @return The index of the first minimum in the PDF after the specified
     * index.
     */
    double findFirstMinimumAfterIndex(const cv::Mat &pdf, const cv::Mat &g,
                                      int index);

    /**
     * Compute an envelope to visualize the Gaussian Mixture Model (GMM)
     * distribution.
     *
     * This function calculates an envelope that illustrates the Gaussian
     * Mixture Model (GMM) distribution, utilizing provided covariance matrices,
     * means, weights, and other parameters. The generated envelope offers a
     * comprehensive overview of the GMM distribution across a defined range of
     * input values. The envelope is constructed by selecting the maximum value
     * attained by the Gaussian components at each point within the specified
     * range.
     *
     * @param covs A collection of covariance matrices, one for each Gaussian
     * component within the GMM.
     * @param means A matrix containing the mean vectors for each GMM component.
     * @param weights The mixture weights assigned to each GMM component.
     * @param numPoints The number of sampling points used to construct the
     * envelope.
     * @param N_G The total number of Gaussian components within the GMM.
     * @return A matrix representing the envelope that visualizes the GMM
     * distribution.
     */
    cv::Mat createEnvelope(const std::vector<cv::Mat> &covs,
                           const cv::Mat &means, const cv::Mat &weights,
                           int numPoints, int N_G);

    /**
     * Find local minima of the envelope, starting from the threshold in
     * the x axis.
     *
     * This function identifies local minima within an envelope signal, such as
     * the one computed in `createEnvelope`. The analysis is conducted with
     * respect to the corresponding input values in the `g` parameter, starting
     * from a previously computed threshold (T_G in the paper).
     *
     * @param envelope The envelope of the Gaussians identifying the green
     * chromaticity distribution.
     * @param g The corresponding input values (linspace) associated with the
     * envelope signal.
     * @param T_G The threshold used to determine significant minima.
     * @return A vector of `LocalMinimum` structures, each containing the value
     * and corresponding input position of a local minimum.
     */
    std::vector<LocalMinimum> findLocalMinima(const cv::Mat &envelope,
                                              const cv::Mat &g,
                                              const double threshold);

    /**
     * Calculate the chromatic distortion matrix based on image data.
     *
     * This function computes a chromatic distortion matrix by analyzing input
     * images `imageGCA` and `imageOpen`. The distortion analysis considers a
     * threshold `T_G` and a set of local minima in the envelope signal,
     * provided as `minima`. First all pixels are grouped in the right clusters,
     * then distorsion is computed with respect to the mean color of the
     * cluster.
     *
     * @param imageGCA The image representing chromaticity after preprocessing.
     * @param imageOpen The image after applying an opening operation.
     * @param T_G The threshold used for distortion analysis.
     * @param minima A vector of `LocalMinimum` structures, containing
     * significant minima in the envelope signal.
     * @return A matrix representing the computed chromatic distortion matrix.
     */
    cv::Mat chromaticDistortionMatrix(const cv::Mat &imageGCA,
                                      const cv::Mat &imageOpen,
                                      const double threshold,
                                      const std::vector<LocalMinimum> &minima);

    /**
     * Generate a binary mask based on chromaticity values.
     *
     * This function creates a binary mask by thresholding the input
     * chromaticity values provided in the `chromaticity` matrix. Pixels with
     * chromaticity values exceeding the specified `threshold` are marked as
     * field (255), while others are set to (0).
     *
     * @param threshold The threshold value for chromaticity.
     * @param chromaticity A matrix containing chromaticity values.
     * @return A binary mask highlighting regions with chromaticity values above
     * the threshold.
     */
    cv::Mat computeMask1(const double threshold, const cv::Mat &chromaticity);

    /**
     * Generate a refined binary mask using chromatic distortion
     * analysis.
     *
     * This function refines a binary mask, represented by `mask1`, using
     * chromatic distortion analysis based on the specified threshold `T_C` and
     * chromatic distortion matrix `cd_matrix`. It enhances the initial mask
     * `mask1` by considering chromatic distortion information.
     *
     * @param T_C The threshold value for chromatic distortion analysis.
     * @param mask1 The initial binary mask.
     * @param cd_matrix The chromatic distortion matrix.
     * @return A refined binary mask highlighting regions with significant
     * chromatic distortion.
     */

    cv::Mat computeMask2(const double T_C, const cv::Mat &mask1,
                         const cv::Mat &cd_matrix);

    /**
     * Calculate the chromatic distortion between two RGB color vectors.
     *
     * This static function computes the chromatic distortion between two RGB
     * color vectors, represented as `v` and `u`. It measures the dissimilarity
     * between the color vectors and provides a quantitative assessment of
     * chromatic distortion.
     *
     * @param v The first RGB color vector.
     * @param u The second RGB color vector.
     * @return The computed chromatic distortion between the input color
     * vectors.
     */
    static float computeChromaticDistortion(cv::Vec3b v, cv::Vec3b u);

    /**
     * Calculate the dot product between two RGB color vectors.
     *
     * This static function computes the dot product between two RGB color
     * vectors, represented as `v` and `u`. It provides a measure of similarity
     * or alignment between the color vectors.
     *
     * @param v The first RGB color vector.
     * @param u The second RGB color vector.
     * @return The computed dot product between the input color vectors.
     */
    static float dotProduct(cv::Vec3b v, cv::Vec3b u);
};
