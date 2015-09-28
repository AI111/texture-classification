//
// Created by sasha on 26.09.15.
//

#ifndef UNTITLED_LBP_H
#define UNTITLED_LBP_H
#include <opencv2/opencv.hpp>
#include <limits>

using namespace cv;
using namespace std;

namespace lbp {

// templated functions
    template <typename _Tp>
    void OLBP_(const cv::Mat& src, cv::Mat& dst);

    template <typename _Tp>
    void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

    template <typename _Tp>
    void VARLBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

// wrapper functions
    void OLBP(const Mat& src, Mat& dst);
    void ELBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);
    void VARLBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);

// Mat return type functions
    Mat OLBP(const Mat& src);
    Mat ELBP(const Mat& src, int radius = 1, int neighbors = 8);
    Mat VARLBP(const Mat& src, int radius = 1, int neighbors = 8);

}
#endif //UNTITLED_LBP_H
