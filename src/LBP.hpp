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

    template <typename _Tp>
    void OLBP_(const cv::Mat& src, cv::Mat& dst);

    void drawHist(Mat& src,const String& name);

    template <typename _Tp>
    void drawHist_(Mat& src,const String& name);

    void OLBP(const Mat& src, Mat& dst);

    Mat OLBP(const Mat& src);

    void showHistogram(Mat& img);


}
#endif //UNTITLED_LBP_H
