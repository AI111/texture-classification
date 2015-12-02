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
    Mat OLBP(const Mat& src);
    void OLBP(const Mat& src, Mat& dst);
    void drawHist(Mat& src,const String& name);
    template <typename _Tp>
    void drawHist_(Mat& src,const String& name);
    void showHistogram(Mat& img);
    //template <typename _Tp>
    //void dftShift_(Mat& src,Mat& dst);
    void dftShift(Mat& src,Mat& dst);
//    void dftShift(cuda::GpuMat& src,cuda::GpuMat& dst);
    Mat calcEntrop1(Mat &src,int size);
    Mat calcEntrop2(Mat &src,int size);



}
#endif //UNTITLED_LBP_H
