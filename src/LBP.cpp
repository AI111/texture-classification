//
// Created by sasha on 26.09.15.
//

#include "LBP.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;

template <typename _Tp>
void lbp::OLBP_(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    for(int i=1;i<src.rows-1;i++) {
        for(int j=1;j<src.cols-1;j++) {
            _Tp center = src.at<_Tp>(i,j);
            unsigned char code = 0;
            code |= (src.at<_Tp>(i-1,j-1) > center) << 7;
            code |= (src.at<_Tp>(i-1,j) > center) << 6;
            code |= (src.at<_Tp>(i-1,j+1) > center) << 5;
            code |= (src.at<_Tp>(i,j+1) > center) << 4;
            code |= (src.at<_Tp>(i+1,j+1) > center) << 3;
            code |= (src.at<_Tp>(i+1,j) > center) << 2;
            code |= (src.at<_Tp>(i+1,j-1) > center) << 1;
            code |= (src.at<_Tp>(i,j-1) > center) << 0;
            dst.at<unsigned char>(i-1,j-1) = code;
        }
    }
}

void lbp::OLBP(const Mat& src, Mat& dst) {
    switch(src.type()) {
        case CV_8SC1: OLBP_<char>(src, dst); break;
        case CV_8UC1: OLBP_<unsigned char>(src, dst); break;
        case CV_16SC1: OLBP_<short>(src, dst); break;
        case CV_16UC1: OLBP_<unsigned short>(src, dst); break;
        case CV_32SC1: OLBP_<int>(src, dst); break;
        case CV_32FC1: OLBP_<float>(src, dst); break;
        case CV_64FC1: OLBP_<double>(src, dst); break;
    }
}


Mat lbp::OLBP(const Mat& src) { Mat dst; OLBP(src, dst); return dst; }


void ::lbp::drawHist(Mat &src,const String& name) {

    Mat hist;
    int histSize = 64;
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    normalize(src, src, 0, 255, NORM_MINMAX, CV_8UC1);
    calcHist( &src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    std::cout<<"HIST ARR\n"<<hist<<"\nHIST ARR SIZE = "<<hist.rows<<" x "<<hist.cols<<std::endl;
    int hist_w = 800; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
              Scalar( 255, 0, 0), 2, 8, 0  );

    }
    imshow(name, histImage );
    histImage.release();
}
