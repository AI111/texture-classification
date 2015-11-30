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

template <typename _Tp>
void ::lbp::drawHist_(Mat &h,const String& name) {
    Mat hist = h.clone();
    int histSize = hist.rows;
   // std::cout<<"HIST ARR\n"<<hist<<"\nHIST ARR SIZE = "<<hist.rows<<" x "<<hist.cols<<std::endl;
    int hist_w = 800; int hist_h = 300;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 0; i < histSize; i++ )
    {

//        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
//              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
//              Scalar( 255, 255, 255), 2, 8, 0  );
        rectangle( histImage, Point( bin_w*(i), hist_h) ,
              Point( bin_w*(i+1), hist_h - cvRound(hist.at<_Tp>(i)) ),
              Scalar( 0, 0, 0), 1, 8, 0);

    }
    namedWindow(name, CV_WINDOW_AUTOSIZE );
    imshow(name, histImage );
    histImage.release();
}

void::lbp::drawHist(Mat &h,const String& name) {
    switch(h.type()) {
        case CV_8SC1: drawHist_<char>(h, name); break;
        case CV_8UC1: drawHist_<unsigned char>(h, name); break;
        case CV_16SC1: drawHist_<short>(h, name); break;
        case CV_16UC1: drawHist_<unsigned short>(h, name); break;
        case CV_32SC1: drawHist_<int>(h, name); break;
        case CV_32FC1: drawHist_<float>(h, name); break;
        case CV_64FC1: drawHist_<double>(h, name); break;
    }
}

void::lbp:: showHistogram(Mat& img)
{
    int bins = 256;             // number of bins
    int nc = img.channels();    // number of channels

    vector<Mat> hist(nc);       // histogram arrays

    // Initalize histogram arrays
    for (int i = 0; i < hist.size(); i++)
        hist[i] = Mat::zeros(1, bins, CV_32SC1);

    // Calculate the histogram of the image
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            for (int k = 0; k < nc; k++)
            {
                uchar val = nc == 1 ? img.at<uchar>(i,j) : img.at<Vec3b>(i,j)[k];
                hist[k].at<int>(val) += 1;
            }
        }
    }

    // For each histogram arrays, obtain the maximum (peak) value
    // Needed to normalize the display later
    int hmax[3] = {0,0,0};
    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < bins-1; j++)
            hmax[i] = hist[i].at<int>(j) > hmax[i] ? hist[i].at<int>(j) : hmax[i];
    }

    const char* wname[3] = { "blue", "green", "red" };
    Scalar colors[3] = { Scalar(255,0,0), Scalar(0,255,0), Scalar(0,0,255) };

    vector<Mat> canvas(nc);

    // Display each histogram in a canvas
    for (int i = 0; i < nc; i++)
    {
        canvas[i] = Mat::ones(125, bins, CV_8UC3);

        for (int j = 0, rows = canvas[i].rows; j < bins-1; j++)
        {
            line(
                    canvas[i],
                    Point(j, rows),
                    Point(j, rows - (hist[i].at<int>(j) * rows/hmax[i])),
                    nc == 1 ? Scalar(200,200,200) : colors[i],
                    1, 8, 0
            );
        }

        imshow(nc == 1 ? "value" : wname[i], canvas[i]);
    }
}