//
// Created by sasha on 26.09.15.
//

#include "LBP.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

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
//template <typename _Tp>
void::lbp::dftShift(Mat& I,Mat& dst){

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    //imshow("DFT result",complexI);
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    dst = planes[0];

    dst += Scalar::all(1);                    // switch to logarithmic scale
    log(dst, dst);

    // crop the spectrum, if it has an odd number of rows or columns
    dst = dst(Rect(0, 0, dst.cols & -2, dst.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = dst.cols/2;
    int cy = dst.rows/2;

    Mat q0(dst, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(dst, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(dst, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(dst, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(dst, dst, 0, 255, NORM_MINMAX); // Transform the matrix with float values into a
}

float entropy(Mat seq, Size size, int index)
{
    int cnt = 0;
    float entr = 0;
    float total_size = size.height * size.width; //total size of all symbols in an image

    for(int i=0;i<index;i++)
    {
        float sym_occur = seq.at<float>(0, i); //the number of times a sybmol has occured
        if(sym_occur>0) //log of zero goes to infinity
        {
            cnt++;
            entr += (sym_occur/total_size)*(log2(total_size/sym_occur));
        }
    }
    //cout<<"cnt: "<<entr<<endl;
    return entr;

}

// myEntropy calculates relative occurrence of different symbols within given input sequence using histogram
Mat myEntropy(Mat seq, int histSize)
{

    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat hist;

    /// Compute the histograms:
    calcHist( &seq, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

    return hist;
}
float entropy(Mat &src){

    if (src.channels()==3) cvtColor(src,src,CV_BGR2GRAY);
    /// Establish the number of bins
    int histSize = 256;
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    /// Compute the histograms:
    Mat hist;
    calcHist( &src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    hist /= src.total();

    Mat logP;
    cv::log(hist,logP);

    return  (float)-1*sum(hist.mul(logP)).val[0];

}
Mat lbp::calcEntrop1(Mat &src,int size){
    Mat ans = Mat(size,size,CV_32FC1);
    int window_size=round(src.cols/size);;
    int end_size=src.cols%window_size;
//    cout<<ans.size()<<endl;
//    cout<<ans<<endl;
    for (int i = 0; i <size ; i++) {
        for (int j = 0; j <size ; j++) {
            //int w=(i==(size-1)||j==(size-1))?end_size:window_size;
            int w = (j==size-1?end_size:window_size);
            int h = (i==size-1?end_size:window_size);
            Mat tmp=Mat(src,Rect(j*window_size,i*window_size,w,h));
            Mat newTmp1;
            tmp.convertTo(newTmp1,CV_32F);
            pow(newTmp1,2,newTmp1);
            //cout<<"Pow 2"<<endl<<newTmp1<<endl;
            double d = sum(newTmp1)[0]/(tmp.cols*tmp.rows);
            //cout<<d<<endl;
            ans.at<float>(i,j)=(float)d;//(float)d;
        }
    }
    return ans;
}
Mat lbp::calcEntrop2(Mat &src,int size){
    Mat ans = Mat(size,size,CV_32FC1);
    int window_size=round(src.cols/size);
    int end_size=src.cols%window_size;
//    cout<<ans.size()<<endl;
//    cout<<ans<<endl;
    for (int i = 0; i <size ; i++) {
        for (int j = 0; j <size ; j++) {
            //int w=(i==(size-1)||j==(size-1))?end_size:window_size;
            int w = (j==size-1?end_size:window_size);
            int h = (i==size-1?end_size:window_size);
            Mat tmp=Mat(src,Rect(j*window_size,i*window_size,w,h));
            Mat hist = myEntropy(tmp, 256);

            // cout<<hist<<endl;

//            Mat newTmp1;
//            tmp.convertTo(newTmp1,CV_32F);
//            pow(newTmp1,2,newTmp1);
//            //cout<<"Pow 2"<<endl<<newTmp1<<endl;
//            double d = sum(newTmp1)[0]/(tmp.cols*tmp.rows);

            ans.at<float>(i,j)=entropy(hist,tmp.size(), 256);//(float)d;
        }
    }
    return ans;
}
//void::lbp::dftShift(cuda::GpuMat& I,cuda::GpuMat& dst){
//
//    Mat padded;                            //expand input image to optimal size
//    int m = getOptimalDFTSize( I.rows );
//    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
//    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
//    cuda::GpuMat i,j;
//    i.upload(Mat_<float>(padded));
//    j.upload(Mat::zeros(padded.size(), CV_32F));
//    cuda::GpuMat planes[] = {i,j};
//    cuda::GpuMat complexI;
//    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//    cuda::dft(complexI, complexI,complexI.size());            // this way the result may fit in the source matrix
//    //imshow("DFT result",complexI);
//    // compute the magnitude and switch to logarithmic scale
//    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
//    dst = planes[0];
//
//    dst += Scalar::all(1);                    // switch to logarithmic scale
//    log(dst, dst);
//
//    // crop the spectrum, if it has an odd number of rows or columns
//    dst = dst(Rect(0, 0, dst.cols & -2, dst.rows & -2));
//
//    // rearrange the quadrants of Fourier image  so that the origin is at the image center
//    int cx = dst.cols/2;
//    int cy = dst.rows/2;
//
//    cuda::GpuMat q0(dst, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
//    cuda::GpuMat q1(dst, Rect(cx, 0, cx, cy));  // Top-Right
//    cuda::GpuMat q2(dst, Rect(0, cy, cx, cy));  // Bottom-Left
//    cuda::GpuMat q3(dst, Rect(cx, cy, cx, cy)); // Bottom-Right
//
//    cuda::GpuMat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
//    q0.copyTo(tmp);
//    q3.copyTo(q0);
//    tmp.copyTo(q3);
//
//    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
//    q2.copyTo(q1);
//    tmp.copyTo(q2);
//
//    normalize(dst, dst, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
//
//}
//
//void::lbp::dftShift(Mat& src,Mat& dst) {
//    switch(src.type()) {
//        case CV_8SC1: dftShift_<char>( src, dst); break;
//        case CV_8UC1: dftShift_<unsigned char>( src, dst); break;
//        case CV_16SC1: dftShift_<short>( src, dst); break;
//        case CV_16UC1: dftShift_<unsigned short>( src, dst); break;
//        case CV_32SC1: dftShift_<int>( src, dst); break;
//        case CV_32FC1: dftShift_<float>( src, dst); break;
//        case CV_64FC1: dftShift_<double>( src, dst); break;
//    }
//}
