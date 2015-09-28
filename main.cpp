#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "src/LBP.hpp"
using namespace cv;
using namespace std;

int main() {
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat image,hist;
    Mat dst; // image after preprocessing
    Mat lbp; // lbp image
    image = imread( "/home/sasha/Downloads/H2difUwqaMo.jpg", 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }


    namedWindow("Display Image", WINDOW_AUTOSIZE );
    cvtColor(image, dst, CV_BGR2GRAY);
    lbp::OLBP(dst, lbp);
    normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);

    calcHist( &lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
              Scalar( 255, 255, 255), 2, 8, 0  );

    }

    imshow("original", image);
    imshow("lbp", lbp);

    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );

    waitKey(0);
    return 0;
}