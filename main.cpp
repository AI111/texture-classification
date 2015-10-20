#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "src/LBP.hpp"
using namespace cv;
using namespace std;

int main() {
    int histSize = 256;
    string path[] ={"/home/sasha/ClionProjects/texture classification/res/AJdocBFYGTw.jpg",
                    "/home/sasha/ClionProjects/texture classification/res/dSPT4uKT2eE.jpg"
    };
    int imgSize=2;
    Mat image[imgSize];
    Mat dst[imgSize];
    Mat lbp[imgSize];
    for (int i = 0; i < imgSize; i++) {
        image[i] = imread( path[i], 1 );
        if ( !image[i].data )
        {
            printf("No image data \n");
            return -1;
        }
        cvtColor(image[i], dst[i], CV_BGR2GRAY);
        lbp::OLBP(dst[i], lbp[i]);
        normalize(lbp[i], lbp[i], 0, 255, NORM_MINMAX, CV_8UC1);

        lbp::drawHist(lbp[i],"img "+i);
        imshow("original "+i, image[i]);
        imshow("lbp "+i, lbp[i]);
    }




//    calcHist( &lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
//
//    int hist_w = 512; int hist_h = 400;
//    int bin_w = cvRound( (double) hist_w/histSize );
//
//    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );
//
//    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
//    for( int i = 1; i < histSize; i++ )
//    {
//        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
//              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
//              Scalar( 255, 0, 0), 2, 8, 0  );
//
//    }


//    imshow("calcHist Demo", histImage );

    waitKey(0);
    return 0;
}