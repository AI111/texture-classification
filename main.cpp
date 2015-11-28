#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "src/histogram.hpp"
#include "src/LBP.hpp"
//#define DEBUG
using namespace cv;
using namespace std;


int histSize = 256;

vector<vector<string>> data({
{"/home/sasha/ClionProjects/texture-classification/res/good/4aftENndm_4.jpg",
 "/home/sasha/ClionProjects/texture-classification/res/good/AJdocBFYGTw.jpg",
 "/home/sasha/ClionProjects/texture-classification/res/good/H2difUwqaMo.jpg",
 "/home/sasha/ClionProjects/texture-classification/res/good/ntaSw0ri3x0.jpg"
},
{"/home/sasha/ClionProjects/texture-classification/res/bad/dSPT4uKT2eE.jpg",
"/home/sasha/ClionProjects/texture-classification/res/bad/FJtA4nR9f5Q.jpg",
"/home/sasha/ClionProjects/texture-classification/res/bad/Jfebkg-mWEg.jpg",
"/home/sasha/ClionProjects/texture-classification/res/bad/xOPEc8D3xpM.jpg"}});

vector<vector<Mat>>imeges(data.size());
vector<Mat>histograms(data.size());

int loadImg(vector <Mat> &img, vector <string> &path) {
    for (auto &pwd : path) {
        Mat im = imread(pwd, 1);
        if (!im.data) {
            printf("No image data \n");
            return -1;
        }
        img.push_back(im);
    }
    return 0;
}
void convertIMG(Mat &img) {
#ifdef DEBUG
    imshow("lbp0",img);
#endif
    cvtColor(img, img, CV_BGR2GRAY);
#ifdef DEBUG
    imshow("lbp1",img);
#endif
    normalize(img,img, 0, 255, NORM_MINMAX, CV_8UC1);
#ifdef DEBUG
    imshow("lbp2",img);
#endif
    Mat dst;
    lbp::OLBP(img,dst);
    img.release();
    img=dst;
    cout<<"CHANELS "<<dst.channels()<<endl;
#ifdef DEBUG
    imshow("lbp3",dst);
#endif
}
void releseDate(){
}
Mat calcHistograms(vector<Mat> &imgs,int histSize,const float * histRange){
    //vector<Mat> ans(imgs.size());
    Mat tmp,ans;
    for (int i = 0; i <imgs.size() ; i++) {
        calcHist( &imgs[i], 1, 0, Mat(),tmp, 1, &histSize, &histRange, true, false );
        if(ans.data){
            hconcat(ans, tmp, ans);
        }else{
            ans=tmp;
        }
        // a.push_back(ans[i]);
    }
    //cout<<ans<<"\n "<<ans.size()<<endl;
    return ans;
}
void calcVariance(Mat &src,vector<double> &mean, vector<double> &dst){
    //dst=Mat(1,src.rows,CV_32F);
    Mat ans,m;
    dst = vector<double>(src.rows);
    mean = vector<double>(src.rows);

    for (int rowIndex = 0; rowIndex < src.rows; rowIndex++) {
        meanStdDev(src.row(rowIndex),m, ans);
        dst[rowIndex]=ans.at<double>(0,0);
        mean[rowIndex]=m.at<double>(0,0);
    }
}
int main() {
//    float d[2][4]=  {{1,2,3,4},{4,2,3,7}};
//    float d1[1][4]= {{7,1,8,0}};
//
//
//
//    Mat a = Mat(2,4,CV_32F,d);
//    Mat b = Mat(1,4,CV_32F,d1);
////    Mat variance,m;
////    max(a,b,m);
////    //Mat mask=Mat::ones(3,1,CV_8UC1);
////    cout<<a<<endl;
////    cout<<b<<endl;
////    cout<<m<<endl;
//
//    Mat A = Mat::eye(3,3,CV_32F), B;
//    sortIdx(b, B, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
//    cout<<b<<endl;
//    cout<<B<<endl;

//    Scalar mean,stddev;
//    meanStdDev(a.col(1),mean,stddev);
//
//    cout<<"MEAN\n"<<mean<<endl;
//    cout<<"Variance\n"<<stddev<<endl;


    for (int classIndex = 0; classIndex < data.size(); classIndex++) {
        if (loadImg(imeges[classIndex], data[classIndex])) {
            printf("No image data \n");
            return -1;
        }
    }
    for(vector<Mat> &imgs : imeges){
        for(Mat &mat: imgs){
            convertIMG(mat);
        }
    }
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    for (int classIndex = 0; classIndex < data.size(); classIndex++) {
        histograms[classIndex]=calcHistograms(imeges[classIndex],histSize,histRange);
    }
    vector<double> ans,sick,meanGood,meanSick;
    calcVariance(histograms[0],meanGood,ans);
    calcVariance(histograms[1],meanSick,sick);


    Mat a = Mat(ans);
    Mat b =Mat(sick);
    //cout<<a<<endl;
    lbp::drawHist(a,"good");
    lbp::drawHist(b,"sick");
    Mat div = abs(a-b);
    lbp::drawHist(div,"div");





//    compareHist()
//   double compared = compareHist(histograms[0].row(0),histograms[0].row(1),0);
//    cout<<endl<<compared<<endl;
//
//
//
////    lbp::drawHist(histograms[0][0],"hist 1");
////    imshow("lbp final",histograms[0][0]);
//    cout<<histograms[0];
//    Mat tr;
//    transpose(histograms[0],tr);
//    cout<<tr;
//    meanStdDev()
    waitKey(0);

//    int imgSize=2;
//    Mat image[imgSize];
//    Mat dst[imgSize];
//    Mat lbp[imgSize];
//    for (int i = 0; i < imgSize; i++) {
// float       image[i] = imread( path[i], 1 );
//        if ( !image[i].data )
//        {
//            printf("No image data \n");
//            return -1;
//        }
//        cvtColor(image[i], dst[i], CV_BGR2GRAY);
//        lbp::OLBP(dst[i], lbp[i]);
//        normalize(lbp[i], lbp[i], 0, 255, NORM_MINMAX, CV_8UC1);
//
//        lbp::drawHist(lbp[i],"img "+i);
//        imshow("original "+i, image[i]);
//        imshow("lbp "+i, lbp[i]);
//    }
//
//
//
//
////    calcHist( &lbp, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );
////
////    int hist_w = 512; int hist_h = 400;
////    int bin_w = cvRound( (double) hist_w/histSize );
////
////    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 255,255,255) );
////
////    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
////    for( int i = 1; i < histSize; i++ )
////    {
////        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
////              Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
////              Scalar( 255, 0, 0), 2, 8, 0  );
////
////    }
//
//
////    imshow("calcHist Demo", histImage );
//
//    waitKey(0);
    return 0;
}