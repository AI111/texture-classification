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
    calcHist( &imgs[0], 1, 0, Mat(),ans, 1, &histSize, &histRange, true, false );
    for (int i = 1; i <imgs.size() ; i++) {
        calcHist( &imgs[i], 1, 0, Mat(),tmp, 1, &histSize, &histRange, true, false );
            hconcat(ans, tmp, ans);
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
    Mat im;
    //im=imread("/home/sasha/ClionProjects/texture-classification/res/lena.jpg",1);
    imshow("img",imeges[0][0]);
    //imwrite("/home/sasha/ClionProjects/texture-classification/res/lbp_good_1.jpg",imeges[0][0]);
    Mat img = imeges[0][0],h=histograms[0].col(0);
    lbp::showHistogram(img);

    lbp::drawHist(h,"my hist");





    Mat a = Mat(ans);
//    Mat b =Mat(sick);
//    //cout<<a<<endl;
    lbp::drawHist(a,"good");

    switch(a.type()) {
        case CV_8SC1: cout<<"type = CV_8SC1"<<endl; break;
        case CV_16SC1: cout<<"type = CV_16SC1"<<endl; break;
        case CV_16UC1: cout<<"type = CV_16UC1"<<endl; break;
        case CV_32SC1: cout<<"type = CV_32SC1"<<endl; break;
        case CV_32FC1: cout<<"type = CV_32FC1"<<endl; break;
        case CV_64FC1: cout<<"type = CV_64FC1"<<endl; break;

    }

//    lbp::drawHist(b,"sick");
//    Mat div = abs(a-b);
//    lbp::drawHist(div,"div");



    waitKey(0);


    return 0;
}