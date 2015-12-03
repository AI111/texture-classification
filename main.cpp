#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "src/histogram.hpp"
#include "src/LBP.hpp"
#include "src/MyData.h"
//#define DEBUG
using namespace cv;
using namespace std;

//vector<vector<string>> data({
//                                     {"/home/sasha/ClionProjects/texture-classification/res/good/4aftENndm_4.jpg",
//                                        "/home/sasha/ClionProjects/texture-classification/res/good/AJdocBFYGTw.jpg",
//                                        "/home/sasha/ClionProjects/texture-classification/res/good/H2difUwqaMo.jpg",
//                                        "/home/sasha/ClionProjects/texture-classification/res/good/ntaSw0ri3x0.jpg"
//                                       },
//                                     {"/home/sasha/ClionProjects/texture-classification/res/bad/dSPT4uKT2eE.jpg",
//                                       "/home/sasha/ClionProjects/texture-classification/res/bad/FJtA4nR9f5Q.jpg",
//                                       "/home/sasha/ClionProjects/texture-classification/res/bad/Jfebkg-mWEg.jpg",
//                                       "/home/sasha/ClionProjects/texture-classification/res/bad/xOPEc8D3xpM.jpg"}});

static void write(FileStorage &fs, const std::string &, const MyData &x) {
    x.write(fs);
}

static void read(const FileNode &node, MyData &x, const MyData &default_value = MyData()) {
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

int histSize = 256;
MyData configData;

vector<vector<Mat>> imeges(2);
vector<Mat> testData;
vector<Mat> histograms(2);

int loadImg(vector<Mat> &img, vector<string> &path) {
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

    cvtColor(img, img, CV_BGR2GRAY);
    if (configData.LBP) {
        Mat dst;
        lbp::OLBP(img, dst);
        img.release();
        img = dst;
    }

}

void releseDate() {
    for (vector<Mat> &v:imeges) {
        for (Mat &img:v) {
            img.release();
        }
    }
    for (Mat &img:testData) {
        img.release();
    }
}

void drawAllHist(vector<Mat> histograms) {
    for (int i = 0; i < histograms.size(); i++) {
        for (int j = 0; j < histograms[i].cols; ++j) {
            Mat t = histograms[i].col(j);
            lbp::drawHist(t, "hist class " + to_string(i) + " img # " + to_string(j));
        }
    }
}

Mat calcHistograms(vector<Mat> &imgs, int histSize, const float *histRange) {
    //vector<Mat> ans(imgs.size());
    Mat tmp, ans;
    calcHist(&imgs[0], 1, 0, Mat(), ans, 1, &histSize, &histRange, true, false);
    for (int i = 1; i < imgs.size(); i++) {
        calcHist(&imgs[i], 1, 0, Mat(), tmp, 1, &histSize, &histRange, true, false);
        hconcat(ans, tmp, ans);
        // a.push_back(ans[i]);
    }
    //cout<<ans<<"\n "<<ans.size()<<endl;
    return ans;
}

void calcVariance(Mat &src, vector<double> &mean, vector<double> &dst) {
    //dst=Mat(1,src.rows,CV_32F);
    Mat ans, m;
    dst = vector<double>(src.rows);
    mean = vector<double>(src.rows);

    for (int rowIndex = 0; rowIndex < src.rows; rowIndex++) {
        meanStdDev(src.row(rowIndex), m, ans);
        dst[rowIndex] = ans.at<double>(0, 0);
        mean[rowIndex] = m.at<double>(0, 0);
    }
}


void calcVariance1(Mat &src, vector<double> &mean, vector<double> &dst) {
    //dst=Mat(1,src.rows,CV_32F);
    Mat ans, m;
    dst = vector<double>(src.rows);
    mean = vector<double>(src.rows);

    for (int rowIndex = 0; rowIndex < src.rows; rowIndex++) {
        double scalar = cv::mean(src.row(rowIndex)).val[0];

        //sum(cv::pow(),2);
        //meanStdDev(src.row(rowIndex),m, ans);
        Mat x = (src.row(rowIndex) - scalar);
        pow(x, 2, x);
        dst[rowIndex] = sum(x).val[0];
        mean[rowIndex] = scalar;
    }
}

Mat spatial_histogram(Mat &img, int histSize, const float *histRange, int w_s, int h_s) {
    Mat tmp, ans;
    int windows = w_s * h_s;
    int window_w = img.cols / w_s;
    int window_h = img.rows / h_s;
    Mat M = Mat(img, Rect(0, 0, window_w, window_h));
    histSize /= windows;
    calcHist(&M, 1, 0, Mat(), ans, 1, &histSize, &histRange, true, false);
    for (int j = 1; j < windows; j++) {
        M = Mat(img, Rect((j % w_s) * window_w, ((int) (j / w_s)) * window_h, window_w, window_h));
        calcHist(&M, 1, 0, Mat(), tmp, 1, &histSize, &histRange, true, false);
        vconcat(ans, tmp, ans);
    }
    return ans;

}

int main(int argc, char **argv) {

    if (argc < 2) {
        cout << "set config .yml file path " << argc << endl;
        return -1;
    }
    cout << argc << endl;
    string file_url = argv[1];//"test.yml";

    FileStorage fs;
    cout << endl << "Reading: " << endl;
    fs.open(file_url, FileStorage::READ);
    fs["MyData"] >> configData;
    cout << endl << "Config Data\n" << configData << endl;
    if (loadImg(imeges[0], configData.goodData)) {
        printf("No image data \n");
        return -1;
    }
    if (loadImg(imeges[1], configData.sickData)) {
        printf("No image data \n");
        return -1;
    }
    if (loadImg(testData, configData.testData)) {
        printf("No image data \n");
        return -1;
    }

    for (vector<Mat> &imgs : imeges) {
        for (Mat &mat: imgs) {
            convertIMG(mat);
        }
    }
    for (Mat &mat: testData) {
        convertIMG(mat);
    }
    for (int i = 0; i < imeges.size(); i++) {
        for (int j = 0; j < imeges[i].size(); j++) {
            // imshow("img "+to_string(i)+" img # "+to_string(j),imeges[i][j]);
            Mat dst;//(512,512,CV_32F);
            if (configData.entropy) {
                Mat entrop = lbp::calcEntrop2(imeges[i][j], configData.entropAnsSize);
                normalize(entrop, entrop, 0, 255, NORM_MINMAX);
                lbp::dftShift(entrop, dst);
            } else {
                lbp::dftShift(imeges[i][j], dst);
            }

            ///imshow("entrop 2" ,entro);
            //cout<<entrop;

            //imshow("spectrum",dst);
            //cout<<dst;
            double min, max;
            minMaxLoc(dst, &min, &max);
            //cout<<"Max ="<<max<<endl;
//            max*=configData.thresholdTresh;//MaxLmaxoc(dst,&min,&max);
            //cout<<"Max ="<<max<<endl;
            threshold(dst, dst, max * configData.thresholdTresh, max, cv::ThresholdTypes::THRESH_TOZERO);
            //imshow("Threshold spectrum",dst);
            medianBlur(dst, dst, configData.medianMaskSize);

            if (configData.binarization)
                threshold(dst, dst, max * configData.binarizationThreshold, 1, cv::ThresholdTypes::THRESH_BINARY);
            cout << "SUM = " << sum(dst) << endl;

            imshow("spectrum class " + to_string(i) + " img # " + to_string(j), dst);
        }
    }

//    float range[] = { 0, 256 } ;
//    const float* histRange = { range };
//    imshow("original img",imeges[0][0]);
////    cout<<imeges[0][0]<<endl;
//   Mat entr=calcEntrop2(imeges[0][0],64);
//    imshow("entrop",entr);

//        Mat a = Mat::ones(8,8,CV_32FC1);
//    cout<<a<<endl;
//
//    for (int classIndex = 0; classIndex < data.size(); classIndex++) {
//        histograms[classIndex]=calcHistograms(imeges[classIndex],histSize,histRange);
//    }
//    vector<double> ans,sick,meanGood,meanSick;
//    calcVariance(histograms[0],meanGood,ans);
//    calcVariance(histograms[1],meanSick,sick);

//    for (int i = 0; i <imeges.size() ; i++) {
//        for (int j = 0; j <imeges[i].size() ; j++) {
//            Mat dst=spatial_histogram(imeges[i][j],histSize,histRange,2,2);
//            lbp::drawHist(dst,"spatial_histogram class "+to_string(i)+" # "+to_string(j));
//        }
//    }

//

//
//    Mat a = Mat(ans);

//    Mat b =Mat(sick);
//    Mat aMean = Mat(meanGood);
//    Mat bMean =Mat(meanSick);
//    Mat div = abs(a-b);
//    Mat divM = abs(aMean-bMean);
//
//    double min ,max;
//    Point maxL, minL;
//    minMaxLoc(div,&min,&max,&minL,&maxL);
//    cout<<"MAX = "<<max<<endl<<"location ="<<maxL<<endl;
//    cout<<div.at<double>(maxL)<<endl;
//    //cout<<"divMean\n"<<divM;
//    cout<<"good mean "<<aMean.at<double>(maxL)<<endl;
//    cout<<"sick mean "<<bMean.at<double>(maxL)<<endl;
//    minMaxLoc(divM,&min,&max,&minL,&maxL);
//    cout<<"MAX = "<<max<<endl<<"location ="<<maxL<<endl;
//    cout<<div.at<double>(maxL)<<endl;
//
//    cout<<"good mean "<<aMean.at<double>(maxL)<<endl;
//    cout<<"sick mean "<<bMean.at<double>(maxL)<<endl;

//    //cout<<histograms[0]<<endl<<endl<<histograms[1];
////    //cout<<a<<endl;
//    cout<<"VARIANCE\n"<<a;
//    lbp::drawHist(a,"good");
//    lbp::drawHist(aMean,"good Mean");
//
//    lbp::drawHist(b,"sick");
//    lbp::drawHist(bMean,"sick Mean");
//
//    lbp::drawHist(div,"div");

    // waitKey(0);


    waitKey(0);
    releseDate();
    return 0;
}