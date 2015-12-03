//
// Created by sasha on 12/3/15.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include "src/MyData.h"

using namespace cv;
using namespace std;
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
static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}

int main(){
    string filename = "test1.yml";
vector<cv::String> v({"/home/sasha/ClionProjects/texture-classification/res/good/4aftENndm_4.jpg",
                        "/home/sasha/ClionProjects/texture-classification/res/good/AJdocBFYGTw.jpg",
                        "/home/sasha/ClionProjects/texture-classification/res/good/H2difUwqaMo.jpg",
                        "/home/sasha/ClionProjects/texture-classification/res/good/ntaSw0ri3x0.jpg"
                       });
    bool a =true;
    cout<<"bool a = "<<a<<endl;
//FileStorage fs(filename,FileStorage::WRITE);
//    cout<<v.size()<<endl;
//    fs << "Mapping";                              // text - mapping
//    fs << "{" ;
//    for(String &str:v){
//        fs<<"texture"<<str;
//    }
//    fs << "}";
//  fs.release();

//    FileStorage fs("test.yml", FileStorage::WRITE);
//
//        MyData data1;
//        data1.goodData=data[0];
//        data1.sickData=data[1];
//        data1.LBP=true;
//        data1.entropy=true;
//        data1.entropAnsSize=64;
//        data1.thresholdTresh=0.491;
//        data1.medianMaskSize=5;
//    fs<<"MyData"<<data1;
////    fs << "textures" << "[";
////    String string1;
////    for( int i = 0; i < v.size(); i++ )
////    {
////        fs << "{:" << "url" << v[i]<< "}";
////    }
////    fs << "]";
//    fs.release();
    //cout<<data1;

//TO READ
////vector<string> myKpVec;
//    FileStorage fs("test.yml",FileStorage::READ);
//////    FileNode features = fs["Mapping"];
////    FileNode features = fs["textures"];
////    FileNodeIterator it = features.begin(), it_end = features.end();
////    int idx = 0;
////    std::vector<String> lbpval;
////
////// iterate through a sequence using FileNodeIterator
////    for( ; it != it_end; ++it, idx++ )
////    {
////        cout << "feature #" << idx << ": ";
////        cout << "url=" << (*it).name() <<" "<< (cv::String)(*it)["url"]<<endl;
////    }
//    MyData a;
//    fs["MyData"]>>a;
//    cout<<a.goodData.size();
//    fs.release();
//    cout<<a.LBP<<" "<<a.entropy;
return 0;
}