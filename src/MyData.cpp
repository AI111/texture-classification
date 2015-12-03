//
// Created by sasha on 12/2/15.
//

#include "MyData.h"

void MyData::read(const FileNode &node) {
    readVector(goodData,node,"DataClass1");
    readVector(sickData,node,"DataClass2");
    readVector(testData,node,"TestData");
//    string s=(string)node["DataClass1"];
//    goodData.clear();
//    boost::split(goodData, s, boost::is_any_of("\n"));
//    cout<<"DataClass1"<<endl<<s<<endl<<goodData.size()<<endl;
//    sickData.clear();
//    s=(string)node["DataClass2"];
//    boost::split(sickData, s, boost::is_any_of("\n"));
//    cout<<"DataClass2"<<endl<<s<<endl<<sickData.size()<<endl;
//    s=(string)node["TestData"];
//    testData.clear();
//    boost::split(testData, s, boost::is_any_of("\n"));
//    cout<<"TestData"<<endl<<s<<endl<<testData.size()<<endl;

}

void MyData::write(FileStorage &fs)const {
fs<<"{";
    concatVector(goodData,fs,"DataClass1");
        concatVector(sickData,fs,"DataClass2");
    concatVector(testData,fs,"TestData");

    fs<<"}";
}

//ostream &MyData::operator<<(ostream &out, const MyData &m) {
//    return <#initializer#>;
//}
void MyData::concatVector(const vector<string> &vec,FileStorage &fs,const string name)const {
    fs<<name<<"[";
    for (int i = 0; i <vec.size() ; i++) {
        fs << "{:" << "url" << vec[i]<< "}";
    }
    fs<<"]";

}
void MyData::readVector(vector<string> &vec,const FileNode &fn,const string name){
    FileNode features = fn[name];
    FileNodeIterator it = features.begin(), it_end = features.end();
    int idx = 0;

// iterate through a sequence using FileNodeIterator
    for( ; it != it_end; ++it, idx++ )
    {
        vec.push_back((cv::String)(*it)["url"]);
        cout << "feature #" << idx << ": ";
        cout << "url=" << (*it).name() <<" "<< (cv::String)(*it)["url"]<<endl;
    }
}
