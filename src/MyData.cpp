//
// Created by sasha on 12/2/15.
//

#include "MyData.h"

void MyData::read(const FileNode &node) {

    string s=(string)node["DataClass1"];
    goodData.clear();
    boost::split(goodData, s, boost::is_any_of("\n"));
    cout<<"DataClass1"<<endl<<s<<endl<<goodData.size()<<endl;
    sickData.clear();
    s=(string)node["DataClass2"];
    boost::split(sickData, s, boost::is_any_of("\n"));
    cout<<"DataClass2"<<endl<<s<<endl<<sickData.size()<<endl;
    s=(string)node["TestData"];
    testData.clear();
    boost::split(testData, s, boost::is_any_of("\n"));
    cout<<"TestData"<<endl<<s<<endl<<testData.size()<<endl;

}

void MyData::write(FileStorage &fs)const {

    fs << "{" << "DataClass1"<<concatVector(goodData)<<"DataClass2"<<concatVector(sickData)<< "TestData"<<concatVector(testData)<<"}";
}

//ostream &MyData::operator<<(ostream &out, const MyData &m) {
//    return <#initializer#>;
//}
string MyData::concatVector(const vector<string> &vec)const {
    string tmp;
    for (int i = 0; i <vec.size() ; ++i) {
        tmp.append(i<vec.size()-1?vec[i]+"\n":vec[i]);
    }

    return tmp;
}
