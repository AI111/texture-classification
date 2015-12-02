//
// Created by sasha on 12/2/15.
//
#include <string>
#include <opencv2/core/core.hpp>
#include <boost/algorithm/string.hpp>

#ifndef TEXTURE_CLASSIFICATION_MYDATA_H
#define TEXTURE_CLASSIFICATION_MYDATA_H

using namespace std;
using namespace cv;
class MyData {

public:   // Data Members
//    string id;
    vector<string> goodData;
    vector<string> sickData;
    vector<string> testData;
    string concatVector(const vector<string> &vec);

public:
    void read(const FileNode& node);
    void write(FileStorage& fs)const;//Read serialization for this class
//    static ostream& operator<<(ostream& out, const MyData& m);
    string concatVector(vector<string> &vec) const;

    string concatVector(const vector<string> &vec) const;
};


#endif //TEXTURE_CLASSIFICATION_MYDATA_H
