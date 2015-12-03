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

public:
    vector<string> goodData;
    vector<string> sickData;
    vector<string> testData;
    bool LBP;
    bool entropy;
    bool binarization;
    int entropAnsSize;
    double thresholdTresh;
    double binarizationThreshold;
    int medianMaskSize;
public:
    void read(const FileNode& node);
    void write(FileStorage& fs)const;//Read serialization for this class
    string concatVector(vector<string> &vec) const;

    string concatVector(const vector<string> &vec) const;

    void concatVector(const vector<string> &vec, FileStorage &fs) const;

    void concatVector(const vector<string> &vec, FileStorage &fs, const string name) const;

    void readVector(vector<string> &vec, const FileNode &fn, const string name);
};

std::ostream& operator<<(std::ostream &strm, const MyData &a);
#endif //TEXTURE_CLASSIFICATION_MYDATA_H
