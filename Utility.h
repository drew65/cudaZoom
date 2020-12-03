// Utility.h file
#ifndef UTILITY_H__
#define UTILITY_H__

#include <iostream>
#include <iomanip>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
//#include <opencv2/opencv.hpp>
using namespace std;
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

/*
cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }
*/
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    cout << "CUDA error at: " << file << ":" << line << std::endl;
    cout << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


#endif
