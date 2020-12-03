#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/videoio.hpp"
//#include "opencv2/opencv.hpp"
//#include <opencv2>
#include <iostream>
#include <stdio.h>
#include <set>
#include "Utility.h"
#include "kernels.h"


//NB  If for some reason the cmake does not compile the project then the following line compiles on Unix
//nvcc `pkg-config --cflags opencv` -o camfeed camfeed.cu `pkg-config --libs opencv`
//NB the project works with openCV2 it works on CV3 by uncomenting the #include "opencv2/videoio.hpp" line and adding -lopencv_videoio to the compile commands
//NB -lnppi may be needed to link the NVIDIA Performance Primitives lib (NPP)

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)


using namespace cv;
using namespace std;



string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main()
{
    //bool gray = 0;
    bool press =0;
    set<int> uKey;
    uKey.insert(114);
    uKey.insert(118);
    uKey.insert(100);
    uKey.insert(103);
    uKey.insert(102);
    uKey.insert(99);
    uKey.insert(98);

    VideoCapture cap(-1);     // get 'any' cam
    if (!cap.isOpened()) {
      return 0;
    }
    Mat frame;
    if ( ! cap.read(frame) ){
        return 0;
    }
    string label;

    string im_type = type2str(frame.type());
    int numRows = frame.rows;
    int numCols = frame.cols;
    int numChannels = frame.channels();
    int numPixels = numRows * numCols;
    int rotate = 0;
    cout << "frame size " << frame.size() << "   Type " << frame.type() << "  " << im_type.c_str() <<  "   " << frame.depth() << "    " << frame.channels() << endl;
    int k, last_k = -1;

    while( cap.isOpened() )   // check if we succeeded
    {
        Mat frame, frame2;
        uchar *h_out;

        if ( ! cap.read(frame) )
            break;

        int switchTest = 0;
        k=-1;
        k = waitKey(1);
        if (k > 254)   k = k & 0xff;
        set<int>::iterator it = uKey.find(k);
        if(it != uKey.end()) {
          press = 1;
          switchTest = k;
        } else {
          press = 0;
          switchTest = last_k;
        }

        switch(switchTest) {

          case 114:
              cout << endl << "TODO Zoom in" << endl;//key r
          case 118:
              cout << endl << "TODO Zoom out" << endl;   // key v
              if (press) {
                if (k == last_k) {
                  // update transVar
                } else {
                  //last_k = k;
                  // reset transVar
                }
                last_k = k;
              }
              //last_k = k;
              // add cuda function
            frame2 = frame;
            break;

          case 100:
            //cout << endl << "Rotate counter clockwise" << endl;  // key d
          case 103:
            cout << endl << "Rotate" << endl;  // key g
            if (press) {
              if (last_k == 100 || last_k == 103) {
                // update transVar
                if (k == 100) rotate += 5;
                else rotate -= 5;
              } else {
                  // reset transVar
                  if (k == 100) rotate = 5;
                  else rotate = -5;
              }
              last_k = k;
            }
            // add cuda function
            frame2 = frame;
            h_out = (uchar*)malloc(numPixels * sizeof(uchar));
            memset(h_out,0 ,numPixels);
            rotation(frame, h_out, numRows, numCols, numChannels, rotate);
            frame2 =  Mat(numRows, numCols, CV_8UC1, h_out);
            break;
          case 102:
              cout << endl << "Return to normal image" << endl; // key f
              if (press) {
                if (k == last_k) {
                  // update transVar
                } else {
                  // reset transVar
                }
                last_k = k;
              }
              rotate = 0;
              frame2 = frame;

            break;
          case 99:
            cout << endl << "This is a test Kernel" << endl; // key c
            if (press) {
              if (k == last_k) {
                  // update transVar
              } else {
                  //last_k = k;
                  // reset transVar
              }
              last_k = k;
          }
          //gray = 0;
          h_out = (uchar*)malloc(numPixels * numChannels * sizeof(uchar));
          memset(h_out,0 ,numPixels * numChannels);
          kernel_test(frame, h_out,  numRows, numCols, numChannels);
          frame2 =  Mat(numRows, numCols, CV_8UC3, h_out);
          free(h_out);
          break;
        case 98:
          cout << endl << "Black & White" << endl;  // key b
            if (press) {
              if (k == last_k) {
              // update transVar
              } else {
              //last_k = k;
              // reset transVar
              }
            last_k = k;
          }
            //gray = 1;
            h_out = (uchar*)malloc(numPixels * sizeof(uchar));
            memset(h_out,0 ,numPixels);
            gray_test(frame, h_out,  numRows, numCols, numChannels);
            frame2 =  Mat(numRows, numCols, CV_8UC1, h_out);
            free(h_out);
            break;
          default:
            cout << endl << "Default" << endl;
            frame2 = frame;
          }  // end switch

        imshow("output image",frame2);

        //cout << "key = " << k <<  "  last_k " << last_k << endl;
        if ( k==27 )
            break;
    }  // end while loop
    return 0;
}
