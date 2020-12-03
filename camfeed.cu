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

//nvcc `pkg-config --cflags opencv` -o camfeed camfeed.cu `pkg-config --libs opencv` -lnppig

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
    uKey.insert(101);
    uKey.insert(99);
    uKey.insert(98);

    VideoCapture cap(-1);     // get 'any' cam
    if (!cap.isOpened()) {
      return 0;
    }
    Mat frame, frame_t;

    if ( ! cap.read(frame) ){
        return 0;
    }
    string label;

    string im_type = type2str(frame.type());
    int numRows = frame.rows;
    int numCols = frame.cols;
    int x = numCols/4;
    int y = numRows/4;
    int numChannels = frame.channels();
    int numPixels = numRows * numCols;
    int rotate = 0;
    int zoom = 0;
    cout << "frame size " << frame.size() << "   Type " << frame.type() << "  " << im_type.c_str() <<  "   " << frame.depth() << "    " << frame.channels() << endl;
    int k, last_k = -1;

    while( cap.isOpened() )   // check if we succeeded
    {
        Mat frame, frame2;
        vector<Mat> channels;
        uchar *h_out;
        uchar *b_out, *g_out, *r_out;

        if ( ! cap.read(frame) )
            break;

        int switchTest = 0;
        k=-1;
        k = waitKey(1);
        if (k > 254)   k = k & 0xff;
        //cout << "k= " << k << endl;
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
              cout << endl << "Zoom in B&W" << endl;//key r
          case 118:
              cout << endl << "Zoom out B&W" << endl;   // key v
              if (press) {
                if (last_k == 114 || last_k == 118) {
                  // update zoom variable
                  if (k == 114) zoom += 1;
                  else if (zoom >0) zoom -= 1;
                  else zoom = 0;
                } else {
                  // reset zoom variable
                  if (k == 114) zoom = 1;
                  else zoom = 0;
                }
                last_k = k;
              }
              //last_k = k;
              // add cuda function
            if (zoom > 0) {
              uchar *grey_out;
              grey_out = (uchar*)malloc(numPixels * sizeof(uchar));
              memset(grey_out,0 ,numPixels);
              grey_zoomOut(frame, grey_out, x, y, numRows/2, numCols/2, numChannels, zoom);
              frame2 =  Mat(numRows, numCols, CV_8UC1, grey_out);
              free(grey_out);

            } else {
            frame2 = frame;
            }
            break;

          case 100:
            //cout << endl << "Rotate counter clockwise" << endl;  // key d
          case 103:
            cout << endl << "Rotate" << endl;  // key g
            if (press) {
              if (last_k == 100 || last_k == 103) {
                // update rotation variable
                if (k == 100) rotate += 5;
                else rotate -= 5;
              } else {
                  // reset rotation variable
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
          case 101:
            cout << endl << "Zoom in" << endl;//key e
          case 99:
            cout << endl << "Zoom out" << endl;   // key c
            if (press) {
              if (last_k == 101 || last_k == 99) {
                // update zoom variable
                if (k == 101) zoom += 1;
                else if (zoom >0) zoom -= 1;
                else zoom = 0;
              } else {
                // reset zoom variable
                if (k == 101) zoom = 1;
                else zoom = 0;
              }
              last_k = k;
            }
            //last_k = k;
            // add cuda function
          //The following algorithm is for blue, green, red colour zooming.

          if (zoom > 0) {
            b_out = (uchar*)malloc(numPixels * sizeof(uchar));
            g_out = (uchar*)malloc(numPixels * sizeof(uchar));
            r_out = (uchar*)malloc(numPixels * sizeof(uchar));
            memset(b_out,0 ,numPixels);
            memset(g_out,0 ,numPixels);
            memset(r_out,0 ,numPixels);

            zoomOut(frame, b_out, g_out, r_out, x, y, numRows/2, numCols/2, numChannels, zoom);

            frame_t =  Mat(numRows, numCols, CV_8UC1, b_out);
            channels.push_back(frame_t);
            frame_t =  Mat(numRows, numCols, CV_8UC1, g_out);
            channels.push_back(frame_t);
            frame_t =  Mat(numRows, numCols, CV_8UC1, r_out);
            channels.push_back(frame_t);
            merge(channels,frame2);
            //frame2 =  Mat(numRows, numCols, CV_8UC3, h_out);
            free(b_out);
            free(g_out);
            free(r_out);

          } else {
          frame2 = frame;
          }
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
