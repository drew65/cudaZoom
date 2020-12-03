//kernels.h
#ifndef KERNELS_H__
#define KERNELS_H__

#include <iostream>
#include <iomanip>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "gputimer.h"
#include "nppi.h"

using namespace cv;
using namespace std;

__global__ void resize(uchar* d_in, uchar*  const d_out, int x, int y, int numRows, int numCols) {
  //crop input pixels kernel
  //int out_idy = blockIdx.x;
  //int out_idx = threadIdx.x;
  int out_id = (blockIdx.x * blockDim.x + threadIdx.x);
  //int in_idy = blockIdx.x + y;
  //int in_idx = threadIdx.x + x;
  int in_id = (blockIdx.x + y) * numCols + (threadIdx.x +x);
  d_out[out_id] = d_in[in_id];

}

void enlarge_cpu(uchar* d_in, uchar*  const d_out, int x, int y, int in_numRows, int in_numCols, int out_numRows, int out_numCols) {
  //enlarge algorithm run of cpu.
  int out_id, in_id;
  for (int i=0; i<out_numRows; i++) {
    for (int j=0; j<out_numCols; j++) {
      out_id = ((i * in_numCols) + j) * 2;
      in_id = ((i+y)*in_numCols) +(j+x);
      d_out[out_id] = d_in[in_id];
    }
  }


}
/* The following three kernels implement the Directional Cubic Convolution Interpolation (DCCI),
which is an edge-directed image scaling algorithm created by Dengwen Zhou and Xiaoliu Shen
By taking into account the edges in animage, this algorithm reduces artifacts common to other scaling algorithms.
For example, staircase artifacts on diagonal lines and curves are eliminated.
The algorithm resizes an image to 2x its original dimensions, minus 1.
This is a cuda implementation of the original greyscale algorithm, working seperatlly on each blue, green, red channel.
*/
// Copy the original pixels to the output image, with gaps between the pixels.
__global__ void enlarge(uchar* d_in, uchar*  const d_out, int x, int y, int numRows, int numCols) {
  //spread out input pixels over x2 grid kernel
  int out_id = ((blockIdx.x * numCols) + threadIdx.x) * 2;
  int in_id = ((blockIdx.x + y) * numCols) + (threadIdx.x +x);
  d_out[out_id] = d_in[in_id];

}

// Calculate the pixels for the diagonal gaps
__global__ void fill_diagonals(uchar *d_in, int numRows, int numCols) {
  int x, y, idx, d1=0, d2=0, idx1, idx2, temp;
  y = blockIdx.x *2 +1;
  x = threadIdx.x *2 +1;
  idx = y * numCols + x;
  if ((y-3 >= 0) && (y+3 <= numRows) && (x-3 >= 0) && (x+3 <= numCols)) {
    for (int j=-3; j<2; j+=2){
      for (int i=-1; i<4; i+=2){
        idx1 = (y+j) *numCols + (x+i);
        idx2 = (y+j+1) *numCols + (x+i-1);
        d1 +=  __sad(d_in[idx1], d_in[idx2], 0);
      }
    }
    for (int j=-3; j<2; j+=2){
      for (int i=-3; i<2; i+=2){
        idx1 = (y+j) *numCols + (x+i);
        idx2 = (y+j+1) *numCols + (x+i+1);
        d2 +=  __sad(d_in[idx1], d_in[idx2], 0);
      }
    }
    if (100 *(1+d1) > 115 * (1+d2)) {
      temp = (-1*d_in[((y-3)*numCols+(x-3))]+9*d_in[(y-1)*numCols+x-1]+9*d_in[(y+1)*numCols+x+1]-1*d_in[(y+3)*numCols+x+3])/16;
      if (temp > 255)   temp = temp & 0xff;
      d_in[idx] = temp;
    } else if  (100 *(1+d2) > 115 * (1+d1)) {
      temp = (-1*d_in[((y+3)*numCols+(x-3))]+9*d_in[(y+1)*numCols+x-1]+9*d_in[(y-1)*numCols+x+1]-1*d_in[(y-3)*numCols+x+3])/16;
      if (temp > 255)   temp = temp & 0xff;
      d_in[idx] = temp;
    } else {
      float w1 = 1/(1+powf(d1,5));
      float w2 = 1/(1+powf(d2,5));
      float weight1 = w1/(w1+w2);
      float weight2 = w2/(w1+w2);
      float drp = (-1*d_in[((y-3)*numCols+(x-3))]+9*d_in[(y-1)*numCols+x-1]+9*d_in[(y+1)*numCols+x+1]-1*d_in[(y+3)*numCols+x+3])/16;
      float urp = (-1*d_in[((y+3)*numCols+(x-3))]+9*d_in[(y+1)*numCols+x-1]+9*d_in[(y-1)*numCols+x+1]-1*d_in[(y-3)*numCols+x+3])/16;
      temp = (drp*weight1 + urp*weight2);
      if (temp > 255)   temp = temp & 0xff;
      d_in[idx] = temp;
    }
  }
}

// Calculate the pixels for the remaining horizontal and vertical gaps
__global__ void fill_horiz_vertic(uchar *d_in, int numRows, int numCols) {
  int x, y, idx, d1=0, d2=0, idx2, temp;
  y = blockIdx.x;
  if(blockIdx.x & 1) {
    x = threadIdx.x *2;
  } else {
    x = threadIdx.x *2 +1;
  }
  idx2 = y * numCols + x;
  if ((y-3 >= 0) && (y+3 <= numRows) && (x-3 >= 0) && (x+3 <= numCols)) {
    //calculate horizontal edge strength
    idx = (y-2) * numCols + (x+1);
    d1 += __sad(d_in[idx], d_in[idx-2], 0);
    idx = (y-1) * numCols + x;
    d1 += __sad(d_in[idx], d_in[idx-2], 0);
    d1 += __sad(d_in[idx+2], d_in[idx], 0);
    idx = (y * numCols) + x+3;
    d1 += __sad(d_in[idx], d_in[idx-2], 0);
    d1 += __sad(d_in[idx-2], d_in[idx-4], 0);
    d1 += __sad(d_in[idx-4], d_in[idx-6], 0);
    idx = (y+1) * numCols + x+2;
    d1 += __sad(d_in[idx], d_in[idx-2], 0);
    d1 += __sad(d_in[idx-2], d_in[idx-4], 0);
    idx = (y+2) * numCols + x+1;
    d1 += __sad(d_in[idx], d_in[idx-2], 0);

    //calculate vertical edge strength
    idx = (y+1) * numCols + (x-2);
    d1 += __sad(d_in[idx], d_in[idx-2*numCols], 0);
    idx = (y+2) * numCols + x-1;
    d1 += __sad(d_in[idx], d_in[idx-2*numCols], 0);
    d1 += __sad(d_in[idx-2*numCols], d_in[idx-4*numCols], 0);
    idx = (y * numCols) + x+3;
    d1 += __sad(d_in[idx], d_in[idx-2*numCols], 0);
    d1 += __sad(d_in[idx-2*numCols], d_in[idx-4*numCols], 0);
    d1 += __sad(d_in[idx-4*numCols], d_in[idx-6*numCols], 0);
    idx = (y+2) * numCols + x+1;
    d1 += __sad(d_in[idx], d_in[idx-2*numCols], 0);
    d1 += __sad(d_in[idx-2*numCols], d_in[idx-4*numCols], 0);
    idx = (y+1) * numCols + x+2;
    d1 += __sad(d_in[idx], d_in[idx-2*numCols], 0);

    if (100 *(1+d1) > 115 * (1+d2)) {
      temp = (-1*d_in[((y-3)*numCols+x)]+9*d_in[(y-1)*numCols+x]+9*d_in[(y+1)*numCols+x]-1*d_in[(y+3)*numCols+x])/16;
      if (temp > 255)   temp = temp & 0xff;
      d_in[idx2] = temp;
    } else if  (100 *(1+d2) > 115 * (1+d1)) {
      temp = (-1*d_in[(y*numCols+(x-3))]+9*d_in[y*numCols+x-1]+9*d_in[y*numCols+x+1]-1*d_in[y*numCols+x+3])/16;
      if (temp > 255)   temp = temp & 0xff;
      d_in[idx2] = temp;
    } else {
      float w1 = 1/(1+powf(d1,5));
      float w2 = 1/(1+powf(d2,5));
      float weight1 = w1/(w1+w2);
      float weight2 = w2/(w1+w2);
      float vp = (-1*d_in[((y-3)*numCols+x)]+9*d_in[(y-1)*numCols+x]+9*d_in[(y+1)*numCols+x]-1*d_in[(y+3)*numCols+x])/16;
      float hp = (-1*d_in[(y*numCols+(x-3))]+9*d_in[y*numCols+x-1]+9*d_in[y*numCols+x+1]-1*d_in[y*numCols+x+3])/16;
      temp = (vp*weight1 + hp*weight2);
      if (temp > 255)   temp = temp & 0xff;
      d_in[idx2] = temp;
    }
  }
}

// Split a bgr image to individual b, g, r arrays.
__global__ void bgr_to_b_g_r(uchar*  bgr_in, uchar*  const b_out, uchar* const g_out, uchar* const r_out, int numRows, int numCols) {
  //split bgr to b,g,r kernel
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  b_out[idx] = bgr_in[3*idx];
  g_out[idx] = bgr_in[3*idx+1];
  r_out[idx] = bgr_in[3*idx+2];
}

// Change a ngr image to greyscale
__global__ void bgr_to_greyscale(uchar*  bgrImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //applying the formula: output = .299f * R + .587f * G + .114f * B;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //uchar *p = &bgrImage[idx];
  greyImage[idx] = bgrImage[3*idx+2] * .299f + bgrImage[3*idx+1] * .587f + bgrImage[3*idx] * .114f;
}

void kernel_test(Mat  frame, uchar * b_out, uchar * g_out,  uchar * r_out, int numRows, int numCols, int numChannels) {
  // This calls an initial test kernel
  GpuTimer timer;
  //uchar4* inputImage;
  uchar *d_in, *d_b_out, *d_g_out, *d_r_out;
  int numPixels = numRows * numCols;
  int numBytes = numPixels * numChannels;
  uchar *p = frame.ptr<uchar>(0);
  const dim3 blockSize(numRows, 1, 1);  //
  const dim3 gridSize( numCols, 1, 1);  //

  checkCudaErrors(cudaFree(0));
  checkCudaErrors(cudaMalloc((void **) &d_in,  sizeof(uchar) * numBytes));
  checkCudaErrors(cudaMalloc((void **) &d_b_out, sizeof(uchar) * numPixels));
  checkCudaErrors(cudaMalloc((void **) &d_g_out, sizeof(uchar) * numPixels));
  checkCudaErrors(cudaMalloc((void **) &d_r_out, sizeof(uchar) * numPixels));
  checkCudaErrors(cudaMemset((void *) d_b_out, 115, numPixels* sizeof(uchar))); //make sure memory is clean
  checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));

  timer.Start();
  bgr_to_b_g_r<<<gridSize, blockSize>>>(d_in, d_b_out, d_g_out, d_r_out, numRows, numCols);
  timer.Stop();

  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(b_out, d_b_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(g_out, d_g_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(r_out, d_r_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
  cout << "bgr kernel  " << timer.Elapsed() << " ms\n";
  cudaFree(d_in);
  cudaFree(d_b_out);
  cudaFree(d_g_out);
  cudaFree(d_r_out);
}

void gray_test(Mat  frame, unsigned char * h_out, int numRows, int numCols, int numChannels) {
  GpuTimer timer;
  uchar *d_in;
  uchar *d_greyImage;
  int numBytes = numRows * numCols * numChannels;
  uchar *p = frame.ptr<uchar>(0);
  const dim3 blockSize(numRows, 1, 1);  //
  const dim3 gridSize( numCols, 1, 1);  //

  checkCudaErrors(cudaFree(0));
  checkCudaErrors(cudaMalloc((void **) &d_in,  sizeof(uchar) * numBytes));
  checkCudaErrors(cudaMalloc((void **) &d_greyImage, sizeof(uchar) * numRows * numCols));
  checkCudaErrors(cudaMemset((void *) d_greyImage, 115, numRows * numCols* sizeof(uchar))); //make sure no memory is left laying around
  checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));

  timer.Start();
  bgr_to_greyscale<<<blockSize, gridSize>>>(d_in, d_greyImage, numRows, numCols);
  timer.Stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(h_out,  d_greyImage, sizeof(uchar) * numRows * numCols, cudaMemcpyDeviceToHost));
  cout << "gray kernel  " << timer.Elapsed() << " ms\n";
  cudaFree(d_in);
  cudaFree(d_greyImage);
}

// Function to setup and call the rotation kernel nppiRotate_8u_C1R on greyscale image.
void rotation(Mat frame, unsigned char * h_out, int numRows, int numCols, int numChannels, int rotate) {

  // change to greyscale
  gray_test(frame, h_out, numRows, numCols, numChannels);
  GpuTimer timer;
  uchar *d_in, *d_out;

  checkCudaErrors(cudaFree(0));
  checkCudaErrors(cudaMalloc((void **) &d_in,  sizeof(uchar) * numRows * numCols));
  checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(uchar) * numRows * numCols));
  checkCudaErrors(cudaMemset((void *) d_out, 115, numRows * numCols* sizeof(uchar))); //make sure no memory is left laying around
  checkCudaErrors(cudaMemcpy( d_in,  h_out, sizeof(uchar) * numRows * numCols, cudaMemcpyHostToDevice));
  NppiSize size;
  size.width = numCols;
  size.height = numRows;
  NppiRect in_rect;
  in_rect.x = 0;
  in_rect.y = 0;
  in_rect.width = 3*numCols/4;
  in_rect.height = 3*numRows/4;

  NppiRect out_rect;
  out_rect.x = 0;
  out_rect.y = 0;
  out_rect.width = numCols;
  out_rect.height = numRows;

  timer.Start();
  nppiRotate_8u_C1R((Npp8u*) d_in, size, numCols, in_rect, (Npp8u*) d_out, numCols, out_rect, rotate, numCols/2, numRows/2, NPPI_INTER_NN);
  timer.Stop();
  checkCudaErrors(cudaMemcpy(h_out,  d_out, sizeof(uchar) * numRows * numCols, cudaMemcpyDeviceToHost));
  cout << "rotation kernel  " << timer.Elapsed() << " ms\n";
  cudaFree(d_in);
  cudaFree(d_out);

}

// Function to call kernels to zoom out on a bgr image.  it increments the zoom algorithm "Directional Cubic Convolution Interpolation x2" 'zoom' times.
void zoomOut(Mat frame, uchar * b_out, uchar * g_out,  uchar * r_out, int x, int y, int numRows, int numCols, int numChannels, int zoom) {
  // define vars
  uchar *d_in, *d_b_out, *d_g_out, *d_r_out;
  int numPixels = frame.rows * frame.cols;
  int numBytes = numPixels  * numChannels;
  uchar *p = frame.ptr<uchar>(0);
  uchar *z_d_b_out, *z_d_g_out, *z_d_r_out;
  int zoomRows = frame.rows;//
  int zoomCols = frame.cols;//
  int zoomPixels = zoomRows * zoomCols;
  const dim3 blockSize(frame.rows, 1, 1);  //
  const dim3 gridSize(frame.cols, 1, 1);  //
  const dim3 z_blockSize(numRows, 1, 1);  //(numRows+4, 1, 1)
  const dim3 z_gridSize(numCols, 1, 1);  //(numCols+6, 1, 1)
  // set memory on device
  checkCudaErrors(cudaFree(0));
  checkCudaErrors(cudaMalloc((void **) &d_in,  sizeof(uchar) * numBytes));
  checkCudaErrors(cudaMalloc((void **) &d_b_out, sizeof(uchar) * numPixels));
  checkCudaErrors(cudaMalloc((void **) &d_g_out, sizeof(uchar) * numPixels));
  checkCudaErrors(cudaMalloc((void **) &d_r_out, sizeof(uchar) * numPixels));
  checkCudaErrors(cudaMemset((void *) d_b_out, 115, numPixels* sizeof(uchar))); //make sure memory is clean
  checkCudaErrors(cudaMalloc((void **) &z_d_b_out, sizeof(uchar) * zoomPixels));
  checkCudaErrors(cudaMalloc((void **) &z_d_g_out, sizeof(uchar) * zoomPixels));
  checkCudaErrors(cudaMalloc((void **) &z_d_r_out, sizeof(uchar) * zoomPixels));
  checkCudaErrors(cudaMemset((void *) z_d_b_out, 0, zoomPixels* sizeof(uchar)));
  checkCudaErrors(cudaMemset((void *) z_d_g_out, 0, zoomPixels* sizeof(uchar)));
  checkCudaErrors(cudaMemset((void *) z_d_r_out, 0, zoomPixels* sizeof(uchar)));
  // copy frame to device
  checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));
  // split channels to blue, green and red arrays
  bgr_to_b_g_r<<<gridSize, blockSize>>>(d_in, d_b_out, d_g_out, d_r_out, frame.rows, frame.cols);
  // for each channel
    // copy pixels to new image
  int zoom_count = zoom;
  while (zoom_count > 0) {
    // enlarge a defined area of the image x2, copy pixels to new files
    enlarge<<<z_blockSize, z_gridSize>>>(d_b_out, z_d_b_out, x, y, zoomRows, zoomCols);// x-3, y-2
    enlarge<<<z_blockSize, z_gridSize>>>(d_g_out, z_d_g_out, x, y, zoomRows, zoomCols);// x-3, y-2
    enlarge<<<z_blockSize, z_gridSize>>>(d_r_out, z_d_r_out, x, y, zoomRows, zoomCols);// x-3, y-2
    // fill in diagonnals
    // extrapolate new diagonal pixels between existing pixels
    fill_diagonals<<<z_blockSize, z_gridSize>>>(z_d_b_out, zoomRows, zoomCols);
    fill_diagonals<<<z_blockSize, z_gridSize>>>(z_d_g_out, zoomRows, zoomCols);
    fill_diagonals<<<z_blockSize, z_gridSize>>>(z_d_r_out, zoomRows, zoomCols);
    // fill in horzontals/verticals
    // extrapolate new horizontal and vertical pixels between existing pixels
    fill_horiz_vertic<<<blockSize, z_gridSize>>>(z_d_b_out, zoomRows, zoomCols);
    fill_horiz_vertic<<<blockSize, z_gridSize>>>(z_d_g_out, zoomRows, zoomCols);
    fill_horiz_vertic<<<blockSize, z_gridSize>>>(z_d_r_out, zoomRows, zoomCols);

    // check to see if zoom again
    if (zoom_count > 1) {
      // zoom in again, so copy bgr images back to input files of enlarge kernel
      checkCudaErrors(cudaMemcpy(d_b_out, z_d_b_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToDevice));
      checkCudaErrors(cudaMemcpy(d_g_out, z_d_g_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToDevice));
      checkCudaErrors(cudaMemcpy(d_r_out, z_d_r_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToDevice));
    } else {
      // copy resulting zoom images back to host, one for each bgr channel
      checkCudaErrors(cudaMemcpy(b_out, z_d_b_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(g_out, z_d_g_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(r_out, z_d_r_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
    }
    zoom_count -= 1;
  //cout << "bgr kernel  " << timer.Elapsed() << " ms\n";
  } // end while loop
  // free device memory
  cudaFree(d_in);
  cudaFree(d_b_out);
  cudaFree(d_g_out);
  cudaFree(d_r_out);
  cudaFree(z_d_b_out);
  cudaFree(z_d_g_out);
  cudaFree(z_d_r_out);
}

void grey_zoomOut(Mat frame, uchar * grey_out, int x, int y, int z_numRows, int z_numCols, int numChannels, int zoom) {
  // define vars
  uchar *d_in, *d_out;
  //int zoomPixels = (numRows+4) * (numCols+6);
  int numPixels = frame.rows * frame.cols;
  int numBytes = numPixels  * numChannels;
  uchar *p = frame.ptr<uchar>(0);
  checkCudaErrors(cudaFree(0));
  checkCudaErrors(cudaMalloc((void **) &d_in,  sizeof(uchar) * numBytes));
  checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(uchar) * numPixels));

  checkCudaErrors(cudaMemset((void *) d_out, 115, numPixels* sizeof(uchar))); //make sure memory is clean
  //checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));
  // split channelsfill_diagonals
  const dim3 blockSize(frame.rows, 1, 1);  //
  const dim3 gridSize(frame.cols, 1, 1);  //
  checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));
  //bgr_to_b_g_r<<<gridSize, blockSize>>>(d_in, d_b_out, d_g_out, d_r_out, frame.rows, frame.cols);
  bgr_to_greyscale<<<blockSize, gridSize>>>(d_in, d_out, frame.rows, frame.cols);
  // for each channel
  uchar *z_d_out;
  int zoomRows = frame.rows;//+(4*2);
  int zoomCols = frame.cols;//+(6*2);
  int zoomPixels = zoomRows * zoomCols;
  checkCudaErrors(cudaMalloc((void **) &z_d_out, sizeof(uchar) * zoomPixels));
  checkCudaErrors(cudaMemset((void *) z_d_out, 0, zoomPixels* sizeof(uchar)));
  int zoom_count = zoom;
  while (zoom_count > 0) {
    // copy pixels to new image
    const dim3 z_blockSize(z_numRows, 1, 1);  //(numRows+4, 1, 1)
    const dim3 z_gridSize(z_numCols, 1, 1);  //(numCols+6, 1, 1)
    //cout << "zoomRows " << zoomRows << "   zoomCols  " << zoomCols << "   zoom_count " << zoom_count << "   z_numCols  " << z_numCols <<  "  x = " << x << "  y = " << y << endl;
    enlarge<<<z_blockSize, z_gridSize>>>(d_out, z_d_out, x, y, zoomRows, zoomCols);// x-3, y-2
    cudaDeviceSynchronize();
    // fill in diagonnals
    fill_diagonals<<<z_blockSize, z_gridSize>>>(z_d_out, zoomRows, zoomCols);
    // fill in horzontals/verticals
    fill_horiz_vertic<<<blockSize, z_gridSize>>>(z_d_out, zoomRows, zoomCols);
    // resize and merge channels
    if (zoom_count > 1) {
      checkCudaErrors(cudaMemcpy(d_out, z_d_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToDevice));
    } else {
    checkCudaErrors(cudaMemcpy(grey_out, z_d_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
    }
    zoom_count -= 1;
  }
  //cout << "bgr kernel  " << timer.Elapsed() << " ms\n";
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(z_d_out);
}

void grey_zoomOut_cpu(Mat frame, uchar * z_grey_out, int x, int y, int z_numRows, int z_numCols, int numChannels, int zoom) {
  // define vars
  uchar *d_in, *d_out, *grey_out;
  //int zoomPixels = (numRows+4) * (numCols+6);
  int numPixels = frame.rows * frame.cols;
  int numBytes = numPixels  * numChannels;
  grey_out = (uchar*)malloc(numPixels * sizeof(uchar));
  memset(grey_out,0 ,numPixels);
  uchar *p = frame.ptr<uchar>(0);
  checkCudaErrors(cudaFree(0));
  checkCudaErrors(cudaMalloc((void **) &d_in,  sizeof(uchar) * numBytes));
  checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(uchar) * numPixels));

  checkCudaErrors(cudaMemset((void *) d_out, 115, numPixels* sizeof(uchar))); //make sure memory is clean
  //checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));
  // split channels
  const dim3 blockSize(frame.rows, 1, 1);  //
  const dim3 gridSize(frame.cols, 1, 1);  //
  checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));
  //bgr_to_b_g_r<<<gridSize, blockSize>>>(d_in, d_b_out, d_g_out, d_r_out, frame.rows, frame.cols);
  bgr_to_greyscale<<<gridSize, blockSize>>>(d_in, d_out, frame.rows, frame.cols);

  checkCudaErrors(cudaMemcpy(grey_out, d_out, sizeof(uchar) * numPixels, cudaMemcpyDeviceToHost));
  // for each channel
  int zoomRows = frame.rows;//+(4*2);
  int zoomCols = frame.cols;//+(6*2);
  //int zoomPixels = zoomRows * zoomCols;

    // copy pixels to new image

  enlarge_cpu(grey_out, z_grey_out, x, y, zoomRows, zoomCols, z_numRows, z_numCols);
  free(grey_out);
  cudaFree(d_in);
  cudaFree(d_out);
    // fill in diagonnals
    // fill in horzontals/verticals
  // resize and merge channels
}

#endif
