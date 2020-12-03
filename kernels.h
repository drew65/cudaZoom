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

__global__ void bgr_to_bgr(uchar*  bgr_in, uchar*  const bgr_out, int numRows, int numCols) {
  //test kernel
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  bgr_out[idx] = bgr_in[idx];
}

__global__ void bgr_to_greyscale(uchar*  bgrImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //applying the formula: output = .299f * R + .587f * G + .114f * B;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //uchar *p = &bgrImage[idx];
  greyImage[idx] = bgrImage[3*idx+2] * .299f + bgrImage[3*idx+1] * .587f + bgrImage[3*idx] * .114f;
}

void kernel_test(Mat  frame, uchar * h_out, int numRows, int numCols, int numChannels) {
  // This calls an initial test kernel
  GpuTimer timer;
  //uchar4* inputImage;
  uchar *d_in, *d_out;
  int numBytes = numRows * numCols * numChannels;
  uchar *p = frame.ptr<uchar>(0);
  const dim3 blockSize(numRows * numChannels, 1, 1);  //
  const dim3 gridSize( numCols, 1, 1);  //

  checkCudaErrors(cudaFree(0));
  checkCudaErrors(cudaMalloc((void **) &d_in,  sizeof(uchar) * numBytes));
  checkCudaErrors(cudaMalloc((void **) &d_out, sizeof(uchar) * numBytes));
  checkCudaErrors(cudaMemset((void *) d_out, 115, numBytes* sizeof(uchar))); //make sure memory is clean
  checkCudaErrors(cudaMemcpy( d_in,  p, sizeof(uchar) * numBytes, cudaMemcpyHostToDevice));

  timer.Start();
  bgr_to_bgr<<<gridSize, blockSize>>>(d_in, d_out, numRows, numCols);
  timer.Stop();

  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(uchar) * numBytes, cudaMemcpyDeviceToHost));
  cout << "bgr kernel  " << timer.Elapsed() << " ms\n";
  cudaFree(d_in);
  cudaFree(d_out);
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
  bgr_to_greyscale<<<gridSize, blockSize>>>(d_in, d_greyImage, numRows, numCols);
  timer.Stop();
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(h_out,  d_greyImage, sizeof(uchar) * numRows * numCols, cudaMemcpyDeviceToHost));
  cout << "gray kernel  " << timer.Elapsed() << " ms\n";
  cudaFree(d_in);
  cudaFree(d_greyImage);
}

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

#endif
