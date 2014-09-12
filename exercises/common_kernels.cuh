// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2014, September 8 - October 10
// ###
// ###
// ### Maria Klodt, Jan Stuehmer, Mohamed Souiai, Thomas Moellenhoff
// ###
// ###
// ###

// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070

//#include "common_kernels.h"
#include <cuda_runtime.h>

//https://devtalk.nvidia.com/default/topic/487686/how-to-split-cuda-code/  solution !!!


//                       in             out      out
__device__ __forceinline__ void d_gradient(float *imgIn, float *v1, float *v2, int w, int h, int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x>=w || y>=h) return;

    int xPlus = x + 1;
    if(xPlus>=w) xPlus=w-1;

    int yPlus = y + 1;
    if(yPlus>=h) yPlus=h-1;

    for (int i = 0; i < nc; ++i)
    {
        v1[x+ y*w +i*w*h]=imgIn[xPlus+ y*w + i*w*h]-imgIn[x+ y*w + i*w*h];
        v2[x+ y*w +i*w*h]=imgIn[x+ yPlus*w + i*w*h]-imgIn[x+ y*w + i*w*h];

    }
}

__global__ __forceinline__ void gradient(float *imgIn, float *v1, float *v2, int w, int h, int nc){
    d_gradient(imgIn, v1, v2, w, h, nc);
}

//                         in        in         out
__device__ __forceinline__ void d_divergence(float *v1, float *v2, float *imgOut, int w, int h, int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x>=w || y>=h) return;

    for (int i = 0; i < nc; ++i)
    {
        float backv1_x=v1[x+ y*w +i*w*h];
        if(x>0) backv1_x -=v1[(x-1)+ y*w + i*w*h];
        float backv2_y=v2[x+ y*w + i*w*h];
        if(y>0) backv2_y -=v2[x+ (y-1)*w + i*w*h];
        imgOut[x+ y*w +i*w*h]=backv1_x+backv2_y;
    }
}

__global__ __forceinline__ void divergence(float *v1, float *v2, float *imgOut, int w, int h, int nc){
    d_divergence(v1, v2, imgOut, w, h, nc);
}

//                     in           out
__device__ __forceinline__ void l2norm(float *imgIn, float *imgOut, int w, int h, int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x>=w || y>=h) return;

    float c=0;

    for (int i = 0; i < nc; ++i)
    {
        c+=powf(imgIn[x+ y*w +i*w*h],2);
    }

    c=sqrtf(c);

    imgOut[x+ y*w]=c; //channel is 0 -> grayscale
}


__device__ __forceinline__ void d_convolutionGPU(float *imgIn, float *kernel, float *imgOut, int w, int h, int nc, int kernelSize){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t k = kernelSize;

    int r=k/2;

    //check for boundarys of the block
    if(x>=w || y>=h) return; 

    //iterate over all channels
    for(unsigned int c=0;c<nc;c++) {
        float sum=0;
        //do convolution
        for(unsigned int i=0;i<k;i++){
            unsigned int x_new;
            //clamping x
            if(x+r<i) x_new=0;
            else if(x+r-i>=w) x_new=w-1;
            else x_new=x+r-i;
            for(unsigned int j=0;j<k;j++){
                //clamping y
                unsigned int y_new;
                if(y+r<j)
                    y_new=0;
                else if(y+r-j>=h)
                    y_new=h-1;
                else
                    y_new=y+r-j;
                sum+=kernel[i+j*k]*imgIn[x_new+y_new*w+w*h*c];
            }
        }
        imgOut[x+w*y+w*h*c]=sum;
    }
}

__global__ __forceinline__ void convolutionGPU(float *imgIn, float *kernel, float *imgOut, int w, int h, int nc, int kernelSize){
    d_convolutionGPU(imgIn, kernel, imgOut, w, h, nc, kernelSize);
}




__device__ __forceinline__ void computeSpatialDerivatives(float *d_img, float *d_dx, float *d_dy, int w, int h, int nc) {

  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;

  // if outside of image --> return
  if (x >= w || y >= h)
    return;

  // calc indices
  int xPlus1 = x+1;
  int xMinus1 = x-1;

  int yPlus1 = y+1;
  int yMinus1 = y-1;

  // do clamping
  xPlus1 = max(min(xPlus1, w-1), 0);
  xMinus1 = min(max(xMinus1, 0), w-1);

  yPlus1 = max(min(yPlus1, h-1), 0);
  yMinus1 = min(max(yMinus1, 0), h-1);
  
  // calc derivatives
  for (int c = 0; c < nc; ++c) {
    // x-derivatives
    d_dx[x + y*w + c*w*h] = (3*d_img[xPlus1 + yPlus1*w + c*w*h] +
                             10*d_img[xPlus1 + y*w + c*w*h] +
                             3*d_img[xPlus1 + yMinus1*w + c*w*h] -
                             3*d_img[xMinus1 + yPlus1*w + c*w*h] -
                             10*d_img[xMinus1 + y*w + c*w*h] -
                             3*d_img[xMinus1 + yMinus1*w + c*w*h]) / 32.0f;

    // y-derivatives
    d_dy[x + y*w + c*w*h] = (3*d_img[xPlus1 + yPlus1*w + c*w*h] +
                             10*d_img[x + yPlus1*w + c*w*h] +
                             3*d_img[xMinus1 + yPlus1*w + c*w*h] -
                             3*d_img[xPlus1 + yMinus1*w + c*w*h] -
                             10*d_img[x + yMinus1*w + c*w*h] -
                             3*d_img[xMinus1 + yMinus1*w + c*w*h]) / 32.0f;
  }
}

__device__ __forceinline__ void createStructureTensor(float *d_dx, float *d_dy, float *d_m11, float *d_m12, float *d_m22, int w, int h, int nc) {
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= w || y >= h)
    return;

  for(int c = 0; c < nc; ++c) {
      d_m11[x + y * w] += d_dx[x + y * w + c*w*h] * d_dx[x + y * w + c*w*h];
      d_m12[x + y * w] += d_dx[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
      d_m22[x + y * w] += d_dy[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
  }
}

__device__ __forceinline__ void createStructureTensorLayered(float *d_dx, float *d_dy, float *d_m11_12_22, int w, int h, int nc) {
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= w || y >= h)
    return;

  for(int c = 0; c < nc; ++c) {
      d_m11_12_22[x + y * w] += d_dx[x + y * w + c*w*h] * d_dx[x + y * w + c*w*h];
      d_m11_12_22[x + y * w + w*h] += d_dx[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
      d_m11_12_22[x + y * w + w*h*2] += d_dy[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
  }
}

//float4 used as 2x2 matrix
// x  y
// z  w

//float2 used for eigenvalues
// x=lambda1, y=lambda2

//float4 used for eigenvectore
//eig
// e1x  e2x
// e1y  e2y
//http://en.wikipedia.org/wiki/Eigenvalue_algorithm#2.C3.972_matrices
__device__ __forceinline__ void compute_eig(float4 m, float *lambda1, float *lambda2, float2 *e1, float2 *e2) { //notice e1,2 are arrays
  float trace = m.x + m.w;
  float sqrt_term = sqrtf(trace*trace - 4*(m.x*m.w - m.y*m.z));


  (*lambda1) = (trace + sqrt_term) / 2.0f; //lambda1  //larger eigenvalue
  (*lambda2) = (trace - sqrt_term) / 2.0f; //lambda2  //smaller eigenvalue QUESTION: since sqrt_term >= 0 always, it must follow that lambda1 > lambda2;

  //only calculating the first column of 
  //[eig1 | x*eig1]=A-lambda2*I;
  // e1->x=m.x-lambda2;
  // //[eig2 | x*eig2]=A-lambda1*I;
  // e2->x=m.x-lambda1;

  e1->y=m.z; //same for both, since -I*lambda is 0 at the off diagonals   //eigenvector corresponding to larger eigenvalue is first column (NOT ROW!!) of float4
  e2->y=m.z;   //eigenvector corresponding to smaller eigenvalue is second column (NOT ROW!!) of float4
}