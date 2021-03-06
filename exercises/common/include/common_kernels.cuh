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
#include <math.h>
#include <iostream>

using namespace std;

//https://devtalk.nvidia.com/default/topic/487686/how-to-split-cuda-code/  solution !!!
const float eps=0.00001;

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




__global__ __forceinline__ void computeSpatialDerivatives(float *d_img, float *d_dx, float *d_dy, int w, int h, int nc) {

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

//neeeds memset 0 beforehand!!!
__global__ __forceinline__ void createStructureTensor(float *d_dx, float *d_dy, float *d_m11, float *d_m12, float *d_m22, int w, int h, int nc) {
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= w || y >= h)
    return;

  for(int c = 0; c < nc; ++c) {
    if (c == 0) {
      d_m11[x + y * w] = d_dx[x + y * w + c*w*h] * d_dx[x + y * w + c*w*h];
      d_m12[x + y * w] = d_dx[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
      d_m22[x + y * w] = d_dy[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];      
    } else {
      d_m11[x + y * w] += d_dx[x + y * w + c*w*h] * d_dx[x + y * w + c*w*h];
      d_m12[x + y * w] += d_dx[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
      d_m22[x + y * w] += d_dy[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
    }
  }
}

//neeeds memset 0 beforehand!!!
__global__ __forceinline__ void createStructureTensorLayered(float *d_dx, float *d_dy, float *d_m11_12_22, int w, int h, int nc) {
  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= w || y >= h)
    return;

  for(int c = 0; c < nc; ++c) {
    if (c == 0) {
      d_m11_12_22[x + y * w] = d_dx[x + y * w + c*w*h] * d_dx[x + y * w + c*w*h];
      d_m11_12_22[x + y * w + w*h] = d_dx[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
      d_m11_12_22[x + y * w + w*h*2] = d_dy[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
    } else {
      d_m11_12_22[x + y * w] += d_dx[x + y * w + c*w*h] * d_dx[x + y * w + c*w*h];
      d_m11_12_22[x + y * w + w*h] += d_dx[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];
      d_m11_12_22[x + y * w + w*h*2] += d_dy[x + y * w + c*w*h] * d_dy[x + y * w + c*w*h];      
    }
  }
}


//scales a vector so that its largest component is 1;
__host__ __device__ __forceinline__ void unitScale(float2 *v) { //notice e1,2 are arrays
  if(v->x > v->y){
    v->y /= v->x;
    v->x = 1;
  }else{
    v->x /= v->y;
    v->y = 1;
  }
}


//working eigenvalue computation with all special cases (received with thanks from another group)
__host__ __device__ __forceinline__ void cuda_eig(float a, float b, float c, float d, float *L1, float *L2, float *V1_1, float *V1_2, float *V2_1, float *V2_2)
{
    float T = a + d; 
    float D = a*d - b*c;
    
    *L1 = (T/2.0f) + sqrtf(T*T/4.0f - D);
    *L2 = (T/2.0f) - sqrtf(T*T/4.0f - D);

    if (abs(c) > eps)
    {
        if (abs(*L1-d) - abs(c) > 0.0f)
        {
            *V1_1 = (*L1-d)/abs(*L1-d);
            *V1_2 = c/abs(*L1-d);
        }
        else
        {
            *V1_1 = (*L1-d)/abs(c);
            *V1_2 = c/abs(c);
        }
        if (abs(*L2-d) - abs(c) > 0.0f)
        {
            *V2_1 = (*L2-d)/abs(*L2-d);
            *V2_2 = c/abs(*L2-d);
        }
        else
        {
            *V2_1 = (*L2-d)/abs(c);
            *V2_2 = c/abs(c);
        }

    }
    else if (abs(b) > eps)
    {
        if (abs(*L1-a) - abs(b) > 0.0f)
        {
        *V1_1 = b/abs(*L1-a);
        *V1_2 = (*L1-a)/abs(*L1-a);
        }
        else
        {
        *V1_1 = b/abs(b);
        *V1_2 = (*L1-a)/abs(b);

        }

        if (abs(*L2-a) - abs(b) > 0.0f)
        {
        *V2_1 = b/abs(*L2-a);
        *V2_2 = (*L2-a)/abs(*L2-a);
        }
        else
        {
        *V2_1 = b/abs(b);
        *V2_2 = (*L2-a)/abs(b);
        }

    }
    else //if ((abs(c) <= eps) && (abs(b) <= eps))
    {
        *V1_1 = 1.0f;
        *V1_2 = 0.0f;
        *V2_1 = 0.0f;
        *V2_2 = 1.0f;
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
//a=m.x, b=m.y, c=m.z, d=m.w
//
//makes channel 2 of diffusion tensor (m22) 0!!!
__host__ __device__ __forceinline__ void compute_eig(float4 m, float *lambda1, float *lambda2, float2 *e1, float2 *e2) { //notice e1,2 are arrays
  float trace = m.x + m.w;
  float sqrt_term = sqrtf(trace*trace - 4*(m.x*m.w - m.y*m.z));


  float T = m.x + m.w; 
  float D = m.x*m.w - m.y*m.z;
    
  *lambda1 = (T/2.0f) + sqrtf(T*T/4.0f - D);
  *lambda2 = (T/2.0f) - sqrtf(T*T/4.0f - D);


  //(*lambda1) = (trace + sqrt_term) / 2.0f; //lambda1  //larger eigenvalue
  //(*lambda2) = (trace - sqrt_term) / 2.0f; //lambda2  //smaller eigenvalue QUESTION: since sqrt_term >= 0 always, it must follow that lambda1 > lambda2;

  //www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
  if(abs(m.z)>eps){ //c!=0
    e2->x = ((*lambda2)-m.w); //-d
    e1->x = ((*lambda1)-m.w); //-d
    e2->y=m.z;
    e1->y=m.z;
  }else if(abs(m.y)>eps){ //b!=0
    e2->y = ((*lambda2)-m.x); //-d
    e1->y = ((*lambda1)-m.x); //-d
    e2->x=m.y;
    e1->x=m.y;
  }else{
    e2->x = 0;
    e1->x = 1;
    e2->y = 1;
    e1->y = 0;
  }

  unitScale(e1);
  unitScale(e2);

  //only calculating the first column of 
  //[eig1 | x*eig1]=A-lambda2*I;
  //e1->x = m.x - (*lambda2);
  //[eig2 | x*eig2]=A-lambda1*I;
  //e2->x = m.x - (*lambda1);

  //e1->y=m.z; //same for both, since -I*lambda is 0 at the off diagonals   //eigenvector corresponding to larger eigenvalue is first column (NOT ROW!!) of float4
  //e2->y=m.z;   //eigenvector corresponding to smaller eigenvalue is second column (NOT ROW!!) of float4
}

// 2x2 matrix times scalar multiplication, result stored in m
__host__ __device__ __forceinline__ void mul(float s, float4 *m) { //notice e1,2 are arrays
  m->x = s * m->x;
  m->y = s * m->y;
  m->z = s * m->z;
  m->w = s * m->w;
}

// 2x2 matrix times scalar multiplication, result stored in m
__host__ __device__ __forceinline__ void add(float4 l, float4 *m) { //notice e1,2 are arrays
  m->x = l.x + m->x;
  m->y = l.y + m->y;
  m->z = l.z + m->z;
  m->w = l.w + m->w;
}
__host__ __device__ float4 operator+(const float4 & a, const float4 & b) {

   return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);

}

// 2x2 matrix times 2x1 vector multiplication -> 2x1 vector stored in input v value
__host__ __device__ __forceinline__ void mulVec(float4 m, float2 *v) { //notice e1,2 are arrays
  float tempx = m.x * v->x + m.y * v->y;
  v->y = m.z * v->x + m.w * v->y;
  v->x = tempx;
}
__host__ __device__ float2 operator*(const float4 & m, const float2 & v) {

   return make_float2(m.x * v.x + m.y * v.y, m.z * v.x + m.w * v.y);

}

//so float4 as matrix represenation can be used with cout
__host__ __device__ ostream& operator<<(ostream& os, const float4& m) {
  os << m.x << " \t" << m.y << std::endl << m.z << " \t" << m.w << std::endl;
  return os;
}

//so float2 as column vector represenation can be used with cout
__host__ __device__ ostream& operator<<(ostream& os, const float2& v) {
  os << v.x << std::endl << v.y << std::endl;
  return os;
}

// outer product of 2x1 vector with itself -> 2x2 matrix
__host__ __device__ __forceinline__ void outer(float2 v, float4 *m) { //notice e1,2 are arrays
  m->x = v.x * v.x;
  m->y = v.x * v.y;
  m->z = v.y * v.x;
  m->w = v.y * v.y;
}

__host__ __device__ __forceinline__ float4 calcG(float lambda1, float lambda2, float2 e1, float2 e2, float C, float alpha){
    float4 e1_2; outer(e1,&e1_2);
    float4 e2_2; outer(e2,&e2_2);

    mul(alpha,&e1_2); //mu1 = alpha

    float mu2=alpha;
    float lambdaDiff=lambda1-lambda2; //always positive since l1>l2;
    if(abs(lambdaDiff) > eps){ //alpha since we use floating point arithmetic
        mu2 = alpha + (1-alpha) * expf( -C / (lambdaDiff*lambdaDiff) );
    }
    mul(mu2,&e2_2);

    float4 G = e1_2 + e2_2;  //own operator in common_kernels.cuh
    //add(e1_2,&e2_2);G=e2_2;

    return G;
}

__host__ __device__ __forceinline__ float4 calcG2(float lambda1, float lambda2, float2 e1, float2 e2, float C, float alpha){

    float mu1=alpha;
    float mu2=alpha;

    float lambdaDiff=lambda1-lambda2; //always positive since l1>l2;
    if(abs(lambdaDiff) > eps){ //alpha since we use floating point arithmetic
        mu2 = alpha + ((1.0f-alpha) * expf( -C / (lambdaDiff*lambdaDiff) )) ;
    }

    float4 G;

    G.x = mu1*e1.x*e1.x + mu2*e2.x*e2.x;
    G.y = mu1*e1.x*e1.y + mu2*e2.x*e2.y;
    G.z = mu1*e1.y*e1.x + mu2*e2.y*e2.x;
    G.w = mu1*e1.y*e1.y + mu2*e2.y*e2.y;

    return G;
}

//from other group
__host__ __device__ __forceinline__ float4 calcG3(float L1, float L2, float2 e1, float2 e2, float C, float alpha){

    float mu1, mu2;
    mu1 = alpha;
    if (abs(L1-L2)<eps)
    {
        mu2 = alpha;
    }
    else
    {
        mu2 = alpha + (1.0f-alpha)*expf(-C/((L1-L2)*(L1-L2)));
    }

    float4 G;
    G.x = mu1*e1.x*e1.x + mu2*e2.x*e2.x;
    G.y=G.z = mu1*e1.x*e1.y + mu2*e2.x*e2.y; 
    G.w = mu1*e1.y*e1.y + mu2*e2.y*e2.y;
}
