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
// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, , p070

#include <cuda_runtime.h>
#include <iostream>
using namespace std;



// cuda error checking
#define CUDA_CHECK cuda_check(__FILE__,__LINE__)
void cuda_check(string file, int line)
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        cout << endl << file << ", line " << line << ": " << cudaGetErrorString(e) << " (" << e << ")" << endl;
        exit(1);
    }
}

__device__ void add_arrays(float *d_a, float *d_b, float *d_c, size_t n) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid < n)
    d_c[tid] = d_a[tid] + d_b[tid];
}

__global__ void kernel_call(float *d_a, float *d_b, float *d_c, size_t n) {
  add_arrays(d_a, d_b, d_c, n);
}

int main(int argc, char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 20;
    
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    
    for(int i=0; i<n; i++)
    {
        h_a[i] = i;
        h_b[i] = (i%5)+1;
        h_c[i] = 0;
    }

    // CPU computation
    for(int i=0; i<n; i++) h_c[i] = h_a[i] + h_b[i];

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    cout << endl;
    // init c
    for(int i=0; i<n; i++) h_c[i] = 0;
    
    // GPU computation
    // ###
    // ### TODO: Implement the array addition on the GPU, store the result in "c"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "aux.h"

    // allocate memory on GPU
    size_t size_arr = n * sizeof(float);
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size_arr);
    CUDA_CHECK;

    cudaMalloc(&d_b, size_arr);
    CUDA_CHECK;
    
    cudaMalloc(&d_c, size_arr);
    CUDA_CHECK;

    // copy the stuff from h_a, h_b to d_a, d_b
    cudaMemcpy(d_a, h_a, size_arr, cudaMemcpyHostToDevice);
    CUDA_CHECK;
    
    cudaMemcpy(d_b, h_b, size_arr, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // setup grid/block size
    dim3 block_size = dim3(128, 1, 1);
    dim3 grid_size = dim3((n + block_size.x - 1) / block_size.x, 1, 1);    

    // do the kernel call
    kernel_call<<<grid_size, block_size>>>(d_a, d_b, d_c, size_arr);
    CUDA_CHECK;

    cudaMemcpy(h_c, d_c, size_arr, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // cuda cleanup
    cudaFree(d_a);
    CUDA_CHECK;

    cudaFree(d_b);
    CUDA_CHECK;

    cudaFree(d_c);
    CUDA_CHECK;
    
    // print result
    cout << "GPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}



