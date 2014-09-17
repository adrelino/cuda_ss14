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
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070


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

__device__ void square_arr(float* d_a, size_t n) {
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (tid < n)
    d_a[tid] = d_a[tid] * d_a[tid];
}

__global__ void kernel_call(float *d_a, size_t n) {
  square_arr(d_a, n);
}

int main(int argc,char **argv)
{
    // alloc and init input arrays on host (CPU)
    int n = 10;
    float *h_a = new float[n];
    for(int i=0; i<n; i++) h_a[i] = i;

    // CPU computation
    for(int i=0; i<n; i++)
    {
        float val = h_a[i];
        val = val*val;
        h_a[i] = val;
    }

    // print result
    cout << "CPU:"<<endl;
    for(int i=0; i<n; i++) cout << i << ": " << h_a[i] << endl;
    cout << endl;
    
    // GPU computation
    // reinit data
    for(int i=0; i<n; i++) h_a[i] = i;

    // ###
    // ### TODO: Implement the "square array" operation on the GPU and store the result in "a"
    // ###
    // ### Notes:
    // ### 1. Remember to free all GPU arrays after the computation
    // ### 2. Always use the macro CUDA_CHECK after each CUDA call, e.g. "cudaMalloc(...); CUDA_CHECK;"
    // ###    For convenience this macro is defined directly in this file, later we will only include "helper.h"
    float *d_a;
    size_t size_of_array = n * sizeof(float);
    
    // allocate data for array on GPU
    cudaMalloc(&d_a, size_of_array);
    CUDA_CHECK;
    
    // copy the data onto GPU
    cudaMemcpy(d_a, h_a, size_of_array, cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // compute the appropiate dimensions for the grid/block
    dim3 block_size = dim3(128, 1, 1);
    dim3 grid_size = dim3((n + block_size.x - 1) / block_size.x, 1, 1);
 
    // launch the kernel - for the first stuff only 10 threads
    kernel_call<<< grid_size, block_size>>>(d_a, n);
    CUDA_CHECK;

    // copy the stuff back
    cudaMemcpy(h_a, d_a, size_of_array, cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // free the stuff
    cudaFree(d_a);
    CUDA_CHECK;    

    // print result
    cout << "GPU:" << endl;
    for(int i=0; i<n; i++) cout << i << ": " << h_a[i] << endl;
    cout << endl;

    // free CPU arrays
    delete[] h_a;
}



