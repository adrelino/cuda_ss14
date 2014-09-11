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

#include "aux.h"
#include "cublas_v2.h"

string get_cublas_error(cublasStatus_t stat) {
  switch(stat)
    {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }

  return "Unknown error";
}

void cublas_check(cublasStatus_t stat) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    cerr << "Received error: " << get_cublas_error(stat) << endl;
  }
}

int main(int argc, char **argv)
{
  cudaDeviceSynchronize(); CUDA_CHECK;
  
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
  // allocate memory on GPU
  size_t size_arr = n * sizeof(float);
  float *d_a, *d_b;

  cudaMalloc(&d_a, size_arr);
  CUDA_CHECK;

  cudaMalloc(&d_b, size_arr);
  CUDA_CHECK;
    
  // CUBLAS stuff
  cublasStatus_t stat;
  cublasHandle_t handle;    
  // get CUBLAS context
  stat = cublasCreate(&handle);
  cublas_check(stat);
    
  // copy the stuff from h_a, h_b to d_a, d_b
  stat = cublasSetVector(n, sizeof(*h_a), h_a, 1, d_a, 1);
  cublas_check(stat);
    
  stat = cublasSetVector(n, sizeof(*h_b), h_b, 1, d_b, 1);
  cublas_check(stat);
    
  // run the cublas algorithm
  float alph = 1.0f;
  stat = cublasSaxpy(handle, n, &alph, d_a, 1, d_b, 1);
  cublas_check(stat);    

  // get vector back from CUBLAS
  stat = cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);
  cublas_check(stat);    
    
  // cuda cleanup
  cudaFree(d_a);
  CUDA_CHECK;

  cudaFree(d_b);
  CUDA_CHECK;

  // cublas cleanup
  cublasDestroy(handle);
    
  // print result
  cout << "GPU using CUBLAS library:"<<endl;
  for(int i=0; i<n; i++) cout << i << ": " << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << endl;
  cout << endl;

  // free CPU arrays
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
}



