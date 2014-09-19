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
// ###
// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070

// USAGE: ./ex15/main -length 100000 -repeats 1000
// createArray ändern um input zu ändern

/* 
bsp ausgabe:

./ex15/main -length 100000 -repeats 1000

repeats: 1000
length: 100000
cpu result: 100000
gpu result: 100000
cublas result: 100000
avg time cpu: 0.61 ms
avg time gpu: 3.46 ms
avg time gpu allocfree: 0.49 ms
avg time cublas: 0.29 ms

*/

#include <helper.h>
#include <iostream>
#include <math.h>
#include "cublas_v2.h"

using namespace std;

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

__host__ float sumArrayCPU(float *in, size_t n){
    float result=0;
    for(size_t i=0; i<n; i++){
        result+=in[i];
    }
    return result;
}

__global__ void sumArray(float *input, float *results, size_t n){
    extern __shared__ float sdata[];
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int tx = threadIdx.x;


    //load input into shared
    float x=0;
    if(i<n) x=input[i];
    sdata[tx]=x;
    __syncthreads();

    if (i>=n) return;

    //reduction
    for(int offset = blockDim.x/2; offset >0; offset >>=1){
        if(tx < offset){
            //add a partial sum upstream to our own
            sdata[tx] += sdata[tx+offset];
        }
        __syncthreads();
    }

    if(tx==0){
        results[blockIdx.x]=sdata[0];
    }
}

void createArray(float* array, size_t length){
    for(size_t i=0;i<length;i++){
        array[i]=1;
    }
}

float GetAverage(float dArray[], int iSize) {
    float dSum = dArray[0];
    for (int i = 1; i < iSize; ++i) {
        dSum += dArray[i];
    }
    return dSum/iSize;
}

int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;

    // ### Define your own parameters here as needed    
    float length=1000;
    getParam("length", length, argc, argv);
    cout << "length: " << length << endl;


    float *input, *output, *middle, *cublasout, cpuout;
    input=(float*)malloc(length*sizeof(float));
    output=(float*)malloc(length*sizeof(float));
    middle=(float*)malloc(length*sizeof(float));
    cublasout=(float*)malloc(length*sizeof(float));


    createArray(input, length);

    
    float *tc, *tg, *tg2, *tcu;
    tc=(float*)malloc(repeats*sizeof(float));
    tg=(float*)malloc(repeats*sizeof(float));
    tg2=(float*)malloc(repeats*sizeof(float));
    tcu=(float*)malloc(repeats*sizeof(float));

    for(int i=0;i<repeats;i++){
//CPU:
        Timer timercpu, timergpu, timergpu2, timercublas; 

        timercpu.start();

        cpuout=sumArrayCPU(input,length);

        timercpu.end();  
        tc[i] = timercpu.get();  

//GPU:
        timergpu.start();

        float *d_input, *d_output, *d_middle, *d_cublas;
        cudaMalloc(&d_input, length * sizeof(float) ); CUDA_CHECK;
        cudaMemcpy(d_input, input, length * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMalloc(&d_output, length * sizeof(float) ); CUDA_CHECK;
        cudaMemset(d_output, 0, length * sizeof(float)); CUDA_CHECK; 
        cudaMalloc(&d_middle, length * sizeof(float) ); CUDA_CHECK;
        cudaMemset(d_middle, 0, length * sizeof(float)); CUDA_CHECK; 
        cudaMalloc(&d_cublas, length * sizeof(float) ); CUDA_CHECK;
        cudaMemset(d_cublas, 0, length * sizeof(float)); CUDA_CHECK; 

        dim3 block = dim3(128,1,1);
        dim3 grid = dim3((length + block.x - 1 ) / block.x, 1, 1);
        size_t smBytes = block.x * sizeof(float);

        timergpu2.start();

        sumArray <<<grid,block, smBytes>>> (d_input, d_middle, length); CUDA_CHECK;
        cudaDeviceSynchronize(); CUDA_CHECK;
        sumArray <<<grid,block, smBytes>>> (d_middle, d_output, length); CUDA_CHECK;
        cudaDeviceSynchronize(); CUDA_CHECK;

        timergpu2.end();
        tg2[i] = timergpu2.get();

//CUBLAS:

        cublasStatus_t stat;
        cublasHandle_t handle;

        stat = cublasCreate(&handle);
        cublas_check(stat);

        timercublas.start();

        stat=cublasSetVector(length, sizeof(*input), input, 1, d_cublas, 1);cublas_check(stat);


        stat = cublasSasum(handle, length, d_cublas, 1, cublasout);cublas_check(stat);

        timercublas.end();
        tcu[i] = timercublas.get();

        cublasDestroy(handle);

        cudaMemcpy(middle, d_middle, length * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaMemcpy(output, d_output, length * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
        cudaFree(d_output); CUDA_CHECK;
        cudaFree(d_input); CUDA_CHECK;
        cudaFree(d_middle); CUDA_CHECK;
        cudaFree(d_cublas); CUDA_CHECK;

        timergpu.end();  
        tg[i] = timergpu.get();  
    }

    cout << "cpu result: " << cpuout << endl;

    // //print blockwise addition
    // for(int i=0;i<length;i++){
    //     if(middle[i]==0) break;
    //     cout << middle[i] << ", ";
    // }
    // cout << endl;
    cout << "gpu result: " << sumArrayCPU(output,length) << endl;
    cout << "cublas result: " << cublasout[0] << endl;

    cout << "avg time cpu: " << GetAverage(tc, repeats)*1000 << " ms" << endl;
    cout << "avg time gpu: " << GetAverage(tg, repeats)*1000 << " ms" << endl;
    cout << "avg time gpu allocfree: " << GetAverage(tg2, repeats)*1000 << " ms" << endl;
    cout << "avg time cublas: " << GetAverage(tcu, repeats)*1000 << " ms" << endl;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



