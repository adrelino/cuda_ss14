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


#include "aux.h"
#include <iostream>
#include <math.h>
//#include <stdio.h>
using namespace std;

__host__ float sumArrayCPU(float *in, size_t n){
    float result=0;
    for(size_t i=0; i<n; i++){
        result+=in[i];
    }
    return result;
}

__global__ void sumArray(float *input, float *results, size_t n){
    size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i>=n) return;

    results[0]=input[0];
}

void createArray(float* output, size_t length){
    for(size_t i=0;i<length;i++){
        output[i]=i;
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
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    
    float length=10000;
    getParam("length", length, argc, argv);
    cout << "length: " << length << endl;


    float *input, *output;
    input=(float*)malloc(length*sizeof(float));
    output=(float*)malloc(length*sizeof(float));

    
    float *tc, *tg, *tg2;
    tc=(float*)malloc(repeats*sizeof(float));
    tg=(float*)malloc(repeats*sizeof(float));
    tg2=(float*)malloc(repeats*sizeof(float));

    for(int i=0;i<repeats;i++){
	//CPU:
	Timer timercpu, timergpu, timergpu2; timercpu.start();
	
	sumArrayCPU(input,length);
	
	timercpu.end();  
	tc[i] = timercpu.get();  // elapsed time in seconds

	//GPU:
	timergpu.start();
	
	float *d_input, *d_output;
	cudaMalloc(&d_input, length * sizeof(float) );
	CUDA_CHECK;
	cudaMemcpy(d_input, input, length * sizeof(float), cudaMemcpyHostToDevice);
	CUDA_CHECK;
	cudaMalloc(&d_output, length * sizeof(float) );
	CUDA_CHECK;
	
	timergpu2.start();

	dim3 block = dim3(128,1,1);
	dim3 grid = dim3((length + block.x - 1 ) / block.x, 1, 1);
	sumArray <<<grid,block>>> (d_input, d_output, length);
	CUDA_CHECK;
	cudaDeviceSynchronize();
	
	timergpu2.end();
	tg2[i] = timergpu2.get();
	
	CUDA_CHECK;
	cudaMemcpy(output, d_output, length * sizeof(float), cudaMemcpyDeviceToHost);
	CUDA_CHECK;
	cudaFree(d_output);
	CUDA_CHECK;
	cudaFree(d_input);
	CUDA_CHECK;
	
	timergpu.end();  
	tg[i] = timergpu.get();  // elapsed time in seconds
    }
    
    cout << "avg time cpu: " << GetAverage(tc, repeats)*1000 << " ms" << endl;
    cout << "avg time gpu: " << GetAverage(tg, repeats)*1000 << " ms" << endl;
    cout << "avg time gpu allocfree: " << GetAverage(tg2, repeats)*1000 << " ms" << endl;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



