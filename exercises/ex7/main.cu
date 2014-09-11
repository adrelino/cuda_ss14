// ### Adrian's
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
// ### TODO: For every student of your group, please provide here:
// ###
// ### name, email, login username (for example p123)
// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, , p070


#include "aux.h"
#include <iostream>
#include <math.h>
//#include <stdio.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA

typedef struct Params {
    int shw;
    int shh;
    int w;
    int h;
    int nc;
    int r;
} Params;

// create texture for storing input image
texture<float, 2, cudaReadModeElementType> texRef;

// we are moving the kernel into constant memory of GPU
#define MAX_RADIUS 20

const int KERNEL_SIZE = 2*MAX_RADIUS+1;

__constant__ int kernelSize;
__constant__ float kernel[KERNEL_SIZE*KERNEL_SIZE];

cv::Mat createKernel(float sigma){
    int r = ceil(3*sigma);

    if (r > MAX_RADIUS)
      r = MAX_RADIUS;

    float sigma2=powf(sigma,2);

    cv::Mat k = cv::Mat::zeros (KERNEL_SIZE,KERNEL_SIZE,CV_32FC1);

    for (int i = 0; i <= r; ++i)
    {
        for (int j = 0; j <= r; ++j)
        {
            float value=1/(2*M_PI*sigma2) * expf( -( powf(i,2)+powf(j,2) ) / (2*sigma2) );
            k.at<float>(MAX_RADIUS+i,MAX_RADIUS+j)=value;
            k.at<float>(MAX_RADIUS-i,MAX_RADIUS+j)=value;
            k.at<float>(MAX_RADIUS+i,MAX_RADIUS-j)=value;
            k.at<float>(MAX_RADIUS-i,MAX_RADIUS-j)=value;
        }
    }

    float s = sum(k)[0];
    k/=s;

    return k;
}

__global__ void convolutionGlobal(float *imgIn, float *imgOut, Params params){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t k = kernelSize;
    int h=params.h;
    int w=params.w;
    int nc=params.nc;


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

__global__ void convolutionShared(float *imgIn, float *imgOut, Params params){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    extern __shared__ float shmem[];

    int r = MAX_RADIUS;
    int nc=params.nc;
    int w=params.w;
    int h=params.h;
    int shw=params.shw;
    int shh=params.shh;


    int tx=threadIdx.x;
    int bx=blockIdx.x;
    int ty=threadIdx.y;
    int by=blockIdx.y;   
    int bdx=blockDim.x;
    int bdy=blockDim.y;   

    for(unsigned int c=0;c<nc;c++,__syncthreads()) {

        ////////
        //step 1: copy data into shared memory, with clamping padding
        //
        for(int pt=tx+bdx*ty ; pt<shw*shh ;pt+=bdx*bdy){
            int xi = (pt % shw) + (bx *bdx - r);
            int yi = (pt / shw) + (by *bdy - r);

            xi = max(min(xi,w-1),0);
            yi = max(min(yi,h-1),0);

            float val=imgIn[xi + yi*w + c*w*h];

            shmem[pt] = val; 
        }

        __syncthreads();


        ///////
        //step 2: convolution, no more clamping needed
        //
        if(x>=w || y>=h) continue; //check for block border only AFTER copying to shared mem (goes over block borders)

        float sum=0;

        //convolution using adrian + markus indexing
	for(int i=0;i<kernelSize;i++){
	  for(int j=0;j<kernelSize;j++){
	    int x_new=threadIdx.x+i;
	    int y_new=threadIdx.y+j;
	    sum+=kernel[i+j*kernelSize]*shmem[x_new+y_new*shw];
	  }
	}
        imgOut[x+w*y+w*h*c]=sum;
    }
}

__global__ void convolutionTexture(float *imgOut, Params params) {
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t k = kernelSize;

    int h=params.h;
    int w=params.w;
    int nc=params.nc;

    int r=kernelSize/2;

    if(x>=w || y>=h) return; //check for blocks

    for(int c=0;c<nc;c++) {
        float sum=0;
        for(int i=0;i<k;i++){
            int x_new;
            if(x+r<i) x_new=0.5f;
            else if(x+r-i>=w) x_new=w-0.5f;
            else x_new=x+r-i+0.5f;


            for(int j=0;j<k;j++){
                int y_new;
                if(y+r-j<0)
                    y_new=c*h+0.5f;
                else if(y+r-j>=h)
                    y_new=(c+1)*h-0.5f;
                else
                    y_new=y+r-j+c*h+0.5f;

                sum+=kernel[i+j*k]*tex2D(texRef, x_new, y_new);
            }
        }
        imgOut[x+w*y+w*h*c]=sum;
    }
}

__host__ float calc_average_time(float *arr, int n) {
  float cum_sum = 0.0f;
  for (int i = 0; i < n; ++i)
    cum_sum += arr[i];

  return cum_sum / n;
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

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif
    
    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;
    
    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed    

    float sigma=3.0f;
    getParam("sigma", sigma, argc, argv);
    if(sigma<0) sigma=3.0f;
    cout << "sigma: " << sigma << endl;


    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
  	cv::VideoCapture camera(0);
  	if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
  	camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
  	camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;
    
#else
    
    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }
    
#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels

    // Set the output image format
    cv::Mat mShared(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mGlobal(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    cv::Mat mTexture(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    size_t n = (size_t)w*h*nc;
    float *imgIn  = new float[n];
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgShared = new float[n];
    float *imgGlobal = new float[n];
    float *imgTexture = new float[n];

    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
    // Get camera image
    camera >> mIn;
    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
#endif

    // show input image
    showImage("Input image", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered(imgIn, mIn);

    float *imgKernel  = new float[KERNEL_SIZE*KERNEL_SIZE];
    cv::Mat mKernel = createKernel(sigma);
    convert_mat_to_layered(imgKernel,mKernel);

    assert(mKernel.rows == mKernel.cols);

    float *d_imgIn;
    float *d_imgShared, *d_imgGlobal, *d_imgTexture;

    dim3 block = dim3(32,4,1);
    dim3 grid = dim3((w + block.x - 1 ) / block.x,(h + block.y - 1 ) / block.y, 1);

    Params params;
    params.r=MAX_RADIUS;
    params.shw = (block.x + 2*params.r);
    params.shh = (block.y + 2*params.r);
    params.w = w;
    params.h = h;
    params.nc = nc;

    size_t smBytes = params.shw * params.shh * sizeof(float);

    cudaMalloc(&d_imgIn, n * sizeof(float) );CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;

    // copy constants
    cudaMemcpyToSymbol(kernelSize, &KERNEL_SIZE, sizeof(int), 0, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpyToSymbol(kernel, imgKernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), 0, cudaMemcpyHostToDevice); CUDA_CHECK;

    cudaMalloc(&d_imgShared, n * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_imgGlobal, n * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_imgTexture, n * sizeof(float) ); CUDA_CHECK;

    // now set up the texture stuff
    texRef.addressMode[0] = cudaAddressModeClamp;
    texRef.addressMode[1] = cudaAddressModeClamp;
    texRef.filterMode = cudaFilterModePoint;
    texRef.normalized = false;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    cudaBindTexture2D(NULL, &texRef, d_imgIn, &desc, w, nc * h, w * sizeof(d_imgIn[0]));
    CUDA_CHECK;

    // do convolution with shared memory
    float *time_shared = new float[repeats];
    float *time_global = new float[repeats];
    float *time_texture = new float[repeats];
    
    for (int i = 0; i < repeats; ++i) {
      Timer timer;

      timer.start();
      convolutionShared<<<grid,block,smBytes>>> (d_imgIn, d_imgShared, params);CUDA_CHECK;
      timer.end();
      time_shared[i] = timer.get();

      cudaDeviceSynchronize(); CUDA_CHECK;

      timer.start();
      convolutionGlobal<<<grid,block>>>(d_imgIn, d_imgGlobal, params); CUDA_CHECK;
      timer.end();
      time_global[i] = timer.get();

      cudaDeviceSynchronize(); CUDA_CHECK;

      timer.start();
      convolutionTexture<<<grid,block>>>(d_imgTexture, params); CUDA_CHECK;
      timer.end();
      time_texture[i] = timer.get();

      cudaDeviceSynchronize(); CUDA_CHECK;
    }

    cout << "avg time convolution shared: " << calc_average_time(time_shared, repeats) * 1000 << " ms" << endl;
    cout << "avg time convolution global: " << calc_average_time(time_global, repeats) * 1000 << " ms" << endl;
    cout << "avg time convolution texture: " << calc_average_time(time_texture, repeats) * 1000 << " ms" << endl;

    delete[] time_shared;
    delete[] time_global;
    delete[] time_texture;

    cudaMemcpy(imgTexture, d_imgTexture, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(imgGlobal, d_imgGlobal, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(imgShared, d_imgShared, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;    

    // unbind texture
    cudaUnbindTexture(texRef);CUDA_CHECK;

    cudaFree(d_imgIn);CUDA_CHECK;
    cudaFree(d_imgShared);CUDA_CHECK;
    cudaFree(d_imgGlobal);CUDA_CHECK;
    cudaFree(d_imgTexture);CUDA_CHECK;

    convert_layered_to_mat(mShared, imgShared);
    showImage("Convolution Shared Memory", mShared, 100+w+40, 100);

    convert_layered_to_mat(mGlobal, imgGlobal);
    showImage("Convolution Global Memory", mGlobal, 100+2*w+40, 100);

    convert_layered_to_mat(mTexture, imgTexture);
    showImage("Convolution Texture Memory", mTexture, 100+3*w+40, 100);

    // convert_layered_to_mat(mTexture, imgTexture);
    // showImage("Convolution Texture Memory", mTexture, 100+3*w+40, 100);

    //cv::Mat blurred=convolution(k,mIn);
    // show output image: first convert to interleaved opencv format from the layered raw array
    //showImage("Blurred", blurred, 100+w+40, 100);
    //std::cout<<"after showing blurred image"<<std::cout;
    
    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif

    // save input and result
    //cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    //cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgKernel;

    delete[] imgShared;
    delete[] imgTexture;
    delete[] imgGlobal;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
