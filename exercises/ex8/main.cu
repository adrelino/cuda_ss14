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

// uncomment to use the camera
//#define CAMERA

cv::Mat kernel(float sigma){
    int r = ceil(3*sigma);
    float sigma2=powf(sigma,2);

    cv::Mat kernel(2*r+1,2*r+1,CV_32FC1);

    for (int i = 0; i <= r; ++i)
    {
        for (int j = 0; j <= r; ++j)
        {
            float value=1/(2*M_PI*sigma2) * expf( -( powf(i,2)+powf(j,2) ) / (2*sigma2) );
            kernel.at<float>(r+i,r+j)=value;
            kernel.at<float>(r-i,r+j)=value;
            kernel.at<float>(r+i,r-j)=value;
            kernel.at<float>(r-i,r-j)=value;
        }
    }

    float s = sum(kernel)[0];
    kernel/=s;

    return kernel;
}

// TODO: move it back to __device__
__global__ void convolutionGPU(float *imgIn, float *GK, float *imgOut, int w, int h, int nc, int wk, int hk){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t k = wk;//==hk

    int rx=wk/2;
    int ry=hk/2;


    if(x>=w || y>=h) return; //check for blocks

    for(unsigned int c=0;c<nc;c++) {
        float sum=0;
        for(unsigned int i=0;i<k;i++){
            unsigned int x_new;
            if(x+rx<i) x_new=rx;
            else if(x+rx-i>=w) x_new=w+rx-1;
            else x_new=x+rx-i;
            for(unsigned int j=0;j<k;j++){
                unsigned int y_new;
                if(y+ry<j) y_new=0;
                else if(y+ry-j>=h) y_new=h+ry-1;
                else y_new=y+ry-j;
                sum+=GK[i+j*k]*imgIn[x_new+y_new*w+w*h*c];
                // if(sum<0) cout << "fuck" << endl;
            }
        }
        imgOut[x+w*y+w*h*c]=sum;
    }
}

__global__ void computeSpatialDerivatives(float *d_img, float *d_dx, float *d_dy, int w, int h, int nc) {

  size_t x = threadIdx.x + blockDim.x * blockIdx.x;
  size_t y = threadIdx.y + blockDim.y * blockIdx.y;

  // if outside of image --> return
  if (x > w || y > h)
    return;

  // calc indices
  int xPlus1 = x+1;
  int xMinus1 = x-1;

  int yPlus1 = y+1;
  int yMinus1 = y-1;

  // do clamping
  if (xPlus1 >= w)
    xPlus1 = w-1;
  if (yPlus1 >= h)
    yPlus1 = h-1;

  if (xMinus1 < 0)
    xMinus1 = 0;
  if (yMinus1 < 0)
    yMinus1 = 0;

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

__global__ void calcStructureTensor(float *d_imgIn, float *GK, int w, int h, int nc, float *d_m11, float *d_m12, float *d_m22) {

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
    float sigma = 1.0f;
    getParam("sigma", sigma, argc, argv);
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
    cout << "image: " << w << " x " << h << endl;

    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers

    cv::Mat mSmooth(h,w,mIn.type());
    cv::Mat mImgV1(h, w, mIn.type());
    cv::Mat mImgV2(h, w, mIn.type());

    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    size_t n = (size_t)w*h*nc;
    float *imgIn  = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

    // smoothed image
    float *imgSmooth = new float[(size_t)w*h*mSmooth.channels()];

    // derivatives in x-direction
    float *imgV1 = new float[(size_t)w*h*mImgV1.channels()];

    // derivatives in y-direction
    float *imgV2 = new float[(size_t)w*h*mImgV2.channels()];

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

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);


    // create kernel for smoothing
    cv::Mat k = kernel(sigma);
    int wk = k.cols;
    int hk = k.rows;
    size_t nr_pixels_kernel = wk * hk;

    float *imgKernel = new float[nr_pixels_kernel];
    convert_mat_to_layered(imgKernel, k);

    // GPU computation

    // transfer data to GPU
    float *d_imgIn, *d_imgKernel, *d_imgS, *d_imgV1, *d_imgV2;

    cudaMalloc(&d_imgIn, n * sizeof(float));
    CUDA_CHECK;

    cudaMalloc(&d_imgKernel, nr_pixels_kernel * sizeof(float));
    CUDA_CHECK;

    cudaMalloc(&d_imgS, n * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_imgV1, n * sizeof(float)); CUDA_CHECK;

    cudaMalloc(&d_imgV2, n * sizeof(float)); CUDA_CHECK;

    cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK;

    cudaMemcpy(d_imgKernel, imgKernel, nr_pixels_kernel * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CHECK;

    // prepare grid
    dim3 block_size = dim3(32,4,1);
    dim3 grid_size = dim3((w + block_size.x - 1 ) / block_size.x,(h + block_size.y - 1 ) / block_size.y, 1);

    // first, smooth image using GPU
    convolutionGPU <<<grid_size, block_size>>> (d_imgIn, d_imgKernel, d_imgS, w, h, nc, wk, hk);
    CUDA_CHECK;

    cudaDeviceSynchronize();
    CUDA_CHECK;

    // debug - get the stuff back
    cudaMemcpy(imgSmooth, d_imgS, n * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK;

    // second, create derivatives
    computeSpatialDerivatives<<<grid_size, block_size>>>(d_imgIn, d_imgV1, d_imgV2, w, h, nc);
    CUDA_CHECK;

    cudaDeviceSynchronize();
    CUDA_CHECK;

    // debug - get derivatives back
    cudaMemcpy(imgV1, d_imgV1, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
    cudaMemcpy(imgV2, d_imgV2, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

    // free stuff
    cudaFree(d_imgIn); CUDA_CHECK;
    cudaFree(d_imgS); CUDA_CHECK;
    cudaFree(d_imgKernel); CUDA_CHECK;

    cudaFree(d_imgV1); CUDA_CHECK;
    cudaFree(d_imgV2); CUDA_CHECK;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    convert_layered_to_mat(mSmooth, imgSmooth);
    showImage("Smoothed Image", mSmooth, 100+w+40, 100);

    convert_layered_to_mat(mImgV1, imgV1);
    showImage("x-Derivative", mImgV1, 100+2*w+40, 100);

    convert_layered_to_mat(mImgV2, imgV2);
    showImage("y-Derivative", mImgV2, 100+3*w+40, 100);


#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    delete[] imgKernel;

    delete[] imgSmooth;
    delete[] imgV1;
    delete[] imgV2;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
