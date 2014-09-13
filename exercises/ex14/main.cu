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

#include "common_kernels.cuh"
#include "opencv_helpers.h"

using namespace std;

__host__ __device__ void g_hat_diffusivity(float s, float* g, int functionType, float eps){
    if(functionType == 1) //G_CONSTANT_1){
        (*g) = 1.0f;
    else if(functionType == 2) //G_INVERSE){
        (*g) = (1.0f / max(eps,s)); //eps 
    else if(functionType == 3) //G_EXP){
        (*g)= (expf( -powf(s,2.0) / eps ) / eps); //eps
    else
        (*g)=1.0;
}

__global__ void computeDiffusivity(float *d_dx, float *d_dy, float *g, int w, int h, int nc, float eps) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= w || y >= h)
    return;

  float s=0;
  for (int i = 0; i < nc; ++i)
    {
      int ind=x+ y*w +i*w*h;
      float dxu=d_dx[ind];
      s+=dxu*dxu;
      float dyu=d_dx[ind];
      s+=dyu*dyu;
    }
  s=sqrtf(s); //we dont use the squared norm, so take root  

  // now compute diffusivity
  float g_cur;
  g_hat_diffusivity(s, &g_cur, 2, eps);
  g[x + y*w] = g_cur;
}

// f is input image
// u is current iteration step
// ui is the resulting iteration
__global__ void computeUpdateStep(float *f, float *u, float *g, float *ui, int w, int h, int nc, float lambda) {
  // compute gr, gl, gu and gd
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (x >= w || y >= h)
    return;

  float gr = (x+1 < w) ? g[x + y*w] : 0.0f;
  float gl = (x > 0) ? g[(x-1) + y*w] : 0.0f;
  float gu = (y+1 < h) ? g[x + y*w] : 0.0f;
  float gd = (y > 0) ? g[x + (y-1)*w] : 0.0f;


  // do clamping for u  
  int xPlus1 = min(x+1,w-1);
  int yPlus1 = min(y+1,h-1);
  int xMinus1 = max(x-1,0);
  int yMinus1 = max(y-1,0);

  // update u
  for (int c = 0; c < nc; ++c) {
    u[x + y*w + c*w*h] = (2*f[x + y*w + c*w*h] +
		  lambda * gr * u[xPlus1 + y*w + c*w*h] +
		  lambda * gl * u[xMinus1 + y*w + c*w*h] +
		  lambda * gu * u[x + yPlus1*w + c*w*h] +
		  lambda * gd * u[x + yMinus1*w + c*w*h]) / (2 + lambda * (gr + gl + gu + gd));
  }
}
 
// uncomment to use the camera
//#define CAMERA
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

    float sigma = 0.1;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;

    // int N = 200;    
    int N = 200;
    getParam("N", N, argc, argv);
    cout << "Iterations N: " << N << endl;

    float lambda = 0.5f;
    getParam("lambda", lambda, argc, argv);
    cout << "Lambda : " << lambda << endl;

    float eps = 0.01;
    getParam("eps", eps, argc, argv);
    cout << "eps: " << eps << endl;

    int delay = 1;
    getParam("delay", delay, argc, argv);
    cout << "Delay: " << delay << endl;

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
    cv::Mat mCur(h,w,mIn.type());
    cv::Mat mDx(h, w, mIn.type());
    cv::Mat mDy(h, w, mIn.type());
    cv::Mat mDiffusivity(h, w, CV_32FC1);

    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed

    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    size_t n = (size_t)w*h*nc;
    float *imgIn  = new float[n];
    float *imgCur  = new float[n];
    float *imgDx = new float[n];
    float *imgDy = new float[n];
    float *imgDiffusivity = new float[w * h];
    
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];

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

    cv::Mat mNoisy = mIn.clone();
    addNoise(mNoisy, sigma);
    float *imgNoisy = new float[n];
    convert_mat_to_layered(imgNoisy, mNoisy);

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    showImage("Noisy", mNoisy, 100+w+40, 100);
    
    // GPU COMPUTATION
    float *d_imgIn, *d_imgOut;
    float *d_imgCur;
    float *d_dx, *d_dy;
    float *d_diffusivity;

    for (int i = 0; i < repeats; ++i) {
      cudaMalloc(&d_imgIn, n * sizeof(float)); CUDA_CHECK;
      cudaMemcpy(d_imgIn, imgNoisy, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

      // u0 = imgIn;
      cudaMalloc(&d_imgCur, n * sizeof(float)); CUDA_CHECK;
      cudaMemcpy(d_imgCur, imgNoisy, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

      cudaMalloc(&d_dx, n * sizeof(float)); CUDA_CHECK;
      cudaMalloc(&d_dy, n * sizeof(float)); CUDA_CHECK;
      
      cudaMalloc(&d_imgOut, n * sizeof(float)); CUDA_CHECK;
      // diffusivity just has one channel
      cudaMalloc(&d_diffusivity, w * h * sizeof(float)); CUDA_CHECK;

      dim3 blockSize(32, 4, 1);
      dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y, 1);
      
      // call the kernels
      for (int j = 0; j < N; ++j) {
	gradient<<<gridSize, blockSize>>>(d_imgCur, d_dx, d_dy, w, h, nc); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;

	// compute diffusivity
	computeDiffusivity<<<gridSize, blockSize>>>(d_dx, d_dy, d_diffusivity, w, h, nc, eps); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;

	// compute update step
	computeUpdateStep<<<gridSize, blockSize>>>(d_imgIn, d_imgCur, d_diffusivity, d_imgCur, w, h, nc, lambda); CUDA_CHECK;
	cudaDeviceSynchronize(); CUDA_CHECK;

	// copy stuff back to display step
	cudaMemcpy(imgCur, d_imgCur, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
	cudaMemcpy(imgDiffusivity, d_diffusivity, w * h * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

	cout << "Iteration: " << j << endl;
	// convert_layered_to_mat(mCur, imgCur);
	// convert_layered_to_mat(mDiffusivity, imgDiffusivity);
	// cout << "Iteration " << j << endl;
	// showImage("u^i", mCur, 100, 100 + h + 40);
	// imagesc("Diffusivity", mDiffusivity, 100+2*w+40, 100);

	// // pause a little bit
	// char key = cv::waitKey(delay);
	// if (static_cast<int>(key) == 27)
	//   break;
      }
      
      // copy stuff back
      cudaMemcpy(imgDx, d_dx, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;
      cudaMemcpy(imgDy, d_dy, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

      cudaMemcpy(imgOut, d_imgCur, n * sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK;

      cudaFree(d_imgIn); CUDA_CHECK;
      cudaFree(d_dx); CUDA_CHECK;
      cudaFree(d_dy); CUDA_CHECK;
      cudaFree(d_diffusivity); CUDA_CHECK;
    }
    

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Result", mOut, 100+2*w+40, 100);

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
    delete[] imgCur;
    delete[] imgNoisy;

    delete[] imgDx;
    delete[] imgDy;
    delete[] imgDiffusivity;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
