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


#include <helper.h>
#include <iostream>
#include <math.h>
#include <stdio.h>

#include <common_kernels.cuh>
#include <opencv_helpers.h>

using namespace std;

__global__ void calcHistogramGlobal(float *imgIn, int* hist, int w, int h, int nc){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= w || y >= h)
        return;

    float k=0;
    for (int c=0; c<nc; c++){
        int i=x+y*w+c*w*h;
        k=imgIn[i]*255.f;
        int ki=lroundf(k)+256*c;
        atomicAdd(&hist[ki], 1);
    }
}

__global__ void calcHistogram(float *imgIn, int* hist, int w, int h, int nc){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int xb = threadIdx.x;
    int yb = threadIdx.y;

    extern __shared__ int s_hist[];

    if(xb < 32 && yb < 8*nc) s_hist[xb+yb*32]=0;

    __syncthreads();

    if (x < w && y < h) {
        float k=0;
        for (int c=0; c<nc; c++){
            int i=x+y*w+c*w*h;
            k=imgIn[i]*255.f;
            int ki=k+256*c;
            atomicAdd(&s_hist[ki], 1);
        }
    }

    __syncthreads();


    if(xb < 32 && yb < 8*nc) atomicAdd(&hist[xb+yb*32],s_hist[xb+yb*32]);
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
    int *hist = new int[256*nc];
    
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    // float *imgOut = new float[(size_t)w*h*mOut.channels()];

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

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // GPU COMPUTATION
    float *d_imgIn;
    int *d_hist, *d_hist2;
    float *tg, *ts;
    ts=(float*)malloc(repeats*sizeof(float));
    tg=(float*)malloc(repeats*sizeof(float));
    
    for (int i = 0; i < repeats; ++i) {
        Timer timerglobal, timershared; 

        cudaMalloc(&d_imgIn, n * sizeof(float)); CUDA_CHECK;
        cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;
        cudaMalloc(&d_hist, 256 * nc * sizeof(int)); CUDA_CHECK;
        cudaMemset(d_hist, 0, 256 * nc * sizeof(int)); CUDA_CHECK;
        cudaMalloc(&d_hist2, 256 * nc * sizeof(int)); CUDA_CHECK;
        cudaMemset(d_hist2, 0, 256 * nc * sizeof(int)); CUDA_CHECK;

        dim3 blockSize(32, 32, 1);
        dim3 gridSize((max(w,32) + blockSize.x - 1) / blockSize.x, (max(h,nc*8) + blockSize.y - 1) / blockSize.y, 1);
        size_t smBytes = 256 * nc * sizeof(int);
        
        timerglobal.start();
        calcHistogramGlobal<<<gridSize, blockSize>>>(d_imgIn, d_hist2, w, h, nc); CUDA_CHECK;
        timerglobal.end();
        tg[i] = timerglobal.get();

        timershared.start();
        calcHistogram<<<gridSize, blockSize, smBytes>>>(d_imgIn, d_hist, w, h, nc); CUDA_CHECK;
        timershared.end(),
        ts[i] = timershared.get();

        cudaDeviceSynchronize(); CUDA_CHECK;

        cudaMemcpy(hist, d_hist, 256 * nc * sizeof(int), cudaMemcpyDeviceToHost); CUDA_CHECK;

        cudaFree(d_imgIn); CUDA_CHECK;
        cudaFree(d_hist); CUDA_CHECK;
    }

    char windowTitle[256];
    
    for(int c=0; c<nc; c++){
        // cout << "channel nr. " << c << ":" << endl;
        // for(int i=256*c; i<256*(c+1); i++){
        //     cout << hist[i] << ", ";
        // }
        // cout << endl;
        int *histc=new int[256];
        std::copy (hist+256*c, hist+256*(c+1), histc);
        sprintf(windowTitle, "Histogram channel %d", c);
        showHistogram256(windowTitle, histc, 100+w, c*200);
    }
    cout << "avg time global memory: " << GetAverage(tg, repeats)*1000 << " ms" << endl;
    cout << "avg time shared memory: " << GetAverage(ts, repeats)*1000 << " ms" << endl;

    // show output image: first convert to interleaved opencv format from the layered raw array
    // convert_layered_to_mat(mOut, imgOut);
    // showImage("Result", mOut, 100+2*w+40, 100);

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif
    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    // cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] hist;
    // delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
