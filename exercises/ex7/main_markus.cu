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

float gaussian(float x, float y, float sigma){
    return expf(-(x*x+y*y)/(2*sigma*sigma))/2.0f/M_PI/sigma/sigma;
}

void createKernel(float sigma, float* GK, int k){
    int k2=k/2;

    float sum=0;
    for(int i=0;i<k;i++){
        for(int j=0;j<k;j++){
            float tmp=gaussian(i-k2,j-k2, sigma);
            sum+=tmp;
            GK[i+j*k]=tmp;
        }
    }
    for(int i=0;i<k;i++){
        for(int j=0;j<k;j++){
            GK[i+j*k]/=sum;
        }
    }
}

//                       in             out      out
__global__ void convolutionGPU(float *imgIn, float *imgKernel, float *imgOut, int w, int h, int nc, int k){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    size_t x_block = threadIdx.x;
    size_t y_block = threadIdx.y;

    int r=k/2;

    //NOTE: BLOCK NICHT QUADRATISCH
    int shw=blockDim.x+2*r;
    int shh=blockDim.y+2*r;
    int stepx=shw/blockDim.x+1;
    int stepy=shh/blockDim.y+1;

    int2 topleft;
    topleft.x=blockIdx.x*blockDim.x-r;
    topleft.y=blockIdx.y*blockDim.y-r;

    extern __shared__ float s_data[];

    for(unsigned int c=0;c<nc;c++) {
        //fill shared memory
        for(unsigned int i=0;i<stepx;i++){
            int i_new=topleft.x+x_block*stepx+i;
            // if(i_new<topleft.x) continue;
            // else 
            if(i_new>=topleft.x+shw) continue;
            else if(i_new<0) i_new=0;
            else if(i_new>=w) i_new=w-1;

            for(unsigned int j=0;j<stepy;j++){
                int j_new=topleft.y+y_block*stepy+j;
                // if(j_new<topleft.y) continue;
                // else 
                if(j_new>=topleft.y+shh) continue;
                else if(j_new<0) j_new=0;
                else if(j_new>=h) j_new=h-1;

                s_data[x_block*stepx+i+(y_block*stepy+j)*shw]=imgIn[i_new + w * j_new + h*w*c];
            }
        }
        
        __syncthreads();
        
        if(x>=w || y>=h) continue; 

        //convolution
        float sum=0;

        for(unsigned int i=0;i<k;i++){
            int x_new=x_block+k-i;
            if(x_new>=shw) x_new=shw-1;
            for(unsigned int j=0;j<k;j++){
                int y_new=y_block+k-j;
                if(y_new>=shh) y_new=shh-1;
                sum+=imgKernel[i+j*k]*s_data[x_new+y_new*shw];
            }
        }
        imgOut[x+w*y+w*h*c]=sum;

        // alt: output=input
        // imgOut[x+w*y+w*h*c]=s_data[(x_block+r)+(y_block+r)*shw];

        __syncthreads();
    }
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
    if(sigma<=0) sigma=3.0f;
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
    cout << "image: " << w << " x " << h << " nc="<<nc <<endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers



    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    size_t n = (size_t)w*h*nc;
    float *imgIn  = new float[n];
    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[n];

    // size_t n1 = (size_t)w*h*1;
    // float *imgKernel  = new float[n1];








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

    // cv::Mat M=kernel(sigma);

    // imagesc("Kernel", M);
    size_t r=ceil(3*sigma);
    size_t k=2*r+1;
    size_t k2=k*k;
    float *GK=new float[k2];
    createKernel(sigma, GK, k);

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    std::cout<<"after showing input image"<<std::cout;




    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered(imgIn, mIn);
    // convert_mat_to_layered(imgKernel,M);

	//GPU:

    // int k=M.cols;
    // int hk=k.rows;

	float *d_imgIn, *d_imgOut, *d_imgKernel;
	cudaMalloc(&d_imgIn, n * sizeof(float) );CUDA_CHECK;
	cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMalloc(&d_imgKernel, k*k * sizeof(float) );CUDA_CHECK;
    cudaMemcpy(d_imgKernel, GK, k*k * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMalloc(&d_imgOut, n * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_imgOut, 0, n * sizeof(float)); CUDA_CHECK;

	dim3 block = dim3(32,8,1);
	dim3 grid = dim3((w + block.x - 1 ) / block.x,(h + block.y - 1 ) / block.y, 1);

    cout <<"grids: "<< grid.x<< "x" <<grid.y<<endl;

	size_t smBytes = (block.x+r+r) * (block.y+r+r) * block.z * sizeof(float);

    convolutionGPU <<<grid,block,smBytes>>> (d_imgIn, d_imgKernel, d_imgOut, w, h, nc, k);CUDA_CHECK;
	cudaDeviceSynchronize();CUDA_CHECK;


	cudaMemcpy(imgOut, d_imgOut, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;

	cudaFree(d_imgIn);CUDA_CHECK;
    cudaFree(d_imgOut);CUDA_CHECK;
    cudaFree(d_imgKernel);CUDA_CHECK;

    convert_layered_to_mat(mOut, imgOut);
    showImage("Convolution GPU", mOut, 100+2*w+40, 100);


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
    delete[] imgOut;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



