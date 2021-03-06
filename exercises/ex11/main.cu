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

// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070


//invoce like: ./ex11/main -i ../images/flowers.png -functionType 2 -delay 1
#include <helper.h>
#include <iostream>
#include <math.h>
//#include <stdio.h>
using namespace std;
#include <opencv_helpers.h>
#include <common_kernels.cuh>

// uncomment to use the camera
//#define CAMERA

//the update step for the image u_n -> u_n+1
__global__ void update(float tau, float *u_n, float *div, int w, int h, int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x>=w || y>=h) return;

    for (int i = 0; i < nc; ++i){
        u_n[x+ y*w +i*w*h]+=tau*div[x+ y*w +i*w*h];
    }
}

//compute the diffusivity value
__host__ __device__ void g_hat_diffusivity(float s, float* g, int functionType, float eps){
    if(functionType == 1) //G_CONSTANT_1){
        (*g) = 1.0f;
    else if(functionType == 2) //G_INVERSE){
        (*g) = (1.0f / max(eps,s)); //eps 
    else if(functionType == 3) //G_EXP){
        (*g)= (expf( -powf(s,2.0f) / eps ) / eps); //eps
    else
        (*g)=1.0f;
}

//step 2 and 3 combined in one kernel. stores the result back into v1 and v2
__global__ void diffusivity(float* v1, float* v2, int w, int h, int nc, int functionType, float eps){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x>=w || y>=h) return;

    //calculates the frobenius norm of 1*2 (gray) or 3*2 (color) matrices
    float s=0;
    for (int i = 0; i < nc; ++i)
    {
        int ind=x+ y*w +i*w*h;
        float dxu=v1[ind];
        s+=dxu*dxu;
        float dyu=v2[ind];
        s+=dyu*dyu;
    }
    s=sqrtf(s); //we dont use the squared norm, so take root

    float g;
    g_hat_diffusivity(s,&g,functionType,eps);

    for (int i = 0; i < nc; ++i)
    {
        //store result again in v1,v2
        v1[x+ y*w +i*w*h]*=g; 
        v2[x+ y*w +i*w*h]*=g;
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
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-N <N>] [-tau <tau>] [-delay <delay>] [-functionType <1,2,3>] [-gray]" << endl; return 1; }
#endif

    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed
    //iteration steps on CPU 
    int N = 200;
    getParam("N", N, argc, argv);
    cout << "N: " << N <<"  [CPU iterations] "<<endl;

    float tau = 0.2;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    int delay = 5;
    getParam("delay", delay, argc, argv);
    cout << "delay: " << delay << " ms"<<"    [use -delay 0 to step with keys]"<<endl;

    int functionType = 1;
    getParam("functionType", functionType, argc, argv);
    cout << "functionType: " << functionType << "    [ G_CONSTANT_1 = 1   ,   G_INVERSE = 2    ,     G_EXP = 3 ]"<<endl;

    float eps=0.001; //define eps
    getParam("eps", eps, argc, argv);
    cout << "eps: " << eps << endl;

    //check if tau is not too large;
    float g0;
    g_hat_diffusivity(0,&g0, functionType, eps);
    float tauMax=0.25f/g0;

    // enable gaussian comparison
    bool doGaussianComparison = false;
    getParam("gauss", doGaussianComparison, argc, argv);
    cout << "doGaussianComparison: " << doGaussianComparison << endl;

    if(tau>tauMax){
        cout << "tau: " << tau <<"    is to big for convergence, setting tau to 0.25*g(0)         new    tau: "<<tauMax<< endl;
        tau=tauMax;
    }

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
    cv::Mat mOut2(h,w,mIn.type()); 
    cv::Mat mOut3(h,w,mIn.type()); 

    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    cv::Mat mOut4(h,w,mIn.type());    // mOut will be a grayscale image, 1 layer
    cv::Mat mConvolution(h,w,mIn.type());
    // ### Define your own output images here as needed

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
    float *imgOut2 = new float[n];
    float *imgDivergence = new float[n];
    float *imgLaplacian = new float[n]; //only one channel

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

    float *tg;
    tg=(float*)malloc(N*sizeof(float));

	//GPU:
    int i=0;
    Timer timergpu; 

    float *d_imgIn, *d_v2, *d_v1, *d_divergence, *d_imgConvolution;

    cudaMalloc(&d_imgIn, n * sizeof(float) );CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;

    cudaMalloc(&d_v1, n * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_v2, n * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_divergence, n * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_imgConvolution, n * sizeof(float)); CUDA_CHECK;

    dim3 block = dim3(32,4,1);
    dim3 grid = dim3((w + block.x - 1 ) / block.x,(h + block.y - 1 ) / block.y, 1);

    bool isRunning=true;

    // compare to convolution with gaussian with sigma = sqrt(2*tau*total_iterations)

    
    if (doGaussianComparison) {
      cout << "Doing convolution with Gaussian..." << endl;
      float *imgConvolution = new float[n];
      size_t totalIterations = N;
      float sigma = sqrtf(2*tau*totalIterations);

      cv::Mat gaussianKernel = kernel(sigma);
      size_t kernelSize = gaussianKernel.rows * gaussianKernel.cols;

      float *imgKernel = new float[kernelSize];    
      float *d_imgKernel;
      
      convert_mat_to_layered(imgKernel, gaussianKernel);
      cudaMalloc(&d_imgKernel, kernelSize * sizeof(float)); CUDA_CHECK;
      cudaMemcpy(d_imgKernel, imgKernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK;

      // do convolution
      convolutionGPU<<<grid,block>>>(d_imgIn, d_imgKernel, d_imgConvolution, w, h, nc, gaussianKernel.rows); CUDA_CHECK;
    
      cudaDeviceSynchronize();CUDA_CHECK;
      cudaMemcpy(imgConvolution, d_imgConvolution, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
      convert_layered_to_mat(mConvolution, imgConvolution);

      cudaFree(d_imgConvolution);
      cudaFree(d_imgKernel);

      delete[] imgKernel;
      delete[] imgConvolution;      
    }
    
	for (; i < N && isRunning; ++i)
    {
        timergpu.start();
        // gpuEntry<<<grid,block>>> (d_imgIn, d_v1, d_v2, d_divergence, w, h, nc, NN, tau, functionType, eps);CUDA_CHECK;
        gradient<<<grid,block>>> (d_imgIn, d_v1, d_v2, w, h, nc);CUDA_CHECK;cudaDeviceSynchronize();
        diffusivity<<<grid,block>>>(d_v1,d_v2,w,h,nc,functionType,eps);CUDA_CHECK;cudaDeviceSynchronize();
        divergence<<<grid,block>>> (d_v1,d_v2,d_divergence, w, h, nc);CUDA_CHECK;cudaDeviceSynchronize();
        update<<<grid,block>>>(tau,d_imgIn,d_divergence,w,h,nc);CUDA_CHECK;cudaDeviceSynchronize();
        cudaDeviceSynchronize();
        timergpu.end();
        tg[i] = timergpu.get();

        cudaMemcpy(imgLaplacian, d_imgIn, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
        
        convert_layered_to_mat(mOut4, imgLaplacian);
        showImage("u^i", mOut4, 120+40, 100);
        cout<<"iteration: "<<i<<endl;
        char key=cv::waitKey(delay);
        int keyN=key;
        //cout<<"-----------"<<key<<"    "<<keyN<<endl;
        if(keyN == 27){
            cout<<"leaving iteration loop at i: "<<i<<"   total iterations: "<<i<<endl;
            isRunning=false;
        }
    }



    cudaMemcpy(imgOut, d_v1, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(imgOut2, d_v2, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(imgDivergence, d_divergence, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(imgLaplacian, d_imgIn, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;

    cudaFree(d_v1);CUDA_CHECK;
    cudaFree(d_v2);CUDA_CHECK;
    cudaFree(d_divergence);CUDA_CHECK;
    cudaFree(d_imgIn);CUDA_CHECK;

    float ms=GetAverage(tg, i)*1000;
    cout << "avg time for one gpu iteration: "<<ms<<" ms"<<endl;

    // show input image
    showImage("u^0", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    imagesc("Gradient_X", mOut, 100+w+40, 100);

    convert_layered_to_mat(mOut2, imgOut2);
    imagesc("Gradient_Y", mOut2, 100+2*w+40, 100);

    convert_layered_to_mat(mOut3, imgDivergence);
    imagesc("Divergence", mOut3, 100+3*w+40, 100);

    convert_layered_to_mat(mOut4, imgLaplacian);
    std::stringstream ss;
    ss<<"u^" <<i<<", tau:"<<tau;
    showImage(ss.str(), mOut4, 100+4*w+40, 100);

    if (doGaussianComparison) {
      showImage("Convolution with Gaussian", mConvolution, 100+5*w+40, 100);

      cv::Mat diffResults = mOut4 - mConvolution;
      imagesc("Subtracted Gaussian", diffResults, 100+6*w+40, 100);
    }

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else

    cout<<"---[pres any key to exit]---"<<endl;

    // wait for key inputs
    cv::waitKey(0);
#endif

    cout<<"exiting"<<endl;

    // save input and result
    //cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    //cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] imgDivergence;
    delete[] imgOut2;
    


    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}
