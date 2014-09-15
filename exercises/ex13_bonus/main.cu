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


//invoce like: ./ex11/main -i ../images/flowers.png -C 2 -delay 1
#include "aux.h"
#include <iostream>
#include <math.h>
#include "common_kernels.cuh"
#include "opencv_helpers.h"
#include <stdio.h>
//#include <stdio.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA


__device__ void cuda_eig(float a, float b, float c, float d, float *L1, float *L2, float *V1_1, float *V1_2, float *V2_1, float *V2_2)
{
    float T = a + d; 
    float D = a*d - b*c;
    float eps = 0.00001;
    
    *L1 = (T/2.0f) + sqrtf(T*T/4.0f - D);
    *L2 = (T/2.0f) - sqrtf(T*T/4.0f - D);

    if (abs(c) > eps)
    {
        if (abs(*L1-d) - abs(c) > 0.0f)
        {
            *V1_1 = (*L1-d)/abs(*L1-d);
            *V1_2 = c/abs(*L1-d);
        }
        else
        {
            *V1_1 = (*L1-d)/abs(c);
            *V1_2 = c/abs(c);
        }
        if (abs(*L2-d) - abs(c) > 0.0f)
        {
            *V2_1 = (*L2-d)/abs(*L2-d);
            *V2_2 = c/abs(*L2-d);
        }
        else
        {
            *V2_1 = (*L2-d)/abs(c);
            *V2_2 = c/abs(c);
        }

    }
    else if (abs(b) > eps)
    {
        if (abs(*L1-a) - abs(b) > 0.0f)
        {
        *V1_1 = b/abs(*L1-a);
        *V1_2 = (*L1-a)/abs(*L1-a);
        }
        else
        {
        *V1_1 = b/abs(b);
        *V1_2 = (*L1-a)/abs(b);

        }

        if (abs(*L2-a) - abs(b) > 0.0f)
        {
        *V2_1 = b/abs(*L2-a);
        *V2_2 = (*L2-a)/abs(*L2-a);
        }
        else
        {
        *V2_1 = b/abs(b);
        *V2_2 = (*L2-a)/abs(b);
        }

    }
    else //if ((abs(c) <= eps) && (abs(b) <= eps))
    {
        *V1_1 = 1.0f;
        *V1_2 = 0.0f;
        *V2_1 = 0.0f;
        *V2_2 = 1.0f;
    } 
    
}


__global__ void diffusion_tensor(float* d_structSmooth, float* d_diffusionTensor, float* d_le1, float* d_le2, int w, int h, float C, float alpha)
{
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x>=w || y>=h) return;

    float L1, L2, V1_1, V1_2, V2_1, V2_2;
    float mu1, mu2;
    float eps = 0.00001;

    float4 m;
    m.x=d_structSmooth[x + w*y]; //m11
    m.y=d_structSmooth[x + w*y + w*h]; //m12
    m.z=m.y;                            //m21 == m12
    m.w=d_structSmooth[x + w*y + w*h*2]; //m22
    

    cuda_eig(m.x, m.y, m.z, m.w, &L1, &L2, &V1_1, &V1_2, &V2_1, &V2_2);

    mu1 = alpha;

    if (abs(L1-L2)<eps)
    {
        mu2 = alpha;
    }
    else
    {
        mu2 = alpha + (1.0f-alpha)*expf(-C/((L1-L2)*(L1-L2)));
    }

    d_diffusionTensor[x + y*w] = mu1*V1_1*V1_1 + mu2*V2_1*V2_1;
    d_diffusionTensor[x + y*w + w*h] = mu1*V1_1*V1_2 + mu2*V2_1*V2_2; 
    d_diffusionTensor[x + y*w + w*h*2] = mu1*V1_2*V1_2 + mu2*V2_2*V2_2;

}

//13.1
__global__ void diffusionTensorFromStructureTensor(float* d_structSmooth, float* d_diffusionTensor,float* d_le1, float* d_le2, int w, int h, float C, float alpha){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x>=w || y>=h) return;

    //b)
    float4 m;
    m.x=d_structSmooth[x + w*y]; //m11
    m.y=d_structSmooth[x + w*y + w*h]; //m12
    m.z=m.y;                            //m21 == m12
    m.w=d_structSmooth[x + w*y + w*h*2]; //m22

    //if(threadIdx.x==50 && threadIdx.y==50){
        //std::cout<<"matrix:"<<m.x<<" "<<m.y<<" "<<m.z<<" "<<m.w<<std::endl;
    //}
    
    float lambda1,lambda2;
    float2 e1,e2;
    compute_eig(m, &lambda1, &lambda2, &e1, &e2);

    d_le1[x + y*w] = lambda1;
    d_le1[x + y*w + w*h] = e1.x; //==G.z ??
    d_le1[x + y*w + w*h*2] = e1.y;

    d_le2[x + y*w] = lambda2;
    d_le2[x + y*w + w*h] = e2.x; //==G.z ??
    d_le2[x + y*w + w*h*2] = e2.y;

    //c)
    float4 G = calcG(lambda1,lambda2,e1,e2,C,alpha);

    d_diffusionTensor[x + y*w] = G.x;
    d_diffusionTensor[x + y*w + w*h] = G.y; //==G.z ??
    d_diffusionTensor[x + y*w + w*h*2] = G.w;
}

//result stored again in v1,v2
__global__ void diffusivity(float* v1, float* v2, float* d_diffusionTensor, int w, int h, int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x>=w || y>=h) return;

    float4 G;
    G.x = d_diffusionTensor[x + y*w];
    G.y = G.z = d_diffusionTensor[x + y*w + w*h];
    G.w = d_diffusionTensor[x + y*w + 2*w*h];

    for (int i = 0; i < nc; ++i)
    {
        float2 nabla_u;
        nabla_u.x=v1[x+ y*w +i*w*h];
        nabla_u.y=v2[x+ y*w +i*w*h];

        float2 vec= G * nabla_u; //matrix -> vector product

        //store result again in gradient
        v1[x+ y*w +i*w*h]=vec.x;
        v2[x+ y*w +i*w*h]=vec.y; 
    }
}


//the update step for the image u_n -> u_n+1
__global__ void update(float tau, float *u_n, float *div, int w, int h, int nc){
    size_t x = threadIdx.x + blockDim.x * blockIdx.x;
    size_t y = threadIdx.y + blockDim.y * blockIdx.y;

    if(x>=w || y>=h) return;

    for (int i = 0; i < nc; ++i){
        u_n[x+ y*w +i*w*h]=u_n[x+ y*w +i*w*h]+tau*div[x+ y*w +i*w*h];
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
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-N <N>] [-tau <tau>] [-delay <delay>] [-C <1,2,3>] [-gray]" << endl; return 1; }
#endif

    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed

    //iteration steps on CPU 
    int N = 100;
    getParam("N", N, argc, argv);
    cout << "N: " << N <<"  [CPU iterations] "<<endl;

    float tau = 0.15;
    getParam("tau", tau, argc, argv);
    cout << "tau: " << tau << endl;

    float sigma = 0.5f;
    getParam("sigma", sigma, argc, argv);
    cout << "sigma: " << sigma << endl;

    float rho = 3.0f;
    getParam("rho", rho, argc, argv);
    cout << "rho: " << rho << endl;

    int delay = 0;
    getParam("delay", delay, argc, argv);
    cout << "delay: " << delay << " ms"<<"    [use -delay 0 to step with keys]"<<endl;

    float C = 5e-6f;
    getParam("C", C, argc, argv);
    cout << "C: " << C <<endl;

    float alpha=0.01f; //define alpha
    getParam("alpha", alpha, argc, argv);
    cout << "alpha: " << alpha << endl;


    //check if tau is not too large;
    float tauMax=0.25f;

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

    cv::Mat G_sigma=kernel(sigma);
    //imagesc("Kernel sigma", G_sigma, 100, 200);
    float *imgKernel_sigma  = new float[G_sigma.rows * G_sigma.cols];
    convert_mat_to_layered(imgKernel_sigma,G_sigma);

    cv::Mat G_rho=kernel(rho);
    //imagesc("Kernel rho", G_rho, 100, 200);
    float *imgKernel_rho  = new float[G_rho.rows * G_rho.cols];
    convert_mat_to_layered(imgKernel_rho,G_rho);


    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    size_t n = (size_t)w*h*nc;
    float *imgIn  = new float[n];

    size_t n3 = (size_t)w*h*3;


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

	float *d_imgIn, *d_v2, *d_v1, *d_divergence, *d_struct, *d_structSmooth, *d_imgKernel_sigma, *d_imgS, *d_imgKernel_rho, *d_diffusionTensor, *d_le1, *d_le2;

	cudaMalloc(&d_imgIn, n * sizeof(float) );CUDA_CHECK;
	cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;

    cudaMalloc(&d_imgS, n * sizeof(float) );CUDA_CHECK;
    cudaMemset(d_imgS, 0, n * sizeof(float) );CUDA_CHECK;
    cudaMalloc(&d_v1, n * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_v1, 0, n * sizeof(float) );CUDA_CHECK;
	cudaMalloc(&d_v2, n * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_v2, 0, n * sizeof(float) );CUDA_CHECK;    
    cudaMalloc(&d_divergence, n * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_divergence, 0,  n * sizeof(float) ); CUDA_CHECK;



    cudaMalloc(&d_struct, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_struct, 0, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_structSmooth, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_structSmooth, 0, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_diffusionTensor, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_diffusionTensor, 0,  n3 * sizeof(float) ); CUDA_CHECK;


    cudaMalloc(&d_le1, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_le1, 0, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_le2, n3 * sizeof(float) ); CUDA_CHECK;
    cudaMemset(d_le2, 0,  n3 * sizeof(float) ); CUDA_CHECK;


    cudaMalloc(&d_imgKernel_sigma, (size_t) G_sigma.cols * G_sigma.rows * sizeof(float) ); CUDA_CHECK;
    cudaMemcpy(d_imgKernel_sigma, imgKernel_sigma, G_sigma.cols * G_sigma.rows * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
    //d_imagesc("d_imgKernel_sigma",d_imgKernel_sigma,G_sigma.cols,G_sigma.rows,1,false,true);

    cudaMalloc(&d_imgKernel_rho, (size_t) G_rho.cols * G_rho.rows * sizeof(float) );CUDA_CHECK;
    cudaMemcpy(d_imgKernel_rho, imgKernel_rho, (size_t) G_rho.cols * G_rho.rows * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;
    //d_imagesc("d_imgKernel_rho",d_imgKernel_rho,G_rho.cols,G_rho.rows,1,false,true);

	dim3 block = dim3(32,8,1);
	dim3 grid = dim3((w + block.x - 1 ) / block.x,(h + block.y - 1 ) / block.y, 1);

    bool isRunning=true;

            //presmooth input image with sigma
        convolutionGPU<<<grid, block>>>(d_imgIn, d_imgKernel_sigma, d_imgS, w, h, nc, G_sigma.cols); CUDA_CHECK;
        cudaDeviceSynchronize();CUDA_CHECK;

        //d_imagesc("d_imgS",d_imgS, w, h, nc);
        //cv::waitKey(0);

        computeSpatialDerivatives<<<grid, block>>>(d_imgS,d_v1,d_v2, w, h, nc);
        cudaDeviceSynchronize();CUDA_CHECK;

        //d_imagesc("d_v1",d_v1, w, h, nc);
        //d_imagesc("d_v2",d_v1, w, h, nc);
        //cv::waitKey(0);
        
        //a)
        createStructureTensorLayered<<<grid, block>>>(d_v1,d_v2,d_struct, w, h, nc); //createStructureTensorLayered
        cudaDeviceSynchronize();CUDA_CHECK;

        //d_imagesc("d_struct",d_struct, w, h, nc, true);
        //d_imagesc("d_imgKernel_rho",d_imgKernel_rho,G_rho.cols,G_rho.rows,1,false,true);
        //cv::waitKey(0);

        //postsmooth structure tensor with rho
        convolutionGPU<<<grid, block>>>(d_struct,d_imgKernel_rho, d_structSmooth, w, h, nc, G_rho.cols); CUDA_CHECK;
        cudaDeviceSynchronize();CUDA_CHECK;

        d_imagesc("d_structSmooth",d_structSmooth, w, h, nc, true);
        cv::waitKey(0);

        //b and c)
        diffusion_tensor /*diffusionTensorFromStructureTensor*/ <<<grid, block>>>(d_structSmooth, d_diffusionTensor, d_le1, d_le2, w, h, C, alpha); 
        cudaDeviceSynchronize();CUDA_CHECK;

        //d_imagesc("d_le1",d_le1, w, h, nc, true);
        //d_imagesc("d_le2",d_le2, w, h, nc, true);

        d_imagesc("d_diffusionTensor",d_diffusionTensor, w, h, nc, true);
        cv::waitKey(0);


	for (; i < N && isRunning; ++i)
    {
        timergpu.start();
        imagescReset();


        //now use normal gradient, not rotational invariant one;
        gradient<<<grid, block>>>(d_imgIn,d_v1,d_v2, w, h, nc);
        cudaDeviceSynchronize();CUDA_CHECK;

        diffusivity<<<grid,block>>>(d_v1,d_v2,d_diffusionTensor,w,h,nc);
        cudaDeviceSynchronize();CUDA_CHECK;

        //d_imagesc("d_v1-diffusivity",d_v1, w, h, nc);
        //d_imagesc("d_v2-diffusivity",d_v1, w, h, nc);
        //cv::waitKey(0);

        divergence<<<grid,block>>>(d_v1,d_v2,d_divergence, w, h, nc);
        cudaDeviceSynchronize();CUDA_CHECK;

        //d_imagesc("d_divergence",d_divergence, w, h, nc);
        //cv::waitKey(0);

        update<<<grid,block>>>(tau,d_imgIn,d_divergence,w,h,nc);
        cudaDeviceSynchronize();CUDA_CHECK;

        timergpu.end();
        tg[i] = timergpu.get();


        cout<<"iteration: "<<i<<endl;
        /*char key=cv::waitKey(delay);
        int keyN=key;
        //cout<<"-----------"<<key<<"    "<<keyN<<endl;
        if(keyN == 27 || key == 'q' || key == 'Q'){
            cout<<"leaving iteration loop at i: "<<i<<"   total iterations: "<<i<<endl;
            isRunning=false;
        }*/
    }

    d_imagesc("d_imgIn",d_imgIn, w, h, nc);


    cudaFree(d_imgIn);CUDA_CHECK;
    cudaFree(d_imgS);CUDA_CHECK;
	cudaFree(d_v1);CUDA_CHECK;
    cudaFree(d_v2);CUDA_CHECK;
    cudaFree(d_struct);CUDA_CHECK;
    cudaFree(d_structSmooth);CUDA_CHECK;
    cudaFree(d_divergence);CUDA_CHECK;
    cudaFree(d_imgKernel_sigma);CUDA_CHECK;
    cudaFree(d_imgKernel_rho);CUDA_CHECK;
    cudaFree(d_diffusionTensor);CUDA_CHECK;



    float ms=GetAverage(tg, i)*1000;
    cout << "avg time for one gpu iteration: "<<ms<<" ms"<<endl;

#ifdef CAMERA
    // end of camera loop
    }
#else

    cout<<"---[pres any key to exit]---"<<endl;
    // wait for key inputs
    cv::waitKey(0);
#endif

    cout<<"exiting"<<endl;

    // free allocated arrays
    delete[] imgIn;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}