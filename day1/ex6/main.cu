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

//ex1
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

    //std::cout<<"kernel:"<<std::endl;
    //std::cout<<kernel<<std::endl;

    return kernel;
}

//ex2
void imagesc(std::string name, cv::Mat mat){
    double min,max;
    cv::minMaxLoc(mat,&min,&max);
    cv::Mat  kernel_prime = mat/max;
    showImage(name, kernel_prime, 50,50);
}

//ex3
cv::Mat convolution(cv::Mat k, cv::Mat u){
  // width and height image
    int w=u.cols;
    int h=u.rows;

    // width and height kernel
    int wk=k.cols;
    int hk=k.rows;

    int rx=wk/2;
    int ry=hk/2;

    cv::Mat out(h,w,u.type());

    // loop over all pixels
    for (int x = 0; x < w; ++x)
      {
	for (int y = 0; y < h; ++y)
	  {
	    float val=0;

	    // do convolution for every pixel
	    for (int i = 0; i < wk; ++i)
	      {
		for (int j = 0; j < hk; ++j)
		  {
		    int y_index = y-j+ry;
		    int x_index = x-i+rx;

		    // check indices - do clamping if necessary
		    if (y_index < 0)
		      y_index = 0;
		    else if(y_index >= h)
		      y_index = h-1;

		    if (x_index < 0)
		      x_index = 0;
		    else if (x_index >= w)
		      x_index = w-1;

		    val+=k.at<float>(j,i)*u.at<float>(y_index,x_index);			    		
		  }
	      }
	    out.at<float>(y,x)=val;
	  }
      }
    return out;
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
    cv::Mat mOut4(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
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

    size_t n_OneChannel = (size_t)w*h*1;
    float *imgLaplacian = new float[n_OneChannel]; //only one channel






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

    float sigma = 4.0f;
    cv::Mat k=kernel(sigma);
    
    imagesc("Kernel", k);


    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    std::cout<<"after showing input image"<<std::cout;


    cv::Mat blurred=convolution(k,mIn);

    // show output image: first convert to interleaved opencv format from the layered raw array
    showImage("Blurred", blurred, 100+w+40, 100);

    std::cout<<"after showing blurred image"<<std::cout;

    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    /*convert_mat_to_layered (imgIn, mIn);

	//GPU:
	
	float *d_imgIn, *d_v2, *d_v1, *d_divergence, *d_laplacian;

	cudaMalloc(&d_imgIn, n * sizeof(float) );CUDA_CHECK;
	cudaMemcpy(d_imgIn, imgIn, n * sizeof(float), cudaMemcpyHostToDevice);CUDA_CHECK;

    cudaMalloc(&d_v1, n * sizeof(float) ); CUDA_CHECK;
	cudaMalloc(&d_v2, n * sizeof(float) ); CUDA_CHECK;
    cudaMalloc(&d_divergence, n * sizeof(float) ); CUDA_CHECK;

    cudaMalloc(&d_laplacian, n_OneChannel * sizeof(float)); CUDA_CHECK; //notice: only one channel


	dim3 block = dim3(32,8,1);
	dim3 grid = dim3((w + block.x - 1 ) / block.x,(h + block.y - 1 ) / block.y, 1);

    cout <<"grids: "<< grid.x<< "x" <<grid.y<<endl;
	
    gradient <<<grid,block>>> (d_imgIn, d_v1, d_v2, w, h, nc);CUDA_CHECK;
	cudaDeviceSynchronize();CUDA_CHECK;
    divergence<<<grid,block>>> (d_v1, d_v2, d_divergence, w, h, nc);CUDA_CHECK;
    cudaDeviceSynchronize();CUDA_CHECK;
    l2norm<<<grid,block>>> (d_divergence, d_laplacian, w, h, nc);CUDA_CHECK;
    cudaDeviceSynchronize();CUDA_CHECK;


	cudaMemcpy(imgOut, d_v1, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(imgOut2, d_v2, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(imgDivergence, d_divergence, n * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(imgLaplacian, d_laplacian, n_OneChannel * sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;

	cudaFree(d_v1);CUDA_CHECK;
    cudaFree(d_v2);CUDA_CHECK;
    cudaFree(d_divergence);CUDA_CHECK;
	cudaFree(d_imgIn);CUDA_CHECK;
    cudaFree(d_laplacian);CUDA_CHECK;
*/




/*
    convert_layered_to_mat(mOut2, imgOut2);
    showImage("Gradient_Y", mOut2, 100+2*w+40, 100);

    convert_layered_to_mat(mOut3, imgDivergence);
    showImage("Divergence", mOut3, 100+3*w+40, 100);

    convert_layered_to_mat(mOut4, imgLaplacian);
    showImage("Laplacian", mOut4, 100+4*w+40, 100);
    */

    // ### Display your own output images here as needed

#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif



/*
    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;
    delete[] imgDivergence;
    delete[] imgOut2;
    */

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



