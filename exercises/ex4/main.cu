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

// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070


#include "helper.h"
#include <iostream>
using namespace std;

// uncomment to use the camera
//#define CAMERA

__device__ void img_thresholding(float *d_imgIn, float *d_imgOut, size_t w, size_t h, size_t nc, float thresh) {
  
  // get the pixel id
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  
  size_t nr_pixels = w * h;

  if (x > w-1 || y > h-1)
    return;

  int pxid = x + y*w;
  float val = 0.0f;
    
  for(int i=0; i < nc; ++i)
    val += d_imgIn[pxid + i * nr_pixels];

  val /= nc;
    
  if (val >= thresh) {
    // red channel
    d_imgOut[pxid + (0 * nr_pixels)] = 1.0f;
    // green channel
    d_imgOut[pxid + (1 * nr_pixels)] = 0.0f;
    // blue channel
    d_imgOut[pxid + (2 * nr_pixels)] = 0.0f;    
  }
  else {
    // red channel
    d_imgOut[pxid + (0 * nr_pixels)] = 0.0f;
    // green channel
    d_imgOut[pxid + (1 * nr_pixels)] = 0.3f;
    // blue channel
    d_imgOut[pxid + (2 * nr_pixels)] = 0.7f;        
  }
}

__global__ void kernel_call(float *d_imgIn, float *d_imgOut, size_t w, size_t h, size_t nc, float thresh) {
  img_thresholding(d_imgIn, d_imgOut, w, h, nc, thresh);
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
    // get the threshold parameter from the command line
    float thresh = 0.13f;
    getParam("thresh", thresh, argc, argv);

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
    size_t w = static_cast<size_t>(mIn.cols);         // width
    size_t h = static_cast<size_t>(mIn.rows);         // height
    size_t nc = static_cast<size_t>(mIn.channels());  // number of channels
    cout << "image: " << w << " x: " << h << endl;
    
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
    size_t size_input_image = (size_t)w*h*nc;
    float *imgIn  = new float[size_input_image];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    size_t size_output_image = (size_t)w*h*mOut.channels();
    float *imgOut = new float[size_output_image];

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

    float *time_with_memory_calls = new float[repeats];
    float *time_gpu_only = new float[repeats];    
    
    for (int i=0; i < repeats; ++i) {
      Timer timer_memory_calls, timer_gpu_only;
      timer_memory_calls.start();
      
      // ###
      // ###
      // ### Main computation
      // ###
      // ###
      // create array for cuda
      float *d_imgIn, *d_imgOut;

      size_t nr_bytes_input_image = size_input_image * sizeof(float);
      cudaMalloc(&d_imgIn, nr_bytes_input_image);
      CUDA_CHECK;

      size_t nr_bytes_output_image = size_output_image * sizeof(float);
      cudaMalloc(&d_imgOut, nr_bytes_output_image);
      CUDA_CHECK;

      cudaMemset(d_imgOut, 0, nr_bytes_output_image);
      CUDA_CHECK;

      // copy layered images to device
      cudaMemcpy(d_imgIn, imgIn, nr_bytes_input_image, cudaMemcpyHostToDevice);
      CUDA_CHECK;

      timer_gpu_only.start();

      // compute the appropiate dimensions for the grid/block
      dim3 block_size = dim3(32, 4, 1);
      dim3 grid_size = dim3((w + block_size.x - 1) / block_size.x, (h + block_size.y - 1) / block_size.y, 1);
    
      kernel_call<<<grid_size, block_size>>>(d_imgIn, d_imgOut, w, h, nc, thresh);
      CUDA_CHECK;
      cudaDeviceSynchronize();

      timer_gpu_only.end();
      // save time in seconds
      time_gpu_only[i] = timer_gpu_only.get();
      
      // copy back from cuda memory
      cudaMemcpy(imgOut, d_imgOut, nr_bytes_output_image, cudaMemcpyDeviceToHost);
      CUDA_CHECK;

      // cudaFree
      cudaFree(d_imgIn);
      CUDA_CHECK;

      cudaFree(d_imgOut);
      CUDA_CHECK;

      timer_memory_calls.end();
      time_with_memory_calls[i] = timer_memory_calls.get();  // elapsed time in seconds
    }

    cout << "avg time gpu: " << calc_average_time(time_with_memory_calls, repeats)*1000 << " ms" << endl;
    cout << "avg time gpu alloc free: " << calc_average_time(time_gpu_only, repeats)*1000 << " ms" << endl;

    delete[] time_with_memory_calls;
    delete[] time_gpu_only;

    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

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

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



