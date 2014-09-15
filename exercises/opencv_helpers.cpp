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

// ### Dennis Mack, dennis.mack@tum.de, p060
// ### Adrian Haarbach, haarbach@in.tum.de, p077
// ### Markus Schlaffer, markus.schlaffer@in.tum.de, p070

#include "opencv_helpers.h"
#include "aux.h"
#include <iostream>

using namespace std;

cv::Mat kernel(float sigma){
    int r = ceil(3*sigma);
    return kernel(sigma, r);
}

cv::Mat kernel(float sigma, int r){
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

void imagesc(std::string title, cv::Mat mat, int x, int y){
    double min,max;
    cv::minMaxLoc(mat,&min,&max);

    int nc=mat.channels();

    cv::Mat mask = cv::Mat(mat != mat);

    cout<<"imagesc: "<< title<<"    nc= "<<nc<<" #nans="<<cv::sum(mask)[0]<< " min="<<min<< " max="<<max<<endl;
    //cout<<mat(cv::Rect(mat.cols/2-2,mat.rows/2-2,4,4))<<endl;

    cv::Mat  kernel_prime = mat/max; //(mat-min)/(max-min);
    cv::minMaxLoc(kernel_prime,&min,&max);
    cout<<"imagesc: "<< title<<"    nc= "<<nc<<" #nans="<<cv::sum(mask)[0]<< " min="<<min<< " max="<<max<<endl;


    showImage(title, kernel_prime, x,y);
}

const int screenW=1600;
const int screenH=1200;
int currentx=0;
int currenty=0;


void imagesc(std::string title, cv::Mat mat){
    if(currentx+mat.cols>screenW){
        cout<<"imagesc, next row"<<endl;
        currentx=0;
        currenty+=mat.rows; //todo, should be rows of mat from last call to this function, this assumes all are same size
    }
    if(currenty+mat.rows>screenH){
        cout<<"imagesc, screen full, starting at top left again"<<endl;
        currentx=0;
        currenty=0;
    }
    imagesc(title,mat,currentx,currenty);
    currentx+=mat.cols;
}

void imagescReset(){
    currentx=0;
    currenty=0; 
}

float GetAverage(float dArray[], int iSize) {
    float dSum = dArray[0];
    for (int i = 1; i < iSize; ++i) {
        dSum += dArray[i];
    }
    return dSum/iSize;
}

void d_imagesc(std::string name,float* d_imgIn, int w, int h, int nc){
    d_imagesc(name,d_imgIn,w,h,nc,false,false);
}

void d_imagesc(std::string name,float* d_imgIn, int w, int h, int nc, bool splitChannels){
    d_imagesc(name,d_imgIn,w,h,nc,splitChannels,false);
};

void d_imagesc(std::string name,float* d_imgIn, int w, int h, int nc, bool splitChannels, bool resize){
    size_t n = (size_t)w*h*nc;
    float *imgIn  = new float[n];

    cudaMemcpy(imgIn, d_imgIn, nc*w*h* sizeof(float), cudaMemcpyDeviceToHost);CUDA_CHECK;
    
    cv::Mat imgOpenCV;
    if(nc==1){
        imgOpenCV = cv::Mat(h,w,CV_32FC1);
    }else if(nc==3){
        imgOpenCV = cv::Mat(h,w,CV_32FC3);
    }

    if(resize){
        cv::Mat dst,tmp;
        tmp=imgOpenCV;
        dst=tmp;
        while(true){
            pyrUp(tmp, dst, cv::Size(tmp.cols*2,tmp.rows*2));
            if(dst.cols>=128){
                imgOpenCV=dst;
                break;
            }
        }
        //cv::resize(imgOpenCV, bigger);
    }

    convert_layered_to_mat(imgOpenCV, imgIn);
    if(splitChannels){
        cv::Mat channel[nc];
        cv::split(imgOpenCV,channel);
        for(int i=0; i<nc; i++){
            std::stringstream ss;
            ss<<name<<" -- Channel : "<<i;
            imagesc(ss.str(), channel[i]);
        }
    }else{
        imagesc(name, imgOpenCV);
    }
    
    delete[] imgIn;
}