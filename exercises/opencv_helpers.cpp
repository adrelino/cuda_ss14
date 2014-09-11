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

void imagesc(std::string name, cv::Mat mat, int x, int y){
    double min,max;
    cv::minMaxLoc(mat,&min,&max);
    cv::Mat  kernel_prime = mat/max;
    showImage(name, kernel_prime, x,y);
}

float GetAverage(float dArray[], int iSize) {
    float dSum = dArray[0];
    for (int i = 1; i < iSize; ++i) {
        dSum += dArray[i];
    }
    return dSum/iSize;
}