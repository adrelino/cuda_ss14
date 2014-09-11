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


#ifndef OPENCV_HELPERS_H
#define OPENCV_HELPERS_H

#include <opencv2/imgproc/imgproc.hpp>

cv::Mat kernel(float sigma); //gaussian kernel, r=ceil(3*sigma)
cv::Mat kernel(float sigma, int r); //gaussian kernel
void imagesc(std::string name, cv::Mat mat, int x, int y);  //like matlab's imagesc scaled (best for grayscale images)
float GetAverage(float dArray[], int iSize); //gets average of float array





#endif  // OPENCV_HELPERS_H
