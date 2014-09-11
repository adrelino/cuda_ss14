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


#ifndef COMMON_KERNELS_H
#define COMMON_KERNELS_H

#include <cuda_runtime.h>

__device__ void gradient(float *imgIn, float *v1, float *v2, int w, int h, int nc);
__device__ void divergence(float *v1, float *v2, float *imgOut, int w, int h, int nc);
__device__ void l2norm(float *imgIn, float *imgOut, int w, int h, int nc);
__device__ void convolutionGPU(float *imgIn, float *kernel, float *imgOut, int w, int h, int nc, int kernelSize);
__device__ void computeSpatialDerivatives(float *d_img, float *d_dx, float *d_dy);
__device__ void createStructureTensor(float *d_dx, float *d_dy, float *d_m11, float *d_m12, float *d_m22);
__device__ void compute_eigenvalues2x2(float a, float b, float c, float d, float *lambda1, float *lambda2);
__device__ void compute_eig(float4 m, float2 *eigval, float4 *eigvec);






#endif  // COMMON_KERNELS_H
