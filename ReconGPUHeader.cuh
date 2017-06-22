/********************************************************************************************/
/* ReconGPUHeader.cuh																		*/
/* Copyright 2015, Xintek Inc., All rights reserved											*/
/********************************************************************************************/
#ifndef _RECONGPUHEADER_CUH_
#define _RECONGPUHEADER_CUH_

//Define the textures used by the reconstruction algorithms
texture<float, 2, cudaReadModeElementType> textImage;
texture<float, 2, cudaReadModeElementType> textError;
texture<float, 2, cudaReadModeElementType> textSino;

//Define data buffer
unsigned short * d_Proj;
float * d_Norm;
float * d_Image;
float * d_Image2;
float * d_GradIm;
float * d_Error;
float * d_Sino;
float * d_Pro;
float * d_PriorIm;
float * d_dp;
float * d_dpp;
float * d_alpha;
float * d_DerivGradIm;
float * d_GradNorm;

//Define Cuda arrays
cudaArray * d_Sinogram;

//Define a number of constants
__device__ __constant__ int d_Px;
__device__ __constant__ int d_Py;
__device__ __constant__ int d_Nx;
__device__ __constant__ int d_Ny;
__device__ __constant__ int d_MPx;
__device__ __constant__ int d_MPy;
__device__ __constant__ int d_MNx;
__device__ __constant__ int d_MNy;
__device__ __constant__ float d_HalfPx2;
__device__ __constant__ float d_HalfPy2;
__device__ __constant__ float d_HalfNx2;
__device__ __constant__ float d_HalfNy2;
__device__ __constant__ int d_Nz;
__device__ __constant__ int d_Views;
__device__ __constant__ float d_PitchPx;
__device__ __constant__ float d_PitchPy;
__device__ __constant__ float d_PitchPxInv;
__device__ __constant__ float d_PitchPyInv;
__device__ __constant__ float d_PitchNx;
__device__ __constant__ float d_PitchNy;
__device__ __constant__ float d_PitchNxInv;
__device__ __constant__ float d_PitchNyInv;
__device__ __constant__ float d_PitchNz;
__device__ __constant__ float d_alpharelax;
__device__ __constant__ float d_rmax;
__device__ __constant__ int d_Z_Offset;

#endif
