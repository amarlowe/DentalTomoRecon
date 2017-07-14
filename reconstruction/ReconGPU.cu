/********************************************************************************************/
/* ReconGPU.cu																				*/
/* Copyright 2016, XinRay Inc., All rights reserved											*/
/********************************************************************************************/

/********************************************************************************************/
/* Version: 1.3																				*/
/* Date: November 7, 2016																	*/
/* Author: Brian Gonzales																	*/
/* Project: TomoD IntraOral Tomosynthesis													*/
/********************************************************************************************/

/********************************************************************************************/
/* This software contrains source code provided by NVIDIA Corporation.						*/
/********************************************************************************************/

/********************************************************************************************/
/* Include the general header and a gpu specific header										*/
/********************************************************************************************/
#include "TomoRecon.h"

//TV constants
float eplison = 1e-12f;
float rmax = 0.95f;
float alpha = 0.95f;

/********************************************************************************************/
/* Helper functions																			*/
/********************************************************************************************/

#define ChkErr(x) {												\
	cudaError_t Error = x;										\
	if(Error != cudaSuccess){									\
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(Error) << "\n";	\
		return Tomo_CUDA_err;\
	}\
}

TomoError cuda_assert(const cudaError_t code, const char* const file, const int line) {
	if (code != cudaSuccess) {
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(code) << "\n";
		return Tomo_CUDA_err;
	}
	else return Tomo_OK;
}

TomoError cuda_assert_void(const char* const file, const int line) {
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(code) << "\n";
		return Tomo_CUDA_err;
	}
	else return Tomo_OK;
}

#define PXL_KERNEL_THREADS_PER_BLOCK  256

surface<void, cudaSurfaceType2D> displaySurface;
surface<void, cudaSurfaceType2D> imageSurface;

union pxl_rgbx_24
{
	uint1       b32;

	struct {
		unsigned  r : 8;
		unsigned  g : 8;
		unsigned  b : 8;
		unsigned  na : 8;
	};
};

//Define the textures used by the reconstruction algorithms
texture<float, cudaTextureType2D, cudaReadModeElementType> textImage;
texture<float, cudaTextureType2D, cudaReadModeElementType> textError;
texture<float, cudaTextureType2D, cudaReadModeElementType> textSino;

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

/********************************************************************************************/
/* GPU Function specific functions															*/
/********************************************************************************************/

template<typename T>
__global__ void resizeKernel(T* input, int wIn, int hIn, int wOut, int hOut, double maxVar) {
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % wOut;
	const int y = idx / wOut;

	float scaleWidth = (float)wIn / (float)wOut;
	float scaleHeight = (float)hIn / (float)hOut;
	
	float sum = 0;
	int i = x*scaleWidth;
	int j = y*scaleHeight;
	if (i > 0 && j > 0 && i < wIn && j < hIn)
		sum = input[j*wIn + i];

	sum = sum / maxVar * 255;

	union pxl_rgbx_24 rgbx;
	rgbx.na = 0xFF;
	rgbx.r = sum;
	rgbx.g = sum;
	rgbx.b = sum;

	surf2Dwrite(rgbx.b32,
		displaySurface,
		x * sizeof(rgbx),
		y,
		cudaBoundaryModeZero); // squelches out-of-bound writes
}

__global__ void resizeKernelTex(int wIn, int hIn, int wOut, int hOut, double maxVar, int index) {
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % wOut;
	const int y = idx / wOut;

	float scaleWidth = (float)wIn / (float)wOut;
	float scaleHeight = (float)hIn / (float)hOut;

	float sum = 0;
	int i = x*scaleWidth;
	int j = y*scaleHeight;
	if (i > 0 && j > 0 && i < wIn && j < hIn)
		sum = tex2D(textImage, (float)i + 0.5f, (float)j + 0.5f + (float)(index * hIn));;

	sum = sum / maxVar * 255;

	union pxl_rgbx_24 rgbx;
	rgbx.na = 0xFF;
	rgbx.r = sum;
	rgbx.g = sum;
	rgbx.b = sum;

	surf2Dwrite(rgbx.b32,
		displaySurface,
		x * sizeof(rgbx),
		y,
		cudaBoundaryModeZero); // squelches out-of-bound writes
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to normalize the reconstruction
__global__ void ProjectionNorm(float * Norm, int view, float ex, float ey, float ez){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py)){
		//Define image location
		float dx1 = ((float)i - (d_HalfPx2 - 0.5f)) * d_PitchPx;
		float dy1 = ((float)j - (d_HalfPy2 - 0.5f)) * d_PitchPy;
		float dx2 = ez / sqrtf((dx1 - ex)*(dx1 - ex) + ez*ez);
		float dy2 = ez / sqrtf((dy1 - ey)*(dy1 - ey) + ez*ez);
		float scale = 1.0f / (dx2*dx2);

		//Define image location
		dx1 = (((float)i - (d_HalfPx2)) * d_PitchPx - ex)*d_PitchNxInv;
		dy1 = (((float)j - (d_HalfPy2)) * d_PitchPy - ey)*d_PitchNyInv;
		dx2 = (((float)i - (d_HalfPx2 - 1.0f)) * d_PitchPx - ex) *d_PitchNxInv;
		dy2 = (((float)j - (d_HalfPy2 - 1.0f)) * d_PitchPy - ey)*d_PitchNyInv;

		//Get first x and y location
		float x1 = dx1 + (d_HalfNx2 - 0.5f) + ex * d_PitchNxInv;
		float y1 = dy1 + (d_HalfNy2 - 0.5f) + ey * d_PitchNyInv;
		float x2 = dx2 + (d_HalfNx2 - 0.5f) + ex * d_PitchNxInv;
		float y2 = dy2 + (d_HalfNy2 - 0.5f) + ey * d_PitchNyInv;

		float Pro = 0;

		//Add slice offset to starting offset
		x1 += d_PitchNz * dx1 / ez * d_Z_Offset;
		x2 += d_PitchNz * dx2 / ez * d_Z_Offset;
		y1 += d_PitchNz * dy1 / ez * d_Z_Offset;
		y2 += d_PitchNz * dy2 / ez * d_Z_Offset;
			

		//Project by stepping through the image one slice at a time
		for (int z = 0; z < d_Nz; z++){
			//Get the next n and x
			x1 += d_PitchNz * dx1 / ez;
			x2 += d_PitchNz * dx2 / ez;
			y1 += d_PitchNz * dy1 / ez;
			y2 += d_PitchNz * dy2 / ez;

			//Get the first and last pixels in x and y the ray passes through
			int xx1 = (int)floorf(min(x1, x2));
			int xx2 = (int)ceilf(max(x1, x2));
			int yy1 = (int)floorf(min(y1, y2));
			int yy2 = (int)ceilf(max(y1, y2));

			//Get the length of the ray in the slice in x and y
			float dist = 1.0f / ((x2 - x1)*(y2 - y1));

			//Set the first x value to the first pixel
			float xs = x1;

			//Cycle through pixels x and y and used to calculate projection
			for (int x = xx1; x < xx2; x++){
				float ys = y1;
				float xend = min((float)(x + 1), x2);

				for (int y = yy1; y < yy2; y++) {
					float yend = min((float)(y + 1), y2);

					//Calculate the weight as the overlap in x and y
					float weight = scale * ((xend - xs)*(yend - ys)*dist);
					Pro += weight;

					ys = yend;
				}
				xs = xend;
			}
		}

		Norm[(j + view*d_MPy)*d_MPx + i] = Pro;
	}
}

__global__ void AverageProjectionNorm(float * Norm, float * Norm2){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py)){
		float count = 0;

		//Get the value of the projection over all views
		for (int n = 0; n < d_Views; n++) count += Norm[(j + n*d_MPy)*d_MPx + i];

		//For each view calculate the precent contribution to the total
		for (int n = 0; n < d_Views; n++){
			float val = Norm[(j + n*d_MPy)*d_MPx + i];

			float nVal = 0;

			if (count > 0) nVal = val / count;

			Norm2[(j + n*d_MPy)*d_MPx + i] = nVal;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
//Functions to do initial correction of raw data: log and scatter correction
__global__ void LogCorrectProj(float * Sino, int view, unsigned short *Proj, float MaxVal){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py)){
		//Log correct the sample size
		float sample = (float)Proj[j*d_Px + i];
		float val = logf(MaxVal) - logf(sample);
		if (sample > MaxVal) val = 0.0f;

		Sino[(j + view*d_MPy)*d_MPx + i] = val;
	}
}

__global__ void ApplyGaussianBlurX(float * Sino, float * BlurrX, int view){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py)){
		int N = 18;
		float sigma = -0.5f / (float)(6 * 6);
		float blur = 0;
		float norm = 0;
		//Use a neighborhood of 6 sigma
		for (int n = -N; n <= N; n++){
			if (((n + i) >= 0) && (n + i < d_Px)){
				float weight = __expf((float)(n*n)*sigma);
				norm += weight;
				blur += weight * Sino[(j + view*d_MPy)*d_MPx + i];
			}
		}
		if (norm == 0) norm = 1.0f;
		BlurrX[j*d_MPx + i] = blur / norm;
	}
}

__global__ void ApplyGaussianBlurY(float * BlurrX, float * BlurrY){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py)){
		int N = 18;
		float sigma = -0.5f / (float)(6 * 6);
		float blur = 0;
		float norm = 0;
		//Use a neighborhood of 6 sigma
		for (int n = -N; n <= N; n++){
			if (((n + j) >= 0) && (n + j < d_Ny)){
				float weight = __expf((float)(n*n)*sigma);
				norm += weight;
				blur += weight * BlurrX[j*d_MPx + i];
			}
		}
		if (norm == 0) norm = 1.0f;
		BlurrY[j*d_MPx + i] = blur / norm;
	}
}

__global__ void ScatterCorrect(float * Sino, unsigned short * Proj, float * BlurXY, int view, float MaxVal){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py)){
		//Log correct the sample size
		float sample = (float)Sino[(j + view*d_MPy)*d_MPx + i];
		float blur = (float)BlurXY[j*d_MPx + i];
		float val = sample +0.1f*__expf(blur);
		Sino[(j + view*d_MPy)*d_MPx + i] = val;
		Proj[j*d_Px + i] = (unsigned short)(val / MaxVal * (float)USHRT_MAX );
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//START part of the reconstuction code
__global__ void ProjectImage(float * Sino, float * Norm, float *Image, float *Error, int view, float ex, float ey, float ez){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//within image boudary
	if ((i < d_Px) && (j < d_Py)) {
		//Check to make sure the ray passes through the image
		float NP = Norm[(j + view*d_MPy)*d_MPx + i];
		float err = 0;

		if (NP != 0) {
			//Define image location
			float dx1 = ((float)i - (d_HalfPx2 - 0.5f)) * d_PitchPx;
			float dy1 = ((float)j - (d_HalfPy2 - 0.5f)) * d_PitchPy;
			float dx2 = ez / sqrtf((dx1 - ex)*(dx1 - ex) + ez*ez);
			float dy2 = ez / sqrtf((dy1 - ey)*(dy1 - ey) + ez*ez);
			float scale = 1.0f / (dx2*dy2);

			//Define image location
			dx1 = (((float)i - (d_HalfPx2)) * d_PitchPx - ex)*d_PitchNxInv;
			dy1 = (((float)j - (d_HalfPy2)) * d_PitchPy - ey)*d_PitchNyInv;
			dx2 = (((float)i - (d_HalfPx2 - 1.0f)) * d_PitchPx - ex) *d_PitchNxInv;
			dy2 = (((float)j - (d_HalfPy2 - 1.0f)) * d_PitchPy - ey)*d_PitchNyInv;

			float Pro = 0.0f;
			float count = 0.0f;

			float x1 = dx1 + (d_HalfNx2 - 0.5f) + ex * d_PitchNxInv;
			float y1 = dy1 + (d_HalfNy2 - 0.5f) + ey * d_PitchNyInv;
			float x2 = dx2 + (d_HalfNx2 - 0.5f) + ex * d_PitchNxInv;
			float y2 = dy2 + (d_HalfNy2 - 0.5f) + ey * d_PitchNyInv;

			//Add slice offset
			x1 += ((((float)d_PitchNz)*dx1) / ez) * (float)d_Z_Offset;
			x2 += ((((float)d_PitchNz)*dx2) / ez) * (float)d_Z_Offset;
			y1 += ((((float)d_PitchNz)*dy1) / ez) * (float)d_Z_Offset;
			y2 += ((((float)d_PitchNz)*dy2) / ez) * (float)d_Z_Offset;


			//Step through the image space by slice in z direction
			for (int z = 0; z < d_Nz; z++) {
				//Get the next n and x
				x1 += (((float)d_PitchNz)*dx1) / ez;
				x2 += (((float)d_PitchNz)*dx2) / ez;
				y1 += (((float)d_PitchNz)*dy1) / ez;
				y2 += (((float)d_PitchNz)*dy2) / ez;

				//Get the first and last pixels in x and y the ray passes through
				int xMin = (int)floorf(min(x1, x2));
				int xMax = (int)ceilf(max(x1, x2));
				int yMin = (int)floorf(min(y1, y2));
				int yMax = (int)ceilf(max(y1, y2));

				//Get the length of the ray in the slice in x and y
				float dist = 1.0f / fabsf((x2 - x1)*(y2 - y1));

				//Set the first x value to the first pixel
				float xs = x1;

				//Cycle through pixels x and y and used to calculate projection
				for (int x = xMin; x < xMax; x++) {
					float ys = y1;
					float xend = min((float)(x + 1), x2);

					for (int y = yMin; y < yMax; y++) {
						float yend = min((float)(y + 1), y2);
						//float yend = min(y + 1, yMax);

						//Calculate the weight as the overlap in x and y
						float weight = scale*((xend - xs)*(yend - ys)*dist);

						int nx = min(max(x, 0), d_Nx - 1);
						int ny = min(max(y, 0), d_Ny - 1);
						Pro += tex2D(textImage, nx + 0.5f, ny + 0.5f + z*d_MNy) * weight;
						count += weight;

						ys = yend;
					}//y loop
					xs = xend;
				}//x loop
			}//z loop

			 //If the ray passes through the image region get projection error
			//err = (tex2D(textSino, (float)i + 0.5f, (float)j + 0.5f + view*MPy) - Pro*NP) / (max(count, 1.0f));
			err = (Sino[i + d_MPx*(j + view*d_MPy)] - Pro*NP) / (max(count, 1.0f));
		}//Norm check

		 //Add Calculated error to an error image to back project
		Error[j*d_MPx + i] = err;
	}//image boudary check
}

__global__ void BackProjectError(float * IM, float * IM2, float * error, float beta, int view, float ex, float ey, float ez){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny)){
		//for (int k = 0; k < d_Nz; k++) {
		//int k = 0;
			//Define the direction in z to get r
			float r = (ez) / (((float)(k + d_Z_Offset)*d_PitchNz) + ez);

			//Use r to get detecor x and y
			float dx1 = ex + r * (((float)i - (d_HalfNx2))*d_PitchNx - ex);
			float dy1 = ey + r * (((float)j - (d_HalfNy2))*d_PitchNy - ey);
			float dx2 = ex + r * (((float)i - (d_HalfNx2 - 1.0f))*d_PitchNx - ex);
			float dy2 = ey + r * (((float)j - (d_HalfNy2 - 1.0f))*d_PitchNy - ey);

			//Use detector x and y to get pixels
			float x1 = dx1 * d_PitchPxInv + (d_HalfPx2 - 0.5f);
			float x2 = dx2 * d_PitchPxInv + (d_HalfPx2 - 0.5f);
			float y1 = dy1 * d_PitchPyInv + (d_HalfPy2 - 0.5f);
			float y2 = dy2 * d_PitchPyInv + (d_HalfPy2 - 0.5f);

			//Get the first and last pixels in x and y the ray passes through
			int xMin = max((int)floorf(min(x1, x2)), 0);
			int	xMax = min((int)ceilf(max(x1, x2)), d_Px - 1);
			int yMin = max((int)floorf(min(y1, y2)), 0);
			int yMax = min((int)ceilf(max(y1, y2)), d_Py - 1);

			//Get the length of the ray in the slice in x and y
			float dist = 1.0f / fabsf((x2 - x1)*(y2 - y1));

			//Set a normalization and pixel value to 0
			float N = 0;
			float val = 0.0f;

			//Set the first x value to the first pixel
			float ezz = 1.0f / (ez*ez);
			float xx = (d_HalfPx2 - 0.5f)*d_PitchPx - ex;
			float yy = (d_HalfPy2 - 0.5f)*d_PitchPy - ey;

			float xs = x1;
			//Cycle through pixels x and y and used to calculate projection
			for (int x = xMin; x < xMax; x++){
				//int x = (xMin + xMax) / 2;
				float ys = y1;
				float xend = min((float)(x + 1), x2);

				for (int y = yMin; y < yMax; y++){
					float yend = min((float)(y + 1), y2);

					//Calculate the weight as the overlap in x and y
					float weight = ((xend - xs))*((yend - ys))*dist;

					//Calculate the scaling of a ray from the center of the pixel
					//to the detector
					float cos_alpha = sqrtf(((float)x - xx)*((float)x - xx) + ez*ez);
					float cos_gamma = sqrtf(((float)y - yy)*((float)y - yy) + ez*ez);
					float scale = (cos_alpha*cos_gamma)*ezz * weight;

					//Update the value based on the error scaled and save the scale
					if (x < d_Nx - 1 && y < d_Ny - 1 && x > 0 && y > 0) {
						val += tex2D(textError, (float)x + 0.5f, (float)y + 0.5f) *scale;
						//val += error[x + y*MNx] * scale;
						N += scale;
					}
					ys = yend;
				}//y loop
				xs = xend;
			}//x loop
			//Get the current value of image
			float update = beta*val / N;

			if (N > 0) {
				float uval = IM[(j + k*d_MNy)*d_MNx + i];
				IM[(j + k*d_MNy)*d_MNx + i] = uval + update;
				IM2[(j + k*d_MNy)*d_MNx + i] = update;
			}
			else IM2[(j + k*d_MNy)*d_MNx + i] = -10.0f;
		//}//z loop
	}
	else IM2[(j + k*d_MNy)*d_MNx + i] = -10.0f;
}

__global__ void CorrectEdgesY(float * IM, float * IM2){
	//Define pixel location in x, z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int k = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < d_Nx && k < d_Nz)
	{
		float count1 = 0, count2 = 0;
		int j1 = 0, j2 = 0;

#pragma unroll
		for (int j = 0; j < d_Ny / 4; j++)
		{
			if (IM2[(j + k*d_MNy)*d_MNx + i] != -10.0f) break;
			count1++;
			j1++;
		}

#pragma unroll
		for (int j = 1; j <= d_Ny / 4; j++)
		{
			if (IM2[(d_MNy - j + k*d_MNy)*d_MNx + i] != -10.0f) break;
			count2++;
			j2++;
		}

		if (j1 < d_Ny / 4) {
			float avgVal = 0;
			int count = min(20, j1);
			for (int j = 0; j < count; j++)
				avgVal += IM2[(j1 + 1 + j + k*d_MNy)*d_MNx + i];

			avgVal = avgVal / (float)(count);

			for (int j = 0; j < j1; j++) {
				float val = IM[(j + k*d_MNy)*d_MNx + i] + avgVal;
				IM[(j + k*d_MNy)*d_MNx + i] = val;
			}
		}
		if (j2 < d_Ny / 4) {
			float avgVal = 0;
			int count = min(20, j2);
			for (int j = 1; j <= count; j++)
				avgVal += IM2[(d_MNy - (j + j2) + k*d_MNy)*d_MNx + i];

			avgVal = avgVal / (float)(count);
			for (int j = 1; j <= j2; j++) {
				float val = IM[(d_MNy - j + k*d_MNy)*d_MNx + i] + avgVal;
				IM[(d_MNy - j + k*d_MNy)*d_MNx + i] = val;
			}
		}

		//smooth over edge
		float data[7] = { 0, 0, 0, 0, 0, 0, 0 };
		int nn = 0;
		if (j1 < d_Ny / 4 && j1 > 0) {
			for (int n = j1 - 3; n <= j1 + 3; n++) {
				float val = 0;
				float count = 0;
				for (int m = n - 3; m <= n + 3; m++) {
					if (m >= 0 && m < d_Ny) {
						float weight = __expf(-(fabsf(m - n)) / 1.0f);
						val += weight*IM[(m + k*d_MNy)*d_MNx + i];
						count += weight;
					}
				}
				if (count == 0) count = 1;
				data[nn] = val / count;
				nn++;
			}
			nn = 0;
			for (int n = j1 - 3; n <= j1 + 3; n++) {
				if (n >= 0 && n < d_Ny) IM[(n + k*d_MNy)*d_MNx + i] = data[nn];
				nn++;
			}
		}

		nn = 0;
		if (j2 < d_Ny / 4 && j2 > 0) {
			for (int n = j2 - 3; n <= j2 + 3; n++) {
				float val = 0;
				float count = 0;
				for (int m = n - 3; m <= n + 3; m++) {
					if (m < d_Ny && m >0) {
						float weight = __expf(-(fabsf(m - n)) / 1.0f);
						val += weight*IM[(d_MNy - m + k*d_MNy)*d_MNx + i];
						count += weight;
					}
				}
				if (count == 0) count = 1;
				data[nn] = val / count;
				nn++;
			}
			nn = 0;
			for (int n = j2 - 3; n <= j2 + 3; n++) {
				if (n < d_Ny && n >0) IM[(d_MNy - n + k*d_MNy)*d_MNx + i] = data[nn];
				nn++;
			}
		}
	}
}


__global__ void CorrectEdgesX(float * IM, float * IM2){
	//Define pixel location in x, z
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int k = blockDim.y * blockIdx.y + threadIdx.y;
	if (j < d_Ny && k < d_Nz){
		float count1 = 0, count2 = 0;
		int i1 = 0, i2 = 0;

#pragma unroll
		for (int i = 0; i < d_Nx / 8; i++)
		{
			if (IM2[(j + k*d_MNy)*d_MNx + i] != -10.0f) break;
			count1++;
			i1++;
		}

#pragma unroll
		for (int i = 1; i <= d_Nx / 8; i++)
		{
			if (IM2[(j + k*d_MNy)*d_MNx + d_MNx - i] != -10.0f) break;
			count2++;
			i2++;
		}

		if (i1 < d_Nx / 8) {
			float avgVal = 0;
			int count = min(20, i1);
			for (int i = 0; i < count; i++)
				avgVal += IM2[(j + k*d_MNy)*d_MNx + i + i1 + 1];

			avgVal = avgVal / (float)(count);

			for (int i = 0; i < i1; i++) {
				float val = IM[(j + k*d_MNy)*d_MNx + i] + avgVal;
				IM[(j + k*d_MNy)*d_MNx + i] = val;
			}
		}
	
		if (i2 < d_Nx / 8) {
			float avgVal = 0;
			int count = min(20, i2);
			for (int i = 1; i <= count; i++)
				avgVal += IM2[(j + k*d_MNy)*d_MNx + d_MNx - (i + i2)];

			avgVal = avgVal / (float)(count);
			for (int i = 1; i <= i2; i++) {
				float val = IM[(j + k*d_MNy)*d_MNx + d_MNx - i] + avgVal;
				IM[(j + k*d_MNy)*d_MNx + d_MNx - i] = val;
			}
		}

		//smooth over edge
		float data[7] = { 0, 0, 0, 0, 0, 0, 0 };
		int nn = 0;
		if (i1 < d_Nx / 8 && i1 > 0) {
			for (int n = i1 - 3; n <= i1 + 3; n++) {
				float val = 0;
				float count = 0;
				for (int m = n - 3; m <= n + 3; m++) {
					if (m >= 0 && m < d_Nx) {
						float weight = __expf(-(fabsf(m - n)) / 1.0f);
						val += weight*IM[(j + k*d_MNy)*d_MNx + m];
						count += weight;
					}
				}
				if (count == 0) count = 1;
				data[nn] = val / count;
				nn++;
			}
			nn = 0;
			for (int n = i1 - 3; n <= i1 + 3; n++) {
				if (n >= 0 && n < d_Nx) IM[(j + k*d_MNy)*d_MNx + n] = data[nn];
				nn++;
			}
		}

		nn = 0;
		if (i2 < d_Nx / 8 && i2 > 0) {
			for (int n = i2 - 3; n <= i2 + 3; n++) {
				float val = 0;
				float count = 0;
				for (int m = n - 3; m <= n + 3; m++) {
					if (m < d_Nx && m >0) {
						float weight = __expf(-(fabsf(m - n)) / 1.0f);
						val += weight*IM[(j+ k*d_MNy)*d_MNx + d_MNx - m];
						count += weight;
					}
				}
				if (count == 0) count = 1;
				data[nn] = val / count;
				nn++;
			}
			nn = 0;
			for (int n = i2 - 3; n <= i2 + 3; n++) {
				if (n < d_Nx && n >0) IM[(j + k*d_MNy)*d_MNx + d_MNx - n] = data[nn];
				nn++;
			}
		}
	}
}


__global__ void NormalizeImage(float * IM)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	float val = IM[(j + k*d_MNy)*d_MNx + i];
	IM[(j + k*d_MNy)*d_MNx + i] = __saturatef(val);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Object Segmentation and Artifact reduction
__global__ void SobelEdgeDetection(float * IM)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Restrict sobel edge detection to center only to reduce impact of artifacts
	if ((i < 3*d_Nx/4 ) && (j < 3*d_Ny/4) && (i > d_Nx/4) && (j > d_Ny/4))
	{
		float val1 = tex2D(textImage, i - 0.5f, j + 1.5f + k*d_MNy);
		val1 += 2.0f*tex2D(textImage, i + 0.5f, j + 1.5f + k*d_MNy);
		val1 += tex2D(textImage, i + 1.5f, j + 1.5f + k*d_MNy);
		val1 -= tex2D(textImage, i - 0.5f, j - 0.5f + k*d_MNy);
		val1 -= 2.0f*tex2D(textImage, i + 0.5f, j - 0.5f + k*d_MNy);
		val1 -= tex2D(textImage, i + 1.5f, j - 0.5f + k*d_MNy);

		float val2 = tex2D(textImage, i + 1.5f, j - 0.5f + k*d_MNy);
		val2 += 2.0f*tex2D(textImage, i + 1.5f, j + 0.5f + k*d_MNy);
		val2 += tex2D(textImage, i + 1.5f, j + 1.5f + k*d_MNy);;
		val2 -= tex2D(textImage, i - 0.5f, j - 0.5f + k*d_MNy);
		val2 -= 2.0f*tex2D(textImage, i - 0.5f, j + 0.5f + k*d_MNy);
		val2 -= tex2D(textImage, i - 0.5f, j + 1.5f + k*d_MNy);

		float val = sqrt(val1 * val1 + val2 * val2);
		IM[(j + k*d_MNy)*d_MNx + i] = val;
	}
}


__global__ void SumSobelEdges(float * Image, int sizeIM, float * MaxVal)
{
	//Define shared memory to read all the threads
	extern __shared__ float data[];

	//define the thread and block location
	int thread = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int j = blockIdx.y *blockDim.y + threadIdx.y;
	int gridSize = blockDim.x * 2 * gridDim.x;

	//Declare a sum value and iterat through grid to sum
	float Sum = 0;
	for (int n = i; n + blockDim.x < sizeIM; n += gridSize) {
		float val1 = Image[j*sizeIM + n];
		float val2 = Image[j*sizeIM + n + blockDim.x];
		Sum += val1 + val2;
	}

	//Each thread puts its local sum into shared memory
	data[thread] = Sum;
	__syncthreads();

	//Do reduction in shared memory
	if (thread < 512) { data[thread] = Sum = Sum + data[thread + 512]; }
	__syncthreads();

	if (thread < 256) { data[thread] = Sum = Sum + data[thread + 256]; }
	__syncthreads();

	if (thread < 128) { data[thread] = Sum = Sum + data[thread + 128]; }
	__syncthreads();

	if (thread < 64) { data[thread] = Sum = Sum + data[thread + 64]; }
	__syncthreads();

	if (thread < 32) { data[thread] = Sum = Sum + data[thread + 32]; }
	__syncthreads();

	if (thread < 16) { data[thread] = Sum = Sum + data[thread + 16]; }
	__syncthreads();

	if (thread < 8) { data[thread] = Sum = Sum + data[thread + 8]; }
	__syncthreads();

	if (thread < 4) { data[thread] = Sum = Sum + data[thread + 4]; }
	__syncthreads();

	if (thread < 2) { data[thread] = Sum = Sum + data[thread + 2]; }
	__syncthreads();

	if (thread < 1) { data[thread] = Sum = Sum + data[thread + 1]; }
	__syncthreads();

	//write the result for this block to global memory
	if (thread == 0) {
		MaxVal[j] = data[0];
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
// Total Variation Minization functions
__global__ void DerivOfGradIm(float * Deriv, float ep)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j <d_Ny) && (k < d_Nz))
	{
		//Read all of the image intensities
		float I = tex2D(textImage, i + 0.5f, j + 0.5f + k*d_MNy);
		float Ix1 = I - tex2D(textImage, i - 0.5f, j + 0.5f + k*d_MNy);
		float Iy1 = I - tex2D(textImage, i + 0.5f, j - 0.5f + k*d_MNy);
		float Iz1 = I - tex2D(textImage, i + 0.5f, j + 0.5f + (k - 1)*d_MNy);
		float Ix2 = tex2D(textImage, i + 1.5f, j + 0.5f + k*d_MNy);
		float Iy2 = tex2D(textImage, i + 0.5f, j + 1.5f + k*d_MNy);
		float Iz2 = tex2D(textImage, i + 0.5f, j + 0.5f + (k + 1)*d_MNy);
		float Ixy = Ix2 - tex2D(textImage, i + 1.5f, j - 0.5f + k*d_MNy);
		float Ixz = Ix2 - tex2D(textImage, i + 1.5f, j + 0.5f + (k - 1)*d_MNy);
		float Iyx = Iy2 - tex2D(textImage, i - 0.5f, j + 1.5f + k*d_MNy);
		float Iyz = Iy2 - tex2D(textImage, i + 0.5f, j + 1.5f + (k - 1)*d_MNy);
		float Izx = Iz2 - tex2D(textImage, i - 0.5f, j + 0.5f + (k + 1)*d_MNy);
		float Izy = Iz2 - tex2D(textImage, i + 0.5f, j - 0.5f + (k + 1)*d_MNy);

		//Calculate the four difference ratios
		float TV1 = (2.0f*Ix1 + 2.0f*Iy1 + 2.0f*Iz1) / sqrtf(Ix1*Ix1 + Iy1*Iy1 + Iz1*Iz1 + ep);

		float TV2 = (2.0f*(Ix2 - I)) / sqrtf((Ix2 - I)*(Ix2 - I) + Ixy*Ixy + Ixz*Ixz + ep);

		float TV3 = (2.0f*(Iy2 - I)) / sqrtf((Iy2 - I)*(Iy2 - I) + Iyx*Iyx + Iyz*Iyz + ep);

		float TV4 = (2.0f*(Iz2 - I)) / sqrtf(Izx*Izx + Izy*Izy + (Iz2 - I)*(Iz2 - I) + ep);

		//Set the TV value at the specific pixel location
		Deriv[(j + k*d_MNy)*d_MNx + i] = TV1 - TV2 - TV3 - TV4;
	}
}

__global__ void GetGradIm(float * IM)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j <d_Ny) && (k < d_Nz))
	{
		//Read all of the image intensities
		float I = tex2D(textImage, i + 0.5f, j + 0.5f + k*d_MNy);
		float Ix1 = I - tex2D(textImage, i - 0.5f, j + 0.5f + k*d_MNy);
		float Iy1 = I - tex2D(textImage, i + 0.5f, j - 0.5f + k*d_MNy);
		float Iz1 = I - tex2D(textImage, i + 0.5f, j + 0.5f + (k - 1)*d_MNy);
		float sgn = 1.0f;
		if (Ix1 + Iy1 + Iz1 < 0) sgn = -1.0f;
		//Set the TV value at the specific pixel location
		IM[(j + k*d_MNy)*d_MNx + i] = sgn*sqrtf(Ix1*Ix1 + Iy1*Iy1 + Iz1*Iz1);// / 3.0f;
																			
	}

}

__global__ void UpdateImageEstimate(float * IM, float * TV, float *TV2)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny))
	{
		float val1 = IM[(j + k*d_MNy)*d_MNx + i] -
			0.00025f* (TV[(j + k*d_MNy)*d_MNx + i] + TV2[(j + k*d_MNy)*d_MNx + i]);
		IM[(j + k*d_MNy)*d_MNx + i] =  __saturatef(val1);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to create synthetic projections from reconstructed data
__global__ void CreateSyntheticProjection(unsigned short * Proj, float * Win, float *Im, float MaxVal)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		float Pro = 0;
		float MaxV = 0;
		//Step through the image space by slice in z direction
		//	#pragma unroll
		for (int z = 0; z < d_Nz; z++)
		{
			if (Win[z] > 0.5f) {
				Pro += Win[z] * max((Im[(j + z*d_MNy)*d_MNx + i] - 0.025f), 0.0f);
				MaxV += Win[z];
			}
		}
		Proj[j*d_Px + i] = (unsigned short)(Pro * USHRT_MAX / (MaxV));
	}
}

__global__ void CreateSyntheticProjectionNew(unsigned short * Proj, float* Win, float * StIm, float *Im, float MaxVal, int cz)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		float Pro = 0;
		float MaxV = 0;

		//Step through the image space by sli ce in z direction
		//	#pragma unroll
		float mean = (max(Im[(j + cz*d_MNy)*d_MNx + i] - 0.025f, 0.0f)
			+ max(Im[(j + (cz - 1)*d_MNy)*d_MNx + i] - 0.025f, 0.0f)
			+ max(Im[(j + (cz + 1)*d_MNy)*d_MNx + i] - 0.025f, 0.0f)) / 3.0f;

		float std = StIm[j*d_MNx + i];

		for (int z = 0; z < d_Nz; z++)
		{
			float val = max(Im[(j + z*d_MNy)*d_MNx + i] - 0.025f, 0.0f);
			float diff = (val - mean);

			//float WW = Win[z];
			//if (WW == 0) WW = 1;
			float scale = 1.0f / ((((abs(diff) / std))*((abs(diff) / std))));
			if (diff == 0) scale = 1.0f;

			//Pro += max((val - 0.025f), 0.0f) * scale;
			Pro += val*scale;
			MaxV += scale;
		}
		float update_val = Pro / MaxV;
		if (MaxV == 0) update_val = 0;

		Proj[j*d_Px + i] = (unsigned short)(update_val * USHRT_MAX);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to set the reconstructions to save and copy back to cpu
__global__ void GetMaxImageVal(float * Image, int sizeIM, float * MaxVal)
{
	//Define shared memory to read all the threads
	extern __shared__ float data[];

	//define the thread and block location
	int thread = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int j = blockIdx.y *blockDim.y + threadIdx.y;
	int gridSize = blockDim.x * 2 * gridDim.x;

	//Declare a sum value and iterat through grid to sum
	float val = 0;
	for (int n = i; n + blockDim.x < sizeIM; n += gridSize) {
		float val1 = Image[j*sizeIM + n];
		float val2 = Image[j*sizeIM + n + blockDim.x];
		val = max(val, max(val1, val2));
	}

	//Each thread puts its local sum into shared memory
	data[thread] = val;
	__syncthreads();

	//Do reduction in shared memory
	if (thread < 512) { data[thread] = val = max(val, data[thread + 512]); }
	__syncthreads();

	if (thread < 256) { data[thread] = val = max(val, data[thread + 256]); }
	__syncthreads();

	if (thread < 128) { data[thread] = val = max(val, data[thread + 128]); }
	__syncthreads();

	if (thread < 64) { data[thread] = val = max(val, data[thread + 64]); }
	__syncthreads();

	if (thread < 32) { data[thread] = val = max(val, data[thread + 32]); }
	__syncthreads();

	if (thread < 16) { data[thread] = val = max(val, data[thread + 16]);; }
	__syncthreads();

	if (thread < 8) { data[thread] = val = max(val, data[thread + 8]); }
	__syncthreads();

	if (thread < 4) { data[thread] = val = max(val, data[thread + 4]); }
	__syncthreads();

	if (thread < 2) { data[thread] = val = max(val, data[thread + 2]); }
	__syncthreads();

	if (thread < 1) { data[thread] = val = max(val, data[thread + 1]); }
	__syncthreads();

	//write the result for this block to global memory
	if (thread == 0) {
		MaxVal[j] = data[0];
	}
}

__global__ void CopyImages(unsigned short * ImOut, float * ImIn, float maxVal){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny))
	{
		float val = ImIn[(j + k*d_MNy)*d_MNx + i] - 0.015f;
		unsigned short val2 = (unsigned short)floorf(__saturatef(val / maxVal) * USHRT_MAX - 1);
		ImOut[(j + k*d_Ny)*d_Nx + i] = val2;
	}
}


/********************************************************************************************/
/* Function to interface the CPU with the GPU:												*/
/********************************************************************************************/
template<typename T>
TomoError TomoRecon::resizeImage(T* in, int wIn, int hIn, cudaArray_t out, int wOut, int hOut, double maxVar) {
	cuda(BindSurfaceToArray(displaySurface, out));

	const int blocks = (wOut * hOut + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

	if (blocks > 0)
		KERNELCALL2(resizeKernel, blocks, PXL_KERNEL_THREADS_PER_BLOCK, in, wIn, hIn, wOut, hOut, maxVar);

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to Initialize the GPU and set up the reconstruction normalization

//Function to define the reconstruction structure
void TomoRecon::DefineReconstructSpace(){
	//Define a new recon data pointer and define size
	Sys->Recon = new ReconGeometry;

	Sys->Recon->Pitch_z = Sys->SysGeo.ZPitch;
	Sys->Recon->Pitch_x = Sys->Proj->Pitch_x;
	Sys->Recon->Pitch_y = Sys->Proj->Pitch_y;
	Sys->Recon->Slice_0_z = Sys->SysGeo.ZDist;
	Sys->Recon->Nx = Sys->Proj->Nx;
	Sys->Recon->Ny = Sys->Proj->Ny;
	Sys->Recon->Nz = Sys->Proj->Nz;
	Sys->Recon->Mean = 210;
	Sys->Recon->Width = 410;
	Sys->Recon->MaxVal = 1.0f;

	if (Sys->UsrIn->SmoothEdge == 1) {
		Sys->Recon->Pitch_z = 2.0;
		Sys->Recon->Slice_0_z = 0;
	}


	//Define the size of the reconstruction memory and allocate a buffer in memory
	//this is local memory not GPU memory
	int size_slice = Sys->Recon->Nx * Sys->Recon->Ny;
	int size_image = Sys->Recon->Nz * size_slice;
	Sys->Recon->ReconIm = new unsigned short[size_image];

}

//Function to set up the memory on the GPU
TomoError TomoRecon::SetUpGPUMemory(){
	//Get Device Number

	//Reset the GPU as a first step
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cudaStatus = cudaSetDevice(1);
		if (cudaStatus != cudaSuccess) {
			cudaStatus = cudaSetDevice(2);
			if (cudaStatus != cudaSuccess) {
				std::cout << "Error: cudaSetDevice failed!" << std::endl;
				exit(1);
			}
		}
	}

	cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));

	size_t heap_size;
	cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size*16);//increase default heap size, we're running out while debugging
	cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Set the memory size as slightly larger to make even multiple of 16
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
	int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;

	//Define the size of each of the memory spaces on the gpu in number of bytes
	size_t sizeIM = MemR_Nx * MemR_Ny * Sys->Recon->Nz * sizeof(float);
	size_t sizeProj = Sys->Proj->Nx * Sys->Proj->Ny * sizeof(unsigned short);
	size_t sizeSino = MemP_Nx  * MemP_Ny * Sys->Proj->NumViews * sizeof(float);
	size_t sizeError = MemP_Nx  * MemP_Ny * sizeof(float);
	size_t sizeSlice = Sys->Recon->Nz * sizeof(float);

	//Allocate memory on GPU
	cuda(MallocPitch((void**)&d_Image, &imagePitch, MemR_Nx * sizeof(float), MemR_Ny * Sys->Recon->Nz));
	cuda(MallocPitch((void**)&d_Image2, &image2Pitch, MemR_Nx * sizeof(float), MemR_Ny * Sys->Recon->Nz));
	cuda(Malloc((void**)&d_GradIm, sizeIM));
	cuda(Malloc((void**)&d_Proj, sizeProj));
	cuda(MallocPitch((void**)&d_Error, &errorPitch, MemR_Nx * sizeof(float), MemR_Ny));
	cuda(Malloc((void**)&d_Sino, sizeSino));
	cuda(Malloc((void**)&d_Norm, sizeSino));
	cuda(Malloc((void**)&d_Pro, sizeSino));
	cuda(Malloc((void**)&d_PriorIm, sizeIM));
	cuda(Malloc((void**)&d_DerivGradIm, sizeIM));
	cuda(Malloc((void**)&d_GradNorm, sizeSlice));
	cuda(Malloc((void**)&d_dp, sizeSlice));
	cuda(Malloc((void**)&d_dpp, sizeSlice));
	cuda(Malloc((void**)&d_alpha, sizeSlice));

	//Set the values of the image and sinogram to all 0
	cuda(Memset2D(d_Image, imagePitch, 0, MemR_Nx * sizeof(float), MemR_Ny * Sys->Recon->Nz));
	cuda(Memset2D(d_Image2, image2Pitch, 0, MemR_Nx * sizeof(float), MemR_Ny * Sys->Recon->Nz));
	cuda(Memset2D(d_Error, errorPitch, 0, MemR_Nx * sizeof(float), MemR_Ny));
	cuda(Memset(d_GradIm, 0, sizeIM));
	cuda(Memset(d_Sino, 0, sizeSino));
	cuda(Memset(d_Norm, 0, sizeSino));
	cuda(Memset(d_Pro, 0, sizeProj));
	cuda(Memset(d_PriorIm, 0, sizeIM));
	cuda(Memset(d_DerivGradIm, 0, sizeIM));
	cuda(Memset(d_GradNorm, 0, sizeSlice));
	cuda(Memset(d_dp, 0, sizeSlice));
	cuda(Memset(d_dpp, 0, sizeSlice));
	cuda(Memset(d_alpha, 0, sizeSlice));

	//Define the textures
	textImage.filterMode = cudaFilterModePoint;
	textImage.addressMode[0] = cudaAddressModeClamp;
	textImage.addressMode[1] = cudaAddressModeClamp;
	//textImage.channelDesc = cudaCreateChannelDesc<float>();

	textError.filterMode = cudaFilterModePoint;
	textError.addressMode[0] = cudaAddressModeClamp;
	textError.addressMode[1] = cudaAddressModeClamp;

	textSino.filterMode = cudaFilterModePoint;
	textSino.addressMode[0] = cudaAddressModeClamp;
	textSino.addressMode[1] = cudaAddressModeClamp;

	cuda(MallocArray(&d_Sinogram, &textSino.channelDesc, MemP_Nx, MemP_Ny * Sys->Proj->NumViews));
	cuda(BindTextureToArray(textSino, d_Sinogram));

	//Set the TV weighting value to start at a constant value less than 1
	float * AlphaSlice = new float[Sys->Recon->Nz];
	for (int slice = 0; slice < Sys->Recon->Nz; slice++) {
		AlphaSlice[slice] = 0.2f;
	}
	cuda(Memcpy(d_alpha, AlphaSlice, sizeSlice, cudaMemcpyHostToDevice));
	delete[] AlphaSlice;

	float HalfPx = (float)Sys->Proj->Nx / 2.0f;
	float HalfPy = (float)Sys->Proj->Ny / 2.0f;
	float HalfNx = (float)Sys->Recon->Nx / 2.0f;
	float HalfNy = (float)Sys->Recon->Ny / 2.0f;
	float PitchPxInv = 1.0f / Sys->Proj->Pitch_x;
	float PitchPyInv = 1.0f / Sys->Proj->Pitch_y;
	float PitchNxInv = 1.0f / Sys->Recon->Pitch_x;
	float PitchNyInv = 1.0f / Sys->Recon->Pitch_y;
	int Sice_Offset =(int)(((float)Sys->Recon->Slice_0_z)
		/(float)(Sys->Recon->Pitch_z));

	cuda(MemcpyToSymbol(d_Px, &Sys->Proj->Nx, sizeof(int)));
	cuda(MemcpyToSymbol(d_Py, &Sys->Proj->Ny, sizeof(int)));
	cuda(MemcpyToSymbol(d_Nx, &Sys->Recon->Nx, sizeof(int)));
	cuda(MemcpyToSymbol(d_Ny, &Sys->Recon->Ny, sizeof(int)));
	cuda(MemcpyToSymbol(d_MPx, &MemP_Nx, sizeof(int)));
	cuda(MemcpyToSymbol(d_MPy, &MemP_Ny, sizeof(int)));
	cuda(MemcpyToSymbol(d_MNx, &MemR_Nx, sizeof(int)));
	cuda(MemcpyToSymbol(d_MNy, &MemR_Ny, sizeof(int)));

	cuda(MemcpyToSymbol(d_HalfPx2, &HalfPx, sizeof(float)));
	cuda(MemcpyToSymbol(d_HalfPy2, &HalfPy, sizeof(float)));
	cuda(MemcpyToSymbol(d_HalfNx2, &HalfNx, sizeof(float)));
	cuda(MemcpyToSymbol(d_HalfNy2, &HalfNy, sizeof(float)));
	cuda(MemcpyToSymbol(d_Nz, &Sys->Recon->Nz, sizeof(int)));
	cuda(MemcpyToSymbol(d_PitchPx, &Sys->Proj->Pitch_x, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchPy, &Sys->Proj->Pitch_y, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchPxInv, &PitchPxInv, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchPyInv, &PitchPyInv, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchNx, &Sys->Recon->Pitch_x, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchNy, &Sys->Recon->Pitch_y, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchNxInv, &PitchNxInv, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchNyInv, &PitchNyInv, sizeof(float)));
	cuda(MemcpyToSymbol(d_PitchNz, &Sys->Recon->Pitch_z, sizeof(float)));
	cuda(MemcpyToSymbol(d_Views, &Sys->Proj->NumViews, sizeof(int)));
	cuda(MemcpyToSymbol(d_rmax, &rmax, sizeof(float)));
	cuda(MemcpyToSymbol(d_alpharelax, &alpha, sizeof(float)));
	cuda(MemcpyToSymbol(d_Z_Offset, &Sice_Offset, sizeof(int)));

	return Tomo_OK;
}

TomoError TomoRecon::GetReconNorm(){
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Set up GPU kernel thread and block sizes based on image size
	dim3 dimBlockProj(16, 16);
	dim3 dimGridProj(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);

	float ex, ey, ez;

	//Calculate a the projection of a image of all ones
	for (int view = 0; view < Sys->Proj->NumViews; view++){
		//these are the geometry values from the input file
		ex = Sys->SysGeo.EmitX[view];
		ey = Sys->SysGeo.EmitY[view];
		ez = Sys->SysGeo.EmitZ[view];

		//this is a GPU function call that causes the threads to perform the projection norm calculation
		KERNELCALL2(ProjectionNorm, dimGridProj, dimBlockProj, d_Norm, view, ex, ey, ez);
	}

	//Define the size of the sinogram space
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	size_t sizeSino = MemP_Nx * MemP_Ny * Sys->Proj->NumViews * sizeof(float);
	float  * d_NormCpy;
	cuda(Malloc((void**)&d_NormCpy, sizeSino));

	//Calculate the contribution of each view to the total number of views
	KERNELCALL2(AverageProjectionNorm, dimGridProj, dimBlockProj, d_Norm, d_NormCpy);

	cuda(Memcpy(d_Norm, d_NormCpy, sizeSino, cudaMemcpyDeviceToDevice));

	cuda(Free(d_NormCpy));

	// Check the last error to make sure that norm calculations correction worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Calculate projection norm: " << cudaGetErrorString(error) << std::endl;

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions called to control the stages of reconstruction
TomoError TomoRecon::LoadAndCorrectProjections(){
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Set up cuda streams to read in projection data
	cudaStream_t * streams = (cudaStream_t*)malloc(Sys->Proj->NumViews * sizeof(cudaStream_t));

	for (int view = 0; view < Sys->Proj->NumViews; view++) cudaStreamCreate(&(streams[view]));

	//Define memory size of the raw data as unsigned short
	int size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	size_t sizeProj = size_proj * sizeof(unsigned short);

	//define the GPU kernel based on size of "ideal projection"
	dim3 dimBlockProj(32, 32);
	dim3 dimGridProj(Sys->Proj->Nx / 32 + Cx, Sys->Proj->Ny / 32 + Cy);

	//Define the size of the sinogram space
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;
	int size_sino = MemP_Nx * MemP_Ny;
	size_t sizeSino = MemP_Nx  * MemP_Ny * Sys->Proj->NumViews * sizeof(float);

	float * d_SinoBlurX;
	float * d_SinoBlurXY;
	cuda(Malloc((void**)&d_SinoBlurX, size_sino * sizeof(float)));
	cuda(Malloc((void**)&d_SinoBlurXY, size_sino * sizeof(float)));

	//Cycle through each stream and do simple log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) {
		cuda(MemcpyAsync(d_Proj, Sys->Proj->RawData + view*size_proj, sizeProj, cudaMemcpyHostToDevice));

		KERNELCALL4(LogCorrectProj, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, view, d_Proj, 40000.0);

		cuda(Memset(d_SinoBlurX, 0, size_sino * sizeof(float)));

		KERNELCALL4(ApplyGaussianBlurX, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, d_SinoBlurX, view);
		KERNELCALL4(ApplyGaussianBlurY, dimGridProj, dimBlockProj, 0, streams[view], d_SinoBlurX, d_SinoBlurXY);

		KERNELCALL4(ScatterCorrect, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, d_Proj, d_SinoBlurXY, view, log(40000.0f));

		cuda(MemcpyAsync(Sys->Proj->RawData + view*size_proj, d_Proj, sizeProj, cudaMemcpyDeviceToHost));
	}

	cuda(Free(d_SinoBlurX));
	cuda(Free(d_SinoBlurXY));

	//Destroy the cuda streams used for log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) cuda(StreamDestroy(streams[view]));

	//Check the last error to make sure that log correction worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Load and Correct Projections: " << cudaGetErrorString(error) << std::endl;

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to control the SART and TV reconstruction
TomoError TomoRecon::FindSliceOffset(){
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Define the size of the Image space
	int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
	int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	//Set up GPU kernel thread and block sizes based on image size
	dim3 dimBlockIm(16, 8);
	dim3 dimGridIm(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);

	dim3 dimBlockSino(16, 16);
	dim3 dimGridSino(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);

	dim3 dimBlockIm2(16, 16);
	dim3 dimGridIm2(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 16 + Cy, Sys->Recon->Nz);
	dim3 dimGridIm3(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 16 + Cy);

	dim3 dimBlockCorr(32, 4);
	dim3 dimGridCorr(floor(Sys->Recon->Ny / 32), (Sys->Recon->Nz / 4) + 1);

	size_t sizeIm = MemR_Nx*MemR_Ny*Sys->Recon->Nz * sizeof(float);

	dim3 dimGridSum(1, Sys->Recon->Nz);
	dim3 dimBlockSum(1024, 1);
	int sumSize = 1024 * sizeof(float);
	int size_Im = MemR_Nx*MemR_Ny;

	float Beta = 1.0f;
	float ex, ey, ez;

	//Find Center slice
	//cuda(Memset(d_Image, 0, sizeIm));
	//cuda(Memset(d_Image2, 0, sizeIm));

	//Single SART Iteration with larger z_Pitch
	for (int iter = 0; iter < 1; iter++) {
		for (int view = 0; view < Sys->Proj->NumViews; view++){
			ex = Sys->SysGeo.EmitX[view];
			ey = Sys->SysGeo.EmitY[view];
			ez = Sys->SysGeo.EmitZ[view];

			cuda(BindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, imagePitch));

			KERNELCALL2(ProjectImage, dimGridSino, dimBlockSino, d_Image, d_Sino, d_Norm, d_Error, view, ex, ey, ez);

			cuda(BindTexture2D(NULL, textError, d_Error, textImage.channelDesc,MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

			KERNELCALL2(BackProjectError, dimGridIm, dimBlockIm, d_Image, d_Image2, d_Error, Beta, view, ex, ey, ez);
		}//views
	}//iterations
	
	cuda(Memset(d_Image2, 0, sizeIm));

	cuda(BindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	KERNELCALL2(SobelEdgeDetection, dimGridIm2, dimBlockIm2, d_Image2);

	float * d_MaxVal;
	float * h_MaxVal = new float[Sys->Recon->Nz];
	cuda(Malloc((void**)&d_MaxVal, Sys->Recon->Nz * sizeof(float)));
	KERNELCALL3(SumSobelEdges, dimGridSum, dimBlockSum, sumSize, d_Image2, size_Im, d_MaxVal);
	cuda(Memcpy(h_MaxVal, d_MaxVal, Sys->Recon->Nz * sizeof(float), cudaMemcpyDeviceToHost));

	int centerSlice = 0;
	int MaxSum = 0;
	for (int n = 0; n < Sys->Recon->Nz; n++){
		if (h_MaxVal[n] > MaxSum) {
			MaxSum = (int)h_MaxVal[n];
			centerSlice = n;
		}
	}
	//if (centerSlice == 0) centerSlice = floor(Sys->Recon->Nz / 2);
	std::cout << "Center slice is:" << centerSlice << std::endl;
	if (centerSlice < 5) centerSlice = 0;
	Sys->Recon->Pitch_z = Sys->SysGeo.ZPitch;
	int Sice_Offset = (int)(((float)centerSlice*1.25)
		/ ((float)Sys->Recon->Pitch_z));

	std::cout << "New Offset is:" << Sice_Offset << std::endl;
	std::cout << "New Pitch is:" << Sys->Recon->Pitch_z << std::endl;

	cuda(MemcpyToSymbol(&d_Z_Offset, &Sice_Offset, sizeof(int)));
	cuda(MemcpyToSymbol(&d_PitchNz, &Sys->Recon->Pitch_z, sizeof(float)));
	
	cuda(Memset(d_Image2, 0, sizeIm));
	cuda(Memset(d_Image, 0, sizeIm));

	tomo_err_throw(GetReconNorm());

	return Tomo_OK;
}

TomoError TomoRecon::AddTVandTVSquared(){
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Define the size of the Image space
	int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
	int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;

	//Set up GPU kernel thread and block sizes based on image size
	dim3 dimBlockIm(16, 8);
	dim3 dimGridIm(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);

	dim3 dimGridSum(1, Sys->Recon->Nz);
	dim3 dimBlockSum(1024, 1);

	cuda(BindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	KERNELCALL2(DerivOfGradIm, dimGridIm, dimBlockIm, d_DerivGradIm, eplison);

	KERNELCALL2(GetGradIm, dimGridIm, dimBlockIm, d_GradIm);

	cuda(BindTexture2D(NULL, textImage, d_GradIm, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	KERNELCALL2(DerivOfGradIm, dimGridIm, dimBlockIm, d_Image2, eplison);

	KERNELCALL2(UpdateImageEstimate, dimGridIm, dimBlockIm, d_Image, d_DerivGradIm, d_Image2);

	return Tomo_OK;
}

TomoError TomoRecon::ReconUsingSARTandTV(){
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Define the size of the Image space
	int MemR_Nx = (Sys->Recon->Nx / 16 +Cx) * 16;
	int MemR_Ny = (Sys->Recon->Ny / 16 +Cy) * 16;
	int MemP_Nx = (Sys->Proj->Nx / 16 +Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 +Cy) * 16;

	//Set up GPU kernel thread and block sizes based on image size
	dim3 dimBlockIm(16, 8);
	dim3 dimGridIm(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);

	dim3 dimBlockIm2(16, 16);
	dim3 dimGridIm2(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 16 + Cy, Sys->Recon->Nz);
	dim3 dimGridIm3(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 16 + Cy);

	dim3 dimBlockSino(16, 16);
	dim3 dimGridSino(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);

	dim3 dimBlockCorr(32, 4);
	dim3 dimGridCorr((int)floor(Sys->Recon->Ny / 32), (Sys->Recon->Nz / 4) +1);

	size_t sizeIm = MemR_Nx*MemR_Ny*Sys->Recon->Nz * sizeof(float);

	dim3 dimGridSum(1, Sys->Recon->Nz);
	dim3 dimBlockSum(1024, 1);

	float Beta = 1.0f;
	float ex, ey, ez;

	//Find Center slice
	cuda(Memset(d_Image, 0, sizeIm));
	cuda(Memset(d_Image2, 0, sizeIm));

	std::cout << "Reconstruction starting" << std::endl;

	//Do a set number of iterations
	for (int iter = 0; iter < 30; iter++){
		//Do one SART iteration by cycling through all views
		for (int view = 0; view <Sys->Proj->NumViews; view++){
			ex = Sys->SysGeo.EmitX[view];
			ey = Sys->SysGeo.EmitY[view];
			ez = Sys->SysGeo.EmitZ[view];

			cuda(BindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

			KERNELCALL2(ProjectImage, dimGridSino, dimBlockSino, d_Image, d_Sino, d_Norm, d_Error, view, ex, ey, ez);

			cuda(BindTexture2D(NULL, textError, d_Error, textImage.channelDesc, MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

			//KERNELCALL2(BackProjectError, dimGridIm, dimBlockIm, d_Image, d_Image2, Beta, view, ex, ey, ez);

			if(Sys->UsrIn->SmoothEdge == 1)
				KERNELCALL2(CorrectEdgesX, dimGridCorr, dimBlockCorr, d_Image, d_Image2);
		}//views

		tomo_err_throw(AddTVandTVSquared());
		Beta = Beta*0.95f;
	}//iterations

	//cudaDeviceSynchronize();
	
	std::cout << "Recon finised" << std::endl;

	//Code to create a sythnetic projection image
	/*	
	int size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	size_t sizeProj = size_proj * sizeof(unsigned short);
	CreateSyntheticProjection << < dimGridSino, dimBlockSino >> >
			(d_Proj, d_Window, d_Image, log(40000.0f));

	cuda(Memcpy(Sys->Proj->SyntData, d_Proj, sizeProj, cudaMemcpyDeviceToHost));
	*/

	// Check the last error to make sure that reconstruction functions  worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Reconstruction: " << cudaGetErrorString(error) << std::endl;

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to save the images
TomoError TomoRecon::CopyAndSaveImages(){
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Define the size of the real image
	//Set up cuda streams to read in projection data
	size_t sizeIM = Sys->Recon->Nx * Sys->Recon->Ny * Sys->Recon->Nz * sizeof(unsigned short);
	int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
	int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;

	unsigned short * d_ImCpy;
	cuda(Malloc((void**)&d_ImCpy, sizeIM));
	cuda(Memset(d_ImCpy, 0, sizeIM));

	//Define block and grid sizes
	dim3 dimBlockIm(16, 16);
	dim3 dimGridIm(MemR_Nx, MemR_Ny, Sys->Recon->Nz);

	dim3 dimGridSum(1, Sys->Recon->Nz);
	dim3 dimBlockSum(1024, 1);
	int sumSize = 1024 * sizeof(float);
	int size_Im = MemR_Nx * MemR_Ny;

	//Get the max image value
	float * d_MaxVal;
	float * h_MaxVal = new float[Sys->Recon->Nz];
	cuda(Malloc((void**)&d_MaxVal, Sys->Recon->Nz * sizeof(float)));
	KERNELCALL3(GetMaxImageVal, dimGridSum, dimBlockSum, sumSize, d_Image, size_Im, d_MaxVal);
	cuda(Memcpy(h_MaxVal, d_MaxVal, Sys->Recon->Nz * sizeof(float), cudaMemcpyDeviceToHost));

	float MaxVal = 0;
	for (int slice = 0; slice < Sys->Recon->Nz; slice++)
		if (h_MaxVal[slice] > MaxVal) MaxVal = h_MaxVal[slice];

	std::cout << "The max reconstructed value is: " << MaxVal << std::endl;
	Sys->Recon->MaxVal = MaxVal;

	//Copy the image to smaller space
	KERNELCALL2(CopyImages, dimGridIm, dimBlockIm, d_ImCpy, d_Image, MaxVal);
	cuda(Memcpy(Sys->Recon->ReconIm, d_ImCpy, sizeIM, cudaMemcpyDeviceToHost));

	//Remove temporary buffer
	cuda(Free(d_ImCpy));

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions referenced from main
TomoError TomoRecon::SetUpGPUForRecon(){
	//Set up GPU Memory space
	DefineReconstructSpace();

	//Set up GPU memory space
	tomo_err_throw(SetUpGPUMemory());

	//Calulate the reconstruction Normalization for the SART
	tomo_err_throw(GetReconNorm());

	return Tomo_OK;
}

TomoError TomoRecon::Reconstruct(){
	//Get the time and start before projections are corrected
	FILETIME filetime, filetime2;
	GetSystemTimeAsFileTime(&filetime);

	//setup local memory to hold final reconstructed image
	int size_slice = Sys->Recon->Nx * Sys->Recon->Ny;
	int size_image = Sys->Recon->Nz * size_slice;
	memset(Sys->Recon->ReconIm, 0, size_image * sizeof(unsigned short));
	std::cout << "Memory Ready" << std::endl;

	FILETIME filetime3, filetime4;
	GetSystemTimeAsFileTime(&filetime3);
	//Correct data and read onto GPU
	tomo_err_throw(LoadAndCorrectProjections());
	std::cout << "Data Loaded" << std::endl;
	LONGLONG time1, time2;
	GetSystemTimeAsFileTime(&filetime4);
	time1 = (((ULONGLONG)filetime3.dwHighDateTime) << 32) + filetime3.dwLowDateTime;
	time2 = (((ULONGLONG)filetime4.dwHighDateTime) << 32) + filetime4.dwLowDateTime;
	std::cout << "Total LoadAndCorrectProjections time: " << (double)(time2 - time1) / 10000000 << " seconds";
	std::cout << std::endl;

	//Find Center Slice using increased Resolution
	if (Sys->UsrIn->CalOffset == 1)
		tomo_err_throw(FindSliceOffset());

	//Call the reconstruction function
	tomo_err_throw(ReconUsingSARTandTV());

	//Copy the reconstructed images to the CPU
	tomo_err_throw(CopyAndSaveImages());

	//Get and display the total ellasped time for the reconstruction
	GetSystemTimeAsFileTime(&filetime2);
//	LONGLONG time1, time2;
	time1 = (((ULONGLONG)filetime.dwHighDateTime) << 32) + filetime.dwLowDateTime;
	time2 = (((ULONGLONG)filetime2.dwHighDateTime) << 32) + filetime2.dwLowDateTime;
	std::cout << "Total Recon time: " << (double)(time2 - time1) / 10000000 << " seconds";
	std::cout << std::endl;

	std::cout << "Reconstruction finished successfully." << std::endl;

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Fucntion to free the gpu memory after program finishes
TomoError TomoRecon::FreeGPUMemory(void){
	//Free memory allocated on the GPU
	cuda(Free(d_Proj));
	cuda(Free(d_Norm));
	cuda(Free(d_Image));
	cuda(Free(d_Image2));
	cuda(Free(d_GradIm));
	cuda(Free(d_Error));
	cuda(Free(d_Sino));
	cuda(Free(d_Pro));
	cuda(Free(d_PriorIm));
	cuda(Free(d_dp));
	cuda(Free(d_dpp));
	cuda(Free(d_alpha));
	cuda(Free(d_DerivGradIm));
	cuda(Free(d_GradNorm));

	cuda(Free(d_Image));
	cuda(Free(d_Image2));
	cuda(Free(d_Sino));
	cuda(Free(d_Norm));
	cuda(Free(d_Pro));
	cuda(Free(d_Proj));

	cuda(Free(d_PriorIm));
	cuda(Free(d_DerivGradIm));
	cuda(Free(d_GradNorm));
	cuda(Free(d_dp));
	cuda(Free(d_dpp));
	cuda(Free(d_alpha));

	//Unbind the texture array and free the cuda array 
	cuda(FreeArray(d_Sinogram));
	cuda(UnbindTexture(textSino));

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Lower level functions for user interactive debugging
TomoError TomoRecon::LoadProjections(int index) {
	//Define memory size of the raw data as unsigned short
	int size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	size_t sizeProj = size_proj * sizeof(unsigned short);

	cuda(Memcpy(d_Proj, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));

	return Tomo_OK;
}

TomoError TomoRecon::correctProjections() {
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Set up cuda streams to read in projection data
	cudaStream_t * streams = (cudaStream_t*)malloc(Sys->Proj->NumViews * sizeof(cudaStream_t));

	for (int view = 0; view < Sys->Proj->NumViews; view++) cudaStreamCreate(&(streams[view]));

	//Define memory size of the raw data as unsigned short
	int size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	size_t sizeProj = size_proj * sizeof(unsigned short);

	//define the GPU kernel based on size of "ideal projection"
	dim3 dimBlockProj(32, 32);
	dim3 dimGridProj(Sys->Proj->Nx / 32 + Cx, Sys->Proj->Ny / 32 + Cy);

	//Define the size of the sinogram space
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;
	int size_sino = MemP_Nx * MemP_Ny;
	size_t sizeSino = MemP_Nx  * MemP_Ny * Sys->Proj->NumViews * sizeof(float);

	float * d_SinoBlurX;
	float * d_SinoBlurXY;
	cuda(Malloc((void**)&d_SinoBlurX, size_sino * sizeof(float)));
	cuda(Malloc((void**)&d_SinoBlurXY, size_sino * sizeof(float)));

	//Cycle through each stream and do simple log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) {
		cuda(MemcpyAsync(d_Proj, Sys->Proj->RawData + view*size_proj, sizeProj, cudaMemcpyHostToDevice));

		KERNELCALL4(LogCorrectProj, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, view, d_Proj, SHRT_MAX);

		cuda(Memset(d_SinoBlurX, 0, size_sino * sizeof(float)));

		KERNELCALL4(ApplyGaussianBlurX, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, d_SinoBlurX, view);
		KERNELCALL4(ApplyGaussianBlurY, dimGridProj, dimBlockProj, 0, streams[view], d_SinoBlurX, d_SinoBlurXY);

		KERNELCALL4(ScatterCorrect, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, d_Proj, d_SinoBlurXY, view, log(SHRT_MAX) - 1);

		cuda(MemcpyAsync(Sys->Proj->RawData + view*size_proj, d_Proj, sizeProj, cudaMemcpyDeviceToHost));
	}

	cuda(Free(d_SinoBlurX));
	cuda(Free(d_SinoBlurXY));

	//Destroy the cuda streams used for log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) cuda(StreamDestroy(streams[view]));
}

TomoError TomoRecon::test(int index) {
	int size_proj = 0;
	size_t sizeProj = 0;

	switch (currentDisplay) {
	case raw_images2://same as raw_images
	case raw_images:
		size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
		sizeProj = size_proj * sizeof(unsigned short);
		cuda(Memcpy(d_Proj, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));
		//LoadProjections(index);
		resizeImage(d_Proj, Sys->Proj->Nx, Sys->Proj->Ny, *ca, width, height, USHRT_MAX);
		break;
	case sino_images:
		size_proj = 1920 * Sys->Proj->Ny;
		//size_t sizeProj = size_proj * sizeof(float);
		//cuda(Memcpy(d_Sino, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));
		resizeImage(d_Sino+ size_proj*index, 1920, Sys->Proj->Ny, *ca, width, height, log(USHRT_MAX) - 1);
		break;
	case norm_images:
		resizeImage(d_Norm + size_proj*index, 1920, Sys->Proj->Ny, *ca, width, height, 1);
		break;
	case recon_images:
		int Cx = 1;
		int Cy = 1;
		if (Sys->Proj->Nx % 16 == 0) Cx = 0;
		if (Sys->Proj->Ny % 16 == 0) Cy = 0;

		//Define the size of the Image space
		int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
		int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;

		size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny*Sys->Recon->Nz, imagePitch));
		cuda(BindSurfaceToArray(displaySurface, *ca));
		//cuda(BindSurfaceToArray(imageSurface, (cudaArray_t)d_Proj));

		const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

		if (blocks > 0)
			KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, Sys->Proj->Nx, Sys->Proj->Ny, width, height, 0.7, index);
		//resizeImage(d_Image + size_proj*index, Sys->Proj->Nx, Sys->Proj->Ny, *ca, width, height, log(SHRT_MAX) - 1);
		//resizeImage(d_Error, 1920, Sys->Proj->Ny, *ca, width, height, log(SHRT_MAX) - 1);
		break;
	}
	return Tomo_OK;
}

TomoError TomoRecon::reconInit() {
	//setup local memory to hold final reconstructed image
	int size_slice = Sys->Recon->Nx * Sys->Recon->Ny;
	int size_image = Sys->Recon->Nz * size_slice;
	memset(Sys->Recon->ReconIm, 0, size_image * sizeof(unsigned short));

	//Find Center Slice using increased Resolution
	if (Sys->UsrIn->CalOffset == 1)
		tomo_err_throw(FindSliceOffset());
}

TomoError TomoRecon::reconStep() {
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Define the size of the Image space
	int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
	int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	//Set up GPU kernel thread and block sizes based on image size
	dim3 dimBlockIm(16, 8);
	dim3 dimGridIm(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);

	dim3 dimBlockIm2(16, 16);
	dim3 dimGridIm2(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 16 + Cy, Sys->Recon->Nz);
	dim3 dimGridIm3(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 16 + Cy);

	dim3 dimBlockSino(16, 16);
	dim3 dimGridSino(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);

	dim3 dimBlockCorr(32, 4);
	dim3 dimGridCorr((int)floor(Sys->Recon->Ny / 32), (Sys->Recon->Nz / 4) + 1);

	size_t sizeIm = MemR_Nx*MemR_Ny*Sys->Recon->Nz * sizeof(float);

	const int blocks = (Sys->Recon->Nx * Sys->Recon->Ny + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;
	dim3 blocks3(blocks, Sys->Recon->Nz, 1);

	float ex, ey, ez;
	//cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	
	//Do one SART iteration by cycling through all views
	for (int view = 0; view < Sys->Proj->NumViews; view++) {
	//int view = 0;
		ex = Sys->SysGeo.EmitX[view];
		ey = Sys->SysGeo.EmitY[view];
		ez = Sys->SysGeo.EmitZ[view];

		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny*Sys->Recon->Nz, imagePitch));

		//KERNELCALL2(ProjectImage, dimGridIm2, dimBlockIm2, d_Sino, d_Norm, d_Image, d_Error, view, ex, ey, ez);
		KERNELCALL2(ProjectImage, dimGridSino, dimBlockSino, d_Sino, d_Norm, d_Image, d_Error, view, ex, ey, ez);

		cuda(BindTexture2D(NULL, textError, d_Error, cudaCreateChannelDesc<float>(), MemP_Nx, MemP_Ny, errorPitch));

		KERNELCALL2(BackProjectError, dimGridIm2, dimBlockIm2, d_Image, d_Image2, d_Error, Beta, view, ex, ey, ez);
		//KERNELCALL2(BackProjectError, blocks3, PXL_KERNEL_THREADS_PER_BLOCK, d_Image, d_Image2, d_Error, Beta, view, ex, ey, ez);

		//cuda(Memset2D(d_Error, errorPitch, 0, MemR_Nx * sizeof(float), MemR_Ny));

		if (Sys->UsrIn->SmoothEdge == 1)
			KERNELCALL2(CorrectEdgesX, dimGridCorr, dimBlockCorr, d_Image, d_Image2);
	}//views

	tomo_err_throw(AddTVandTVSquared());
	Beta = Beta*0.95f;
	iteration++;
}