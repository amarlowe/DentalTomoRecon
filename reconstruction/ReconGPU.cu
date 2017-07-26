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

TomoError cuda_assert(const cudaError_t code, const char* const file, const int line) {
	if (code != cudaSuccess) {
		std::cout << "Cuda failure " << file << ":" << line << ": " << cudaGetErrorString(code) << "\n";
		return Tomo_CUDA_err;
	}
	else return Tomo_OK;
}

TomoError cuda_assert_void(const char* const file, const int line) {
	cudaError_t code = cudaGetLastError();
	if (code != cudaSuccess) {
		std::cout << "Cuda failure " << file << ":" << line << ": " << cudaGetErrorString(code) << "\n";
		return Tomo_CUDA_err;
	}
	else return Tomo_OK;
}

__device__ static float atomicMax(float* address, float val){
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

__device__ static float atomicMin(float* address, float val) {
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fminf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
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
__device__ __constant__ float d_HalfPx;
__device__ __constant__ float d_HalfPy;
__device__ __constant__ float d_HalfNx;
__device__ __constant__ float d_HalfNy;
__device__ __constant__ int d_Nz;
__device__ __constant__ int d_Views;
__device__ __constant__ float d_PitchPx;
__device__ __constant__ float d_PitchPy;
__device__ __constant__ float d_PitchNx;
__device__ __constant__ float d_PitchNy;
__device__ __constant__ float d_PitchNz;
__device__ __constant__ float d_alpharelax;
__device__ __constant__ float d_rmax;
__device__ __constant__ int d_Z_Offset;
__device__ __constant__ float* d_beamx;
__device__ __constant__ float* d_beamy;
__device__ __constant__ float* d_beamz;
__device__ __constant__ size_t d_errorPitch;
__device__ __constant__ size_t d_imagePitch;
__device__ __constant__ size_t d_sinoPitch;

/********************************************************************************************/
/* GPU Function specific functions															*/
/********************************************************************************************/

template<typename T>
__global__ void resizeKernel(T* input, int wIn, int hIn, int wOut, int hOut, double maxVar) {
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int x = idx % wOut;
	int y = idx / wOut;
	float scale = max((float)wIn / (float)wOut, (float)hIn / (float)hOut);

	float sum = 0;
	int i = (x - (wOut - wIn / scale) / 2)*scale;
	int j = (y - (hOut - hIn / scale) / 2)*scale;
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

__global__ void resizeKernelTex(int wIn, int hIn, int wOut, int hOut, float * maxVar, int index) {
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % wOut;
	const int y = idx / wOut;
	float scale = max((float)wIn / (float)wOut, (float)hIn / (float)hOut);

	float sum = 0;
	int i = (x - (wOut - wIn / scale) / 2)*scale;
	int j = (y - (hOut - hIn / scale) / 2)*scale;
	if (i > 0 && j > 0 && i < wIn && j < hIn)
		sum = tex2D(textImage, (float)i + 0.5f, (float)j + 0.5f + (float)(index * hIn));;

	//sum = sum / *maxVar * 255;
	sum = sum / USHRT_MAX * 255;

	union pxl_rgbx_24 rgbx;
	if (sum > 255) {
		rgbx.na = 0xFF;
		rgbx.r = 255;//flag errors with big red spots
		rgbx.g = 0;
		rgbx.b = 0;
	}
	else {
		rgbx.na = 0xFF;
		rgbx.r = sum;
		rgbx.g = sum;
		rgbx.b = sum;
	}

	surf2Dwrite(rgbx.b32,
		displaySurface,
		x * sizeof(rgbx),
		y,
		cudaBoundaryModeZero); // squelches out-of-bound writes
}

/////////////////////////////////////////////////////////////////////////////////////////////
//Functions to do initial correction of raw data: log and scatter correction
__global__ void LogCorrectProj(float * Sino, int view, unsigned short *Proj, float MaxVal){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i < d_Px) && (j < d_Py)){
		//Log correct the sample size
		float sample = (float)Proj[j*d_Px + i];
		float val = logf(MaxVal) - logf(sample);
		if (sample > MaxVal) val = 0.0f;

		//Sino[(j + view*d_MPy)*d_MPx + i] = val;
		Sino[(j + view*d_MPy)*d_sinoPitch + i] = sample;
	}
}

__global__ void rescale(float * Sino, int view, float * MaxVal, float * MinVal) {
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i < d_Px) && (j < d_Py)) {
		float test = Sino[(j + view*d_MPy)*d_sinoPitch + i];
		if (test > 0) {
			Sino[(j + view*d_MPy)*d_sinoPitch + i] = (test - *MinVal + 1.0f) / *MaxVal * USHRT_MAX;//scale from 1 to max
		}
	}
}

__global__ void ApplyGaussianBlurX(float * Sino, float * BlurrX, int view){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
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

	//Check image boundaries
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

	//Check image boundaries
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
__global__ void ProjectImage(float * Sino, float *Error){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int view = blockDim.z * blockIdx.z + threadIdx.z;

	//geometry cache
	float ex = d_beamx[view];
	float ey = d_beamy[view];
	float ez = d_beamz[view];

	//within image boudary
	if ((i < d_Px) && (j < d_Py)) {
		//Get scale factor
		float dx = ((float)i + 0.5f - d_HalfPx) * d_PitchPx;//Center x offset in mm
		float dy = ((float)j + 0.5f - d_HalfPy) * d_PitchPy;//Center y offset in mm
		float dx2 = ez / sqrtf(pow(dx - ex, 2) + pow(ez, 2));//Z direction vector in xz plane
		float dy2 = ez / sqrtf(pow(dy - ey, 2) + pow(ez, 2));//Z direction vector in yz plane
		float scale = dx2*dy2;

		float Pro = 0.0f;
		float count = 0.0f;

		//Coordinates relative to the recon upper left, geometry independent
		float x = dx / d_PitchNx + d_HalfNx - 0.5;
		float y = dy / d_PitchNy + d_HalfNy - 0.5;

		//Change deltas to offset per stepsize relative to emmiter
		dx = d_PitchNz / ez * (dx - ex) / d_PitchNx;
		dy = d_PitchNz / ez * (dy - ey) / d_PitchNy;

		//Add slice offset
		x += dx * (float)(d_Z_Offset - 1);
		y += dy * (float)(d_Z_Offset - 1);

		//Step through the image space by slice in z direction
		for (int z = 0; z < d_Nz; z++) {
			//Get the next n and x
			x += dx;
			y += dy;
			//Out of bounds and mid pixel interpolation handled by texture call
			//TODO!!!!: bounds are not checked for y direction, make 3d texture or bounds handle here
			if (y > 0 && y < d_MNy && x > 0 && x < d_MNx) {
				float value = tex2D(textImage, x, y + z*d_MNy);
				if (value != 0) {
					Pro += value;
					count += 1.0f;
				}
			}
		}//z loop

		float err;
		if (count == 0) err = 0.0f;
		//else err = Sino[i + d_sinoPitch*(j + view*d_MPy)] * scale - Pro / count;
		else err = Sino[i + d_sinoPitch*(j + view*d_MPy)] - Pro / count;

		//Get the projection error and add calculated error to an error image to back project
		//atomicAdd(&Error[(j + view*d_MPy)*d_MPx + i], err);
		Error[(j + view*d_MPy)*d_errorPitch + i] = err;
		//Error[(j + view*d_MPy)*d_errorPitch + i] = 0.0;
	}//image boudary check
}

__global__ void BackProjectError(float * IM, float beta){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Set a normalization and pixel value to 0
	float error = 0.0f;
	float count = 0.0f;

	//Check image boundaries
	if ((i < d_Nx) && (j < d_Ny)) {
		for (int view = 0; view < NUMVIEWS; view++) {
			//geometry cache (in pixels)
			float ex = d_beamx[view] / d_PitchNx;
			float ey = d_beamy[view] / d_PitchNy;
			float ezInv = d_PitchNz / d_beamz[view];

			float NtoPx = d_PitchNx / d_PitchPx;
			float NtoPy = d_PitchNy / d_PitchPy;


			float dx = (d_Z_Offset + k) * ezInv;
			float x = (i + 0.5 - d_HalfNx + ex * dx) / (1 + dx) * NtoPx - 0.5 + d_HalfPx;
			float y = (j + 0.5 - d_HalfNy + ey * dx) / (1 + dx) * NtoPy - 0.5 + d_HalfPy;

			//Update the value based on the error scaled and save the scale
			if (y > 0 && y < d_MPy && x > 0 && x < d_MPx) {
				float val = tex2D(textError, x, y + view*d_MPy);
				if (val != 0) {
					error += val;
					count++;
				}
			}
		}
	}

	if (count > 0)
		IM[(j + k*d_MNy)*d_MNx + i] += beta*error / count;
}

__global__ void projectSlice(float * IM, int index) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Set a normalization and pixel value to 0
	float error = 0.0f;
	float count = 0.0f;

	//Check image boundaries
	if ((i < d_Nx) && (j < d_Ny)) {
		for (int view = 0; view < NUMVIEWS; view++) {
			//geometry cache (in pixels)
			float ex = d_beamx[view] / d_PitchNx;
			float ey = d_beamy[view] / d_PitchNy;
			float ezInv = d_PitchNz / d_beamz[view];

			float NtoPx = d_PitchNx / d_PitchPx;
			float NtoPy = d_PitchNy / d_PitchPy;

			float dx = (d_Z_Offset + index) * ezInv;
			float x = (i + 0.5 - d_HalfNx + ex * dx) / (1 + dx) * NtoPx - 0.5 + d_HalfPx;
			float y = (j + 0.5 - d_HalfNy + ey * dx) / (1 + dx) * NtoPy - 0.5 + d_HalfPy;

			//Update the value based on the error scaled and save the scale
			if (y > 0 && y < d_MPy && x > 0 && x < d_MPx) {
				float val = tex2D(textError, x, y + view*d_MPy);
				if (val != 0) {
					error += val;
					count++;
				}
			}
		}
	}

	if (count > 0)
		IM[j*d_MNx + i] = error / count;
}

__global__ void BackProjectSliceOff(float * IM, float beta) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int view = blockDim.z * blockIdx.z + threadIdx.z;

	//geometry cache
	float ex = d_beamx[view];
	float ey = d_beamy[view];
	float ez = d_beamz[view];

	//Check image boundaries
	if ((i < d_Nx) && (j < d_Ny)) {
		for (int k = 0; k < d_Nz; k++) {
			//int k = 0;
			//Define the direction in z to get r
			float r = (ez) / (((float)(k + d_Z_Offset)*d_PitchNz) + ez);

			//Use r to get detecor x and y
			float dx1 = ex + r * (((float)i - (d_HalfNx))*d_PitchNx - ex);
			float dy1 = ey + r * (((float)j - (d_HalfNy))*d_PitchNy - ey);
			float dx2 = dx1 + r * d_PitchNx;
			float dy2 = dy1 + r * d_PitchNy;

			//Use detector x and y to get pixels
			float x1 = dx1 / d_PitchPx + (d_HalfPx - 0.5f);
			float x2 = dx2 / d_PitchPx + (d_HalfPx - 0.5f);
			float y1 = dy1 / d_PitchPy + (d_HalfPy - 0.5f);
			float y2 = dy2 / d_PitchPy + (d_HalfPy - 0.5f);

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
			float xx = (d_HalfPx - 0.5f)*d_PitchPx - ex;
			float yy = (d_HalfPy - 0.5f)*d_PitchPy - ey;

			float xs = x1;
			//Cycle through pixels x and y and used to calculate projection
			for (int x = xMin; x < xMax; x++) {
				//int x = (xMin + xMax) / 2;
				float ys = y1;
				float xend = min((float)(x + 1), x2);

				for (int y = yMin; y < yMax; y++) {
					float yend = min((float)(y + 1), y2);

					//Calculate the weight as the overlap in x and y
					float weight = ((xend - xs))*((yend - ys))*dist;

					//Calculate the scaling of a ray from the center of the pixel
					//to the detector
					float cos_alpha = sqrtf(((float)x - xx)*((float)x - xx) + ez*ez);
					float cos_gamma = sqrtf(((float)y - yy)*((float)y - yy) + ez*ez);
					float scale = (cos_alpha*cos_gamma)*ezz * weight;

					//Update the value based on the error scaled and save the scale
					val += tex2D(textError, (float)x + 0.5f, (float)(y + view*d_MPy) + 0.5f) *scale;

					N += scale;
					ys = yend;
				}//y loop
				xs = xend;
			}//x loop
			 //Get the current value of image
			float update = beta*val / N;

			if (N > 0) {
				IM[(j + k*d_MNy)*d_MNx + i] = update;
			}
			else IM[(j + k*d_MNy)*d_MNx + i] = -10.0f;
		}//z loop
	}
	else for (int k = 0; k < d_Nz; k++) IM[(j + k*d_MNy)*d_MNx + i] = -10.0f;
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

__global__ void SumSobelEdges(float * Image, int sizeIM, float * MaxVal){
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
//Functions to create synthetic projections from reconstructed data
__global__ void CreateSyntheticProjection(unsigned short * Proj, float * Win, float *Im, float MaxVal)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
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

__global__ void CreateSyntheticProjectionNew(unsigned short * Proj, float* Win, float * StIm, float *Im, float MaxVal, int cz){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
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
__global__ void GetMaxImageVal(float * Image, int sizeIM, float * MaxVal){
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

	if (thread < 32){
		// Fetch final intermediate sum from 2nd warp
		data[thread] = val = max(val, data[thread + 32]);
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2){
			val = max(val, __shfl_down(val, offset));
		}
	}

	//write the result for this block to global memory
	if (thread == 0) {
		atomicMax(MaxVal, val);
	}
}

__global__ void GetMinImageVal(float * Image, int sizeIM, float * MinVal) {
	//Define shared memory to read all the threads
	extern __shared__ float data[];

	//define the thread and block location
	int thread = threadIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	int j = blockIdx.y *blockDim.y + threadIdx.y;
	int gridSize = blockDim.x * 2 * gridDim.x;

	if (i == 0 && j == 0) *MinVal = (float)SHRT_MAX;
	__syncthreads();

	//Declare a sum value and iterat through grid to sum
	float val = (float)SHRT_MAX;
	for (int n = i; n + blockDim.x < sizeIM; n += gridSize) {
		float val1 = Image[j*sizeIM + n];
		float val2 = Image[j*sizeIM + n + blockDim.x];
		if (val1 == 0.0f) val1 = (float)SHRT_MAX;
		if (val2 == 0.0f) val2 = (float)SHRT_MAX;
		val = min(val, min(val1, val2));
	}

	//Each thread puts its local sum into shared memory
	data[thread] = val;
	__syncthreads();

	//Do reduction in shared memory
	if (thread < 512) { data[thread] = val = min(val, data[thread + 512]); }
	__syncthreads();

	if (thread < 256) { data[thread] = val = min(val, data[thread + 256]); }
	__syncthreads();

	if (thread < 128) { data[thread] = val = min(val, data[thread + 128]); }
	__syncthreads();

	if (thread < 64) { data[thread] = val = min(val, data[thread + 64]); }
	__syncthreads();

	if (thread < 32) {
		// Fetch final intermediate sum from 2nd warp
		data[thread] = val = min(val, data[thread + 32]);
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2) {
			val = min(val, __shfl_down(val, offset));
		}
	}

	//write the result for this block to global memory
	if (thread == 0) {
		atomicMin(MinVal, val);
	}
}

__global__ void CopyImages(unsigned short * ImOut, float * ImIn, float maxVal){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Check image boundaries
	if ((i < d_Nx) && (j < d_Ny)){
		ImOut[(j + k*d_Ny)*d_Nx + i] = (unsigned short)(ImIn[(j + k*d_MNy)*d_MNx + i] / maxVal * SHRT_MAX);
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
	size_t sizeSino = MemP_Nx * MemP_Ny * Sys->Proj->NumViews * sizeof(float);
	size_t sizeError = MemP_Nx * MemP_Ny * sizeof(float);
	size_t sizeSlice = Sys->Recon->Nz * sizeof(float);

	//Allocate memory on GPU
	if (continuousMode) {
		cuda(MallocPitch((void**)&d_Image, &imagePitch, MemR_Nx * sizeof(float), MemR_Ny));
		cuda(Memset2DAsync(d_Image, imagePitch, 0, MemR_Nx * sizeof(float), MemR_Ny));
	}
	else {
		cuda(MallocPitch((void**)&d_Image, &imagePitch, MemR_Nx * sizeof(float), MemR_Ny * Sys->Recon->Nz));
		cuda(Memset2DAsync(d_Image, imagePitch, 0, MemR_Nx * sizeof(float), MemR_Ny * Sys->Recon->Nz));
	}
	
	cuda(Malloc((void**)&d_Proj, sizeProj));
	cuda(MallocPitch((void**)&d_Error, &errorPitch, MemP_Nx * sizeof(float), MemP_Ny * Sys->Proj->NumViews));
	cuda(MallocPitch((void**)&d_Sino, &sinoPitch, MemP_Nx * sizeof(float), MemP_Ny * Sys->Proj->NumViews));
	cuda(Malloc((void**)&d_Pro, sizeSino));
	cuda(Malloc((void**)&beamx, Sys->Proj->NumViews * sizeof(float)));
	cuda(Malloc((void**)&beamy, Sys->Proj->NumViews * sizeof(float)));
	cuda(Malloc((void**)&beamz, Sys->Proj->NumViews * sizeof(float)));
	cuda(Malloc((void**)&d_MaxVal, sizeof(float)));
	cuda(Malloc((void**)&d_MinVal, sizeof(float)));

	//Set the values of the image and sinogram to all 0
	cuda(Memset2DAsync(d_Error, errorPitch, 0, MemP_Nx * sizeof(float), MemP_Ny * Sys->Proj->NumViews));
	cuda(Memset2DAsync(d_Sino, sinoPitch, 0, MemP_Nx * sizeof(float), MemP_Ny * Sys->Proj->NumViews));
	cuda(MemcpyAsync(beamx, Sys->SysGeo.EmitX, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(beamy, Sys->SysGeo.EmitY, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(beamz, Sys->SysGeo.EmitZ, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemsetAsync(d_Pro, 0, sizeProj));

	//Define the textures
	textImage.filterMode = cudaFilterModeLinear;
	textImage.addressMode[0] = cudaAddressModeClamp;
	textImage.addressMode[1] = cudaAddressModeClamp;

	textError.filterMode = cudaFilterModePoint;
	textError.addressMode[0] = cudaAddressModeClamp;
	textError.addressMode[1] = cudaAddressModeClamp;

	textSino.filterMode = cudaFilterModeLinear;
	textSino.addressMode[0] = cudaAddressModeClamp;
	textSino.addressMode[1] = cudaAddressModeClamp;

	cuda(MallocArray(&d_Sinogram, &textSino.channelDesc, MemP_Nx, MemP_Ny * Sys->Proj->NumViews));
	cuda(BindTextureToArray(textSino, d_Sinogram));

	float HalfPx = (float)Sys->Proj->Nx / 2.0f;
	float HalfPy = (float)Sys->Proj->Ny / 2.0f;
	float HalfNx = (float)Sys->Recon->Nx / 2.0f;
	float HalfNy = (float)Sys->Recon->Ny / 2.0f;
	int Sice_Offset =(int)(((float)Sys->Recon->Slice_0_z)
		/(float)(Sys->Recon->Pitch_z));

	cuda(MemcpyToSymbolAsync(d_Px, &Sys->Proj->Nx, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_Py, &Sys->Proj->Ny, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_Nx, &Sys->Recon->Nx, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_Ny, &Sys->Recon->Ny, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_MPx, &MemP_Nx, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_MPy, &MemP_Ny, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_MNx, &MemR_Nx, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_MNy, &MemR_Ny, sizeof(int)));

	cuda(MemcpyToSymbolAsync(d_HalfPx, &HalfPx, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_HalfPy, &HalfPy, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_HalfNx, &HalfNx, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_HalfNy, &HalfNy, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_Nz, &Sys->Recon->Nz, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_PitchPx, &Sys->Proj->Pitch_x, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_PitchPy, &Sys->Proj->Pitch_y, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_PitchNx, &Sys->Recon->Pitch_x, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_PitchNy, &Sys->Recon->Pitch_y, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_PitchNz, &Sys->Recon->Pitch_z, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_Views, &Sys->Proj->NumViews, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_rmax, &rmax, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_alpharelax, &alpha, sizeof(float)));
	cuda(MemcpyToSymbolAsync(d_Z_Offset, &Sice_Offset, sizeof(int)));
	cuda(MemcpyToSymbolAsync(d_beamx, &beamx, sizeof(float*)));
	cuda(MemcpyToSymbolAsync(d_beamy, &beamy, sizeof(float*)));
	cuda(MemcpyToSymbolAsync(d_beamz, &beamz, sizeof(float*)));
	size_t pitch = errorPitch / sizeof(float);
	cuda(MemcpyToSymbolAsync(d_errorPitch, &pitch, sizeof(size_t)));
	pitch = imagePitch / sizeof(float);
	cuda(MemcpyToSymbolAsync(d_imagePitch, &pitch, sizeof(size_t)));
	pitch = sinoPitch / sizeof(float);
	cuda(MemcpyToSymbolAsync(d_sinoPitch, &pitch, sizeof(size_t)));

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

	dim3 dimBlockIm2(32, 8);
	dim3 dimGridIm2(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);
	dim3 dimGridIm3(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Proj->NumViews * Sys->Recon->Nz);//, Sys->Proj->NumViews
	dim3 dimGridIm4(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Proj->NumViews);

	dim3 dimBlockSino(16, 16);
	dim3 dimGridSino(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);//, Sys->Proj->NumViews

	dim3 dimBlockCorr(32, 4);
	dim3 dimGridCorr((int)floor(Sys->Recon->Ny / 32), (Sys->Recon->Nz / 4) + 1);

	size_t sizeIm = MemR_Nx*MemR_Ny*Sys->Recon->Nz * sizeof(float);

	dim3 dimGridSum(1, Sys->Recon->Nz);
	dim3 dimBlockSum(1024, 1);
	int sumSize = 1024 * sizeof(float);
	int size_Im = MemR_Nx*MemR_Ny;

	//Find Center slice
	//Single SART Iteration with larger z_Pitch
	cuda(BindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, imagePitch));

	KERNELCALL2(ProjectImage, dimGridIm4, dimBlockIm2, d_Sino, d_Error);

	cuda(BindTexture2D(NULL, textError, d_Error, textImage.channelDesc, MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

	KERNELCALL2(BackProjectSliceOff, dimGridIm4, dimBlockIm2, d_Image, Beta);
	
	//d_Image2 replaced with d_image. Need to create custom backproject error that reproduces old d_image2 behavior 
	cuda(Memset(d_Image, 0, sizeIm));

	cuda(BindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	KERNELCALL2(SobelEdgeDetection, dimGridIm2, dimBlockIm2, d_Image);

	float * d_MaxVal;
	float * h_MaxVal = new float[Sys->Recon->Nz];
	cuda(Malloc((void**)&d_MaxVal, Sys->Recon->Nz * sizeof(float)));
	KERNELCALL3(SumSobelEdges, dimGridSum, dimBlockSum, sumSize, d_Image, size_Im, d_MaxVal);
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
	
	cuda(Memset(d_Image, 0, sizeIm));

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
	float h_MaxVal;
	KERNELCALL3(GetMaxImageVal, dimGridSum, dimBlockSum, sumSize, d_Image, size_Im, d_MaxVal);
	cuda(Memcpy(&h_MaxVal, d_MaxVal, sizeof(float), cudaMemcpyDeviceToHost));

	std::cout << "The max reconstructed value is: " << h_MaxVal << std::endl;
	Sys->Recon->MaxVal = h_MaxVal;

	//Copy the image to smaller space
	KERNELCALL2(CopyImages, dimGridIm, dimBlockIm, d_ImCpy, d_Image, h_MaxVal);
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

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Fucntion to free the gpu memory after program finishes
TomoError TomoRecon::FreeGPUMemory(void){
	//Free memory allocated on the GPU
	cuda(Free(d_Proj));
	cuda(Free(d_Image));
	cuda(Free(d_Error));
	cuda(Free(d_Sino));
	cuda(Free(d_Pro));

	cuda(Free(beamx));
	cuda(Free(beamy));
	cuda(Free(beamz));

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

	cuda(MemcpyAsync(d_Proj, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));

	return Tomo_OK;
}

TomoError TomoRecon::correctProjections() {
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Define the size of the Image space
	int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
	int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	//Define memory size of the raw data as unsigned short
	int size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	size_t sizeProj = size_proj * sizeof(unsigned short);

	//define the GPU kernel based on size of "ideal projection"
	dim3 dimBlockProj(32, 32);
	dim3 dimGridProj(Sys->Proj->Nx / 32 + Cx, Sys->Proj->Ny / 32 + Cy);

	dim3 dimBlockIm2(32, 8);
	dim3 dimGridIm2(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);
	dim3 dimGridIm3(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Proj->NumViews * Sys->Recon->Nz);//, Sys->Proj->NumViews
	dim3 dimGridIm4(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, 1);//Sys->Proj->NumViews

	//Define the size of the sinogram space
	int size_sino = MemP_Nx * MemP_Ny;
	size_t sizeSino = MemP_Nx  * MemP_Ny * Sys->Proj->NumViews * sizeof(float);

	dim3 dimGridSum(1, 1);
	dim3 dimBlockSum(1024, 1);

	//Cycle through each stream and do simple log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) {
		cuda(MemcpyAsync(d_Proj, Sys->Proj->RawData + view*size_proj, sizeProj, cudaMemcpyHostToDevice));

		KERNELCALL2(LogCorrectProj, dimGridProj, dimBlockProj, d_Sino, view, d_Proj, USHRT_MAX);

		KERNELCALL3(GetMaxImageVal, dimGridSum, dimBlockSum, 1024 * sizeof(float), d_Sino + (MemR_Nx * MemR_Ny)*view, MemR_Nx * MemR_Ny, d_MaxVal);

		KERNELCALL3(GetMinImageVal, dimGridSum, dimBlockSum, 1024 * sizeof(float), d_Sino + (MemR_Nx * MemR_Ny)*view, MemR_Nx * MemR_Ny, d_MinVal);

		KERNELCALL2(rescale, dimGridProj, dimBlockProj, d_Sino, view, d_MaxVal, d_MinVal);
	}

	if (!continuousMode) {
		//Initial projection
		cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), MemP_Nx, MemP_Ny*Sys->Proj->NumViews, sinoPitch));

		KERNELCALL2(BackProjectError, dimGridIm2, dimBlockIm2, d_Image, Beta);
	}
}

TomoError TomoRecon::test(int index) {
	int size_proj = 0;
	size_t sizeProj = 0;

	switch (currentDisplay) {
	case raw_images2://same as raw_images
	case raw_images:
		size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
		sizeProj = size_proj * sizeof(unsigned short);
		cuda(MemcpyAsync(d_Proj, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));
		resizeImage(d_Proj, Sys->Proj->Nx, Sys->Proj->Ny, *ca, width, height, USHRT_MAX);
		break;
	case sino_images:
		size_proj = 1920 * Sys->Proj->Ny;
		//size_t sizeProj = size_proj * sizeof(float);
		//cuda(Memcpy(d_Sino, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));
		//resizeImage(d_Sino+ size_proj*index, 1920, Sys->Proj->Ny, *ca, width, height, log(USHRT_MAX) - 1);
		resizeImage(d_Sino + size_proj*index, 1920, Sys->Proj->Ny, *ca, width, height, USHRT_MAX);
		break;
	case norm_images:
		//skip straight to the first iteration
		//resizeImage(d_Norm + size_proj*index, 1920, Sys->Proj->Ny, *ca, width, height, 1);
		//break;
	case recon_images:
	{
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

		//dim3 dimGridSum(1, Sys->Recon->Nz);
		//dim3 dimBlockSum(1024, 1);
		//KERNELCALL4(GetMaxImageVal, dimGridSum, dimBlockSum, 1024 * sizeof(float), stream, d_Image, MemR_Nx * MemR_Ny, d_MaxVal);
		if (blocks > 0)
			KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, Sys->Proj->Nx, Sys->Proj->Ny, width, height, d_MaxVal, index);

		//float h_MaxVal;
		//cuda(Memcpy(&h_MaxVal, d_MaxVal, sizeof(float), cudaMemcpyDeviceToHost));

		//std::cout << "The max reconstructed value is: " << h_MaxVal << std::endl;
		//resizeImage(d_Image + size_proj*index, Sys->Proj->Nx, Sys->Proj->Ny, *ca, width, height, log(SHRT_MAX) - 1);
		//resizeImage(d_Error, 1920, Sys->Proj->Ny, *ca, width, height, log(SHRT_MAX) - 1);
	}
		break;
	case error_images:
	{
		int Cx = 1;
		int Cy = 1;
		if (Sys->Proj->Nx % 16 == 0) Cx = 0;
		if (Sys->Proj->Ny % 16 == 0) Cy = 0;

		//Define the size of the Image space
		int MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
		int MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;
		int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
		int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

		size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemP_Nx, MemP_Ny*Sys->Proj->NumViews, errorPitch));
		cuda(BindSurfaceToArray(displaySurface, *ca));
		//cuda(BindSurfaceToArray(imageSurface, (cudaArray_t)d_Proj));

		const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

		//dim3 dimGridSum(1, Sys->Proj->NumViews);
		//dim3 dimBlockSum(1024, 1);
		//KERNELCALL4(GetMaxImageVal, dimGridSum, dimBlockSum, 1024 * sizeof(float), stream, d_Error, MemR_Nx * MemR_Ny, d_MaxVal);
		if (blocks > 0)
			KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, Sys->Proj->Nx, Sys->Proj->Ny, width, height, d_MaxVal, index);

		//float h_MaxVal;
		//cuda(Memcpy(&h_MaxVal, d_MaxVal, sizeof(float), cudaMemcpyDeviceToHost));

		//std::cout << "The max reconstructed value is: " << h_MaxVal << std::endl;
	}
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

	dim3 dimBlockIm2(32, 8);
	dim3 dimGridIm2(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);
	dim3 dimGridIm3(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, 1);//Sys->Proj->NumViews * Sys->Recon->Nz
	dim3 dimGridIm4(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Proj->NumViews);

	dim3 dimBlockSino(16, 16);
	dim3 dimGridSino(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);//, Sys->Proj->NumViews

	dim3 dimBlockCorr(32, 4);
	dim3 dimGridCorr((int)floor(Sys->Recon->Ny / 32), (Sys->Recon->Nz / 4) + 1);


	const int blocks = (Sys->Recon->Nx * Sys->Recon->Ny + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;
	dim3 blocks3(blocks, Sys->Recon->Nz, 1);

	//cuda(Memset2D(d_Error, errorPitch, 0, MemR_Nx * sizeof(float), MemR_Ny*Sys->Proj->NumViews));

	cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny*Sys->Recon->Nz, imagePitch));

	KERNELCALL2(ProjectImage, dimGridIm4, dimBlockIm2, d_Sino, d_Error);

	cuda(BindTexture2D(NULL, textError, d_Error, cudaCreateChannelDesc<float>(), MemP_Nx, MemP_Ny*Sys->Proj->NumViews, errorPitch));

	KERNELCALL2(BackProjectError, dimGridIm2, dimBlockIm2, d_Image, Beta);
	//if (Sys->UsrIn->SmoothEdge == 1)
	//	KERNELCALL2(CorrectEdgesX, dimGridCorr, dimBlockCorr, d_Image, d_Image2);

	Beta = Beta*DECAY;
	iteration++;
}

TomoError TomoRecon::singleFrame() {
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Define the size of the Image space
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	dim3 dimBlockIm(32, 8);
	dim3 dimGridIm(Sys->Recon->Nx / 32 + Cx, Sys->Recon->Ny / 8 + Cy, 1);

	//Initial projection
	cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), MemP_Nx, MemP_Ny*Sys->Proj->NumViews, sinoPitch));

	KERNELCALL2(projectSlice, dimGridIm, dimBlockIm, d_Image, sliceIndex);
}

float TomoRecon::getDistance() {
	return Sys->SysGeo.ZDist + Sys->SysGeo.ZPitch*sliceIndex;
}