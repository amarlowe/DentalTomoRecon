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

//Conversion Helpers
__device__ float xP2MM_d(float p) {
	return (p + 0.5 - d_HalfPx) * d_PitchPx;
}

__device__ float yP2MM_d(float p) {
	return (p + 0.5 - d_HalfPy) * d_PitchPy;
}

__device__ float xR2MM_d(float r) {
	return (r + 0.5 - d_HalfNx) * d_PitchNx;
}

__device__ float yR2MM_d(float r) {
	return (r + 0.5 - d_HalfNy) * d_PitchNy;
}

__device__ float xMM2P_d(float m) {
	return m / d_PitchPx - 0.5 + d_HalfPx;
}

__device__ float yMM2P_d(float m) {
	return m / d_PitchPy - 0.5 + d_HalfPy;
}

__device__ float xMM2R_d(float m) {
	return m / d_PitchNx - 0.5 + d_HalfNx;
}

__device__ float yMM2R_d(float m) {
	return m / d_PitchNy - 0.5 + d_HalfNy;
}

//Loop unrolling templates, device functions are mapped to inlines 99% of the time
template<int i> __device__ float convolutionRow(float x, float y, float kernel[KERNELSIZE]){
	return
		tex2D(textImage, x + (float)(KERNELRADIUS - i), y) * kernel[i]
		+ convolutionRow<i - 1>(x, y, kernel);
}

template<> __device__ float convolutionRow<-1>(float x, float y, float kernel[KERNELSIZE]){
	return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y, float kernel[KERNELSIZE]){
	return
		tex2D(textImage, x, y + (float)(KERNELRADIUS - i)) * kernel[i]
		+ convolutionColumn<i - 1>(x, y, kernel);
}

template<> __device__ float convolutionColumn<-1>(float x, float y, float kernel[KERNELSIZE]){
	return 0;
}

//Image metric generators
__global__ void convolutionRowsKernel(float *d_Dst, float kernel[KERNELSIZE]) {
	const int ix = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int iy = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= d_Nx - KERNELRADIUS || iy >= d_Py - KERNELRADIUS || ix < KERNELRADIUS || iy < KERNELRADIUS)
		return;

	d_Dst[MUL_ADD(iy, d_imagePitch, ix)] = convolutionRow<KERNELSIZE>(x, y, kernel);
}

__global__ void convolutionColumnsKernel(float *d_Dst, float kernel[KERNELSIZE]){
	const int ix = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int iy = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= d_Nx - KERNELRADIUS || iy >= d_Py - KERNELRADIUS || ix < KERNELRADIUS || iy < KERNELRADIUS)
		return;

	d_Dst[MUL_ADD(iy, d_imagePitch, ix)] = convolutionColumn<KERNELSIZE>(x, y, kernel);
}

__global__ void squareMag(float *d_Dst, float *src1, float *src2, int pitchIn, int pitchOut) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= d_Nx || y >= d_Py || x < 0 || y < 0)
		return;

	d_Dst[MUL_ADD(y, pitchOut, x)] = pow(src1[MUL_ADD(y, pitchIn, x)],2) + pow(src2[MUL_ADD(y, pitchIn, x)],2);
}

__global__ void squareDiff(float *d_Dst, int view, float xOff, float yOff, int pitchOut) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= d_Nx || y >= d_Py || x < 0 || y < 0)
		return;

	
	d_Dst[MUL_ADD(y, pitchOut, x)] = pow(tex2D(textError, x - xOff, y - yOff + view*d_MPy) - tex2D(textError, x, y + (NUMVIEWS / 2)*d_MPy), 2);
}

//Display functions
template<typename T>
__global__ void resizeKernel(T* input, int wIn, int hIn, int wOut, int hOut, float scale, int xOff, int yOff, int maxVal, int lightOff) {
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	int x = idx % wOut;
	int y = idx / wOut;

	float sum = 0;
	int i = (x - (wOut - wIn / scale) / 2)*scale + xOff;
	int j = (y - (hOut - hIn / scale) / 2)*scale + yOff;
	if (i > 0 && j > 0 && i < wIn && j < hIn)
		sum = input[j*wIn + i];

	//sum = sum / maxVar * 255;
	if (sum > 0)
		sum = (logf((float)USHRT_MAX) - logf(sum)) / (logf((float)USHRT_MAX) - 1) * UCHAR_MAX;
	if (sum > 0) sum += lightOff;
	if (sum < 0) sum = 0;
	if (sum > maxVal) sum = UCHAR_MAX;
	else sum = sum / maxVal * UCHAR_MAX;

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

__global__ void resizeKernelTex(int wIn, int hIn, int wOut, int hOut, int index, float scale, int xOff, int yOff, int maxVal, int lightOff, bool log) {//float * maxVar, 
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % wOut;
	const int y = idx / wOut;
	bool negative = false;

	float sum = 0;
	int i = (x - (wOut - wIn / scale) / 2)*scale + xOff;
	int j = (y - (hOut - hIn / scale) / 2)*scale + yOff;
	if (i > 0 && j > 0 && i < wIn && j < hIn)
		sum = tex2D(textImage, (float)i + 0.5f, (float)j + 0.5f + (float)(index * hIn));;

	if (log) {
		if(sum > 0)
			sum = (logf((float)USHRT_MAX) - logf(sum)) / (logf((float)USHRT_MAX) - 1) * UCHAR_MAX;
		if (sum > 0) sum += lightOff;
		if (sum < 0) {
			negative = true;
			sum = abs(sum);
		}
		if (sum > maxVal) sum = UCHAR_MAX;
		else sum = sum / maxVal * UCHAR_MAX;
	}
	else {
		sum = sum / maxVal * UCHAR_MAX;
		if (sum < 0) {
			negative = true;
			sum = abs(sum);
		}
	}
	

	union pxl_rgbx_24 rgbx;
	if (sum > 255) {
		rgbx.na = 0xFF;
		rgbx.r = 255;//flag errors with big red spots
		rgbx.g = 0;
		rgbx.b = 0;
	}
	else {
		rgbx.na = 0xFF;
		if (negative) {
			rgbx.r = 0;
			rgbx.g = 0;
		}
		else {
			rgbx.r = sum;
			rgbx.g = sum;
		}
		rgbx.b = sum;
	}

	surf2Dwrite(rgbx.b32,
		displaySurface,
		x * sizeof(rgbx),
		y,
		cudaBoundaryModeZero); // squelches out-of-bound writes
}

__global__ void drawSelectionBox(int UX, int UY, int LX, int LY, int wOut) {
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % wOut;
	const int y = idx / wOut;

	if ((x >= UX && x < UX + LINEWIDTH && y >= LY - LINEWIDTH && y < UY + LINEWIDTH) ||
		(x >= LX - LINEWIDTH && x < LX && y >= LY - LINEWIDTH && y < UY + LINEWIDTH) ||
		(y >= UY && y < UY + LINEWIDTH && x >= LX && x < UX) ||
		(y >= LY - LINEWIDTH && y < LY && x >= LX && x < UX)) {
		union pxl_rgbx_24 rgbx;
		rgbx.na = 0xFF;
		rgbx.r = 255;//flag errors with big red spots
		rgbx.g = 0;
		rgbx.b = 0;

		surf2Dwrite(rgbx.b32,
			displaySurface,
			x * sizeof(rgbx),
			y,
			cudaBoundaryModeZero); // squelches out-of-bound writes
	}
}

__global__ void drawSelectionBar(int X, int Y, int wOut, bool vertical) {
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % wOut;
	const int y = idx / wOut;

	if (!vertical && (x >= X && x < X + LINEWIDTH && y >= Y && y < Y + BARHEIGHT) ||
		vertical && (y >= Y && y < Y + LINEWIDTH && x >= X - BARHEIGHT && x < X)) {
		union pxl_rgbx_24 rgbx;
		rgbx.na = 0xFF;
		rgbx.r = 255;//flag errors with big red spots
		rgbx.g = 0;
		rgbx.b = 0;

		surf2Dwrite(rgbx.b32,
			displaySurface,
			x * sizeof(rgbx),
			y,
			cudaBoundaryModeZero); // squelches out-of-bound writes
	}
}

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

//Create the single slice projection image
__global__ void projectSlice(float * IM, float distance) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float values[NUMVIEWS];

	//Set a normalization and pixel value to 0
	float error = 0.0f;
	float count = 0.0f;

	//Check image boundaries
	if ((i >= d_Nx) || (j >= d_Ny)) return;

	for (int view = 0; view < NUMVIEWS; view++) {
		float dz = distance / d_beamz[view];
		float x = xMM2P_d((xR2MM_d(i) + d_beamx[view] * dz) / (1 + dz));
		float y = yMM2P_d((yR2MM_d(j) + d_beamy[view] * dz) / (1 + dz));

		//Update the value based on the error scaled and save the scale
		if (y > 0 && y < d_MPy && x > 0 && x < d_MPx) {
			values[view] = tex2D(textError, x, y + view*d_MPy);
			if (values[view] != 0) {
				error += values[view];
				count++;
			}
		}
	}

	if (count > 0)
		IM[j*d_MNx + i] = error / count;
}

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
		val = max(val, data[thread + 32]);
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
		val = min(val, data[thread + 32]);
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

__global__ void sumReduction(float * Image, int pitch, float * sum, int lowX, int upX, int lowY, int upY) {
	//Define shared memory to read all the threads
	extern __shared__ float data[];

	//define the thread and block location
	const int thread = threadIdx.x;
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	float val;

	if (x >= upX || y >= upY || x < lowX || y < lowY) {
		val = 0.0;
		data[thread] = 0.0;
	}
	else {
		val = Image[y*pitch + x];
		data[thread] = val;
		//Image[y*pitch + x] = 0.0;//test display
	}

	//Each thread puts its local sum into shared memory
	__syncthreads();

	//Do reduction in shared memory
	if (thread < 512) data[thread] = val += data[thread + 512];
	__syncthreads();

	if (thread < 256) data[thread] = val += data[thread + 256];
	__syncthreads();

	if (thread < 128) data[thread] = val += data[thread + 128];
	__syncthreads();

	if (thread < 64) data[thread] = val += data[thread + 64];
	__syncthreads();

	if (thread < 32) {
		// Fetch final intermediate sum from 2nd warp
		data[thread] = val += data[thread + 32];
		// Reduce final warp using shuffle
		for (int offset = warpSize / 2; offset > 0; offset /= 2) {
			val += __shfl_down(val, offset);
		}
	}

	//write the result for this block to global memory
	if (thread == 0 && val > 0) {
		atomicAdd(sum, val);
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

	if (blocks > 0) {
		KERNELCALL2(resizeKernel, blocks, PXL_KERNEL_THREADS_PER_BLOCK, in, wIn, hIn, wOut, hOut, scale, xOff, yOff, UCHAR_MAX / pow(LIGHTFACTOR, light), LIGHTOFFFACTOR*lightOff);
		if (baseX >= 0 && currX >= 0)
			KERNELCALL4(drawSelectionBox, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, max(baseX, currX), max(baseY, currY), min(baseX, currX), min(baseY, currY), width);
		if(lowX >= 0)
			KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, lowX, lowY, width, vertical);
		if (upX >= 0)
			KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, upX, upY, width, vertical);
	}

	return Tomo_OK;
}

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
	//Normalize Geometries
	float baselinex = Sys->SysGeo.EmitX[NUMVIEWS / 2];
	float baseliney = Sys->SysGeo.EmitY[NUMVIEWS / 2];
	for (int i = 0; i < NUMVIEWS; i++) {
		Sys->SysGeo.EmitX[i] -= baselinex;
		Sys->SysGeo.EmitY[i] -= baseliney;
	}

	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";

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

	Cx = 1;
	Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Set the memory size as slightly larger to make even multiple of 16
	MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	MemR_Nx = (Sys->Recon->Nx / 16 + Cx) * 16;
	MemR_Ny = (Sys->Recon->Ny / 16 + Cy) * 16;

	//Define the size of each of the memory spaces on the gpu in number of bytes
	sizeProj = Sys->Proj->Nx * Sys->Proj->Ny * sizeof(unsigned short);
	sizeSino = MemP_Nx * MemP_Ny * Sys->Proj->NumViews * sizeof(float);
	
	cuda(Malloc((void**)&d_Proj, sizeProj));
	cuda(MallocPitch((void**)&d_Sino, &sinoPitch, MemP_Nx * sizeof(float), MemP_Ny * Sys->Proj->NumViews));
	cuda(Malloc((void**)&beamx, Sys->Proj->NumViews * sizeof(float)));
	cuda(Malloc((void**)&beamy, Sys->Proj->NumViews * sizeof(float)));
	cuda(Malloc((void**)&beamz, Sys->Proj->NumViews * sizeof(float)));
	cuda(Malloc((void**)&d_MaxVal, sizeof(float)));
	cuda(Malloc((void**)&d_MinVal, sizeof(float)));

	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";

	//Set the values of the image and sinogram to all 0
	cuda(Memset2DAsync(d_Sino, sinoPitch, 0, MemP_Nx * sizeof(float), MemP_Ny * Sys->Proj->NumViews));
	cuda(MemcpyAsync(beamx, Sys->SysGeo.EmitX, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(beamy, Sys->SysGeo.EmitY, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(beamz, Sys->SysGeo.EmitZ, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));

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

	return Tomo_OK;
}

TomoError TomoRecon::mallocContinuous() {
	contThreads.x = 32;
	contThreads.y = 8;
	contBlocks.x = Sys->Recon->Nx / 32 + Cx;
	contBlocks.y = Sys->Recon->Ny / 8 + Cy;

	distance = Sys->Recon->Slice_0_z;

	sizeIM = MemR_Nx * MemR_Ny * sizeof(float);
	sizeError = MemP_Nx * MemP_Ny * sizeof(float);

	//Set up dispaly and buffer (error) regions
	cuda(MallocPitch((void**)&d_Image, &imagePitch, MemR_Nx * sizeof(float), MemR_Ny));
	cuda(MallocPitch((void**)&d_Error, &errorPitch, MemP_Nx * sizeof(float), MemP_Ny));

	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";

	size_t pitch = errorPitch / sizeof(float);
	cuda(MemcpyToSymbolAsync(d_errorPitch, &pitch, sizeof(size_t)));
	pitch = imagePitch / sizeof(float);
	cuda(MemcpyToSymbolAsync(d_imagePitch, &pitch, sizeof(size_t)));
	pitch = sinoPitch / sizeof(float);
	cuda(MemcpyToSymbolAsync(d_sinoPitch, &pitch, sizeof(size_t)));

	//Setup derivative buffers
	cuda(Malloc(&xDer, sizeIM * sizeof(float)));

	//Set up all kernels
	cuda(Malloc(&d_noop, KERNELSIZE * sizeof(float)));
	cuda(Malloc(&d_gauss, KERNELSIZE * sizeof(float)));
	cuda(Malloc(&d_gaussDer, KERNELSIZE * sizeof(float)));
	cuda(Malloc(&d_gaussDer2, KERNELSIZE * sizeof(float)));
	cuda(Malloc(&d_gaussDer3, KERNELSIZE * sizeof(float)));

	float tempNoop[KERNELSIZE];
	setNOOP(tempNoop);
	cuda(Memcpy(d_noop, tempNoop, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernel[KERNELSIZE];
	setGauss(tempKernel);
	cuda(Memcpy(d_gauss, tempKernel, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer[KERNELSIZE];
	setGaussDer(tempKernelDer);
	cuda(Memcpy(d_gaussDer, tempKernelDer, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer2[KERNELSIZE];
	setGaussDer2(tempKernelDer2);
	cuda(Memcpy(d_gaussDer2, tempKernelDer2, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer3[KERNELSIZE];
	setGaussDer3(tempKernelDer3);
	cuda(Memcpy(d_gaussDer3, tempKernelDer3, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	reconMemSet = true;

	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";

	return Tomo_OK;
}

//Functions to save the images
TomoError TomoRecon::CopyAndSaveImages(){
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

//Functions referenced from main
TomoError TomoRecon::SetUpGPUForRecon(){
	//Set up GPU Memory space
	DefineReconstructSpace();

	//Set up GPU memory space
	tomo_err_throw(SetUpGPUMemory());

	return Tomo_OK;
}

//Fucntion to free the gpu memory after program finishes
TomoError TomoRecon::FreeGPUMemory(void){
	//Free memory allocated on the GPU
	cuda(Free(d_Proj));
	cuda(Free(d_Image));
	cuda(Free(d_Error));
	cuda(Free(d_Sino));

	cuda(Free(beamx));
	cuda(Free(beamy));
	cuda(Free(beamz));

	cuda(Free(d_noop));
	cuda(Free(d_gauss));
	cuda(Free(d_gaussDer));
	cuda(Free(d_gaussDer2));
	cuda(Free(d_gaussDer3));

	cuda(Free(xDer));

	//Unbind the texture array and free the cuda array
	cuda(UnbindTexture(textSino));

	return Tomo_OK;
}

//Lower level functions for user interactive debugging
TomoError TomoRecon::LoadProjections(int index) {
	//Define memory size of the raw data as unsigned short
	int size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	size_t sizeProj = size_proj * sizeof(unsigned short);

	cuda(MemcpyAsync(d_Proj, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));

	return Tomo_OK;
}

TomoError TomoRecon::correctProjections() {
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
}

TomoError TomoRecon::test(int index) {
	int size_proj = 0;
	size_t sizeProj = 0;

	scale = max((float)Sys->Proj->Nx / (float)width, (float)Sys->Proj->Ny / (float)height) * pow(ZOOMFACTOR, -zoom);
	float maxX = (pow(ZOOMFACTOR, zoom) - 1) * (float)Sys->Proj->Nx / 2;
	float maxY = (pow(ZOOMFACTOR, zoom) - 1) * (float)Sys->Proj->Ny / 2;
	if (xOff != 0 && xOff > maxX || xOff < -maxX) xOff = xOff > 0 ? maxX : -maxX;
	if (yOff != 0 && yOff > maxY || yOff < -maxY) yOff = yOff > 0 ? maxY : -maxY;

	switch (currentDisplay) {
	case raw_images2://same as raw_images
	case raw_images:
		{
			size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
			sizeProj = size_proj * sizeof(unsigned short);
			cuda(MemcpyAsync(d_Proj, Sys->Proj->RawData + index*size_proj, sizeProj, cudaMemcpyHostToDevice));
			resizeImage(d_Proj, Sys->Proj->Nx, Sys->Proj->Ny, *ca, width, height, USHRT_MAX);
		}
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
			size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
			cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny*Sys->Recon->Nz, imagePitch));
			cuda(BindSurfaceToArray(displaySurface, *ca));

			const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

			if (blocks > 0) {
				KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, 
					Sys->Proj->Nx, Sys->Proj->Ny, width, height, index, scale, xOff, yOff, UCHAR_MAX/pow(LIGHTFACTOR, light), LIGHTOFFFACTOR*lightOff, derDisplay == no_der);
				if (baseXr >= 0 && currXr >= 0)
					KERNELCALL4(drawSelectionBox, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(max(baseXr, currXr),true), 
						I2D(max(baseYr, currYr),false), I2D(min(baseXr, currXr),true), I2D(min(baseYr, currYr),false), width);
				if (lowXr >= 0)
					KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(lowXr,true), I2D(lowYr,false), width, vertical);
				if (upXr >= 0)
					KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(upXr,true), I2D(upYr,false), width, vertical);
			}
		}
		break;
	case error_images:
		{
			size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
			cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemP_Nx, MemP_Ny*Sys->Proj->NumViews, errorPitch));
			cuda(BindSurfaceToArray(displaySurface, *ca));

			const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

			if (blocks > 0)
				KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, Sys->Proj->Nx, Sys->Proj->Ny, width, height, index, scale, xOff, yOff, USHRT_MAX, LIGHTOFFFACTOR*lightOff, true);
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

	return Tomo_OK;
}

TomoError TomoRecon::singleFrame() {
	//Initial projection
	cuda(Memset2DAsync(d_Image, imagePitch, 0, MemR_Nx * sizeof(float), MemR_Ny));
	cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), MemP_Nx, MemP_Ny*Sys->Proj->NumViews, sinoPitch));

	KERNELCALL2(projectSlice, contBlocks, contThreads, d_Image, distance);

	switch (derDisplay) {
	case no_der:
		break;
	case der_x:
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, d_gaussDer);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, d_Image, d_gauss);
		break;
	case der_y:
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, d_Error, d_gaussDer);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Image, d_gauss);
		break;
	case square_mag:
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, d_gaussDer);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, xDer, d_gauss);

		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, d_Error, d_gaussDer);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Image, d_gauss);

		KERNELCALL2(squareMag, contBlocks, contThreads, d_Image, xDer, d_Image, MemR_Nx, imagePitch/sizeof(float));
		break;
	case slice_diff:
	{
		float xOff = -Sys->SysGeo.EmitX[diffSlice] * distance / Sys->SysGeo.EmitZ[diffSlice] / Sys->Recon->Pitch_x;
		float yOff = -Sys->SysGeo.EmitY[diffSlice] * distance / Sys->SysGeo.EmitZ[diffSlice] / Sys->Recon->Pitch_y;
		KERNELCALL2(squareDiff, contBlocks, contThreads, d_Image, diffSlice, xOff, yOff, imagePitch / sizeof(float));
	}
		break;
	case der2_x:
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, d_gaussDer2);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, d_Image, d_gauss);
		break;
	case der2_y:
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, d_Error, d_gaussDer2);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Image, d_gauss);
		break;
	case der3_x:
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, d_gaussDer2);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, d_Image, d_gauss);
		break;
	case der3_y:
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, imagePitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, d_Error, d_gaussDer3);
		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), MemR_Nx, MemR_Ny, errorPitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Image, d_gauss);
		break;
	}

	return Tomo_OK;
}

inline float TomoRecon::focusHelper() {
	//dim3 dimGridSum(1, Sys->Recon->Nz);
	dim3 dimGridSum(Sys->Recon->Nx / 1024 + Cx, Sys->Recon->Ny + Cy);
	dim3 dimBlockSum(1024, 1);//TODO
	int sumSize = 1024 * sizeof(float);

	//Render new frame
	singleFrame();

	//get the focus metric
	float currentBest;
	cuda(MemsetAsync(d_MaxVal, 0, sizeof(float)));
	//TODO: check boundary conditions
	KERNELCALL3(sumReduction, dimGridSum, dimBlockSum, sumSize, d_Image, MemR_Nx, d_MaxVal, min(baseXr, currXr), max(baseXr, currXr), min(baseYr, currYr), max(baseYr, currYr));
	cuda(Memcpy(&currentBest, d_MaxVal, sizeof(float), cudaMemcpyDeviceToHost));
	return currentBest;
}

//Projection space to recon space
TomoError TomoRecon::P2R(int* rX, int* rY, int pX, int pY, int view){
	float dz = distance / Sys->SysGeo.EmitZ[view];
	*rX = xMM2R(xP2MM(pX) * (1 + dz) - Sys->SysGeo.EmitX[view] * dz);
	*rY = yMM2R(yP2MM(pY) * (1 + dz) - Sys->SysGeo.EmitY[view] * dz);

	return Tomo_OK;
}

//Recon space to projection space
TomoError TomoRecon::R2P(int* pX, int* pY, int rX, int rY, int view) {
	float dz = distance / Sys->SysGeo.EmitZ[view];
	*pX = xMM2P((xR2MM(rX) + Sys->SysGeo.EmitX[view] * dz) / (1 + dz));
	*pY = yMM2P((yR2MM(rY) + Sys->SysGeo.EmitY[view] * dz) / (1 + dz));

	return Tomo_OK;
}

//Image space to on-screen display
TomoError TomoRecon::I2D(int* dX, int* dY, int iX, int iY) {
	float innerOffx = (width - Sys->Proj->Nx / scale) / 2;
	float innerOffy = (height - Sys->Proj->Ny / scale) / 2;

	*dX = (iX - xOff) / scale + innerOffx;
	*dY = (iY - yOff) / scale + innerOffy;

	return Tomo_OK;
}

//Projection space to recon space
int TomoRecon::P2R(int p, int view, bool xDir) {
	float dz = distance / Sys->SysGeo.EmitZ[view];
	if (xDir)
		return xMM2R(xP2MM(p) * (1 + dz) - Sys->SysGeo.EmitX[view] * dz);
	//else
	return yMM2R(yP2MM(p) * (1 + dz) - Sys->SysGeo.EmitY[view] * dz);
}

//Recon space to projection space
int TomoRecon::R2P(int r, int view, bool xDir) {
	float dz = distance / Sys->SysGeo.EmitZ[view];
	if (xDir)
		return xMM2P((xR2MM(r) + Sys->SysGeo.EmitX[view] * dz) / (1 + dz));
	//else
	return yMM2P((yR2MM(r) + Sys->SysGeo.EmitY[view] * dz) / (1 + dz));
}

//Image space to on-screen display
int TomoRecon::I2D(int i, bool xDir) {
	if (xDir) {
		float innerOffx = (width - Sys->Proj->Nx / scale) / 2;

		return (i - xOff) / scale + innerOffx;
	}
	//else
	float innerOffy = (height - Sys->Proj->Ny / scale) / 2;

	return (i - yOff) / scale + innerOffy;
}

//On-screen coordinates to image space
int TomoRecon::D2I(int d, bool xDir) {
	if (xDir) {
		float innerOffx = (width - Sys->Proj->Nx / scale) / 2;

		return (d - innerOffx) * scale + xOff;
	}
	//else
	float innerOffy = (height - Sys->Proj->Ny / scale) / 2;

	return (d - innerOffy) * scale + yOff;
}

TomoError TomoRecon::autoFocus(bool firstRun) {
	static float step;
	static float best;
	static bool linearRegion;
	static bool firstLin = true;
	static float bestDist;
	static int oldLight;

	if (firstRun) {
		step = STARTSTEP;
		best = 0;
		linearRegion = false;
		derDisplay = square_mag;
		oldLight = light;
		light = -100;
		distance = MINDIS;
		bestDist = MINDIS;
	}

	if (continuousMode) {
		float newVal = focusHelper();

		if (!linearRegion) {
			if (newVal > best) {
				best = newVal;
				bestDist = distance;
			}

			distance += step;

			if (distance > MAXDIS) {
				linearRegion = true;
				firstLin = true;
				distance = bestDist;
			}
		}
		else {
			//compare to current
			if (newVal > best) {
				best = newVal;
				bestDist = distance;
				distance += step;
			}
			else {
				if(!firstLin) distance -= step;//revert last move

				//find next step
				step = -step / 2;

				if (abs(step) < LASTSTEP) {
					light = oldLight;
					derDisplay = no_der;
					return Tomo_Done;
				}
				else distance += step;
			}
			
			firstLin = false;
		}

		return Tomo_OK;
	}

	return Tomo_Done;
}

TomoError TomoRecon::autoGeo(bool firstRun) {
	static float step;
	static float best;
	static bool xDir;
	static int view;
	static int oldLight;

	if (firstRun) {
		step = GEOSTART;
		best = FLT_MAX;
		diffSlice = 0;
		xDir = true;
		derDisplay = slice_diff;
		oldLight = light;
		light = -150;
	}

	if (continuousMode) {
		float newVal = focusHelper();
		float newOffset = xDir ? Sys->SysGeo.EmitX[diffSlice] : Sys->SysGeo.EmitY[diffSlice];

		//compare to current
		if (newVal < best) {
			best = newVal;
			newOffset += step;
		}
		else {
			newOffset -= step;//revert last move

			//find next step
			step = -step / 2;
			if (abs(step) < GEOLAST) {
				step = GEOSTART;
				best = FLT_MAX;
				if (xDir) Sys->SysGeo.EmitX[diffSlice] = newOffset;
				else Sys->SysGeo.EmitY[diffSlice] = newOffset;
				xDir = !xDir;
				if (xDir) diffSlice++;//if we're back on xDir, we've made the next step
				if (diffSlice == (NUMVIEWS / 2)) diffSlice++;//skip center, it's what we're comparing
				if (diffSlice >= NUMVIEWS) {
					light = oldLight;
					derDisplay = no_der;
					cuda(MemcpyAsync(beamx, Sys->SysGeo.EmitX, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
					cuda(MemcpyAsync(beamy, Sys->SysGeo.EmitY, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
					return Tomo_Done;
				}
				return Tomo_OK;//skip the rest so values don't bleed into one another
			}

			newOffset += step;
		}

		if (xDir) Sys->SysGeo.EmitX[diffSlice] = newOffset;
		else Sys->SysGeo.EmitY[diffSlice] = newOffset;

		return Tomo_OK;
	}
	
	return Tomo_Done;
}

TomoError TomoRecon::readPhantom(float * resolution) {
	if (vertical) {
		float phanScale = (lowYr - upYr) / (1/LOWERBOUND - 1/UPPERBOUND);
		float * h_xDer2 = (float*)malloc(MemR_Nx*MemR_Ny * sizeof(float));
		cuda(Memcpy(h_xDer2, d_Image, MemR_Nx*MemR_Ny*sizeof(float), cudaMemcpyDeviceToHost));
		//Get x range from the bouding box
		int startX = min(baseXr, currXr);
		int endX = max(baseXr, currXr);
		int thisY = lowYr;//Get beginning y val from tick mark
		while (thisY >= upYr) {//y counts down
			int thisX = startX;
			int negCross = 0;
			bool negativeSpace = false;
			float negAcc = 0;
			while (thisX < endX) {
				if (negativeSpace) {
					float val = h_xDer2[thisY * MemR_Nx + thisX];
					if (val > 0) {
						negativeSpace = false;
						if (negAcc < -INTENSITYTHRESH) {
							negCross++;
						}
					}
					else {
						negAcc += val;
					}
				}
				else {
					float val = h_xDer2[thisY * MemR_Nx + thisX];
					if (val < 0) {
						negativeSpace = true;
						negAcc = val;
					}
				}
				thisX++;
			}
			if (negCross < LINEPAIRS) {
				thisY++;
				break;
			}
			thisY--;
		}
		*resolution = phanScale / (thisY - lowYr + phanScale / LOWERBOUND);
		free(h_xDer2);
		
	}
	else {
		float phanScale = (lowXr - upXr) / (1 / LOWERBOUND - 1 / UPPERBOUND);
		float * h_yDer2 = (float*)malloc(MemR_Nx*MemR_Ny * sizeof(float));
		cuda(Memcpy(h_yDer2, d_Image, MemR_Nx*MemR_Ny * sizeof(float), cudaMemcpyDeviceToHost));
		//Get x range from the bouding box
		int startY = min(baseYr, currYr);
		int endY = max(baseYr, currYr);
		int thisX = lowXr;//Get beginning y val from tick mark
		while (thisX >= upXr) {//y counts down
			int thisY = startY;
			int negCross = 0;
			bool negativeSpace = false;
			float negAcc = 0;
			while (thisY < endY) {
				if (negativeSpace) {
					float val = h_yDer2[thisY * MemR_Nx + thisX];
					if (val > 0) {
						negativeSpace = false;
						if (negAcc < -INTENSITYTHRESH) {
							negCross++;
						}
					}
					else {
						negAcc += val;
					}
				}
				else {
					float val = h_yDer2[thisY * MemR_Nx + thisX];
					if (val < 0) {
						negativeSpace = true;
						negAcc = val;
					}
				}
				thisY++;
			}
			if (negCross < LINEPAIRS) {
				thisX++;
				break;
			}
			thisX--;
		}
		*resolution = phanScale / (thisX - lowXr + phanScale / LOWERBOUND);
		free(h_yDer2);
	}

	return Tomo_OK;
}

inline float TomoRecon::getDistance() {
	return distance;
}

TomoError TomoRecon::initTolerances(std::vector<toleranceData> &data, int numTests, std::vector<float> offsets) {
	//start set as just the combinations
	for (int i = 0; i < NUMVIEWS; i++) {
		int resultsLen = data.size();//size will change every iteration, pre-record it
		int binRep = 1 << i;
		for (int j = 0; j < resultsLen; j++) {
			toleranceData newData = data[j];
			newData.name += "+";
			newData.name += std::to_string(i);
			newData.numViewsChanged++;
			newData.viewsChanged |= binRep;
			data.push_back(newData);
		}

		//add the base
		toleranceData newData;
		newData.name += std::to_string(i);
		newData.numViewsChanged = 1;
		newData.viewsChanged = binRep;
		data.push_back(newData);
	}

	//blow up with the diffent directions
	int combinations = data.size();//again, changing sizes later on
	for (int i = 0; i < combinations; i++) {
		toleranceData baseline = data[i];

		baseline.thisDir = dir_y;
		data.push_back(baseline);

		baseline.thisDir = dir_z;
		data.push_back(baseline);
	}

	//then fill in the set with all the view changes
	combinations = data.size();//again, changing sizes later on
	for (int i = 0; i < combinations; i++) {
		toleranceData baseline = data[i];
		for (int j = 0; j < offsets.size() - 1; j++) {//skip the last
			toleranceData newData = baseline;
			newData.offset = offsets[j];
			data.push_back(newData);
		}

		//the last one is done in place
		data[i].offset = offsets[offsets.size() - 1];
	}

	//finally, put in a control
	toleranceData control;
	control.name += " none";
	control.numViewsChanged = 0;
	control.viewsChanged = 0;
	control.offset = 0;
	data.push_back(control);

	return Tomo_OK;
}

TomoError TomoRecon::testTolerances(std::vector<toleranceData> &data, bool firstRun) {
	static auto iter = data.begin();
	static int oldLight = light;
	if (firstRun) {
		if(vertical) derDisplay = der2_x;
		else derDisplay = der2_y;
		light = -30;
		iter = data.begin();
		return Tomo_OK;
	}
	
	//for (auto iter = data.begin(); iter != data.end(); ++iter) {
	if (iter == data.end()) {
		light = oldLight;
		derDisplay = no_der;
		return Tomo_Done;
	}
		float geo[NUMVIEWS];
		switch (iter->thisDir) {
		case dir_x:
			memcpy(geo, Sys->SysGeo.EmitX, sizeof(float)*NUMVIEWS);
			break;
		case dir_y:
			memcpy(geo, Sys->SysGeo.EmitY, sizeof(float)*NUMVIEWS);
			break;
		case dir_z:
			memcpy(geo, Sys->SysGeo.EmitZ, sizeof(float)*NUMVIEWS);
			break;
		}

		for (int i = 0; i < NUMVIEWS; i++) {
			bool active = ((iter->viewsChanged >> i) & 1) > 0;//Shift, mask and check
			if (!active) continue;
			if (i < NUMVIEWS / 2) geo[i] -= iter->offset;
			else geo[i] += iter->offset;
		}

		//be safe, recopy values to overwrite previous iterations
		switch (iter->thisDir) {
		case dir_x:
			cuda(MemcpyAsync(beamx, geo, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(beamy, Sys->SysGeo.EmitY, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(beamz, Sys->SysGeo.EmitZ, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			break;
		case dir_y:
			cuda(MemcpyAsync(beamx, Sys->SysGeo.EmitX, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(beamy, geo, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(beamz, Sys->SysGeo.EmitZ, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			break;
		case dir_z:
			cuda(MemcpyAsync(beamx, Sys->SysGeo.EmitX, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(beamy, Sys->SysGeo.EmitY, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(beamz, geo, Sys->Proj->NumViews * sizeof(float), cudaMemcpyHostToDevice));
			break;
		}

		singleFrame();

		float readVal;
		readPhantom(&readVal);
		iter->phantomData = readVal;
	//}
		++iter;
	return Tomo_OK;
}

TomoError TomoRecon::setReconBox(int index) {
	P2R(&upXr, &upYr, upX, upY, 0);
	P2R(&lowXr, &lowYr, lowX, lowY, 0);
	P2R(&currXr, &currYr, currX, currY, 0);
	P2R(&baseXr, &baseYr, baseX, baseY, 0);

	return Tomo_OK;
}

//Conversion Helpers
inline float TomoRecon::xP2MM(int p) {
	return (p + 0.5 - Sys->Proj->Nx / 2) * Sys->Proj->Pitch_x;
}

inline float TomoRecon::yP2MM(int p) {
	return (p + 0.5 - Sys->Proj->Ny / 2) * Sys->Proj->Pitch_y;
}

inline float TomoRecon::xR2MM(int r) {
	return (r + 0.5 - Sys->Recon->Nx / 2) * Sys->Recon->Pitch_x;
}

inline float TomoRecon::yR2MM(int r) {
	return (r + 0.5 - Sys->Recon->Ny / 2) * Sys->Recon->Pitch_y;
}

inline int TomoRecon::xMM2P(float m) {
	return m / Sys->Proj->Pitch_x - 0.5 + Sys->Proj->Nx / 2;
}

inline int TomoRecon::yMM2P(float m) {
	return m / Sys->Proj->Pitch_y - 0.5 + Sys->Proj->Ny / 2;
}

inline int TomoRecon::xMM2R(float m) {
	return m / Sys->Recon->Pitch_x - 0.5 + Sys->Recon->Nx / 2;
}

inline int TomoRecon::yMM2R(float m) {
	return m / Sys->Recon->Pitch_y - 0.5 + Sys->Recon->Ny / 2;
}