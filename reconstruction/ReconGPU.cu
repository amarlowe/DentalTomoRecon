/********************************************************************************************/
/* ReconGPU.cu																				*/
/* Copyright 2017, XinRay Inc., All rights reserved											*/
/********************************************************************************************/

#include "TomoRecon.h"

/********************************************************************************************/
/* CUDA specific helper functions															*/
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

/********************************************************************************************/
/* GPU Function specific functions															*/
/********************************************************************************************/

//Conversion Helpers
__host__ __device__ float xP2MM(float p, float Px, float PitchPx) {
	return (p + 0.5f - Px / 2.0f) * PitchPx;
}

__host__ __device__ float yP2MM(float p, float Py, float PitchPy) {
	return (p + 0.5f - Py / 2.0f) * PitchPy;
}

__host__ __device__ float xR2MM(float r, float Rx, float PitchRx) {
	return (r + 0.5f - Rx / 2.0f) * PitchRx;
}

__host__ __device__ float yR2MM(float r, float Ry, float PitchRy) {
	return (r + 0.5f - Ry / 2.0f) * PitchRy;
}

__host__ __device__ float xMM2P(float m, float Px, float PitchPx) {
	return m / PitchPx - 0.5f + Px / 2.0f;
}

__host__ __device__ float yMM2P(float m, float Py, float PitchPy) {
	return m / PitchPy - 0.5f + Py / 2.0f;
}

__host__ __device__ float xMM2R(float m, float Rx, float PitchRx) {
	return m / PitchRx - 0.5f + Rx / 2.0f;
}

__host__ __device__ float yMM2R(float m, float Ry, float PitchRy) {
	return m / PitchRy - 0.5f + Ry / 2.0f;
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
__global__ void convolutionRowsKernel(float *d_Dst, float kernel[KERNELSIZE], params consts) {
	const int ix = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int iy = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= consts.Rx - KERNELRADIUS || iy >= consts.Ry - KERNELRADIUS || ix < KERNELRADIUS || iy < KERNELRADIUS)
		return;

	d_Dst[MUL_ADD(iy, consts.ReconPitchNum, ix)] = convolutionRow<KERNELSIZE>(x, y, kernel);
}

__global__ void convolutionColumnsKernel(float *d_Dst, float kernel[KERNELSIZE], params consts){
	const int ix = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int iy = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;

	if (ix >= consts.Rx - KERNELRADIUS || iy >= consts.Ry - KERNELRADIUS || ix < KERNELRADIUS || iy < KERNELRADIUS)
		return;

	d_Dst[MUL_ADD(iy, consts.ReconPitchNum, ix)] = convolutionColumn<KERNELSIZE>(x, y, kernel);
}

__global__ void squareMag(float *d_Dst, float *src1, float *src2, int pitchIn, int pitchOut, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Rx || y >= consts.Ry || x < 0 || y < 0)
		return;

	d_Dst[MUL_ADD(y, pitchOut, x)] = pow(src1[MUL_ADD(y, pitchIn, x)],2) + pow(src2[MUL_ADD(y, pitchIn, x)],2);
}

__global__ void squareDiff(float *d_Dst, int view, float xOff, float yOff, int pitchOut, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;
	
	d_Dst[MUL_ADD(y, pitchOut, x)] = pow(tex2D(textError, x - xOff, y - yOff + view*consts.Py) - tex2D(textError, x, y + (NUMVIEWS / 2)*consts.Py), 2);
}

//Display functions
__global__ void resizeKernelTex(int wIn, int hIn, int wOut, int hOut, int index, float scale, int xOff, int yOff, bool derDisplay, params consts) {
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % wOut;
	const int y = idx / wOut;
	bool negative = false;
	bool saturate = false;

	float sum = 0;
	int i = (x - (wOut - wIn / scale) / 2)*scale + xOff;
	int j = (y - (hOut - hIn / scale) / 2)*scale + yOff;
	if (i > 0 && j > 0 && i < wIn && j < hIn)
		sum = tex2D(textImage, (float)i + 0.5f, (float)j + 0.5f + (float)(index * hIn));

	if (sum < 0) {
		negative = true;
		sum = abs(sum);
	}

	if (!derDisplay && consts.log) {
		if (sum != 0) {
			float correctedMax = logf(USHRT_MAX);
			sum = (correctedMax - logf(sum + 1)) / correctedMax * USHRT_MAX;
		}
	}
	//else {
		sum = (sum - consts.minVal) / consts.maxVal * UCHAR_MAX;
		saturate = sum > UCHAR_MAX;
	//}

	union pxl_rgbx_24 rgbx;
	if (saturate) {
		rgbx.na = UCHAR_MAX;
		rgbx.r = UCHAR_MAX;//flag errors with big red spots
		rgbx.g = UCHAR_MAX;//0
		rgbx.b = UCHAR_MAX;//0
	}
	else {
		rgbx.na = UCHAR_MAX;
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
__global__ void LogCorrectProj(float * Sino, int view, unsigned short *Proj, float MaxVal, params consts){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i < consts.Px) && (j < consts.Py)){
		//Flip and invert while converting to float
		int x, y;
		if (consts.flip) x = consts.Px - 1 - i;
		else x = i;
		if (consts.orientation) y = consts.Py - 1 - j;
		else y = j;

		Sino[(y + view*consts.Py)*consts.ProjPitchNum + x] = Proj[j*consts.Px + i];
	}
}

__global__ void rescale(float * Sino, int view, float * MaxVal, float * MinVal, params consts) {
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i < consts.Px) && (j < consts.Py)) {
		float test = Sino[(j + view*consts.Py)*consts.ProjPitchNum + i];
		if (test > 0) {
			Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] = (test - *MinVal + 1.0f) / *MaxVal * USHRT_MAX;//scale from 1 to max
		}
	}
}

//Create the single slice projection image
__global__ void projectSlice(float * IM, float distance, params consts) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float values[NUMVIEWS];

	//Set a normalization and pixel value to 0
	float error = 0.0f;
	float count = 0.0f;

	//Check image boundaries
	if ((i >= consts.Rx) || (j >= consts.Ry)) return;

	for (int view = 0; view < NUMVIEWS; view++) {
		float dz = distance / consts.d_Beamz[view];
		float x = xMM2P((xR2MM(i, consts.Rx, consts.PitchRx) + consts.d_Beamx[view] * dz) / (1 + dz), consts.Px, consts.PitchPx);
		float y = yMM2P((yR2MM(j, consts.Ry, consts.PitchRy) + consts.d_Beamy[view] * dz) / (1 + dz), consts.Py, consts.PitchPy);

		//Update the value based on the error scaled and save the scale
		if (y > 0 && y < consts.Py) {
			values[view] = tex2D(textError, x, y + view*consts.Py);
			if (values[view] != 0) {
				error += values[view];
				count++;
			}
		}
	}

	if (count > 0)
		IM[j*consts.ReconPitchNum + i] = error / count;
	else IM[j*consts.ReconPitchNum + i] = 0;
}

//Ruduction and histogram functions
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

template <typename T>
__global__ void histogram256Kernel(unsigned int *d_Histogram, T *d_Data, unsigned int dataCount, params consts) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	int minX = min(consts.baseXr, consts.currXr);
	int maxX = max(consts.baseXr, consts.currXr);
	int minY = min(consts.baseYr, consts.currYr);
	int maxY = max(consts.baseYr, consts.currYr);

	if (i < minX || i > maxX || j < minY || j > maxY) return;

	float data = (unsigned short)abs(d_Data[MUL_ADD(j, consts.ReconPitchNum, i)]);//whatever it currently is, cast it to ushort
	if (consts.log) {
		if (data != 0) {
			float correctedMax = logf(USHRT_MAX);
			data = (correctedMax - logf(data + 1)) / correctedMax * USHRT_MAX;
		}
	}
	atomicAdd(d_Histogram + ((unsigned short)data >> 8), 1);//bin by the upper 256 bits
}

/********************************************************************************************/
/* Function to interface the CPU with the GPU:												*/
/********************************************************************************************/

//Function to set up the memory on the GPU
TomoError TomoRecon::initGPU(const char * gainFile, const char * darkFile, const char * mainFile){
	//init recon space
	Sys->Recon.Pitch_x = Sys->Proj.Pitch_x;
	Sys->Recon.Pitch_y = Sys->Proj.Pitch_y;
	Sys->Recon.Nx = Sys->Proj.Nx;
	Sys->Recon.Ny = Sys->Proj.Ny;

	//Normalize Geometries
	Sys->Geo.IsoX = Sys->Geo.EmitX[NUMVIEWS / 2];
	Sys->Geo.IsoY = Sys->Geo.EmitY[NUMVIEWS / 2];
	Sys->Geo.IsoZ = Sys->Geo.EmitZ[NUMVIEWS / 2];
	for (int i = 0; i < NUMVIEWS; i++) {
		Sys->Geo.EmitX[i] -= Sys->Geo.IsoX;
		Sys->Geo.EmitY[i] -= Sys->Geo.IsoY;
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

	//Thread and block sizes for standard kernel calls (2d optimized)
	contThreads.x = WARPSIZE;
	contThreads.y = MAXTHREADS / WARPSIZE;
	contBlocks.x = (Sys->Recon.Nx + contThreads.x - 1) / contThreads.x;
	contBlocks.y = (Sys->Recon.Ny + contThreads.y - 1) / contThreads.y;

	//Thread and block sizes for reductions (1d optimized)
	reductionThreads.x = MAXTHREADS;
	reductionBlocks.x = (Sys->Recon.Nx + reductionThreads.x - 1) / reductionThreads.x;
	reductionBlocks.y = Sys->Recon.Ny;

	//Set up dispaly and buffer (error) regions
	cuda(MallocPitch((void**)&d_Image, &reconPitch, Sys->Recon.Nx * sizeof(float), Sys->Recon.Ny));
	cuda(MallocPitch((void**)&d_Error, &reconPitch, Sys->Recon.Nx * sizeof(float), Sys->Recon.Ny));
	cuda(MallocPitch((void**)&d_Sino, &projPitch, Sys->Proj.Nx * sizeof(float), Sys->Proj.Ny * Sys->Proj.NumViews));

	reconPitchNum = (int)reconPitch / sizeof(float);

	//Define the size of each of the memory spaces on the gpu in number of bytes
	sizeProj = Sys->Proj.Nx * Sys->Proj.Ny * sizeof(unsigned short);
	sizeSino = projPitch * Sys->Proj.Ny * Sys->Proj.NumViews;
	sizeIM = reconPitch * Sys->Recon.Ny;
	sizeError = reconPitch * Sys->Recon.Ny;
	
	cuda(Malloc((void**)&constants.d_Beamx, Sys->Proj.NumViews * sizeof(float)));
	cuda(Malloc((void**)&constants.d_Beamy, Sys->Proj.NumViews * sizeof(float)));
	cuda(Malloc((void**)&constants.d_Beamz, Sys->Proj.NumViews * sizeof(float)));
	cuda(Malloc((void**)&d_MaxVal, sizeof(float)));
	cuda(Malloc((void**)&d_MinVal, sizeof(float)));

	//Copy geometries
	cuda(MemcpyAsync(constants.d_Beamx, Sys->Geo.EmitX, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(constants.d_Beamy, Sys->Geo.EmitY, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(constants.d_Beamz, Sys->Geo.EmitZ, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));

	//Define the textures
	textImage.filterMode = cudaFilterModeLinear;
	textImage.addressMode[0] = cudaAddressModeClamp;
	textImage.addressMode[1] = cudaAddressModeClamp;

	textError.filterMode = cudaFilterModeLinear;
	textError.addressMode[0] = cudaAddressModeClamp;
	textError.addressMode[1] = cudaAddressModeClamp;

	textSino.filterMode = cudaFilterModeLinear;
	textSino.addressMode[0] = cudaAddressModeClamp;
	textSino.addressMode[1] = cudaAddressModeClamp;

	constants.Px = Sys->Proj.Nx;
	constants.Py = Sys->Proj.Ny;
	constants.Rx = Sys->Recon.Nx;
	constants.Ry = Sys->Recon.Ny;
	constants.PitchPx = Sys->Proj.Pitch_x;
	constants.PitchPy = Sys->Proj.Pitch_y;
	constants.PitchRx = Sys->Recon.Pitch_x;
	constants.PitchRy = Sys->Recon.Pitch_y;
	constants.Views = Sys->Proj.NumViews;
	constants.log = true;
	constants.orientation = true;
	constants.flip = false;

	int pitch = (int)projPitch / sizeof(float);
	constants.ReconPitchNum = reconPitchNum;
	constants.ProjPitchNum = pitch;

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
	cuda(MemcpyAsync(d_noop, tempNoop, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernel[KERNELSIZE];
	setGauss(tempKernel);
	cuda(MemcpyAsync(d_gauss, tempKernel, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer[KERNELSIZE];
	setGaussDer(tempKernelDer);
	cuda(MemcpyAsync(d_gaussDer, tempKernelDer, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer2[KERNELSIZE];
	setGaussDer2(tempKernelDer2);
	cuda(MemcpyAsync(d_gaussDer2, tempKernelDer2, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer3[KERNELSIZE];
	setGaussDer3(tempKernelDer3);
	cuda(MemcpyAsync(d_gaussDer3, tempKernelDer3, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";


	{//Read and correct projections
		unsigned short * RawData = new unsigned short[Sys->Proj.Nx*Sys->Proj.Ny];

		int size_proj = Sys->Proj.Nx * Sys->Proj.Ny;
		size_t sizeProj = size_proj * sizeof(unsigned short);

		//define the GPU kernel based on size of "ideal projection"
		dim3 dimBlockProj(32, 32);
		dim3 dimGridProj((Sys->Proj.Nx + 31) / 32, (Sys->Proj.Ny + 31) / 32);

		dim3 dimGridSum(1, 1);
		dim3 dimBlockSum(1024, 1);

		int projPixels = reconPitchNum * Sys->Proj.Ny;

		FILE * ProjData = NULL;
		std::string ProjPath = mainFile;
		unsigned short * d_Proj;
		cuda(Malloc((void**)&d_Proj, sizeProj));

		//Read the rest of the blank images for given projection sample set 
		for (int view = 0; view < NumViews; view++) {
			ProjPath = ProjPath.substr(0, ProjPath.length() - 5);
			ProjPath += std::to_string(view) + ".raw";

			fopen_s(&ProjData, ProjPath.c_str(), "rb");

			if (ProjData == NULL)
				return Tomo_file_err;

			//Write the reconstructed data into the predefine memory location
			fread(RawData, sizeof(unsigned short), Sys->Proj.Nx * Sys->Proj.Ny, ProjData);
			fclose(ProjData);

			cuda(MemcpyAsync(d_Proj, RawData, sizeProj, cudaMemcpyHostToDevice));

			KERNELCALL2(LogCorrectProj, dimGridProj, dimBlockProj, d_Sino, view, d_Proj, USHRT_MAX, constants);

			KERNELCALL3(GetMaxImageVal, dimGridSum, dimBlockSum, 1024 * sizeof(float), d_Sino + projPixels*view, projPixels, d_MaxVal);

			KERNELCALL3(GetMinImageVal, dimGridSum, dimBlockSum, 1024 * sizeof(float), d_Sino + projPixels*view, projPixels, d_MinVal);

			KERNELCALL2(rescale, dimGridProj, dimBlockProj, d_Sino, view, d_MaxVal, d_MinVal, constants);
		}

		delete[] RawData;
		cuda(Free(d_Proj));
		/*
		int loc1;
		int GainLoc;
		int DarkLoc;
		int vnum = 0;
		//Correct the Raw projections with the gain and dark images
		vnum = 0;
		for (int view = 0; view < NumViews; view++) {
			for (int x = 0; x < Sys->Proj.Nx; x++) {
				for (int y = 0; y < Sys->Proj.Ny; y++)
				{
					loc1 = (y + view*Sys->Proj.Ny)*Sys->Proj.Nx + x;
					GainLoc = y*Sys->Proj.Nx + x + view*Sys->Proj.Nx*Sys->Proj.Ny;
					DarkLoc = y*Sys->Proj.Nx + x;

					unsigned short val = Sys->Norm.ProjBuf[loc1];
					//unsigned short gval = Sys->Norm.GainData[GainLoc];
					//					unsigned short dval = Sys->Norm->DarkData[DarkLoc];


					//					float C_val = (float)(val)-(float)dval;
					//					if (C_val < 1) C_val = 1;

					//					float C_gval = (float)(gval)-(float)dval;

					float C_val = (float)(val);
					//float C_gval = (float)(gval);

					//float n_val = C_val / C_gval;
					//if (C_val + 25 > C_gval) n_val = 0.0;

					int loc3 = (y + view*Sys->Proj.Ny)*Sys->Proj.Nx
						+ Sys->Proj.Nx - (x + 1);

					if (Sys->Proj.Flip == 1) {
						loc3 = (y + view*Sys->Proj.Ny)*Sys->Proj.Nx + x;
					}
					Sys->Norm.CorrBuf[loc3] += C_val;

				}
			}
		}*/
	}

	return Tomo_OK;
}

//Fucntion to free the gpu memory after program finishes
TomoError TomoRecon::FreeGPUMemory(void){
	//Free memory allocated on the GPU
	cuda(Free(d_Image));
	cuda(Free(d_Error));
	cuda(Free(d_Sino));
	cuda(Free(xDer));

	cuda(Free(constants.d_Beamx));
	cuda(Free(constants.d_Beamy));
	cuda(Free(constants.d_Beamz));

	cuda(Free(d_noop));
	cuda(Free(d_gauss));
	cuda(Free(d_gaussDer));
	cuda(Free(d_gaussDer2));
	cuda(Free(d_gaussDer3));

	return Tomo_OK;
}

template <typename T>
TomoError TomoRecon::getHistogram(T * image, unsigned int byteSize, unsigned int *histogram) {
	unsigned int * d_Histogram;

	cuda(Malloc((void **)&d_Histogram, HIST_BIN_COUNT * sizeof(unsigned int)));
	cuda(Memset(d_Histogram, 0, HIST_BIN_COUNT * sizeof(unsigned int)));

	KERNELCALL2(histogram256Kernel, contBlocks, contThreads, d_Histogram, image, byteSize / sizeof(T), constants);

	cuda(Memcpy(histogram, d_Histogram, HIST_BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	cuda(Free(d_Histogram));

	return Tomo_OK;
}

TomoError TomoRecon::test(int index) {
	scale = max((float)Sys->Proj.Nx / (float)width, (float)Sys->Proj.Ny / (float)height) * pow(ZOOMFACTOR, -zoom);
	int maxX = (int)((pow(ZOOMFACTOR, zoom) - 1.0f) * (float)Sys->Proj.Nx / 2.0f);
	int maxY = (int)((pow(ZOOMFACTOR, zoom) - 1.0f) * (float)Sys->Proj.Ny / 2.0f);
	if (xOff != 0 && xOff > maxX || xOff < -maxX) xOff = xOff > 0 ? maxX : -maxX;
	if (yOff != 0 && yOff > maxY || yOff < -maxY) yOff = yOff > 0 ? maxY : -maxY;

	cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys->Recon.Nx, Sys->Recon.Ny, reconPitch));
	cuda(BindSurfaceToArray(displaySurface, *ca));

	const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

	if (blocks > 0) {
		KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream,
			Sys->Proj.Nx, Sys->Proj.Ny, width, height, index, scale, xOff, yOff, derDisplay != no_der, constants);
		if (constants.baseXr >= 0 && constants.currXr >= 0)
			KERNELCALL4(drawSelectionBox, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(max(constants.baseXr, constants.currXr), true),
				I2D(max(constants.baseYr, constants.currYr), false), I2D(min(constants.baseXr, constants.currXr), true), I2D(min(constants.baseYr, constants.currYr), false), width);
		if (lowXr >= 0)
			KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(lowXr, true), I2D(lowYr, false), width, vertical);
		if (upXr >= 0)
			KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(upXr, true), I2D(upYr, false), width, vertical);
	}

	cuda(UnbindTexture(textImage));

	return Tomo_OK;
}

TomoError TomoRecon::singleFrame() {
	//Initial projection
	cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), Sys->Proj.Nx, Sys->Proj.Ny*Sys->Proj.NumViews, projPitch));
	KERNELCALL2(projectSlice, contBlocks, contThreads, d_Image, distance, constants);
	cuda(UnbindTexture(textImage));

	switch (derDisplay) {
	case no_der:
		break;
	case der_x:
		imageKernel(d_gaussDer, d_gauss, d_Image);
		break;
	case der_y:
		imageKernel(d_gauss, d_gaussDer, d_Image);
		break;
	case square_mag:
		imageKernel(d_gaussDer, d_gauss, xDer);
		imageKernel(d_gauss, d_gaussDer, d_Image);

		KERNELCALL2(squareMag, contBlocks, contThreads, d_Image, xDer, d_Image, reconPitchNum, reconPitchNum, constants);
		break;
	case slice_diff:
	{
		float xOff = -Sys->Geo.EmitX[diffSlice] * distance / Sys->Geo.EmitZ[diffSlice] / Sys->Recon.Pitch_x;
		float yOff = -Sys->Geo.EmitY[diffSlice] * distance / Sys->Geo.EmitZ[diffSlice] / Sys->Recon.Pitch_y;
		KERNELCALL2(squareDiff, contBlocks, contThreads, d_Image, diffSlice, xOff, yOff, reconPitchNum, constants);
	}
		break;
	case der2_x:
		imageKernel(d_gaussDer2, d_gauss, d_Image);
		break;
	case der2_y:
		imageKernel(d_gauss, d_gaussDer2, d_Image);
		break;
	case der3_x:
		imageKernel(d_gaussDer3, d_gauss, d_Image);
		break;
	case der3_y:
		imageKernel(d_gauss, d_gaussDer3, d_Image);
		break;
	}

	return Tomo_OK;
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
					singleFrame();
					autoLight();
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
		float newOffset = xDir ? Sys->Geo.EmitX[diffSlice] : Sys->Geo.EmitY[diffSlice];

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
				if (xDir) Sys->Geo.EmitX[diffSlice] = newOffset;
				else Sys->Geo.EmitY[diffSlice] = newOffset;
				xDir = !xDir;
				if (xDir) diffSlice++;//if we're back on xDir, we've made the next step
				if (diffSlice == (NUMVIEWS / 2)) diffSlice++;//skip center, it's what we're comparing
				if (diffSlice >= NUMVIEWS) {
					light = oldLight;
					derDisplay = no_der;
					cuda(MemcpyAsync(constants.d_Beamx, Sys->Geo.EmitX, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
					cuda(MemcpyAsync(constants.d_Beamy, Sys->Geo.EmitY, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
					return Tomo_Done;
				}
				return Tomo_OK;//skip the rest so values don't bleed into one another
			}

			newOffset += step;
		}

		if (xDir) Sys->Geo.EmitX[diffSlice] = newOffset;
		else Sys->Geo.EmitY[diffSlice] = newOffset;

		return Tomo_OK;
	}
	
	return Tomo_Done;
}

TomoError TomoRecon::autoLight() {
	unsigned int histogram[HIST_BIN_COUNT];
	getHistogram(d_Image, reconPitch*Sys->Recon.Ny, histogram);

	int threshold = Sys->Recon.Nx * Sys->Recon.Ny / AUTOTHRESHOLD;
	int i;
	for (i = 0; i < HIST_BIN_COUNT; i++) {
		unsigned int count = histogram[i];
		if (count > threshold) break;
	}
	if (i >= HIST_BIN_COUNT) i = 0;
	constants.minVal = i * UCHAR_MAX;

	//go from the reverse direction for maxval
	for (i = HIST_BIN_COUNT - 1; i >= 0; i--) {
		unsigned int count = histogram[i];
		if (count > threshold) break;
	}
	if (i < 0) i = HIST_BIN_COUNT;
	constants.maxVal = i * UCHAR_MAX;
	if (constants.minVal == constants.maxVal) constants.maxVal += UCHAR_MAX;

	return Tomo_OK;
}

TomoError TomoRecon::readPhantom(float * resolution) {
	if (vertical) {
		float phanScale = (lowYr - upYr) / (1/LOWERBOUND - 1/UPPERBOUND);
		float * h_xDer2 = (float*)malloc(reconPitch*Sys->Recon.Ny);
		cuda(Memcpy(h_xDer2, d_Image, reconPitch*Sys->Recon.Ny, cudaMemcpyDeviceToHost));
		//Get x range from the bouding box
		int startX = min(constants.baseXr, constants.currXr);
		int endX = max(constants.baseXr, constants.currXr);
		int thisY = lowYr;//Get beginning y val from tick mark
		while (thisY >= upYr) {//y counts down
			int thisX = startX;
			int negCross = 0;
			bool negativeSpace = false;
			float negAcc = 0;
			while (thisX < endX) {
				if (negativeSpace) {
					float val = h_xDer2[thisY * reconPitchNum + thisX];
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
					float val = h_xDer2[thisY * reconPitchNum + thisX];
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
		float * h_yDer2 = (float*)malloc(reconPitchNum*Sys->Recon.Ny * sizeof(float));
		cuda(Memcpy(h_yDer2, d_Image, reconPitchNum*Sys->Recon.Ny * sizeof(float), cudaMemcpyDeviceToHost));
		//Get x range from the bouding box
		int startY = min(constants.baseYr, constants.currYr);
		int endY = max(constants.baseYr, constants.currYr);
		int thisX = lowXr;//Get beginning y val from tick mark
		while (thisX >= upXr) {//y counts down
			int thisY = startY;
			int negCross = 0;
			bool negativeSpace = false;
			float negAcc = 0;
			while (thisY < endY) {
				if (negativeSpace) {
					float val = h_yDer2[thisY * reconPitchNum + thisX];
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
					float val = h_yDer2[thisY * reconPitchNum + thisX];
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

TomoError TomoRecon::initTolerances(std::vector<toleranceData> &data, int numTests, std::vector<float> offsets) {
	//start set as just the combinations
	for (int i = 0; i < NUMVIEWS; i++) {
		int resultsLen = (int)data.size();//size will change every iteration, pre-record it
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
	int combinations = (int)data.size();//again, changing sizes later on
	for (int i = 0; i < combinations; i++) {
		toleranceData baseline = data[i];

		baseline.thisDir = dir_y;
		data.push_back(baseline);

		baseline.thisDir = dir_z;
		data.push_back(baseline);
	}

	//then fill in the set with all the view changes
	combinations = (int)data.size();//again, changing sizes later on
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
			memcpy(geo, Sys->Geo.EmitX, sizeof(float)*NUMVIEWS);
			break;
		case dir_y:
			memcpy(geo, Sys->Geo.EmitY, sizeof(float)*NUMVIEWS);
			break;
		case dir_z:
			memcpy(geo, Sys->Geo.EmitZ, sizeof(float)*NUMVIEWS);
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
			cuda(MemcpyAsync(constants.d_Beamx, geo, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(constants.d_Beamy, Sys->Geo.EmitY, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(constants.d_Beamz, Sys->Geo.EmitZ, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			break;
		case dir_y:
			cuda(MemcpyAsync(constants.d_Beamx, Sys->Geo.EmitX, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(constants.d_Beamy, geo, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(constants.d_Beamz, Sys->Geo.EmitZ, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			break;
		case dir_z:
			cuda(MemcpyAsync(constants.d_Beamx, Sys->Geo.EmitX, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(constants.d_Beamy, Sys->Geo.EmitY, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
			cuda(MemcpyAsync(constants.d_Beamz, geo, Sys->Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
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

/****************************************************************************/
/*								Kernel launch helpers						*/
/****************************************************************************/

inline float TomoRecon::focusHelper() {
	//Render new frame
	singleFrame();

	//get the focus metric
	float currentBest;
	cuda(MemsetAsync(d_MaxVal, 0, sizeof(float)));
	//TODO: check boundary conditions
	KERNELCALL3(sumReduction, reductionBlocks, reductionThreads, reductionSize, d_Image, reconPitchNum, d_MaxVal, 
		min(constants.baseXr, constants.currXr), max(constants.baseXr, constants.currXr), min(constants.baseYr, constants.currYr), max(constants.baseYr, constants.currYr));

	cuda(Memcpy(&currentBest, d_MaxVal, sizeof(float), cudaMemcpyDeviceToHost));
	return currentBest;
}

inline TomoError TomoRecon::imageKernel(float xK[KERNELSIZE], float yK[KERNELSIZE], float * output) {
	cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys->Recon.Nx, Sys->Recon.Ny, reconPitch));
	KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, xK, constants);
	cuda(UnbindTexture(textImage));

	cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), Sys->Recon.Nx, Sys->Recon.Ny, reconPitch));
	KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, output, yK, constants);
	cuda(UnbindTexture(textImage));

	return Tomo_OK;
}

TomoError TomoRecon::resetLight() {
	constants.baseXr = 3 * Sys->Recon.Nx / 4;
	constants.baseYr = 3 * Sys->Recon.Ny / 4;
	constants.currXr = Sys->Recon.Nx / 4;
	constants.currYr = Sys->Recon.Ny / 4;

	autoLight();

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	return Tomo_OK;
}

TomoError TomoRecon::resetFocus() {
	constants.baseXr = 3 * Sys->Recon.Nx / 4;
	constants.baseYr = 3 * Sys->Recon.Ny / 4;
	constants.currXr = Sys->Recon.Nx / 4;
	constants.currYr = Sys->Recon.Ny / 4;

	autoFocus(true);
	while (autoFocus(false) == Tomo_OK);

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	return Tomo_OK;
}

/****************************************************************************/
/*									Conversions								*/
/****************************************************************************/

//Projection space to recon space
TomoError TomoRecon::P2R(int* rX, int* rY, int pX, int pY, int view) {
	float dz = distance / Sys->Geo.EmitZ[view];
	*rX = xMM2R(xP2MM(pX, constants.Px, constants.PitchPx) * (1 + dz) - Sys->Geo.EmitX[view] * dz, constants.Rx, constants.PitchRx);
	*rY = yMM2R(yP2MM(pY, constants.Py, constants.PitchPy) * (1 + dz) - Sys->Geo.EmitY[view] * dz, constants.Ry, constants.PitchRy);

	return Tomo_OK;
}

//Recon space to projection space
TomoError TomoRecon::R2P(int* pX, int* pY, int rX, int rY, int view) {
	float dz = distance / Sys->Geo.EmitZ[view];
	*pX = xMM2P((xR2MM(rX, constants.Rx, constants.PitchRx) + Sys->Geo.EmitX[view] * dz) / (1 + dz), constants.Px, constants.PitchPx);
	*pY = yMM2P((yR2MM(rY, constants.Ry, constants.PitchRy) + Sys->Geo.EmitY[view] * dz) / (1 + dz), constants.Py, constants.PitchPy);

	return Tomo_OK;
}

//Image space to on-screen display
TomoError TomoRecon::I2D(int* dX, int* dY, int iX, int iY) {
	float innerOffx = (width - Sys->Proj.Nx / scale) / 2;
	float innerOffy = (height - Sys->Proj.Ny / scale) / 2;

	*dX = (int)((iX - xOff) / scale + innerOffx);
	*dY = (int)((iY - yOff) / scale + innerOffy);

	return Tomo_OK;
}

//Projection space to recon space
int TomoRecon::P2R(int p, int view, bool xDir) {
	float dz = distance / Sys->Geo.EmitZ[view];
	if (xDir)
		return (int)(xMM2R(xP2MM(p, constants.Px, constants.PitchPx) * (1 + dz) - Sys->Geo.EmitX[view] * dz, constants.Rx, constants.PitchRx));
	//else
	return (int)(yMM2R(yP2MM(p, constants.Py, constants.PitchPy) * (1 + dz) - Sys->Geo.EmitY[view] * dz, constants.Ry, constants.PitchRy));
}

//Recon space to projection space
int TomoRecon::R2P(int r, int view, bool xDir) {
	float dz = distance / Sys->Geo.EmitZ[view];
	if (xDir)
		return (int)(xMM2P((xR2MM(r, constants.Rx, constants.PitchRx) + Sys->Geo.EmitX[view] * dz) / (1.0f + dz), constants.Px, constants.PitchPx));
	//else
	return (int)(yMM2P((yR2MM(r, constants.Ry, constants.PitchRy) + Sys->Geo.EmitY[view] * dz) / (1.0f + dz), constants.Py, constants.PitchPy));
}

//Image space to on-screen display
int TomoRecon::I2D(int i, bool xDir) {
	if (xDir) {
		float innerOffx = (width - Sys->Proj.Nx / scale) / 2.0f;

		return (int)((i - xOff) / scale + innerOffx);
	}
	//else
	float innerOffy = (height - Sys->Proj.Ny / scale) / 2.0f;

	return (int)((i - yOff) / scale + innerOffy);
}

//On-screen coordinates to image space
int TomoRecon::D2I(int d, bool xDir) {
	if (xDir) {
		float innerOffx = (width - Sys->Proj.Nx / scale) / 2.0f;

		return (int)((d - innerOffx) * scale + xOff);
	}
	//else
	float innerOffy = (height - Sys->Proj.Ny / scale) / 2.0f;

	return (int)((d - innerOffy) * scale + yOff);
}