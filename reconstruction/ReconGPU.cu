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

union pxl_rgbx_24{
	uint1       b32;

	struct {
		unsigned  r : 8;
		unsigned  g : 8;
		unsigned  b : 8;
		unsigned  na : 8;
	};
};

#define PXL_KERNEL_THREADS_PER_BLOCK  256

surface<void, cudaSurfaceType2D> displaySurface;
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
__global__ void resizeKernelTex(int wIn, int hIn, int wOut, int hOut, float scale, int xOff, int yOff, bool derDisplay, params consts) {
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
		sum = tex2D(textImage, (float)i + 0.5f, (float)j + 0.5f);

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
	sum = (sum - consts.minVal) / consts.maxVal * UCHAR_MAX;
	saturate = sum > UCHAR_MAX;

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
__global__ void LogCorrectProj(float * Sino, int view, unsigned short *Proj, unsigned short *Dark, unsigned short *Gain, params consts){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i < consts.Px) && (j < consts.Py)){
		//Flip and invert while converting to float
		int x, y;
		if (consts.orientation) x = consts.Px - 1 - i;
		else x = i;
		if (consts.flip) y = consts.Py - 1 - j;
		else y = j;

		//float val = (float)Proj[j*consts.Px + i] - (float)Dark[j*consts.Px + i];
		float val = Proj[j*consts.Px + i];

		//bar noise correction
		if (i > 1 && i < consts.Px - 1) {
			float val1 = Proj[j*consts.Px + i - 1];
			float val2 = Proj[j*consts.Px + i + 1];
			float val3 = (val1 + val2) / 2;
			if(abs(val1 - val2) < DIFFTHRESH && abs(val3 - val) > DIFFTHRESH / 2)
				val = val3;
		}
		if (j > 1 && j< consts.Py - 1) {
			float val1 = Proj[(j-1)*consts.Px + i];
			float val2 = Proj[(j+1)*consts.Px + i];
			float val3 = (val1 + val2) / 2;
			if (abs(val1 - val2) < DIFFTHRESH && abs(val3 - val) > DIFFTHRESH / 2)
				val = val3;
		}
		if (val < LOWTHRESH) val = 0.0;

		val /= Gain[j*consts.Px + i];
		if (val > HIGHTHRESH) val = 0.0;

		//if (val / Gain[j*consts.Px + i] > HIGHTHRESH) val = 0.0;

		Sino[(y + view*consts.Py)*consts.ProjPitchNum + x] = val;
	}
}

__global__ void rescale(float * Sino, int view, float * MaxVal, float * MinVal, float * colShifts, float * rowShifts, params consts) {
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i < consts.Px) && (j < consts.Py)) {
		float test = Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] - *MinVal;
		if (test > 0) {
			Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] = (test - colShifts[i] - rowShifts[j]) / *MaxVal * USHRT_MAX;//scale from 1 to max
		}
		else Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] = 0.0;
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
		if (y > 0 && y < consts.Py && x > 0 && x < consts.Px) {
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

__global__ void sumRowsOrCols(float * sum, bool cols, params consts) {
	//Define shared memory to read all the threads
	extern __shared__ float data[];

	//define the thread and block location
	const int thread = threadIdx.x;
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	float val = 0;
	int i = x;
	int limit;
	if (cols) limit = consts.Py;
	else limit = consts.Px;

	while(i < limit){
		if(cols)
			val += tex2D(textSino, y, i);
		else
			val += tex2D(textSino, i, y);
		i += blockDim.x;
	}

	data[thread] = val;

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
	if (thread == 0) {
		sum[y] = val;
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

	float data = abs(d_Data[MUL_ADD(j, consts.ReconPitchNum, i)]);//whatever it currently is, cast it to ushort
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
	Sys.Recon.Pitch_x = Sys.Proj.Pitch_x;
	Sys.Recon.Pitch_y = Sys.Proj.Pitch_y;
	Sys.Recon.Nx = Sys.Proj.Nx;
	Sys.Recon.Ny = Sys.Proj.Ny;

	//Normalize Geometries
	Sys.Geo.IsoX = Sys.Geo.EmitX[NUMVIEWS / 2];
	Sys.Geo.IsoY = Sys.Geo.EmitY[NUMVIEWS / 2];
	Sys.Geo.IsoZ = Sys.Geo.EmitZ[NUMVIEWS / 2];
	for (int i = 0; i < NUMVIEWS; i++) {
		Sys.Geo.EmitX[i] -= Sys.Geo.IsoX;
		Sys.Geo.EmitY[i] -= Sys.Geo.IsoY;
	}

	size_t avail_mem;
	size_t total_mem;
	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";

	//Get Device Number
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
	contBlocks.x = (Sys.Recon.Nx + contThreads.x - 1) / contThreads.x;
	contBlocks.y = (Sys.Recon.Ny + contThreads.y - 1) / contThreads.y;

	//Thread and block sizes for reductions (1d optimized)
	reductionThreads.x = MAXTHREADS;
	reductionBlocks.x = (Sys.Recon.Nx + reductionThreads.x - 1) / reductionThreads.x;
	reductionBlocks.y = Sys.Recon.Ny;

	//Set up dispaly and buffer (error) regions
	cuda(MallocPitch((void**)&d_Image, &reconPitch, Sys.Recon.Nx * sizeof(float), Sys.Recon.Ny));
	cuda(MallocPitch((void**)&d_Error, &reconPitch, Sys.Recon.Nx * sizeof(float), Sys.Recon.Ny));
	cuda(MallocPitch((void**)&d_Sino, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));

	reconPitchNum = (int)reconPitch / sizeof(float);

	//Define the size of each of the memory spaces on the gpu in number of bytes
	sizeProj = Sys.Proj.Nx * Sys.Proj.Ny * sizeof(unsigned short);
	sizeSino = projPitch * Sys.Proj.Ny * Sys.Proj.NumViews;
	sizeIM = reconPitch * Sys.Recon.Ny;
	sizeError = reconPitch * Sys.Recon.Ny;
	
	cuda(Malloc((void**)&constants.d_Beamx, Sys.Proj.NumViews * sizeof(float)));
	cuda(Malloc((void**)&constants.d_Beamy, Sys.Proj.NumViews * sizeof(float)));
	cuda(Malloc((void**)&constants.d_Beamz, Sys.Proj.NumViews * sizeof(float)));
	cuda(Malloc((void**)&d_MaxVal, sizeof(float)));
	cuda(Malloc((void**)&d_MinVal, sizeof(float)));

	//Copy geometries
	cuda(MemcpyAsync(constants.d_Beamx, Sys.Geo.EmitX, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(constants.d_Beamy, Sys.Geo.EmitY, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
	cuda(MemcpyAsync(constants.d_Beamz, Sys.Geo.EmitZ, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));

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

	constants.Px = Sys.Proj.Nx;
	constants.Py = Sys.Proj.Ny;
	constants.Rx = Sys.Recon.Nx;
	constants.Ry = Sys.Recon.Ny;
	constants.PitchPx = Sys.Proj.Pitch_x;
	constants.PitchPy = Sys.Proj.Pitch_y;
	constants.PitchRx = Sys.Recon.Pitch_x;
	constants.PitchRy = Sys.Recon.Pitch_y;
	constants.Views = Sys.Proj.NumViews;
	constants.log = true;
	constants.orientation = false;
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

	ReadProjections(gainFile, darkFile, mainFile);

	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";

	return Tomo_OK;
}

TomoError TomoRecon::ReadProjections(const char * gainFile, const char * darkFile, const char * mainFile) {
	//Read and correct projections
	unsigned short * RawData = new unsigned short[Sys.Proj.Nx*Sys.Proj.Ny];
	unsigned short * DarkData = new unsigned short[Sys.Proj.Nx*Sys.Proj.Ny];
	unsigned short * GainData = new unsigned short[Sys.Proj.Nx*Sys.Proj.Ny];

	float * sumValsVert = new float[NumViews * Sys.Proj.Nx];
	float * sumValsHor = new float[NumViews * Sys.Proj.Ny];
	float * vertOff = new float[NumViews * Sys.Proj.Nx];
	float * horOff = new float[NumViews * Sys.Proj.Ny];
	float * d_SumValsVert;
	float * d_SumValsHor;
	cuda(Malloc((void**)&d_SumValsVert, Sys.Proj.Nx * sizeof(float)));
	cuda(Malloc((void**)&d_SumValsHor, Sys.Proj.Ny * sizeof(float)));

	//define the GPU kernel based on size of "ideal projection"
	dim3 dimBlockProj(32, 32);
	dim3 dimGridProj((Sys.Proj.Nx + 31) / 32, (Sys.Proj.Ny + 31) / 32);

	dim3 dimGridSum(1, 1);
	dim3 dimBlockSum(1024, 1);

	FILE * fileptr = NULL;
	std::string ProjPath = mainFile;
	std::string GainPath = gainFile;
	unsigned short * d_Proj;
	unsigned short * d_Dark;
	unsigned short * d_Gain;
	cuda(Malloc((void**)&d_Proj, sizeProj));
	cuda(Malloc((void**)&d_Dark, sizeProj));
	cuda(Malloc((void**)&d_Gain, sizeProj));

	//Read dark
	fopen_s(&fileptr, darkFile, "rb");

	if (fileptr == NULL)
		return Tomo_file_err;

	fread(DarkData, sizeof(unsigned short), Sys.Proj.Nx * Sys.Proj.Ny, fileptr);
	fclose(fileptr);
	cuda(MemcpyAsync(d_Dark, DarkData, sizeProj, cudaMemcpyHostToDevice));

	constants.baseXr = 0;
	constants.baseYr = 0;
	constants.currXr = Sys.Proj.Nx;
	constants.currYr = Sys.Proj.Ny;

	bool oldLog = constants.log;
	constants.log = false;

	//Read the rest of the blank images for given projection sample set 
	for (int view = 0; view < NumViews; view++) {
		ProjPath = ProjPath.substr(0, ProjPath.length() - 5);
		ProjPath += std::to_string(view) + ".raw";
		GainPath = GainPath.substr(0, GainPath.length() - 5);
		GainPath += std::to_string(view) + ".raw";

		fopen_s(&fileptr, ProjPath.c_str(), "rb");
		if (fileptr == NULL) return Tomo_file_err;
		fread(RawData, sizeof(unsigned short), Sys.Proj.Nx * Sys.Proj.Ny, fileptr);
		fclose(fileptr);

		fopen_s(&fileptr, GainPath.c_str(), "rb");
		if (fileptr == NULL) return Tomo_file_err;
		fread(GainData, sizeof(unsigned short), Sys.Proj.Nx * Sys.Proj.Ny, fileptr);
		fclose(fileptr);

		cuda(MemcpyAsync(d_Proj, RawData, sizeProj, cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(d_Gain, GainData, sizeProj, cudaMemcpyHostToDevice));

		KERNELCALL2(LogCorrectProj, dimGridProj, dimBlockProj, d_Sino, view, d_Proj, d_Dark, d_Gain, constants);

		scanLineDetect(view, d_SumValsVert, sumValsVert + view * Sys.Proj.Nx, vertOff + view * Sys.Proj.Nx, true);
		scanLineDetect(view, d_SumValsHor, sumValsHor + view * Sys.Proj.Ny, horOff + view * Sys.Proj.Ny, false);
	}

	float step;
	float best;
	bool linearRegion;
	bool firstLin = true;
	float bestDist;

	float * scales = new float[NumViews];

	for (int i = 0; i < NumViews; i++) {
		float offLight = 0.0;
		float bestOffLight = 0.0;
		float bestScale = 1.0;
		float thisScale = 1.0;
		bool scaleSwitch = false;
		firstLin = true;
		float scaleStep = 0.01;
		step = 10;
		best = FLT_MAX;
		while (true) {
			float newVal = graphCost(sumValsVert, sumValsHor, i, offLight, thisScale, 0.0);

			//compare to current
			if (newVal < best) {
				if (scaleSwitch) {
					bestScale = thisScale;
					thisScale += scaleStep;
				}
				else {
					bestOffLight = offLight;
					offLight += step;
				}
				best = newVal;
			}
			else {
				if (!firstLin) {
					if(scaleSwitch) thisScale -= scaleStep;//revert last move
					else offLight -= step;//revert last move
				}
				scaleSwitch = !scaleSwitch;
				if (!scaleSwitch) {
					step = -step / 2;//find next step
					scaleStep = -scaleStep / 2;

					if (abs(step) < 0.1)
						break;
					else offLight += step;
				}
				else thisScale += scaleStep;
			}
			firstLin = false;
		}

		scales[i] = bestScale;
		for (int j = 0; j < Sys.Proj.Nx; j++) {
			float r = xP2MM(j, Sys.Proj.Nx, Sys.Proj.Pitch_x) - Sys.Geo.EmitX[i];
			sumValsVert[j + i * Sys.Proj.Nx] = sumValsVert[j + i * Sys.Proj.Nx] * bestScale - 0.0*pow(r, 2) + bestOffLight;
		}
		for (int j = 0; j < Sys.Proj.Ny; j++) {
			float r = yP2MM(j, Sys.Proj.Ny, Sys.Proj.Pitch_y) - Sys.Geo.EmitY[i];
			sumValsHor[j + i * Sys.Proj.Ny] = sumValsHor[j + i * Sys.Proj.Ny] * bestScale - 0.0*pow(r, 2) + bestOffLight;
			horOff[j + i * Sys.Proj.Ny] -= bestOffLight;
		}
	}

	step = STARTSTEP;
	distance = MINDIS;
	bestDist = MINDIS;
	best = FLT_MAX;
	linearRegion = false;
	firstLin = true;

	while (true) {
		float newVal = graphCost(sumValsVert, sumValsHor);

		if (newVal < best) {
			best = newVal;
			bestDist = distance;
		}

		distance += step;

		if (distance > MAXDIS) {
			distance = bestDist;
			break;
		}
	}

	//Normalize projection image lighting
	float maxVal, minVal;
	unsigned int histogram[HIST_BIN_COUNT];
	getHistogram(d_Sino + (NumViews / 2)*projPitch / sizeof(float)*Sys.Proj.Ny, projPitch*Sys.Proj.Ny, histogram);
	autoLight(histogram, 80, &minVal, &maxVal);
	cuda(Memcpy(d_MinVal, &minVal, sizeof(float), cudaMemcpyHostToDevice));

	for (int view = 0; view < NumViews; view++) {
		float scale = maxVal/scales[view];
		cuda(Memcpy(d_MaxVal, &scale, sizeof(float), cudaMemcpyHostToDevice));
		
		{
			std::ofstream FILE1, FILE2;
			std::stringstream outputfile1, outputfile2;
			outputfile1 << "C:\\Users\\jdean\\Downloads\\cudaTV\\cudaTV\\correctedVert" << view << ".txt";
			outputfile2 << "C:\\Users\\jdean\\Downloads\\cudaTV\\cudaTV\\correctedHor" << view << ".txt";
			FILE1.open(outputfile1.str());
			for (int i = 0; i < Sys.Proj.Nx; i++) {
				int x = i + Sys.Geo.EmitX[view] * distance / Sys.Geo.EmitZ[view] / constants.PitchPx;
				if(x < Sys.Proj.Nx && x > 0)
					FILE1 << sumValsVert[x + view * Sys.Proj.Nx] << "\n";
				else FILE1 << 0.0 << "\n";
			}
			FILE1.close();

			FILE2.open(outputfile2.str());
			for (int i = 0; i < Sys.Proj.Ny; i++) {
				int x = i + Sys.Geo.EmitY[view] * distance / Sys.Geo.EmitZ[view] / constants.PitchPy;
				if (x < Sys.Proj.Ny && x > 0)
					FILE2 << sumValsHor[x + view * Sys.Proj.Ny] << "\n";
				else FILE2 << 0.0 << "\n";
			}
			FILE2.close();
		}

		cuda(MemcpyAsync(d_SumValsVert, vertOff + view * Sys.Proj.Nx, Sys.Proj.Nx * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(d_SumValsHor, horOff + view * Sys.Proj.Ny, Sys.Proj.Ny * sizeof(float), cudaMemcpyHostToDevice));

		KERNELCALL2(rescale, dimGridProj, dimBlockProj, d_Sino, view, d_MaxVal, d_MinVal, d_SumValsVert, d_SumValsHor, constants);
	}

	constants.log = oldLog;

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	delete[] scales;
	delete[] RawData;
	delete[] DarkData;
	delete[] GainData;
	delete[] sumValsHor;
	delete[] sumValsVert;
	delete[] vertOff;
	delete[] horOff;
	cuda(Free(d_Proj));
	cuda(Free(d_Dark));
	cuda(Free(d_Gain));
	cuda(Free(d_SumValsHor));
	cuda(Free(d_SumValsVert));

	return Tomo_OK;
}

TomoError TomoRecon::scanLineDetect(int view, float * d_sum, float * sum, float * offset, bool vert) {
	int vectorSize;
	if (vert) vectorSize = Sys.Proj.Nx;
	else vectorSize = Sys.Proj.Ny;

	cuda(BindTexture2D(NULL, textSino, d_Sino + view*Sys.Proj.Ny*projPitch / sizeof(float), cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny, projPitch));
	KERNELCALL3(sumRowsOrCols, dim3(1, vectorSize), reductionThreads, reductionSize, d_sum, vert, constants);
	cuda(Memcpy(sum, d_sum, vectorSize * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef PRINTSCANCORRECTIONS
	float * sumCorr = new float[vectorSize];
	{
		std::ofstream FILE;
		std::stringstream outputfile;
		outputfile << "C:\\Users\\jdean\\Downloads\\cudaTV\\cudaTV\\original" << view << ".txt";
		FILE.open(outputfile.str());
		for (int i = 0; i < vectorSize; i++) {
			sum[i] /= vectorSize;
			FILE << sum[i] << "\n";
			sumCorr[i] = sum[i];
		}
		FILE.close();
	}
#endif

#ifdef CHAMBOLLE
	float *g, *z, *z0, *div, *div0;
	size_t size;
	int N = vectorSize;
	int i, j;
	float alpha;
	float tau = 0.25;
	float lambda = 10000;


	size = N * sizeof(float);
	g = sumCorr;
	div = (float*)malloc(size);
	div0 = (float*)malloc(size);
	z = (float*)malloc(2 * size);
	z0 = (float*)malloc(2 * size);

	for (i = 0; i<N; i++) {
	div0[i] = 0;
	z0[i] = 0;
	}

	#define SWP(a,b) {float *swap=a;a=b;b=swap;}
	for (j = 0; j<200; j++) {
	for (i = 0; i<N; i++) { div[i] = div0[i] - g[i] / lambda; }

	nabla(div, z, N);
	for (i = 0; i<N; i++)
	{
	int ix = 2 * i + 0;
	int iy = 2 * i + 1;
	alpha = 1.0 / (1.0 + tau*sqrtf(z[ix] * z[ix] + z[iy] * z[iy]));

	z[ix] = (z0[ix] + tau*z[ix])*alpha;
	z[iy] = (z0[iy] + tau*z[iy])*alpha;
	}

	diver(z, div, N);
	SWP(z, z0);
	SWP(div, div0);

	}

	for (i = 0; i<N; i++) { sumCorr[i] = g[i] - div0[i] * lambda; }

	free(div);
	free(div0);
	free(z);
	free(z0);
#else
	float *di;
	size_t size;
	int i, j;
	int N = vectorSize;
	size = N * sizeof(float);
	di = (float*)malloc(size);
	float tau = vert ? cConstants.vertTau : cConstants.horTau;

	for (j = 0; j < cConstants.iterations; j++) {
		lapla(sumCorr, di, N);
		for (i = 0; i < N; i++) sumCorr[i] += di[i] * tau;
	}

	free(di);
#endif

#ifdef PRINTSCANCORRECTIONS
	{
		std::ofstream FILE;
		std::stringstream outputfile;
		outputfile << "C:\\Users\\jdean\\Downloads\\cudaTV\\cudaTV\\corrected" << view << ".txt";
		FILE.open(outputfile.str());
		for (int i = 0; i < vectorSize; i++) {
			FILE << sumCorr[i] << "\n";
			offset[i] = sum[i] - sumCorr[i];
			sum[i] = sumCorr[i];
		}
		FILE.close();
	}
	delete[] sumCorr;
#else
	for (int i = 0; i < vectorSize; i++) {
		sum[i] -= sumValsVertCorr[i];
	}
#endif
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

TomoError TomoRecon::draw(int x, int y) {
	//interop update
	display(x, y);
	map(stream);

	scale = max((float)Sys.Proj.Nx / (float)width, (float)Sys.Proj.Ny / (float)height) * pow(ZOOMFACTOR, -zoom);
	checkOffsets(&xOff, &yOff);

	cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys.Recon.Nx, Sys.Recon.Ny, reconPitch));
	cuda(BindSurfaceToArray(displaySurface, ca));

	const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

	if (blocks > 0) {
		KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream,
			Sys.Proj.Nx, Sys.Proj.Ny, width, height, scale, xOff, yOff, derDisplay != no_der, constants);
		if (constants.baseXr >= 0 && constants.currXr >= 0)
			KERNELCALL4(drawSelectionBox, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(max(constants.baseXr, constants.currXr), true),
				I2D(max(constants.baseYr, constants.currYr), false), I2D(min(constants.baseXr, constants.currXr), true), I2D(min(constants.baseYr, constants.currYr), false), width);
		if (lowXr >= 0)
			KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(lowXr, true), I2D(lowYr, false), width, vertical);
		if (upXr >= 0)
			KERNELCALL4(drawSelectionBar, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, I2D(upXr, true), I2D(upYr, false), width, vertical);
	}

	cuda(UnbindTexture(textImage));

	//interop commands to ready buffer
	unmap(stream);
	blit();

	return Tomo_OK;
}

TomoError TomoRecon::singleFrame() {
	//Initial projection
	switch (dataDisplay) {
	case reconstruction:
		cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
		KERNELCALL2(projectSlice, contBlocks, contThreads, d_Image, distance, constants);
		cuda(UnbindTexture(textImage));
		break;
	case projections:
		cuda(Memcpy(d_Image, d_Sino + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		break;
	}

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
		float xOff = -Sys.Geo.EmitX[diffSlice] * distance / Sys.Geo.EmitZ[diffSlice] / Sys.Recon.Pitch_x;
		float yOff = -Sys.Geo.EmitY[diffSlice] * distance / Sys.Geo.EmitZ[diffSlice] / Sys.Recon.Pitch_y;
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

	if (firstRun) {
		step = STARTSTEP;
		distance = MINDIS;
		bestDist = MINDIS;
		best = 0;
		linearRegion = false;
		derDisplay = square_mag;
		singleFrame();

		unsigned int histogram[HIST_BIN_COUNT];
		int threshold = Sys.Recon.Nx * Sys.Recon.Ny / AUTOTHRESHOLD;
		getHistogram(d_Image, reconPitch*Sys.Recon.Ny, histogram);
		autoLight(histogram, threshold, &constants.minVal, &constants.maxVal);
	}

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
				derDisplay = no_der;
				singleFrame();
				unsigned int histogram[HIST_BIN_COUNT];
				int threshold = Sys.Recon.Nx * Sys.Recon.Ny / AUTOTHRESHOLD;
				getHistogram(d_Image, reconPitch*Sys.Recon.Ny, histogram);
				autoLight(histogram, threshold, &constants.minVal, &constants.maxVal);
				return Tomo_Done;
			}
			else distance += step;
		}
			
		firstLin = false;
	}

	return Tomo_OK;
}

//TODO: fix, currently worse than default geo
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
		float newOffset = xDir ? Sys.Geo.EmitX[diffSlice] : Sys.Geo.EmitY[diffSlice];

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
				if (xDir) Sys.Geo.EmitX[diffSlice] = newOffset;
				else Sys.Geo.EmitY[diffSlice] = newOffset;
				xDir = !xDir;
				if (xDir) diffSlice++;//if we're back on xDir, we've made the next step
				if (diffSlice == (NUMVIEWS / 2)) diffSlice++;//skip center, it's what we're comparing
				if (diffSlice >= NUMVIEWS) {
					light = oldLight;
					derDisplay = no_der;
					cuda(MemcpyAsync(constants.d_Beamx, Sys.Geo.EmitX, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
					cuda(MemcpyAsync(constants.d_Beamy, Sys.Geo.EmitY, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
					return Tomo_Done;
				}
				return Tomo_OK;//skip the rest so values don't bleed into one another
			}

			newOffset += step;
		}

		if (xDir) Sys.Geo.EmitX[diffSlice] = newOffset;
		else Sys.Geo.EmitY[diffSlice] = newOffset;

		return Tomo_OK;
	}
	
	return Tomo_Done;
}

TomoError TomoRecon::autoLight(unsigned int histogram[HIST_BIN_COUNT], int threshold, float * minVal, float * maxVal) {
	int i;
	for (i = 0; i < HIST_BIN_COUNT; i++) {
		unsigned int count = histogram[i];
		if (count > threshold) break;
	}
	if (i >= HIST_BIN_COUNT) i = 0;
	*minVal = i * UCHAR_MAX;

	//go from the reverse direction for maxval
	for (i = HIST_BIN_COUNT - 1; i >= 0; i--) {
		unsigned int count = histogram[i];
		if (count > threshold) break;
	}
	if (i < 0) i = HIST_BIN_COUNT;
	*maxVal = i * UCHAR_MAX;
	if (*minVal == *maxVal) *maxVal += UCHAR_MAX;

	return Tomo_OK;
}

TomoError TomoRecon::readPhantom(float * resolution) {
	if (vertical) {
		float phanScale = (lowYr - upYr) / (1/LOWERBOUND - 1/UPPERBOUND);
		float * h_xDer2 = (float*)malloc(reconPitch*Sys.Recon.Ny);
		cuda(Memcpy(h_xDer2, d_Image, reconPitch*Sys.Recon.Ny, cudaMemcpyDeviceToHost));
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
		float * h_yDer2 = (float*)malloc(reconPitchNum*Sys.Recon.Ny * sizeof(float));
		cuda(Memcpy(h_yDer2, d_Image, reconPitchNum*Sys.Recon.Ny * sizeof(float), cudaMemcpyDeviceToHost));
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
	if (firstRun) {
		if(vertical) derDisplay = der2_x;
		else derDisplay = der2_y;
		singleFrame();

		unsigned int histogram[HIST_BIN_COUNT];
		int threshold = Sys.Recon.Nx * Sys.Recon.Ny / AUTOTHRESHOLD;
		getHistogram(d_Image, reconPitch*Sys.Recon.Ny, histogram);
		autoLight(histogram, threshold, &constants.minVal, &constants.maxVal);

		iter = data.begin();
		return Tomo_OK;
	}
	
	if (iter == data.end()) {
		derDisplay = no_der;
		return Tomo_Done;
	}

	float geo[NUMVIEWS];
	switch (iter->thisDir) {
	case dir_x:
		memcpy(geo, Sys.Geo.EmitX, sizeof(float)*NUMVIEWS);
		break;
	case dir_y:
		memcpy(geo, Sys.Geo.EmitY, sizeof(float)*NUMVIEWS);
		break;
	case dir_z:
		memcpy(geo, Sys.Geo.EmitZ, sizeof(float)*NUMVIEWS);
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
		cuda(MemcpyAsync(constants.d_Beamx, geo, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(constants.d_Beamy, Sys.Geo.EmitY, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(constants.d_Beamz, Sys.Geo.EmitZ, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		break;
	case dir_y:
		cuda(MemcpyAsync(constants.d_Beamx, Sys.Geo.EmitX, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(constants.d_Beamy, geo, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(constants.d_Beamz, Sys.Geo.EmitZ, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		break;
	case dir_z:
		cuda(MemcpyAsync(constants.d_Beamx, Sys.Geo.EmitX, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(constants.d_Beamy, Sys.Geo.EmitY, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(constants.d_Beamz, geo, Sys.Proj.NumViews * sizeof(float), cudaMemcpyHostToDevice));
		break;
	}

	singleFrame();

	float readVal;
	readPhantom(&readVal);
	iter->phantomData = readVal;
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
	cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys.Recon.Nx, Sys.Recon.Ny, reconPitch));
	KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, xK, constants);
	cuda(UnbindTexture(textImage));

	cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), Sys.Recon.Nx, Sys.Recon.Ny, reconPitch));
	KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, output, yK, constants);
	cuda(UnbindTexture(textImage));

	return Tomo_OK;
}

TomoError TomoRecon::resetLight() {
	constants.baseXr = 3 * Sys.Recon.Nx / 4;
	constants.baseYr = 3 * Sys.Recon.Ny / 4;
	constants.currXr = Sys.Recon.Nx / 4;
	constants.currYr = Sys.Recon.Ny / 4;

	unsigned int histogram[HIST_BIN_COUNT];
	int threshold = Sys.Recon.Nx * Sys.Recon.Ny / AUTOTHRESHOLD;
	getHistogram(d_Image, reconPitch*Sys.Recon.Ny, histogram);
	autoLight(histogram, threshold, &constants.minVal, &constants.maxVal);

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	return Tomo_OK;
}

TomoError TomoRecon::resetFocus() {
	constants.baseXr = 3 * Sys.Recon.Nx / 4;
	constants.baseYr = 3 * Sys.Recon.Ny / 4;
	constants.currXr = Sys.Recon.Nx / 4;
	constants.currYr = Sys.Recon.Ny / 4;

	autoFocus(true);
	while (autoFocus(false) == Tomo_OK);

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	return Tomo_OK;
}

float TomoRecon::graphCost(float * vertGraph, float * horGraph, int view, float offset, float lightScale, float rSq) {
	float sum = 0.0;
	int sizeX = Sys.Proj.Nx;
	int sizeY = Sys.Proj.Ny;
	int center = NumViews / 2;

	for (int i = 0; i < sizeX; i++) {
		float base = vertGraph[i + center*sizeX];
		if (view == -1) {
			int count = 0;
			float avg = 0.0;
			for (int j = 0; j < NumViews; j++) {
				float x = i + Sys.Geo.EmitX[j] * distance / Sys.Geo.EmitZ[j] / constants.PitchPx;
				if (x < sizeX && x > 0) {
					count++;
					//avg += pow(vertGraph[(int)x + j*sizeX]* lightScale - base, 2);
					avg += abs(vertGraph[(int)x + j*sizeX] * lightScale - base);
				}
			}
			if (count > 0) sum += avg / count;
		}
		else {
			/*float x = i + Sys.Geo.EmitX[view] * distance / Sys.Geo.EmitZ[view] / constants.PitchPx;
			if (x < sizeX && x > 0) {
				sum += pow(vertGraph[(int)x + view*sizeX] + offset - base, 2);
			}*/
		}
	}

	for (int i = 0; i < sizeY; i++) {
		float base = horGraph[i + center*sizeY];
		if (view == -1) {
			/*int count = 0;
			float avg = 0.0;
			for (int j = 0; j < NumViews; j++) {
				float x = i + Sys.Geo.EmitY[j] * distance / Sys.Geo.EmitZ[j] / constants.PitchPy;
				if (x < sizeY && x > 0) {
					count++;
					avg += pow(horGraph[(int)x + j*sizeY] - base, 2);
				}
			}
			if (count > 0) sum += avg / count;*/
		}
		else {
			float x = i + Sys.Geo.EmitY[view] * distance / Sys.Geo.EmitZ[view] / constants.PitchPy;
			float r = yP2MM(i, Sys.Proj.Ny, Sys.Proj.Pitch_y) - Sys.Geo.EmitY[view];
			if (x < sizeY && x > 0) {
				//sum += pow(horGraph[(int)x + view*sizeY]*lightScale + offset - base, 2);
				sum += abs(horGraph[(int)x + view*sizeY] * lightScale + offset + rSq*pow(r,2) - base);
			}
		}
	}

	return sum;
}

/****************************************************************************/
/*									Conversions								*/
/****************************************************************************/

//Projection space to recon space
TomoError TomoRecon::P2R(int* rX, int* rY, int pX, int pY, int view) {
	float dz = distance / Sys.Geo.EmitZ[view];
	*rX = xMM2R(xP2MM(pX, constants.Px, constants.PitchPx) * (1 + dz) - Sys.Geo.EmitX[view] * dz, constants.Rx, constants.PitchRx);
	*rY = yMM2R(yP2MM(pY, constants.Py, constants.PitchPy) * (1 + dz) - Sys.Geo.EmitY[view] * dz, constants.Ry, constants.PitchRy);

	return Tomo_OK;
}

//Recon space to projection space
TomoError TomoRecon::R2P(int* pX, int* pY, int rX, int rY, int view) {
	float dz = distance / Sys.Geo.EmitZ[view];
	*pX = xMM2P((xR2MM(rX, constants.Rx, constants.PitchRx) + Sys.Geo.EmitX[view] * dz) / (1 + dz), constants.Px, constants.PitchPx);
	*pY = yMM2P((yR2MM(rY, constants.Ry, constants.PitchRy) + Sys.Geo.EmitY[view] * dz) / (1 + dz), constants.Py, constants.PitchPy);

	return Tomo_OK;
}

//Image space to on-screen display
TomoError TomoRecon::I2D(int* dX, int* dY, int iX, int iY) {
	float innerOffx = (width - Sys.Proj.Nx / scale) / 2;
	float innerOffy = (height - Sys.Proj.Ny / scale) / 2;

	*dX = (int)((iX - xOff) / scale + innerOffx);
	*dY = (int)((iY - yOff) / scale + innerOffy);

	return Tomo_OK;
}

//Projection space to recon space
int TomoRecon::P2R(int p, int view, bool xDir) {
	float dz = distance / Sys.Geo.EmitZ[view];
	if (xDir)
		return (int)(xMM2R(xP2MM(p, constants.Px, constants.PitchPx) * (1 + dz) - Sys.Geo.EmitX[view] * dz, constants.Rx, constants.PitchRx));
	//else
	return (int)(yMM2R(yP2MM(p, constants.Py, constants.PitchPy) * (1 + dz) - Sys.Geo.EmitY[view] * dz, constants.Ry, constants.PitchRy));
}

//Recon space to projection space
int TomoRecon::R2P(int r, int view, bool xDir) {
	float dz = distance / Sys.Geo.EmitZ[view];
	if (xDir)
		return (int)(xMM2P((xR2MM(r, constants.Rx, constants.PitchRx) + Sys.Geo.EmitX[view] * dz) / (1.0f + dz), constants.Px, constants.PitchPx));
	//else
	return (int)(yMM2P((yR2MM(r, constants.Ry, constants.PitchRy) + Sys.Geo.EmitY[view] * dz) / (1.0f + dz), constants.Py, constants.PitchPy));
}

//Image space to on-screen display
int TomoRecon::I2D(int i, bool xDir) {
	if (xDir) {
		float innerOffx = (width - Sys.Proj.Nx / scale) / 2.0f;

		return (int)((i - xOff) / scale + innerOffx);
	}
	//else
	float innerOffy = (height - Sys.Proj.Ny / scale) / 2.0f;

	return (int)((i - yOff) / scale + innerOffy);
}

//On-screen coordinates to image space
int TomoRecon::D2I(int d, bool xDir) {
	if (xDir) {
		float innerOffx = (width - Sys.Proj.Nx / scale) / 2.0f;

		return (int)((d - innerOffx) * scale + xOff);
	}
	//else
	float innerOffy = (height - Sys.Proj.Ny / scale) / 2.0f;

	return (int)((d - innerOffy) * scale + yOff);
}