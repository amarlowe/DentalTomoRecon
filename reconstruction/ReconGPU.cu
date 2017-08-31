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

__global__ void mag(float *d_Dst, float *src1, float *src2, int pitchIn, int pitchOut, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Rx || y >= consts.Ry || x < 0 || y < 0)
		return;

	d_Dst[MUL_ADD(y, pitchOut, x)] = sqrt(pow(src1[MUL_ADD(y, pitchIn, x)], 2) + pow(src2[MUL_ADD(y, pitchIn, x)], 2));
}

__global__ void squareDiff(float *d_Dst, int view, float xOff, float yOff, int pitchOut, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;
	
	d_Dst[MUL_ADD(y, pitchOut, x)] = pow(tex2D(textError, x - xOff, y - yOff + view*consts.Py) - tex2D(textError, x, y + (NUMVIEWS / 2)*consts.Py), 2);
}

__global__ void add(float* src1, float* src2, float *d_Dst, int pitch, bool useRatio, bool useAbs, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;

	if (useRatio) {
		if (useAbs) {
			float val = consts.log ? abs(src2[MUL_ADD(y, pitch, x)]) : USHRT_MAX - abs(src2[MUL_ADD(y, pitch, x)]);
			d_Dst[MUL_ADD(y, pitch, x)] = (src1[MUL_ADD(y, pitch, x)] + val * consts.ratio) / (abs(consts.ratio) + 1);
		}
		else {
			float val = consts.log ? src2[MUL_ADD(y, pitch, x)] : USHRT_MAX - src2[MUL_ADD(y, pitch, x)];
			d_Dst[MUL_ADD(y, pitch, x)] = (src1[MUL_ADD(y, pitch, x)] + val * consts.ratio) / (abs(consts.ratio) + 1);
		}
	}
	else
		d_Dst[MUL_ADD(y, pitch, x)] = (src1[MUL_ADD(y, pitch, x)] + src2[MUL_ADD(y, pitch, x)]) / 2;
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

	if (consts.log) {
		if (sum != 0) {
			float correctedMax = logf(USHRT_MAX);
			sum = (correctedMax - logf(sum + 1)) / correctedMax * USHRT_MAX;
		}
	}
	sum = (sum - consts.minVal) / (consts.maxVal - consts.minVal) * UCHAR_MAX;
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
			if (consts.showNegative) {
				rgbx.r = 0;
				rgbx.g = 0;
				rgbx.b = sum;
			}
			else {
				rgbx.r = UCHAR_MAX;
				rgbx.g = UCHAR_MAX;
				rgbx.b = UCHAR_MAX;
			}
		}
		else {
			rgbx.r = sum;
			rgbx.g = sum;
			rgbx.b = sum;
		}
		
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
__global__ void LogCorrectProj(float * Sino, int view, unsigned short *Proj, unsigned short *Gain, params consts){
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

		if (val < LOWTHRESH) val = 0.0;

		val /= Gain[j*consts.Px + i];
		if (val > HIGHTHRESH) val = 0.0;
		val *= USHRT_MAX;

		//if (val / Gain[j*consts.Px + i] > HIGHTHRESH) val = 0.0;

		Sino[(y + view*consts.Py)*consts.ProjPitchNum + x] = val;

		//large noise correction
		if (consts.useMaxNoise) {
			//Get a second round to aviod gain correction issues
			__syncthreads();

			if (x > 1 && x < consts.Px - 1) {
				float val1 = Sino[(y + view*consts.Py)*consts.ProjPitchNum + x - 1];
				float val2 = Sino[(y + view*consts.Py)*consts.ProjPitchNum + x + 1];
				float val3 = (val1 + val2) / 2;
				if (abs(val1 - val2) < 2 * consts.maxNoise && abs(val3 - val) > consts.maxNoise)
					val = val3;
			}
			if (y > 1 && y < consts.Py - 1) {
				float val1 = Sino[(y - 1 + view*consts.Py)*consts.ProjPitchNum + x];
				float val2 = Sino[(y + 1 + view*consts.Py)*consts.ProjPitchNum + x];
				float val3 = (val1 + val2) / 2;
				if (abs(val1 - val2) < 2 * consts.maxNoise && abs(val3 - val) > consts.maxNoise)
					val = val3;
			}

			Sino[(y + view*consts.Py)*consts.ProjPitchNum + x] = val;
		}
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
		if (consts.orientation) dz = -dz;//z changes sign when flipped in the x direction
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
		if (data > 0) {
			float correctedMax = logf(USHRT_MAX);
			data = (correctedMax - logf(data + 1)) / correctedMax * USHRT_MAX;
		}
	}
	if (data > USHRT_MAX) data = USHRT_MAX;
	if (data < 0.0f) return;// data = 0.0f;
	atomicAdd(d_Histogram + ((unsigned short)data >> 8), 1);//bin by the upper 256 bits
}

/* div = div0 - g/lambda*/
__global__ void sub0(float *div0, float *div, float *g, float lambda, int nx, int ny) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x < nx && y < ny)
		div[idx] = div0[idx] - g[idx] / lambda;
}

//Chambolle TV denoising
__global__ void gradient(float *g, float *div0, float *div, float *z, float lambda, int nx, int ny) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x<nx && y<ny) {
		if (x<(nx - 1)) z[2 * idx + 0] = (div0[idx + 1] - g[idx + 1] / lambda) - (div0[idx] - g[idx] / lambda);
		else z[2 * idx + 0] = 0;
		if (y<(ny - 1)) z[2 * idx + 1] = (div0[idx + nx] - g[idx + nx] / lambda) - (div0[idx] - g[idx] / lambda);
		else z[2 * idx + 1] = 0;
	}
}

/*z = (z0 + tau z)/(1+tau|z|)*/
__global__ void zupdate(float *z, float *z0, float tau, int nx, int ny) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;
	if (x<nx && y<ny) {
		float a = z[2 * idx + 0];
		float b = z[2 * idx + 1];
		float t = 1 / (1 + tau*sqrtf(a*a + b*b));
		z[2 * idx + 0] = (z0[2 * idx + 0] + tau*z[2 * idx + 0])*t;
		z[2 * idx + 1] = (z0[2 * idx + 1] + tau*z[2 * idx + 1])*t;
	}
}

__global__ void divergence(float *v, float *d, int nx, int ny) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = x + y*nx;

	if (x<nx && y<ny) {
		float AX = 0;
		if ((x<(nx - 1))) AX += v[2 * (idx)+0];
		if ((x>0))      AX -= v[2 * (idx - 1) + 0];

		if ((y<(ny - 1))) AX += v[2 * (idx)+1];
		if ((y>0))      AX -= v[2 * (idx - nx) + 1];

		d[idx] = AX;
	}
}

// u=  (1 - tu) * uold + tu .* ( f + 1/lambda*div(z) );
__global__ void updhgF_SoA(float *f, float *z1, float *z2, float *g, float tf, float invlambda, int nx, int ny){
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;
	float DIVZ;

	if (px<nx && py<ny){
		// compute the divergence
		DIVZ = 0;
		if ((px<(nx - 1))) DIVZ += z1[idx];
		if ((px>0))      DIVZ -= z1[idx - 1];

		if ((py<(ny - 1))) DIVZ += z2[idx];
		if ((py>0))      DIVZ -= z2[idx - nx];

		// update f
		f[idx] = (1 - tf) *f[idx] + tf * (g[idx] + invlambda*DIVZ);
	}

}


// z= zold + tz*lambda* grad(u);	
// and normalize z:  
//n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) ) ); 
// z= z/n;
__global__ void updhgZ_SoA(float *z1, float *z2, float *f, float tz, float lambda, int nx, int ny){
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = px + py*nx;

	if (px<nx && py<ny){
		// compute the gradient
		float a = 0;
		float b = 0;
		float fc = f[idx];
		if (px<(nx - 1)) a = f[idx + 1] - fc;
		if (py<(ny - 1)) b = f[idx + nx] - fc;

		// update z

		a = z1[idx] + tz*lambda*a;
		b = z2[idx] + tz*lambda*b;

		// project
		float t = 0;
		t = sqrtf(a*a + b*b);
		t = (t <= 1 ? 1. : t);

		z1[idx] = a / t;
		z2[idx] = b / t;
	}
}


/********************************************************************************************/
/* Function to interface the CPU with the GPU:												*/
/********************************************************************************************/

//Function to set up the memory on the GPU
TomoError TomoRecon::initGPU(){
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

	//Set up display and buffer regions
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
	cuda(Malloc(&buff1, sizeIM * sizeof(float)));
	cuda(Malloc(&buff2, sizeIM * sizeof(float)));

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

	return Tomo_OK;
}

TomoError TomoRecon::ReadProjections(const char * gainFile, const char * mainFile) {
	//Read and correct projections
	unsigned short * RawData = new unsigned short[Sys.Proj.Nx*Sys.Proj.Ny];
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
	unsigned short * d_Gain;
	cuda(Malloc((void**)&d_Proj, sizeProj));
	cuda(Malloc((void**)&d_Gain, sizeProj));

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

		KERNELCALL2(LogCorrectProj, dimGridProj, dimBlockProj, d_Sino, view, d_Proj, d_Gain, constants);

		
		scanLineDetect(view, d_SumValsVert, sumValsVert + view * Sys.Proj.Nx, vertOff + view * Sys.Proj.Nx, true, cConstants.scanVertEnable);
		scanLineDetect(view, d_SumValsHor, sumValsHor + view * Sys.Proj.Ny, horOff + view * Sys.Proj.Ny, false, cConstants.scanHorEnable);
	}

	float step = 10;
	float best = FLT_MAX;
	float bestRSq;
	float rSq = 0.0;
	bool firstLin = true;
	float bestDist;

	float sumMain = 0.0;
	for (int j = 0; j < Sys.Proj.Ny; j++)
		sumMain += sumValsHor[j + NumViews / 2 * Sys.Proj.Ny];

	for (int i = 0; i < NumViews; i++) {
		if (i == NumViews / 2) continue;
		float sumView = 0.0;
		for (int j = 0; j < Sys.Proj.Ny; j++)
			sumView += sumValsHor[j + i * Sys.Proj.Ny];

		float offset = (sumMain - sumView) / Sys.Proj.Ny;
		for (int j = 0; j < Sys.Proj.Ny; j++) {
			horOff[j + i * Sys.Proj.Ny] -= offset;
		}
	}

	step = STARTSTEP;
	bestDist = MINDIS;
	best = FLT_MAX;
	firstLin = true;

	//Normalize projection image lighting
	float maxVal, minVal;
	unsigned int histogram[HIST_BIN_COUNT];
	getHistogram(d_Sino + (NumViews / 2)*projPitch / sizeof(float)*Sys.Proj.Ny, projPitch*Sys.Proj.Ny, histogram);
	autoLight(histogram, 80, &minVal, &maxVal);
	cuda(Memcpy(d_MinVal, &minVal, sizeof(float), cudaMemcpyHostToDevice));

	float *g, *z1, *z2;
	size_t size;
	int j;
	int nx = projPitch / sizeof(float);
	int ny = Sys.Proj.Ny;
	size = nx*ny * sizeof(float);
	float tz = 2, tf = .2, beta = 0.0001;
	/* allocate device memory */
	cudaMalloc((void **)&g, size);
	cudaMalloc((void **)&z1, size);
	cudaMalloc((void **)&z2, size);

	/* setup a 2D thread grid, with 16x16 blocks */
	/* each block is will use nearby memory*/
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	for (int view = 0; view < NumViews; view++) {
		cuda(MemcpyAsync(d_MaxVal, &maxVal, sizeof(float), cudaMemcpyHostToDevice));
		/*
		{
			std::ofstream FILE1, FILE2;
			std::stringstream outputfile1, outputfile2;
			outputfile1 << "C:\\Users\\jdean\\Downloads\\cudaTV\\cudaTV\\correctedVert" << view << ".txt";
			outputfile2 << "C:\\Users\\jdean\\Downloads\\cudaTV\\cudaTV\\correctedHor" << view << ".txt";
			FILE1.open(outputfile1.str());
			for (int i = 0; i < Sys.Proj.Nx; i++) {
				//int x = i + Sys.Geo.EmitX[view] * distance / Sys.Geo.EmitZ[view] / constants.PitchPx;
				int x = i;
				if(x < Sys.Proj.Nx && x > 0)
					FILE1 << sumValsVert[x + view * Sys.Proj.Nx] << "\n";
				else FILE1 << 0.0 << "\n";
			}
			FILE1.close();

			FILE2.open(outputfile2.str());
			for (int i = 0; i < Sys.Proj.Ny; i++) {
				//int x = i + Sys.Geo.EmitY[view] * distance / Sys.Geo.EmitZ[view] / constants.PitchPy;
				int x = i;
				if (x < Sys.Proj.Ny && x > 0)
					FILE2 << sumValsHor[x + view * Sys.Proj.Ny] << "\n";
				else FILE2 << 0.0 << "\n";
			}
			FILE2.close();
		}*/

		cuda(MemcpyAsync(d_SumValsVert, vertOff + view * Sys.Proj.Nx, Sys.Proj.Nx * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(d_SumValsHor, horOff + view * Sys.Proj.Ny, Sys.Proj.Ny * sizeof(float), cudaMemcpyHostToDevice));

		KERNELCALL2(rescale, dimGridProj, dimBlockProj, d_Sino, view, d_MaxVal, d_MinVal, d_SumValsVert, d_SumValsHor, constants);

		if (useTV) {
			//chaTV(d_Sino + projPitch / sizeof(float) * Sys.Proj.Ny * view, iter, projPitch / sizeof(float), Sys.Proj.Ny, lambda);
			float * input = d_Sino + projPitch / sizeof(float) * Sys.Proj.Ny * view;

			/* Copy input to device*/
			cuda(MemcpyAsync(g, input, size, cudaMemcpyDeviceToDevice));
			cuda(MemsetAsync(z1, 0, size));
			cuda(MemsetAsync(z2, 0, size));

			/* call the functions */
			for (j = 0; j<iter; j++) {
				tz = 0.2 + 0.08*j;
				tf = (0.5 - 5. / (15 + j)) / tz;

				// z= zold + tauz.* grad(u);	
				// and normalize z:  n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) + beta) ); z/=n;
				KERNELCALL2(updhgZ_SoA, n_blocks, block_size, z1, z2, input, tz, 1 / lambda, nx, ny);

				// u=  (1 - tauu*lambda) * uold + tauu .* div(z) + tauu*lambda.*f;
				KERNELCALL2(updhgF_SoA, n_blocks, block_size, input, z1, z2, g, tf, lambda, nx, ny);
			}
		}
	}

	/* free device memory */
	cuda(Free(g));
	cuda(Free(z1));
	cuda(Free(z2));

	constants.log = oldLog;

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	delete[] RawData;
	delete[] GainData;
	delete[] sumValsHor;
	delete[] sumValsVert;
	delete[] vertOff;
	delete[] horOff;
	cuda(Free(d_Proj));
	cuda(Free(d_Gain));
	cuda(Free(d_SumValsHor));
	cuda(Free(d_SumValsVert));

	return Tomo_OK;
}

TomoError TomoRecon::chaTV(float *input, int it, int nx, int ny, float lambda){

	return Tomo_OK;
}

TomoError TomoRecon::WriteDICOMFullData(std::string Path, int slices) {
	//Set up the basic path to the raw projection data
	FILE * ReconData = fopen(Path.c_str(), "ab");

	//Open the path and read data to buffer
	
	if (ReconData == NULL)
		return Tomo_DICOM_err;

	float * RawData = new float[reconPitch / sizeof(float)*Sys.Proj.Ny];
	unsigned short * output = new unsigned short[reconPitch / sizeof(float)*Sys.Proj.Ny];

	//Create the reconstruction volume around the current location
	float oldDistance = distance;
	distance -= slices / 2 * Sys.Geo.ZPitch;
	for (int i = 0; i < slices; i++) {
		singleFrame();
		distance += Sys.Geo.ZPitch;
		cudaMemcpy(RawData, d_Image, reconPitch*Sys.Recon.Ny, cudaMemcpyDeviceToHost);
		for (int j = 0; j < reconPitch / sizeof(float)*Sys.Proj.Ny; j++) {
			float data = RawData[j];
			if (data != 0.0) {
				if (constants.log)
					data = (logf(USHRT_MAX) - logf(data + 1)) / logf(USHRT_MAX) * USHRT_MAX;
				data = (data - constants.minVal) / (constants.maxVal - constants.minVal) * SHRT_MAX;
				if (data > SHRT_MAX) data = SHRT_MAX;
				if (data < 0.0f) data = 0.0f;
			}
			output[j] = (unsigned short)data;
		}
		fwrite(output, sizeof(unsigned short), reconPitch / sizeof(float)*Sys.Proj.Ny, ReconData);
	}
	
	distance = oldDistance;
	singleFrame();
	fclose(ReconData);
	delete[] RawData;
	delete[] output;

	return Tomo_OK;
}

TomoError TomoRecon::scanLineDetect(int view, float * d_sum, float * sum, float * offset, bool vert, bool enable) {
	int vectorSize;
	if (vert) vectorSize = Sys.Proj.Nx;
	else vectorSize = Sys.Proj.Ny;
	float * sumCorr = new float[vectorSize];

	cuda(BindTexture2D(NULL, textSino, d_Sino + view*Sys.Proj.Ny*projPitch / sizeof(float), cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny, projPitch));
	KERNELCALL3(sumRowsOrCols, dim3(1, vectorSize), reductionThreads, reductionSize, d_sum, vert, constants);
	cuda(Memcpy(sum, d_sum, vectorSize * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef PRINTSCANCORRECTIONS
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
#else
	for (int i = 0; i < vectorSize; i++) {
		sum[i] /= vectorSize;
		sumCorr[i] = sum[i];
	}
#endif

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

#ifdef PRINTSCANCORRECTIONS
	{
		std::ofstream FILE;
		std::stringstream outputfile;
		outputfile << "C:\\Users\\jdean\\Downloads\\cudaTV\\cudaTV\\corrected" << view << ".txt";
		FILE.open(outputfile.str());
		for (int i = 0; i < vectorSize; i++) {
			FILE << sumCorr[i] << "\n";
			if(enable)
				offset[i] = sum[i] - sumCorr[i];
			else offset[i] = 0.0;
			sum[i] = sumCorr[i];
		}
		FILE.close();
	}
	
#else
	for (int i = 0; i < vectorSize; i++) {
		if (enable)
			offset[i] = sum[i] - sumCorr[i];
		else offset[i] = 0.0;
		sum[i] = sumCorr[i];
	}
#endif
	delete[] sumCorr;
}

//Fucntion to free the gpu memory after program finishes
TomoError TomoRecon::FreeGPUMemory(void){
	//Free memory allocated on the GPU
	cuda(Free(d_Image));
	cuda(Free(d_Error));
	cuda(Free(d_Sino));
	cuda(Free(buff1));
	cuda(Free(buff2));

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
	case x_mag_enhance:
		imageKernel(d_gaussDer, d_gauss, buff1);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, Sys.Recon.Nx, true, true, constants);
		break;
	case y_mag_enhance:
		imageKernel(d_gauss, d_gaussDer, buff1);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, Sys.Recon.Nx, true, true, constants);
		break;
	case mag_enhance:
		imageKernel(d_gaussDer, d_gauss, buff1);
		imageKernel(d_gauss, d_gaussDer, buff2);
		KERNELCALL2(mag, contBlocks, contThreads, buff1, buff2, buff1, reconPitchNum, reconPitchNum, constants);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, Sys.Recon.Nx, true, false, constants);
		break;
	case x_enhance:
		imageKernel(d_gaussDer, d_gauss, buff1);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, Sys.Recon.Nx, true, false, constants);
		break;
	case y_enhance:
		imageKernel(d_gauss, d_gaussDer, buff1);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, Sys.Recon.Nx, true, false, constants);
		break;
	case both_enhance:
		imageKernel(d_gaussDer, d_gauss, buff1);
		imageKernel(d_gauss, d_gaussDer, buff2);
		KERNELCALL2(add, contBlocks, contThreads, buff1, buff2, buff1, Sys.Recon.Nx, false, false, constants);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, Sys.Recon.Nx, true, false, constants);
		break;
	case der_x:
		imageKernel(d_gaussDer, d_gauss, d_Image);
		break;
	case der_y:
		imageKernel(d_gauss, d_gaussDer, d_Image);
		break;
	case square_mag:
		imageKernel(d_gaussDer, d_gauss, buff1);
		imageKernel(d_gauss, d_gaussDer, d_Image);

		KERNELCALL2(squareMag, contBlocks, contThreads, d_Image, buff1, d_Image, reconPitchNum, reconPitchNum, constants);
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
	static derivative_t oldDisplay;

	if (firstRun) {
		step = STARTSTEP;
		distance = MINDIS;
		bestDist = MINDIS;
		best = 0;
		linearRegion = false;
		oldDisplay = derDisplay;
		derDisplay = square_mag;
		singleFrame();
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
				derDisplay = oldDisplay;
				singleFrame();
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
	int innerThresh = threshold;
	bool emptyHist = false;
	if (histogram == NULL) {
		emptyHist = true;
		histogram = new unsigned int[HIST_BIN_COUNT];
		getHistogram(d_Image, reconPitch*Sys.Recon.Ny, histogram);
		innerThresh = Sys.Recon.Nx * Sys.Recon.Ny / AUTOTHRESHOLD;
		minVal = &constants.minVal;
		maxVal = &constants.maxVal;
	}

	int i;
	for (i = 0; i < HIST_BIN_COUNT; i++) {
		unsigned int count = histogram[i];
		if (count > innerThresh) break;
	}
	if (i >= HIST_BIN_COUNT) i = 0;
	*minVal = i * UCHAR_MAX;

	//go from the reverse direction for maxval
	for (i = HIST_BIN_COUNT - 1; i >= 0; i--) {
		unsigned int count = histogram[i];
		if (count > innerThresh) break;
	}
	if (i < 0) i = HIST_BIN_COUNT;
	*maxVal = i * UCHAR_MAX;
	if (*minVal == *maxVal) *maxVal += UCHAR_MAX;

	if (emptyHist) delete[] histogram;

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
		bool ascend = upYr > lowYr;
		int increment = ascend ? 1 : -1;
		while ((!ascend && thisY >= upYr) || (ascend && thisY <= upYr)) {//y counts down
			int thisX = startX;
			int negCross = 0;
			bool negativeSpace = false;
			float negAcc = 0;
			while (thisX < endX) {
				float val = h_xDer2[thisY * reconPitchNum + thisX];
				h_xDer2[thisY * reconPitchNum + thisX] = val / 10.0f;
				if (negativeSpace) {
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
					if (val < 0) {
						negativeSpace = true;
						negAcc = val;
					}
				}
				thisX++;
			}
			if (negCross < LINEPAIRS) {
				thisY -= increment;
				break;
			}
			thisY += increment;
		}
		*resolution = phanScale / (thisY - lowYr + phanScale / LOWERBOUND);
		
		cuda(Memcpy(d_Image, h_xDer2, reconPitch*Sys.Recon.Ny, cudaMemcpyHostToDevice));
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
		bool ascend = upXr > lowXr;
		int increment = ascend ? 1 : -1;
		while ((!ascend && thisX >= upXr) || (ascend && thisX <= upXr)) {//y counts down
			int thisY = startY;
			int negCross = 0;
			bool negativeSpace = false;
			float negAcc = 0;
			while (thisY < endY) {
				float val = h_yDer2[thisY * reconPitchNum + thisX];
				h_yDer2[thisY * reconPitchNum + thisX] = val / 10.0f;
				if (negativeSpace) {
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
					if (val < 0) {
						negativeSpace = true;
						negAcc = val;
					}
				}
				thisY++;
			}
			if (negCross < LINEPAIRS) {
				thisX -= increment;
				break;
			}
			thisX += increment;
		}
		*resolution = phanScale / (thisX - lowXr + phanScale / LOWERBOUND);

		cuda(Memcpy(d_Image, h_yDer2, reconPitch*Sys.Recon.Ny, cudaMemcpyHostToDevice));
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
		autoLight();
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

	autoLight();

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

float TomoRecon::getMax(float * d_Im) {
	constants.baseXr = 3 * Sys.Recon.Nx / 4;
	constants.baseYr = 3 * Sys.Recon.Ny / 4;
	constants.currXr = Sys.Recon.Nx / 4;
	constants.currYr = Sys.Recon.Ny / 4;

	unsigned int histogram[HIST_BIN_COUNT];
	int threshold = Sys.Recon.Nx * Sys.Recon.Ny / AUTOTHRESHOLD;
	getHistogram(d_Im, reconPitch*Sys.Recon.Ny, histogram);

	int i;
	for (i = HIST_BIN_COUNT - 1; i >= 0; i--) {
		unsigned int count = histogram[i];
		if (count > threshold) break;
	}
	if (i < 0) i = HIST_BIN_COUNT;

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	return i * UCHAR_MAX;
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