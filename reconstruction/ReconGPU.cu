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
texture<float, cudaTextureType3D, cudaReadModeElementType> textRecon;
texture<float, cudaTextureType3D, cudaReadModeElementType> textDelta;
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
	const int pitch = consts.dataDisplay == reconstruction || consts.dataDisplay == iterRecon ? consts.ReconPitchNum : consts.ProjPitchNum;

	if (consts.dataDisplay == reconstruction || consts.dataDisplay == iterRecon) {
		if (ix >= consts.Rx || iy >= consts.Ry)
			return;
		if (ix >= consts.Rx - KERNELRADIUS || ix < KERNELRADIUS ) {// || iy >= consts.Ry - KERNELRADIUS || iy < KERNELRADIUS
			d_Dst[MUL_ADD(iy, pitch, ix)] = 0.0f;
			return;
		}
	}
	else {
		if (ix >= consts.Px || iy >= consts.Py)
			return;
		if (ix >= consts.Px - KERNELRADIUS || ix < KERNELRADIUS) {// || iy >= consts.Py - KERNELRADIUS || iy < KERNELRADIUS
			d_Dst[MUL_ADD(iy, pitch, ix)] = 0.0f;
			return;
		}
	}

	d_Dst[MUL_ADD(iy, pitch, ix)] = convolutionRow<KERNELSIZE>(x, y, kernel);
}

__global__ void convolutionColumnsKernel(float *d_Dst, float kernel[KERNELSIZE], params consts){
	const int ix = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int iy = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	const float  x = (float)ix + 0.5f;
	const float  y = (float)iy + 0.5f;
	const int pitch = consts.dataDisplay == reconstruction || consts.dataDisplay == iterRecon ? consts.ReconPitchNum : consts.ProjPitchNum;

	if (consts.dataDisplay == reconstruction || consts.dataDisplay == iterRecon) {
		if (ix >= consts.Rx || iy >= consts.Ry)
			return;
		if (iy >= consts.Ry - KERNELRADIUS || iy < KERNELRADIUS) {//ix >= consts.Rx - KERNELRADIUS || || ix < KERNELRADIUS
			d_Dst[MUL_ADD(iy, pitch, ix)] = 0.0f;
			return;
		}
	}
	else {
		if (ix >= consts.Px || iy >= consts.Py)
			return;
		if (iy >= consts.Py - KERNELRADIUS || iy < KERNELRADIUS) {//ix >= consts.Px - KERNELRADIUS || || ix < KERNELRADIUS
			d_Dst[MUL_ADD(iy, pitch, ix)] = 0.0f;
			return;
		}
	}

	d_Dst[MUL_ADD(iy, pitch, ix)] = convolutionColumn<KERNELSIZE>(x, y, kernel);
}

__global__ void squareMag(float *d_Dst, float *src1, float *src2, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	bool recon = consts.dataDisplay == reconstruction || consts.dataDisplay == iterRecon;
	int pitch = recon ? consts.ReconPitchNum : consts.ProjPitchNum;
	if ((recon && x >= consts.Rx) || (recon && y >= consts.Ry) || (!recon && x >= consts.Px) || (!recon && y >= consts.Py) || x < 0 || y < 0)
		return;

	d_Dst[MUL_ADD(y, pitch, x)] = pow(src1[MUL_ADD(y, pitch, x)],2) + pow(src2[MUL_ADD(y, pitch, x)],2);
}

__global__ void mag(float *d_Dst, float *src1, float *src2, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	bool recon = consts.dataDisplay == reconstruction || consts.dataDisplay == iterRecon;
	int pitch = recon ? consts.ReconPitchNum : consts.ProjPitchNum;
	if ((recon && x >= consts.Rx) || (recon && y >= consts.Ry) || (!recon && x >= consts.Px) || (!recon && y >= consts.Py) || x < 0 || y < 0)
		return;

	d_Dst[MUL_ADD(y, pitch, x)] = sqrt(pow(src1[MUL_ADD(y, pitch, x)], 2) + pow(src2[MUL_ADD(y, pitch, x)], 2));
}

__global__ void squareDiff(float *d_Dst, int view, float xOff, float yOff, int pitchOut, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;
	
	d_Dst[MUL_ADD(y, pitchOut, x)] = pow(tex2D(textError, x - xOff, y - yOff + view*consts.Py) - tex2D(textError, x, y + (NUMVIEWS / 2)*consts.Py), 2);
}

__global__ void add(float* src1, float* src2, float *d_Dst, bool useRatio, bool useAbs, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);
	bool recon = consts.dataDisplay == reconstruction || consts.dataDisplay == iterRecon;
	int pitch = recon ? consts.ReconPitchNum : consts.ProjPitchNum;
	if ((recon && x >= consts.Rx) || (recon && y >= consts.Ry) || (!recon && x >= consts.Px) || (!recon && y >= consts.Py) || x < 0 || y < 0)
		return;

	if (useRatio) {
		if (useAbs) {
			float val = consts.log ? abs(src2[MUL_ADD(y, pitch, x)]) : USHRT_MAX - abs(src2[MUL_ADD(y, pitch, x)]);
			d_Dst[MUL_ADD(y, pitch, x)] = src1[MUL_ADD(y, pitch, x)] * consts.ratio + val * (1 - consts.ratio);
		}
		else {
			float val = consts.log ? src2[MUL_ADD(y, pitch, x)] : USHRT_MAX - src2[MUL_ADD(y, pitch, x)];
			d_Dst[MUL_ADD(y, pitch, x)] = src1[MUL_ADD(y, pitch, x)] * consts.ratio + val * (1 - consts.ratio);
		}
	}
	else
		d_Dst[MUL_ADD(y, pitch, x)] = (src1[MUL_ADD(y, pitch, x)] + src2[MUL_ADD(y, pitch, x)]) / 2;
}

__global__ void div(float* src1, float* src2, float *d_Dst, int pitch, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;

	d_Dst[MUL_ADD(y, pitch, x)] = 100 * src1[MUL_ADD(y, pitch, x)] / (abs(src2[MUL_ADD(y, pitch, x)]) + 1);
}

__global__ void thresh(float* src1, float* src2, float *d_Dst, int pitch, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;

	if(src1[MUL_ADD(y, pitch, x)] < 50.0f)
		d_Dst[MUL_ADD(y, pitch, x)] = src2[MUL_ADD(y, pitch, x)];
	else d_Dst[MUL_ADD(y, pitch, x)] = 0.0f;
}

__global__ void sub(float* src1, float* src2, float *d_Dst, int pitch, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;

	d_Dst[MUL_ADD(y, pitch, x)] = src1[MUL_ADD(y, pitch, x)] - src2[MUL_ADD(y, pitch, x)];
}

__global__ void invert(float* image, params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Px || y >= consts.Py || x < 0 || y < 0)
		return;

	float val = image[MUL_ADD(y, consts.ProjPitchNum, x)];
	float correctedMax = logf(USHRT_MAX);
	if (val <= 0.0f) image[MUL_ADD(y, consts.ProjPitchNum, x)] = 0.0f;
	else if (val >= USHRT_MAX) image[MUL_ADD(y, consts.ProjPitchNum, x)] = USHRT_MAX;
#ifdef USELOGITER
	else image[MUL_ADD(y, consts.ProjPitchNum, x)] = (correctedMax - logf(val + 1)) / correctedMax * USHRT_MAX;
#else
	else image[MUL_ADD(y, consts.ProjPitchNum, x)] = USHRT_MAX - val;
#endif
}

__global__ void projectSliceZ(float * zBuff[KERNELSIZE], int index, int projIndex, float distance, params consts) {
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
		if (projIndex >= 0) view = projIndex;
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

		if (projIndex >= 0) break;
	}

	if (count > 0)
		zBuff[index][j*consts.ReconPitchNum + i] = error / count;
	else zBuff[index][j*consts.ReconPitchNum + i] = 0;
}

__global__ void zConvolution(float *d_Dst, float * zSrc[KERNELSIZE], float kernel[KERNELSIZE], params consts) {
	const int x = MUL_ADD(blockDim.x, blockIdx.x, threadIdx.x);
	const int y = MUL_ADD(blockDim.y, blockIdx.y, threadIdx.y);

	if (x >= consts.Rx || y >= consts.Ry || x < 0 || y < 0)
		return;

	float out = 0.0f;
	for (int i = 0; i < KERNELSIZE; i++)
		out += zSrc[i][MUL_ADD(y, consts.ReconPitchNum, x)] * kernel[i];

	//if(abs(out) > 10.0f)
		d_Dst[MUL_ADD(y, consts.ReconPitchNum, x)] = abs(out);
	//else d_Dst[MUL_ADD(y, consts.ReconPitchNum, x)] = 0.0f;
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
	if (consts.orientation) i = wIn - 1 - i;
	if (consts.flip) j = hIn - 1 - j;
	if (i > 0 && j > 0 && i < wIn && j < hIn)
		sum = tex2D(textImage, (float)i + 0.5f, (float)j + 0.5f);

	if (sum < 0) {
		negative = true;
		sum = abs(sum);
	}

	if (consts.log) {
		if (sum > 0.0f) {
			float correctedMax = logf(USHRT_MAX);
			sum = (correctedMax - logf(sum + 1)) / correctedMax * USHRT_MAX;
		}
		else sum = consts.minVal;
	}
	sum = (sum - consts.minVal) / (consts.maxVal - consts.minVal) * UCHAR_MAX;
	//if (!consts.log) sum = UCHAR_MAX - sum;
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
				rgbx.r = 0;
				rgbx.g = 0;
				rgbx.b = 0;
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
		int x = i;
		int y = j;

		float val = Proj[j*consts.Px + i];

		if (consts.useGain) {
			val /= (float)Gain[j*consts.Px + i] * (float)consts.exposure / (float)EXPOSUREBASE;
			if (val > HIGHTHRESH) val = 0.0f;
			val *= USHRT_MAX;
		}

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

__global__ void rescale(float * Sino, int view, float * MaxVal, float * MinVal, float * colShifts, float * rowShifts, float scale, params consts) {
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i < consts.Px) && (j < consts.Py)) {
		float test = Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] -*MinVal;
		if (test > 0) {
			test = (test - colShifts[i] - rowShifts[j]) / scale / (*MaxVal - *MinVal) * USHRT_MAX;//scale from 1 to max
			if (test > LOWTHRESH) Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] = test;// && test < ABSHIGHTHRESH
			else Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] = 0.0f;
		}
		else Sino[(j + view*consts.Py)*consts.ProjPitchNum + i] = 0.0f;
	}
}

//Create the single slice projection image
__global__ void projectSlice(float * IM, float distance, params consts) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float value;

	//Set a normalization and pixel value to 0
	float error = 0.0f;
	float count = 0.0f;

	//Check image boundaries
	if ((i >= consts.Rx) || (j >= consts.Ry)) return;

	for (int view = 0; view < NUMVIEWS; view++) {
		float dz = distance / consts.d_Beamz[view];
		float x = xMM2P((xR2MM(i, consts.Rx, consts.PitchRx) + consts.d_Beamx[view] * dz), consts.Px, consts.PitchPx);// / (1 + dz)
		float y = yMM2P((yR2MM(j, consts.Ry, consts.PitchRy) + consts.d_Beamy[view] * dz), consts.Py, consts.PitchPy);

		//Update the value based on the error scaled and save the scale
		if (y > 0 && y < consts.Py && x > 0 && x < consts.Px) {
			value = tex2D(textError, x, y + view*consts.Py);
			float increment = 1.0f;
			if (y < TAPERSIZE) increment *= y / TAPERSIZE;
			if (y > consts.Py - TAPERSIZE) increment *= (consts.Py - y) / TAPERSIZE;
			if (x < TAPERSIZE) increment *= x / TAPERSIZE;
			if (x > consts.Px - TAPERSIZE) increment *= (consts.Px - x) / TAPERSIZE;
			if (value != 0) {
				error += value * increment;
				count += increment;
			}
		}
	}

	if (count > 0)
		IM[j*consts.ReconPitchNum + i] = error / (float)count;
	else IM[j*consts.ReconPitchNum + i] = 0.0f;
}

#ifdef SHOWERROR
__global__ void projectIter(float * oldRecon, int slice, float iteration, bool skipTV, params consts, cudaSurfaceObject_t surfRecon, cudaSurfaceObject_t errorRecon) {
#else
__global__ void projectIter(float * proj, float * oldRecon, int slice, float iteration, bool skipTV, params consts, cudaSurfaceObject_t surfRecon, cudaSurfaceObject_t surfDelta, bool firstRun = false) {
#endif
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Set a normalization and pixel value to 0
	float count = 0.0f;
	float error = 0.0f;
	//float singleVal = 0.0f;

	//Check image boundaries
	if ((i >= consts.Rx) || (j >= consts.Ry)) return;

	for (int view = 0; view < NUMVIEWS; view++) {
		float dz = (consts.startDis + slice * consts.pitchZ) / consts.d_Beamz[view];
		float x = xMM2P((xR2MM(i, consts.Rx, consts.PitchRx) + consts.d_Beamx[view] * dz), consts.Px, consts.PitchPx);// / (1 + dz)
		float y = yMM2P((yR2MM(j, consts.Ry, consts.PitchRy) + consts.d_Beamy[view] * dz), consts.Py, consts.PitchPy);

		//Update the value based on the error scaled and save the scale
		if (y > 0 && y < consts.Py && x > 0 && x < consts.Px) {
			float value = tex2D(textError, x, y + view*consts.Py);
			float increment = 1.0f;
			if (y < TAPERSIZE) increment *= y / TAPERSIZE;
			if (y > consts.Py - TAPERSIZE) increment *= (consts.Py - y) / TAPERSIZE;
			if (x < TAPERSIZE) increment *= x / TAPERSIZE;
			if (x > consts.Px - TAPERSIZE) increment *= (consts.Px - x) / TAPERSIZE;
			if (abs(value) > 0.1f) {
				//float singleTemp = tex2D(textSino, x, y + view*consts.Py);
				error += value * increment;
				count += increment;
				//singleVal += singleTemp * increment;
			}
			//float minTest = proj[(view * consts.Py + j)*consts.ProjPitchNum + i];
			//if (minTest < minimum) minimum = minTest;
		}
	}

	if (count > 0) {
		error /= ((float)count * (float)consts.slices);
		//singleVal /= (float)consts.slices;
	}
	else {
		error = 0.0f;
		//singleVal = 0.0f;
	}

	float returnVal, delta;
	surf3Dread(&returnVal, surfRecon, i * sizeof(float), j, slice);
	surf3Dread(&delta, surfDelta, i * sizeof(float), j, slice);
	
	if (!skipTV) {
		float AX = 0, BX = 0, temp;
		if (i > 0) { 
			temp = oldRecon[i - 1 + j*consts.ReconPitchNum];
			if (temp > 0.1f) BX += temp * TVX;  AX += TVX;
		}
		if (i < consts.Rx - 1) { 
			temp = oldRecon[i + 1 + j*consts.ReconPitchNum];
			if (temp > 0.1f) BX += temp * TVX; AX += TVX;
		}
		if (j > 0) { 
			temp = oldRecon[i + (j - 1)*consts.ReconPitchNum];
			if (temp > 0.1f) BX += temp * TVY; AX += TVY;
		}
		if (j < consts.Ry - 1) { 
			temp = oldRecon[i + (j + 1)*consts.ReconPitchNum];
			if (temp > 0.1f) BX += temp * TVY; AX += TVY; 
		}
		//if (slice > 0) { surf3Dread(&returnVal, surfRecon, i * sizeof(float), j, slice - 1); BX += returnVal * TVZ; AX += TVZ; }
		//if (slice < consts.slices - 1) { surf3Dread(&returnVal, surfRecon, i * sizeof(float), j, slice + 1); BX += returnVal * TVZ; AX += TVZ; }
		//surf3Dread(&returnVal, surfRecon, i * sizeof(float), j, slice);
		if(AX > 0.0f) error += BX - AX*returnVal;
	}

	error += 0.5f*delta;
	returnVal += error;
	returnVal *= 0.97f;
	//returnVal += singleVal - returnVal * 0.01;
	//if (returnVal > 10.0 * singleVal) returnVal = 10.0 * singleVal;
	//returnVal = singleVal;
	if (returnVal > 10000) returnVal = 10000;
	if (returnVal < 0.1f) returnVal = 0.1f;
#ifdef SHOWERROR
	surf3Dwrite(error, errorRecon, i * sizeof(float), j, slice);
#endif

#ifdef RECONDERIVATIVE
	if (count == 0 || returnVal < 0.0f) surf3Dwrite(0.0f, surfRecon, i * sizeof(float), j, slice);
	else surf3Dwrite(returnVal, surfRecon, i * sizeof(float), j, slice);
#else
	surf3Dwrite(error, surfDelta, i * sizeof(float), j, slice);
	surf3Dwrite(returnVal, surfRecon, i * sizeof(float), j, slice);
#endif // RECONDERIVATIVE
}

__global__ void backProject(float * proj, float * error, int view, float iteration, float totalIterations, params consts) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	float value = 0;
	int count = 0;

	//Check image boundaries
	if ((i >= consts.Px) || (j >= consts.Py)) return;

	for (int slice = 0; slice < consts.slices; slice++) {
		float dz = (consts.startDis + slice * consts.pitchZ) / consts.d_Beamz[view];
		float x = xMM2R((xP2MM(i, consts.Px, consts.PitchPx) - consts.d_Beamx[view] * dz), consts.Rx, consts.PitchRx);
		float y = yMM2R((yP2MM(j, consts.Py, consts.PitchPy) - consts.d_Beamy[view] * dz), consts.Ry, consts.PitchRy);

		//Update the value based on the error scaled and save the scale
		if (y >= 0 && y < consts.Ry && x >= 0 && x < consts.Rx) {
			//value += tex2D(textSino, x, y + slice*consts.Ry);
			float returnVal;
			//surf3Dread(&returnVal, surfRecon, x * sizeof(float), y, slice);
			returnVal = tex3D(textRecon, x + 0.5f, y + 0.5f, slice);
			if (returnVal >= 0.1f) count++;
			value += returnVal;
		}
	}
	
	float projVal = proj[j*consts.ProjPitchNum + i];
#ifdef RECONDERIVATIVE
	if (projVal > 0.0f && abs(projVal) <= USHRT_MAX && count > 0) {
		error[j*consts.ProjPitchNum + i] = projVal - (value * (float)consts.Views / (float)count);
	}
#else
	if (projVal > 0.0f && count > 0) {
#ifdef USELOGITER
		float correctedMax = logf(USHRT_MAX);
		projVal = (correctedMax - logf(projVal + 1)) / correctedMax * USHRT_MAX;
#else
#ifdef INVERSEITER
		projVal = USHRT_MAX - projVal;
#endif
#endif
		error[j*consts.ProjPitchNum + i] = (projVal - (value * (float)consts.Views / (float)count));
	}
#endif // RECONDERIVATIVE
	else error[j*consts.ProjPitchNum + i] = 0.0f;
}

__global__ void copySlice(float * image, int slice, params consts, cudaSurfaceObject_t surfRecon, bool invertLogCorrect = false) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i >= consts.Rx) || (j >= consts.Ry)) return;

	float returnVal;
	surf3Dread(&returnVal, surfRecon, i * sizeof(float), j, slice);
	if (invertLogCorrect) {
		if (returnVal > 10) {
			float correctedMax = logf(USHRT_MAX);
			returnVal = (correctedMax - logf(USHRT_MAX - returnVal + 1)) / correctedMax * USHRT_MAX;
		}
	}
	image[j*consts.ReconPitchNum + i] = returnVal;
}

__global__ void invertRecon(int slice, params consts, cudaSurfaceObject_t surfRecon) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i >= consts.Rx) || (j >= consts.Ry)) return;

	float test;
	surf3Dread(&test, surfRecon, i * sizeof(float), j, slice);
	surf3Dwrite(USHRT_MAX - test, surfRecon, i * sizeof(float), j, slice);
}

__global__ void scaleRecon(int slice, unsigned int minVal, unsigned int maxVal, params consts, cudaSurfaceObject_t surfRecon) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i >= consts.Rx) || (j >= consts.Ry)) return;

	float test;
	surf3Dread(&test, surfRecon, i * sizeof(float), j, slice);
	if (test == 0.0f) return;
	test -= minVal;
	float maxV = maxVal - minVal;
	if (test > maxV) test = maxV;
	if (test > 1.0f) {
		surf3Dwrite(test / maxV * (float)USHRT_MAX, surfRecon, i * sizeof(float), j, slice);
	}
	else surf3Dwrite(1.0f, surfRecon, i * sizeof(float), j, slice);
}

__global__ void initArray(int slice, float value, params consts, cudaSurfaceObject_t surfRecon) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Check image boundaries
	if ((i >= consts.Rx) || (j >= consts.Ry)) return;

	surf3Dwrite(value, surfRecon, i * sizeof(float), j, slice);
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

	//if (consts.orientation) i = consts.Px - 1 - i;
	//if (consts.flip) j = consts.Py - 1 - j;

	float data = abs(d_Data[MUL_ADD(j, consts.ReconPitchNum, i)]);//whatever it currently is, cast it to ushort
	//if (data <= 0.0f) return;
	if (consts.log) {
		if (data > 0) {
			float correctedMax = logf(USHRT_MAX);
			data = (correctedMax - logf(data + 1)) / correctedMax * USHRT_MAX;
		}
	}
	if (data > USHRT_MAX) data = USHRT_MAX;
	if (data <= 0.0f) return;// data = 0.0f;
	atomicAdd(d_Histogram + ((unsigned short)data >> 8), 1);//bin by the upper 256 bits
}

__global__ void histogramReconKernel(unsigned int *d_Histogram, int slice, bool useLog, params consts, cudaSurfaceObject_t surfRecon) {
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	int minX = min(consts.baseXr, consts.currXr);
	int maxX = max(consts.baseXr, consts.currXr);
	int minY = min(consts.baseYr, consts.currYr);
	int maxY = max(consts.baseYr, consts.currYr);

	if (i < minX || i >= maxX || j < minY || j >= maxY) return;

	if (consts.orientation) i = consts.Rx - 1 - i;
	if (consts.flip) j = consts.Ry - 1 - j;

	float data = 0;
	surf3Dread(&data, surfRecon, i * sizeof(float), j, slice);
	if (consts.log && useLog) {
		if (data > 0) {
			float correctedMax = logf(USHRT_MAX);
			data = (correctedMax - logf(data + 1)) / correctedMax * USHRT_MAX;
		}
	}
	if (consts.isReconstructing) {
		if (data > 0) {
			float correctedMax = logf(USHRT_MAX);
			data = (correctedMax - logf(USHRT_MAX - data + 1)) / correctedMax * USHRT_MAX;
		}
	}
	if (data > USHRT_MAX) data = USHRT_MAX;
	if (data <= 0.0f) return;// data = 0.0f;
	atomicAdd(d_Histogram + ((unsigned short)data >> 8), 1);//bin by the upper 8 bits
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
		float val = g[idx];
		if(val > 0)
			f[idx] = (1 - tf) *f[idx] + tf * (val + invlambda*DIVZ);
		else f[idx] = 0;
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
		if (px < (nx - 1)) {
			float val = f[idx + 1];
			if(val > 0) a = val - fc;
		}
		if (py < (ny - 1)) {
			float val = f[idx + nx];
			if(val > 0) b = val - fc;
		}

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
	float redFac = 1.0f;
	Sys.Recon.Pitch_x = Sys.Proj.Pitch_x * redFac;
	Sys.Recon.Pitch_y = Sys.Proj.Pitch_y * redFac;
	Sys.Recon.Nx = Sys.Proj.Nx / redFac;
	Sys.Recon.Ny = Sys.Proj.Ny / redFac;

	//Normalize Geometries
	Sys.Geo.IsoX = Sys.Geo.EmitX[NUMVIEWS / 2];
	Sys.Geo.IsoY = Sys.Geo.EmitY[NUMVIEWS / 2];
	Sys.Geo.IsoZ = Sys.Geo.EmitZ[NUMVIEWS / 2];
	for (int i = 0; i < NUMVIEWS; i++) {
		Sys.Geo.EmitX[i] -= Sys.Geo.IsoX;
		Sys.Geo.EmitY[i] -= Sys.Geo.IsoY;
	}

	constants.pitchZ = Sys.Geo.ZPitch;

	//cudaDeviceSynchronize();

#ifdef PRINTMEMORYUSAGE
	size_t avail_mem;
	size_t total_mem;
	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";
#endif // PRINTMEMORYUSAGE

	//Get Device Number
	cudaError_t cudaStatus;
	int deviceCount;
	int failedAttempts = 0;
	cuda(GetDeviceCount(&deviceCount));
	for (int i = 0; i < deviceCount; i++) {
		if (cudaSetDevice(i) == cudaSuccess) break;
		failedAttempts++;
	}
	if (failedAttempts == deviceCount) return Tomo_CUDA_err;

	cuda(StreamCreateWithFlags(&stream, cudaStreamDefault));

	//Thread and block sizes for standard kernel calls (2d optimized)
	contThreads.x = WARPSIZE;
	contThreads.y = MAXTHREADS / WARPSIZE;
	contBlocks.x = (Sys.Proj.Nx + contThreads.x - 1) / contThreads.x;
	contBlocks.y = (Sys.Proj.Ny + contThreads.y - 1) / contThreads.y;

	//Thread and block sizes for reductions (1d optimized)
	reductionThreads.x = MAXTHREADS;
	reductionBlocks.x = (Sys.Proj.Nx + reductionThreads.x - 1) / reductionThreads.x;
	reductionBlocks.y = Sys.Proj.Ny;

	//Set up display and buffer regions
	cuda(MallocPitch((void**)&d_Image, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny));
	cuda(MallocPitch((void**)&d_Sino, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));
	cuda(MallocPitch((void**)&inXBuff, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));
	cuda(MallocPitch((void**)&inYBuff, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));
#ifdef USEITERATIVE
	cuda(MallocPitch((void**)&d_Error, &reconPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));
#else
	cuda(MallocPitch((void**)&d_Error, &reconPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny));
#endif

	reconPitchNum = (int)reconPitch / sizeof(float);
	constants.ReconPitchNum = reconPitchNum;

	//Define the size of each of the memory spaces on the gpu in number of bytes
	sizeProj = Sys.Proj.Nx * Sys.Proj.Ny * sizeof(unsigned short);
	sizeSino = projPitch * Sys.Proj.Ny * Sys.Proj.NumViews;
	sizeIM = projPitch * Sys.Proj.Ny;
	sizeError = projPitch * Sys.Proj.Ny;
	
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

	textRecon.filterMode = cudaFilterModeLinear;
	textRecon.addressMode[0] = cudaAddressModeClamp;
	textRecon.addressMode[1] = cudaAddressModeClamp;

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
	constants.ProjPitchNum = pitch;

	//Setup derivative buffers
	cuda(Malloc(&buff1, sizeIM * sizeof(float)));
	cuda(Malloc(&buff2, sizeIM * sizeof(float)));

#ifdef ENABLEZDER
	//Z buffer
	cuda(MallocPitch((void**)&inZBuff, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));
	cuda(MallocPitch((void**)&maxZVal, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));
	cuda(MallocPitch((void**)&maxZPos, &projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny * Sys.Proj.NumViews));
	float * tempZBuffs[KERNELSIZE];
	cuda(Malloc(&zBuffs, KERNELSIZE * sizeof(float*)));
	for (int i = 0; i < KERNELSIZE; i++) {
		cuda(Malloc(&tempZBuffs[i], sizeIM * sizeof(float)));
	}
	cuda(MemcpyAsync(zBuffs, tempZBuffs, KERNELSIZE * sizeof(float*), cudaMemcpyHostToDevice));
#endif // ENABLEZDER

	//Set up all kernels
	cuda(Malloc(&d_gauss, KERNELSIZE * sizeof(float)));
	cuda(Malloc(&d_gaussDer, KERNELSIZE * sizeof(float)));
	cuda(Malloc(&d_gaussDer2, KERNELSIZE * sizeof(float)));
	//cuda(Malloc(&d_gaussDer3, KERNELSIZE * sizeof(float)));

	float tempKernel[KERNELSIZE];
	setGauss(tempKernel);
	cuda(MemcpyAsync(d_gauss, tempKernel, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer[KERNELSIZE];
	setGaussDer(tempKernelDer);
	cuda(MemcpyAsync(d_gaussDer, tempKernelDer, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	float tempKernelDer2[KERNELSIZE];
	setGaussDer2(tempKernelDer2);
	cuda(MemcpyAsync(d_gaussDer2, tempKernelDer2, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));

	/*float tempKernelDer3[KERNELSIZE];
	setGaussDer3(tempKernelDer3);
	cuda(MemcpyAsync(d_gaussDer3, tempKernelDer3, KERNELSIZE * sizeof(float), cudaMemcpyHostToDevice));*/

#ifdef PRINTMEMORYUSAGE
	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";
#endif // PRINTMEMORYUSAGE

	return Tomo_OK;
}

TomoError TomoRecon::ReadProjections(unsigned short ** GainData, unsigned short ** RawData) {
	//Correct projections
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

	//setStep(1.0);

	//Read the rest of the blank images for given projection sample set 
	for (int view = 0; view < NumViews; view++) {
		cuda(MemcpyAsync(d_Proj, RawData[view], sizeProj, cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(d_Gain, GainData[view], sizeProj, cudaMemcpyHostToDevice));

		KERNELCALL2(LogCorrectProj, dimGridProj, dimBlockProj, d_Sino, view, d_Proj, d_Gain, constants);

		cuda(BindTexture2D(NULL, textSino, d_Sino + view*Sys.Proj.Ny*projPitch / sizeof(float), cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny, projPitch));
		KERNELCALL3(sumRowsOrCols, dim3(1, Sys.Proj.Nx), reductionThreads, reductionSize, d_SumValsVert, true, constants);
		cuda(MemcpyAsync(sumValsVert + view * Sys.Proj.Nx, d_SumValsVert, Sys.Proj.Nx * sizeof(float), cudaMemcpyDeviceToHost));

		cuda(BindTexture2D(NULL, textSino, d_Sino + view*Sys.Proj.Ny*projPitch / sizeof(float), cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny, projPitch));
		KERNELCALL3(sumRowsOrCols, dim3(1, Sys.Proj.Ny), reductionThreads, reductionSize, d_SumValsHor, false, constants);
		cuda(MemcpyAsync(sumValsHor + view * Sys.Proj.Ny, d_SumValsHor, Sys.Proj.Ny * sizeof(float), cudaMemcpyDeviceToHost));
	}

	for (int view = 0; view < NumViews; view++) {
		tomo_err_throw(scanLineDetect(view, d_SumValsVert, sumValsVert + view * Sys.Proj.Nx, vertOff + view * Sys.Proj.Nx, true, cConstants.scanVertEnable));
		tomo_err_throw(scanLineDetect(view, d_SumValsHor, sumValsHor + view * Sys.Proj.Ny, horOff + view * Sys.Proj.Ny, false, cConstants.scanHorEnable));
	}

	//Normalize projection image lighting
	float maxVal, minVal;
	unsigned int histogram[HIST_BIN_COUNT];
	tomo_err_throw(getHistogram(d_Sino + (NumViews / 2)*projPitch / sizeof(float)*Sys.Proj.Ny, projPitch*Sys.Proj.Ny, histogram));
	tomo_err_throw(autoLight(histogram, 1, &minVal, &maxVal));

	for (int view = 0; view < NumViews; view++) {
		if (view == (NumViews / 2)) continue;
		float thisMax, thisMin;
		unsigned int histogram[HIST_BIN_COUNT];
		tomo_err_throw(getHistogram(d_Sino + view*projPitch / sizeof(float)*Sys.Proj.Ny, projPitch*Sys.Proj.Ny, histogram));
		tomo_err_throw(autoLight(histogram, 1, &thisMin, &thisMax));
		if (thisMax > maxVal) maxVal = thisMax;
		if (thisMin < minVal) minVal = thisMin;
	}
	cuda(Memcpy(d_MinVal, &minVal, sizeof(float), cudaMemcpyHostToDevice));

	float *g, *z1, *z2;
	size_t size;
	int j;
	int nx = projPitch / sizeof(float);
	int ny = Sys.Proj.Ny;
	size = nx*ny * sizeof(float);
	float tz = 2, tf = .2, beta = 0.0001;
	/* allocate device memory */
	cuda(Malloc((void **)&g, size));
	cuda(Malloc((void **)&z1, size));
	cuda(Malloc((void **)&z2, size));

	/* setup a 2D thread grid, with 16x16 blocks */
	/* each block is will use nearby memory*/
	dim3 block_size(16, 16);
	dim3 n_blocks((nx + block_size.x - 1) / block_size.x,
		(ny + block_size.y - 1) / block_size.y);

	for (int view = 0; view < NumViews; view++) {
		float bestScale = 1.0f;
		unsigned int histogram2[HIST_BIN_COUNT];
		tomo_err_throw(getHistogram(d_Sino + view*projPitch / sizeof(float)*Sys.Proj.Ny, projPitch*Sys.Proj.Ny, histogram2));

		//manually check range of offset values
		float error = FLT_MAX;
		int bestOffset = -100;
		if (view != NumViews / 2) {
			for (float scale = 0.90f; scale < 1.20f; scale += 0.001f) {
				for (int offset = -20; offset < 10; offset++) {
					//find average error
					float avgError = 0.0f;
					int count = 0;
					for (int test = 0; test < 255; test++) {
						float index2 = test*scale + offset;
						if (index2 >= 0 && index2 < 255) {
							int lower = floor(index2);
							int upper = ceil(index2);
							float intopVal = ((float)upper - index2) * histogram2[lower] + (index2 - (float)lower) * histogram2[upper];
							//avgError += pow((float)histogram[test] - intopVal, 2.0f);
							avgError += abs((float)histogram[test] - intopVal);
							count++;
						}
					}
					//avgError /= (float)count;

					//MAX logic
					if (avgError < error) {
						error = avgError;
						bestOffset = offset;
						bestScale = scale;
					}
				}
			}

			for (int j = 0; j < Sys.Proj.Nx; j++) {
				vertOff[j + view * Sys.Proj.Nx] += bestOffset * 255;//offsets are subtracted
			}
		}

		cuda(MemcpyAsync(d_MaxVal, &maxVal, sizeof(float), cudaMemcpyHostToDevice));

		cuda(MemcpyAsync(d_SumValsVert, vertOff + view * Sys.Proj.Nx, Sys.Proj.Nx * sizeof(float), cudaMemcpyHostToDevice));
		cuda(MemcpyAsync(d_SumValsHor, horOff + view * Sys.Proj.Ny, Sys.Proj.Ny * sizeof(float), cudaMemcpyHostToDevice));

		KERNELCALL2(rescale, dimGridProj, dimBlockProj, d_Sino, view, d_MaxVal, d_MinVal, d_SumValsVert, d_SumValsHor, bestScale, constants);

		if (useTV) {
			//chaTV(d_Sino + projPitch / sizeof(float) * Sys.Proj.Ny * view, iter, projPitch / sizeof(float), Sys.Proj.Ny, lambda);
			float * input = d_Sino + projPitch / sizeof(float) * Sys.Proj.Ny * view;

			/* Copy input to device*/
			cuda(MemcpyAsync(g, input, size, cudaMemcpyDeviceToDevice));
			cuda(MemsetAsync(z1, 0, size));
			cuda(MemsetAsync(z2, 0, size));

			/* call the functions */
			for (j = 0; j < iter; j++) {
				tz = 0.2 + 0.08*j;
				tf = (0.5 - 5. / (15 + j)) / tz;

				// z= zold + tauz.* grad(u);	
				// and normalize z:  n=max(1,sqrt(z(:,:,1).*z(:,:,1) +z(:,:,2).*z(:,:,2) + beta) ); z/=n;
				KERNELCALL2(updhgZ_SoA, n_blocks, block_size, z1, z2, input, tz, 1 / lambda, nx, ny);

				// u=  (1 - tauu*lambda) * uold + tauu .* div(z) + tauu*lambda.*f;
				KERNELCALL2(updhgF_SoA, n_blocks, block_size, input, z1, z2, g, tf, lambda, nx, ny);
			}
		}

		//Get x and y derivatives and save to their own buffers
		cuda(Memcpy(d_Image, d_Sino + view * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		sourceData oldDisplay = constants.dataDisplay;
		constants.dataDisplay = projections;
		tomo_err_throw(imageKernel(d_gaussDer, d_gauss, inXBuff + view * projPitch / sizeof(float) * Sys.Proj.Ny, true));
		tomo_err_throw(imageKernel(d_gauss, d_gaussDer, inYBuff + view * projPitch / sizeof(float) * Sys.Proj.Ny, true));

#ifdef SQUAREMAGINX
		cuda(Memcpy(buff1, inXBuff + view * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		cuda(Memcpy(d_Image, inYBuff + view * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));

		KERNELCALL2(mag, contBlocks, contThreads, inXBuff + view * projPitch / sizeof(float) * Sys.Proj.Ny, buff1, d_Image, constants);
#endif //SQUAREMAGINX

		constants.dataDisplay = oldDisplay;

#ifdef ENABLEZDER
		cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
		for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
			KERNELCALL2(projectSliceZ, contBlocks, contThreads, zBuffs, i + KERNELRADIUS, view, i*Sys.Geo.ZPitch, constants);
		}
		cuda(UnbindTexture(textError));

		KERNELCALL2(zConvolution, contBlocks, contThreads, inZBuff + view * projPitch / sizeof(float) * Sys.Proj.Ny, zBuffs, d_gaussDer, constants);
	}

	/*for (float dis = 0.0f; dis < MAXDIS; dis += Sys.Geo.ZPitch) {
		//Find the normalized z derivative at every step
		cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
		for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
			KERNELCALL2(projectSliceZ, contBlocks, contThreads, zBuffs, i + KERNELRADIUS, -1, dis + i*Sys.Geo.ZPitch, constants);
		}
		cuda(UnbindTexture(textError));

		KERNELCALL2(zConvolution, contBlocks, contThreads, buff1, zBuffs, d_gaussDer, constants);

		tomo_err_throw(project(inZBuff, buff2));

		KERNELCALL2(sub, contBlocks, contThreads, buff2, buff1, d_Image, reconPitchNum, constants);

		//Check if the value is at a maximum, and make sure it was a contributor
	}*/
#else
	}
#endif

	/*for (int beam = 0; beam < 7; beam++) {
		unsigned int histogram2[HIST_BIN_COUNT];
		tomo_err_throw(getHistogram(d_Sino + beam*projPitch / sizeof(float)*Sys.Proj.Ny, projPitch*Sys.Proj.Ny, histogram2));
		std::ofstream outputFile;
		char outFilename[250];
		sprintf(outFilename, "./histogramOut%d.txt", beam);
		outputFile.open(outFilename);
		for (int test = 1; test < 255; test++) outputFile << histogram2[test] << "\n";
		outputFile.close();
	}*/

	/* free device memory */
	cuda(Free(g));
	cuda(Free(z1));
	cuda(Free(z2));

	constants.log = oldLog;

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	delete[] sumValsHor;
	delete[] sumValsVert;
	delete[] vertOff;
	delete[] horOff;
	cuda(Free(d_Proj));
	cuda(Free(d_Gain));
	cuda(Free(d_SumValsHor));
	cuda(Free(d_SumValsVert));

#ifdef ENABLESOLVER
	int numSlices = 20;
	int sqrtNumSl = ceil(sqrt(numSlices));
	int matrixSize = Sys.Proj.Nx * Sys.Proj.Ny / pow(sqrtNumSl, 2) * numSlices * sizeof(float);

	cuda(Malloc(&d_Recon, matrixSize));
	cuda(Memset(d_Recon, 0, matrixSize));

#ifdef PRINTMEMORYUSAGE
	size_t avail_mem;
	size_t total_mem;
	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";
#endif // PRINTMEMORYUSAGE
#endif // ENABLESOLVER

	return Tomo_OK;
}

TomoError TomoRecon::exportRecon(unsigned short * exportData) {
	float * RawData = new float[reconPitch / sizeof(float)*Sys.Proj.Ny];
	int oldProjection = getActiveProjection();

	//Create the reconstruction volume around the current location
	float oldDistance = distance;
	distance -= constants.slices / 2 * Sys.Geo.ZPitch;
	for (int i = 0; i < constants.slices; i++) {
		setActiveProjection(i);
		singleFrame();
		distance += Sys.Geo.ZPitch;
		cuda(Memcpy(RawData, d_Image, reconPitch*Sys.Recon.Ny, cudaMemcpyDeviceToHost));
		for (int j = 0; j < Sys.Recon.Ny; j++) {
			int y = j;
			if (constants.flip) y = Sys.Recon.Ny - 1 - j;
			for (int k = 0; k < Sys.Recon.Nx; k++) {
				float data = RawData[reconPitch / sizeof(float) * j + k];
				if (data != 0.0) {
					//data *= 2.0f;
					if (constants.log) {
						if (data > USHRT_MAX) data = USHRT_MAX;
						data = (logf(USHRT_MAX) - logf(data)) / logf(USHRT_MAX) * USHRT_MAX;
					}

					if (data > USHRT_MAX) data = USHRT_MAX;
					if (data < 0.0f) data = 0.0f;
				}
				int x = k;
				if (constants.orientation) x = Sys.Recon.Nx - 1 - k;
				exportData[Sys.Recon.Nx * (y + Sys.Recon.Ny * i) + x] = (unsigned short)data;
			}
		}
	}

	setActiveProjection(oldProjection);
	distance = oldDistance;
	delete[] RawData;

	tomo_err_throw(singleFrame());

	return Tomo_OK;
}

TomoError TomoRecon::scanLineDetect(int view, float * d_sum, float * sum, float * offset, bool vert, bool enable) {
	int vectorSize;
	if (vert) vectorSize = Sys.Proj.Nx;
	else vectorSize = Sys.Proj.Ny;
	float * sumCorr = new float[vectorSize];

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

	return Tomo_OK;
}

//Fucntion to free the gpu memory after program finishes
TomoError TomoRecon::FreeGPUMemory(void){
	//Free memory allocated on the GPU
	cuda(Free(d_Image));
	cuda(Free(d_Error));
	cuda(Free(d_Sino));
	cuda(Free(buff1));
	cuda(Free(buff2));
	cuda(Free(inXBuff));
	cuda(Free(inYBuff));

	cuda(Free(constants.d_Beamx));
	cuda(Free(constants.d_Beamy));
	cuda(Free(constants.d_Beamz));
	cuda(Free(d_MaxVal));
	cuda(Free(d_MinVal));

	cuda(Free(d_gauss));
	cuda(Free(d_gaussDer));
	cuda(Free(d_gaussDer2));
	//cuda(Free(d_gaussDer3));

#ifdef ENABLESOLVER
	cuda(Free(d_Recon));
#endif

#ifdef ENABLEZDER
	cuda(Free(inZBuff));
	cuda(Free(maxZVal));
	cuda(Free(maxZPos));
	cuda(Free(zBuffs));
	
#endif // ENABLEZDER

	//cuda(DeviceReset());

	if (iterativeInitialized) {
		cuda(DestroySurfaceObject(surfReconObj));
		cuda(Free(d_ReconOld));
		cuda(FreeArray(d_Recon2));
	}

	

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

TomoError TomoRecon::getHistogramRecon(unsigned int *histogram, bool useall = false, bool useLog = true) {
	unsigned int * d_Histogram;

	cuda(Malloc((void **)&d_Histogram, HIST_BIN_COUNT * sizeof(unsigned int)));
	cuda(Memset(d_Histogram, 0, HIST_BIN_COUNT * sizeof(unsigned int)));

	//cuda(BindSurfaceToArray(surfRecon, d_Recon2));
	if (useall) {
		for (int slice = 0; slice < Sys.Recon.Nz; slice++) {
			KERNELCALL2(histogramReconKernel, contBlocks, contThreads, d_Histogram, slice, useLog, constants, surfReconObj);
		}
	}
	else {
		KERNELCALL2(histogramReconKernel, contBlocks, contThreads, d_Histogram, sliceIndex, useLog, constants, surfReconObj);
	}

	cuda(Memcpy(histogram, d_Histogram, HIST_BIN_COUNT * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	cuda(Free(d_Histogram));

	return Tomo_OK;
}

TomoError TomoRecon::draw(int x, int y) {
	//interop update
	display(x, y);
	map(stream);

	if(constants.dataDisplay == projections){
		scale = max((float)Sys.Proj.Nx / (float)width, (float)Sys.Proj.Ny / (float)height) * pow(ZOOMFACTOR, -zoom);
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny, projPitch));
	}
	else {
		scale = max((float)Sys.Recon.Nx / (float)width, (float)Sys.Recon.Ny / (float)height) * pow(ZOOMFACTOR, -zoom);
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys.Recon.Nx, Sys.Recon.Ny, reconPitch));
	}
	checkOffsets(&xOff, &yOff);

	cuda(BindSurfaceToArray(displaySurface, ca));

	const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

	if (blocks > 0) {
		if (constants.dataDisplay == projections) {
			KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream,
				Sys.Proj.Nx, Sys.Proj.Ny, width, height, scale, xOff, yOff, derDisplay != no_der, constants);
		}
		else {
			KERNELCALL4(resizeKernelTex, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream,
				Sys.Recon.Nx, Sys.Recon.Ny, width, height, scale, xOff, yOff, derDisplay != no_der, constants);
		}
		if (constants.baseXr >= 0 && constants.currXr >= 0)
			KERNELCALL4(drawSelectionBox, blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream, max(I2D(constants.baseXr, true), I2D(constants.currXr, true)),
				max(I2D(constants.baseYr, false), I2D(constants.currYr, false)), min(I2D(constants.baseXr, true), I2D(constants.currXr, true)),
				min(I2D(constants.baseYr, false), I2D(constants.currYr, false)), width);
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

TomoError TomoRecon::singleFrame(bool outputFrame, float** output, unsigned int * histogram) {
	//Initial projection
	switch (constants.dataDisplay) {
	case reconstruction:
		if (derDisplay != square_mag) {//only case frequently used that doesn't need this, leads to 3/2x speedup in autofocus
			cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
			KERNELCALL2(projectSlice, contBlocks, contThreads, d_Image, distance, constants);
			cuda(UnbindTexture(textError));
		}
		break;
	case projections:
		cuda(Memcpy(d_Image, d_Sino + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		break;
	case iterRecon:
#ifdef SHOWERROR
		KERNELCALL2(copySlice, contBlocks, contThreads, d_Image, sliceIndex, constants, surfErrorObj, constants.isReconstructing);
#else
		//KERNELCALL2(copySlice, contBlocks, contThreads, d_Image, sliceIndex, constants, surfDeltaObj, constants.isReconstructing);
		KERNELCALL2(copySlice, contBlocks, contThreads, d_Image, sliceIndex, constants, surfReconObj, constants.isReconstructing);
#endif
		break;
	case error:
		cuda(Memcpy2DAsync(d_Image, projPitch, d_Error + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny, cudaMemcpyDeviceToDevice));
		break;
	}

	switch (derDisplay) {
	case no_der:
		break;
	case x_mag_enhance:
		switch (constants.dataDisplay) {
		case reconstruction:
			tomo_err_throw(project(inXBuff, buff1));
			break;
		case projections:
			cuda(Memcpy(buff1, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			break;
		case iterRecon:
			tomo_err_throw(imageKernel(d_gaussDer, d_gauss, buff1, false));
			break;
		case error:
			break;
		}
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, true, true, constants);
		break;
	case y_mag_enhance:
		switch (constants.dataDisplay) {
		case reconstruction:
			tomo_err_throw(project(inYBuff, buff1));
			break;
		case projections:
			cuda(Memcpy(buff1, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			break;
		case iterRecon:
			tomo_err_throw(imageKernel(d_gauss, d_gaussDer, buff1, false));
			break;
		case error:
			break;
		}
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, true, true, constants);
		break;
	case mag_enhance:
		switch (constants.dataDisplay) {
		case reconstruction:
			tomo_err_throw(project(inXBuff, buff1));
			tomo_err_throw(project(inYBuff, buff2));
			break;
		case projections:
			cuda(Memcpy(buff1, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			cuda(Memcpy(buff2, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			break;
		case iterRecon:
			tomo_err_throw(imageKernel(d_gaussDer, d_gauss, buff1, false));
			tomo_err_throw(imageKernel(d_gauss, d_gaussDer, buff2, false));
			break;
		case error:
			break;
		}
		KERNELCALL2(mag, contBlocks, contThreads, buff1, buff2, buff1, constants);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, true, false, constants);
		break;
	case x_enhance:
		switch (constants.dataDisplay) {
		case reconstruction:
			tomo_err_throw(project(inXBuff, buff1));
			break;
		case projections:
			cuda(Memcpy(buff1, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			break;
		case iterRecon:
			tomo_err_throw(imageKernel(d_gaussDer, d_gauss, buff1, false));
			break;
		case error:
			break;
		}
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, true, false, constants);
		break;
	case y_enhance:
		switch (constants.dataDisplay) {
		case reconstruction:
			tomo_err_throw(project(inYBuff, buff1));
			break;
		case projections:
			cuda(Memcpy(buff1, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			break;
		case iterRecon:
			tomo_err_throw(imageKernel(d_gauss, d_gaussDer, buff1, false));
			break;
		case error:
			break;
		}
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, true, false, constants);
		break;
	case both_enhance:
		switch (constants.dataDisplay) {
		case reconstruction:
			tomo_err_throw(project(inXBuff, buff1));
			tomo_err_throw(project(inYBuff, buff2));
			break;
		case projections:
			cuda(Memcpy(buff1, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			cuda(Memcpy(buff2, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			break;
		case iterRecon:
			tomo_err_throw(imageKernel(d_gaussDer, d_gauss, buff1, false));
			tomo_err_throw(imageKernel(d_gauss, d_gaussDer, buff2, false));
			break;
		case error:
			break;
		}
		KERNELCALL2(add, contBlocks, contThreads, buff1, buff2, buff1, false, false, constants);
		KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, true, false, constants);
		break;
	case der_x:
		if (constants.dataDisplay == projections) {
			cuda(Memcpy(d_Image, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		}
		else {
			tomo_err_throw(project(inXBuff, d_Image));
		}
		break;
	case der_y:
		if (constants.dataDisplay == projections) {
			cuda(Memcpy(d_Image, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		}
		else {
			tomo_err_throw(project(inYBuff, d_Image));
		}
		break;
	case square_mag:
		switch (constants.dataDisplay) {
			case reconstruction:
				tomo_err_throw(project(inXBuff, buff1));
				tomo_err_throw(project(inYBuff, d_Image));
				break;
			case projections:
				cuda(Memcpy(buff1, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
				cuda(Memcpy(d_Image, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
				break;
			case iterRecon:
				tomo_err_throw(imageKernel(d_gaussDer, d_gauss, buff1, false));
				tomo_err_throw(imageKernel(d_gauss, d_gaussDer, d_Image, false));
				break;
			case error:
				break;
		}

		KERNELCALL2(squareMag, contBlocks, contThreads, d_Image, buff1, d_Image, constants);
		break;
	case slice_diff:
	{
		float xOff = -Sys.Geo.EmitX[diffSlice] * distance / Sys.Geo.EmitZ[diffSlice] / Sys.Recon.Pitch_x;
		float yOff = -Sys.Geo.EmitY[diffSlice] * distance / Sys.Geo.EmitZ[diffSlice] / Sys.Recon.Pitch_y;
		KERNELCALL2(squareDiff, contBlocks, contThreads, d_Image, diffSlice, xOff, yOff, reconPitchNum, constants);
	}
		break;
	case der2_x:
		tomo_err_throw(imageKernel(d_gaussDer2, d_gauss, d_Image, constants.dataDisplay == projections));
		break;
	case der2_y:
		tomo_err_throw(imageKernel(d_gauss, d_gaussDer2, d_Image, constants.dataDisplay == projections));
		break;
	case der3_x:
		//imageKernel(d_gaussDer3, d_gauss, d_Image);
		break;
	case der3_y:
		//imageKernel(d_gauss, d_gaussDer3, d_Image);
		break;
	case mag_der:
		/*if (dataDisplay == projections) {
			cuda(Memcpy(buff1, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			cuda(Memcpy(buff2, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		}
		else {
			tomo_err_throw(project(inXBuff, buff1));
			tomo_err_throw(project(inYBuff, buff2));
		}

		KERNELCALL2(mag, contBlocks, contThreads, d_Image, buff2, buff1, reconPitchNum, reconPitchNum, constants);*/

		if (constants.dataDisplay == projections) {
			cuda(Memcpy(d_Image, inZBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		}
		else {
			tomo_err_throw(project(inZBuff, d_Image));
		}

		//tomo_err_throw(project(inXBuff, d_Image));

		/*cuda(BindTexture2D(NULL, textError, inXBuff, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
		for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
			KERNELCALL2(projectSliceZ, contBlocks, contThreads, zBuffs, i + KERNELRADIUS, distance + i*Sys.Geo.ZPitch, constants);
		cuda(UnbindTexture(textError));

		KERNELCALL2(zConvolution, contBlocks, contThreads, d_Image, zBuffs, d_gaussDer, constants);*/
		break;
	case z_der_mag:
	{
		/*if (dataDisplay == projections) {
			cuda(Memcpy(buff1, inXBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
			cuda(Memcpy(buff2, inYBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		}
		else {
			tomo_err_throw(project(inXBuff, buff1));
			tomo_err_throw(project(inYBuff, buff2));
		}

		KERNELCALL2(mag, contBlocks, contThreads, buff1, buff2, buff1, reconPitchNum, reconPitchNum, constants);*/

		if (constants.dataDisplay == projections) {
			cuda(Memcpy(d_Image, inZBuff + sliceIndex * projPitch / sizeof(float) * Sys.Proj.Ny, sizeIM, cudaMemcpyDeviceToDevice));
		}
		else {
			cuda(BindTexture2D(NULL, textError, d_Sino, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
			for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
				KERNELCALL2(projectSliceZ, contBlocks, contThreads, zBuffs, i + KERNELRADIUS, -1, distance + i*Sys.Geo.ZPitch, constants);
			}
			cuda(UnbindTexture(textError));

			KERNELCALL2(zConvolution, contBlocks, contThreads, buff1, zBuffs, d_gaussDer, constants);

			tomo_err_throw(project(inZBuff, buff2));

			KERNELCALL2(sub, contBlocks, contThreads, buff2, buff1, d_Image, reconPitchNum, constants);
		}

		//KERNELCALL2(thresh, contBlocks, contThreads, buff2, buff1, d_Image, reconPitchNum, constants);

		/*KERNELCALL2(zConvolution, contBlocks, contThreads, buff1, zBuffs, d_gaussDer2, constants);
		KERNELCALL2(zConvolution, contBlocks, contThreads, buff2, zBuffs, d_gaussDer, constants);
		KERNELCALL2(div, contBlocks, contThreads, buff1, buff2, d_Image, reconPitchNum, constants);*/

		//imageKernel(d_gauss, d_gauss, d_Image);
		//KERNELCALL2(add, contBlocks, contThreads, d_Image, buff1, d_Image, reconPitchNum, true, false, constants);
	}
		break;
	}

	if (outputFrame) {
		*output = new float[Sys.Proj.Nx * Sys.Proj.Ny];

		constants.baseXr = 7 * Sys.Proj.Nx / 8;
		constants.baseYr = 7 * Sys.Proj.Ny / 8;
		constants.currXr = Sys.Proj.Nx / 8;
		constants.currYr = Sys.Proj.Ny / 8;

		cuda(Memcpy2D(*output, Sys.Proj.Nx * sizeof(float), d_Image, projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny, cudaMemcpyDeviceToHost));
		getHistogram(d_Image, reconPitch*Sys.Recon.Ny, histogram);

		constants.baseXr = -1;
		constants.baseYr = -1;
		constants.currXr = -1;
		constants.currYr = -1;
	}

	return Tomo_OK;
}

TomoError TomoRecon::autoFocus(bool firstRun) {
	static float step;
	static float best;
	static bool linearRegion;
	static bool firstLin = true;
	static float bestDist;
	static int bestSlice;
	static derivative_t oldDisplay;

	if (firstRun) {
		step = constants.pitchZ;
		distance = constants.startDis;
		bestDist = constants.startDis;
		sliceIndex = 0;
		bestSlice = sliceIndex;
		best = 0;
		linearRegion = false;
		oldDisplay = derDisplay;
		derDisplay = square_mag;
	}

	float newVal = focusHelper();

	if (!linearRegion) {
		if (newVal > best) {
			best = newVal;
			bestDist = distance;
			bestSlice = sliceIndex;
		}

		distance += step;
		sliceIndex++;

		if (distance > constants.startDis + constants.pitchZ*constants.slices || sliceIndex >= constants.slices) {
			linearRegion = true;
			firstLin = true;
			distance = bestDist;
			sliceIndex = bestSlice;
			if (constants.dataDisplay == iterRecon) {
				derDisplay = oldDisplay;
				singleFrame();
				return Tomo_Done;
			}
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
		derDisplay = square_mag;
	}

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

TomoError TomoRecon::autoLight(unsigned int histogram[HIST_BIN_COUNT], int threshold, float * minVal, float * maxVal) {
	int innerThresh = threshold;
	bool emptyHist = false;
	if (histogram == NULL) {
		emptyHist = true;
		histogram = new unsigned int[HIST_BIN_COUNT];
			tomo_err_throw(getHistogram(d_Image, reconPitch*Sys.Recon.Ny, histogram));
		innerThresh = abs(constants.baseXr - constants.currXr) * abs(constants.baseYr - constants.currYr) / AUTOTHRESHOLD;
		minVal = &constants.minVal;
		maxVal = &constants.maxVal;
	}

	/*std::ofstream outputFile;
	char outFilename[250];
	sprintf(outFilename, "./histogramOutRecon.txt");
	outputFile.open(outFilename);
	for (int test = 1; test < 255; test++) outputFile << histogram[test] << "\n";
	outputFile.close();*/

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
	//start with a control, but make sure
	toleranceData control;
	control.name += " none";
	control.numViewsChanged = 0;
	control.viewsChanged = 0;
	control.offset = 0;
	data.push_back(control);

	//start set as just the combinations
	for (int i = 0; i < NUMVIEWS; i++) {
		int resultsLen = (int)data.size();//size will change every iteration, pre-record it
		int binRep = 1 << i;
		for (int j = 1; j < resultsLen; j++) {
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
	for (int i = 1; i < combinations; i++) {
		toleranceData baseline = data[i];

		baseline.thisDir = dir_y;
		data.push_back(baseline);

		baseline.thisDir = dir_z;
		data.push_back(baseline);
	}

	//then fill in the set with all the view changes
	combinations = (int)data.size();//again, changing sizes later on
	for (int i = 1; i < combinations; i++) {
		toleranceData baseline = data[i];
		for (int j = 0; j < offsets.size() - 1; j++) {//skip the last
			toleranceData newData = baseline;
			newData.offset = offsets[j];
			data.push_back(newData);
		}

		//the last one is done in place
		data[i].offset = offsets[offsets.size() - 1];
	}

	return Tomo_OK;
}

TomoError TomoRecon::testTolerances(std::vector<toleranceData> &data, bool firstRun) {
	static auto iter = data.begin();
	if (firstRun) {
		if(vertical) derDisplay = der2_x;
		else derDisplay = der2_y;
		tomo_err_throw(singleFrame());
		tomo_err_throw(autoLight());
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
	tomo_err_throw(readPhantom(&readVal));
	iter->phantomData = readVal;
	++iter;

	return Tomo_OK;
}

TomoError TomoRecon::initIterative() {
#ifdef PRINTMEMORYUSAGE
	size_t avail_mem;
	size_t total_mem;
	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";
#endif // PRINTMEMORYUSAGE

	iterativeInitialized = true;
	constants.isReconstructing = true;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaExtent vol = { Sys.Recon.Nx, Sys.Recon.Ny, Sys.Recon.Nz };
	cuda(Malloc3DArray(&d_Recon2, &channelDesc, vol, cudaArraySurfaceLoadStore));
	cuda(Malloc3DArray(&d_ReconDelta, &channelDesc, vol, cudaArraySurfaceLoadStore));
#ifdef SHOWERROR
	cuda(Malloc3DArray(&d_ReconError, &channelDesc, vol, cudaArraySurfaceLoadStore));
#endif
	cuda(MallocPitch((void**)&d_ReconOld, &reconPitch, Sys.Recon.Nx * sizeof(float), Sys.Recon.Ny));

	reconPitchNum = (int)reconPitch / sizeof(float);
	constants.ReconPitchNum = reconPitchNum;

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = d_Recon2;
	cuda(CreateSurfaceObject(&surfReconObj, &resDesc));
	resDesc.res.array.array = d_ReconDelta;
	cuda(CreateSurfaceObject(&surfDeltaObj, &resDesc));
#ifdef SHOWERROR
	resDesc.res.array.array = d_ReconError;
	cuda(CreateSurfaceObject(&surfErrorObj, &resDesc));
#endif

#ifdef RECONDERIVATIVE
	cuda(Memcpy2DAsync(d_Error, projPitch, inXBuff, projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny*NUMVIEWS, cudaMemcpyDeviceToDevice));
#else

	for (int view = 0; view < NumViews; view++) {
		cuda(Memcpy2DAsync(d_Error + view * projPitch / sizeof(float) * Sys.Proj.Ny, projPitch, d_Sino + view * projPitch / sizeof(float) * Sys.Proj.Ny, projPitch, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny, cudaMemcpyDeviceToDevice));
		//cuda(Memset2DAsync(d_Error + view * projPitch / sizeof(float) * Sys.Proj.Ny, projPitch, 0, Sys.Proj.Nx * sizeof(float), Sys.Proj.Ny));
#ifdef INVERSEITER
		KERNELCALL2(invert, contBlocks, contThreads, d_Error + view * projPitch / sizeof(float) * Sys.Proj.Ny, constants);
#endif // INVERSEITER
	}
#endif // RECONDERIVATIVE
	
	cuda(BindTexture2D(NULL, textError, d_Error, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
	for (int slice = 0; slice < Sys.Recon.Nz; slice++) {
		KERNELCALL2(initArray, contBlocks, contThreads, slice, 0.0f, constants, surfReconObj);
		KERNELCALL2(initArray, contBlocks, contThreads, slice, DELTA0, constants, surfDeltaObj);
		KERNELCALL2(copySlice, contBlocks, contThreads, d_ReconOld, slice, constants, surfReconObj);
#ifdef SHOWERROR
		KERNELCALL2(projectIter, contBlocks, contThreads, d_ReconOld, slice, 1.0f, true, constants, surfReconObj, surfErrorObj);
#else 
		KERNELCALL2(projectIter, contBlocks, contThreads, d_Sino, d_ReconOld, slice, 1.0f, true, constants, surfReconObj, surfDeltaObj, true);
#endif
	}
	cuda(UnbindTexture(textError));

#ifdef PRINTMEMORYUSAGE
	cudaMemGetInfo(&avail_mem, &total_mem);
	std::cout << "Available memory: " << avail_mem << "/" << total_mem << "\n";
#endif // PRINTMEMORYUSAGE

	return Tomo_OK;
}

TomoError TomoRecon::resetIterative() {
	cuda(DestroySurfaceObject(surfReconObj));
	cuda(Free(d_ReconOld));
	cuda(FreeArray(d_Recon2));

	return initIterative();
}

TomoError TomoRecon::iterStep() {
	static float iteration = 1.0f;

	cuda(BindTextureToArray(textRecon, d_Recon2));
#ifdef RECONDERIVATIVE
	for (int view = 0; view < NumViews; view++) {
		KERNELCALL2(backProject, contBlocks, contThreads, inXBuff + view * projPitch / sizeof(float) * Sys.Proj.Ny, d_Error + view * projPitch / sizeof(float) * Sys.Proj.Ny, view, constants);
	}
#else
	for (int view = 0; view < NumViews; view++) {
		KERNELCALL2(backProject, contBlocks, contThreads, d_Sino + view * projPitch / sizeof(float) * Sys.Proj.Ny, d_Error + view * projPitch / sizeof(float) * Sys.Proj.Ny, view, iteration, ITERATIONS, constants);
	}
#endif // RECONDERIVATIVE
	cuda(UnbindTexture(textRecon));

	cuda(BindTexture2D(NULL, textSino, d_Sino, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
	cuda(BindTexture2D(NULL, textError, d_Error, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch))
	for (int slice = 0; slice < Sys.Recon.Nz; slice++) {
		KERNELCALL2(copySlice, contBlocks, contThreads, d_ReconOld, slice, constants, surfReconObj);
#ifdef SHOWERROR
		KERNELCALL2(projectIter, contBlocks, contThreads, d_ReconOld, slice, iteration, SKIPITERTV, constants, surfReconObj, surfErrorObj);
#else 
		KERNELCALL2(projectIter, contBlocks, contThreads, d_Sino, d_ReconOld, slice, iteration, SKIPITERTV, constants, surfReconObj, surfDeltaObj);
#endif
		/*for (int i = 0; i < TVITERATIONS; i++) {
			KERNELCALL2(copySlice, contBlocks, contThreads, d_ReconOld, slice, constants);
			KERNELCALL2(projectIter, contBlocks, contThreads, d_ReconOld, slice, iteration++, true, constants);
		}*/
	}
	cuda(UnbindTexture(textError));

	iteration++;
	
	return Tomo_OK;
}

TomoError TomoRecon::finalizeIter() {
	//no longer need gradient records
	cuda(FreeArray(d_ReconDelta)); 
	cuda(DestroySurfaceObject(surfDeltaObj));

	constants.isReconstructing = false;

#ifdef INVERSEITER
	for (int slice = 0; slice < Sys.Recon.Nz; slice++)
		KERNELCALL2(invertRecon, contBlocks, contThreads, slice, constants, surfReconObj);
#endif //INVERSEITER

	constants.baseXr = 0;
	constants.baseYr = 0;
	constants.currXr = Sys.Recon.Nx;
	constants.currYr = Sys.Recon.Ny;

	float maxVal, minVal;
	unsigned int histogram[HIST_BIN_COUNT];
	tomo_err_throw(getHistogramRecon(histogram, true, false));
	tomo_err_throw(autoLight(histogram, 20, &minVal, &maxVal));

	/*tomo_err_throw(getHistogramRecon(histogram, true, false));
	std::ofstream outputFile;
	char outFilename[250];
	sprintf(outFilename, "./histogramOutRecon.txt");
	outputFile.open(outFilename);
	for (int test = 1; test < 255; test++) outputFile << histogram[test] / Sys.Recon.Nz << "\n";
	outputFile.close();*/

	//cuda(BindSurfaceToArray(surfRecon, d_Recon2));
	for (int slice = 0; slice < Sys.Recon.Nz; slice++)
		KERNELCALL2(scaleRecon, contBlocks, contThreads, slice, minVal, maxVal, constants, surfReconObj);

	

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

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

inline TomoError TomoRecon::imageKernel(float xK[KERNELSIZE], float yK[KERNELSIZE], float * output, bool projs) {
	if (projs) {
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny, projPitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, xK, constants);
		cuda(UnbindTexture(textImage));

		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny, projPitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, output, yK, constants);
		cuda(UnbindTexture(textImage));
	}
	else {
		cuda(BindTexture2D(NULL, textImage, d_Image, cudaCreateChannelDesc<float>(), Sys.Recon.Nx, Sys.Recon.Ny, reconPitch));
		KERNELCALL2(convolutionRowsKernel, contBlocks, contThreads, d_Error, xK, constants);
		cuda(UnbindTexture(textImage));

		cuda(BindTexture2D(NULL, textImage, d_Error, cudaCreateChannelDesc<float>(), Sys.Recon.Nx, Sys.Recon.Ny, reconPitch));
		KERNELCALL2(convolutionColumnsKernel, contBlocks, contThreads, output, yK, constants);
		cuda(UnbindTexture(textImage));
	}

	return Tomo_OK;
}

inline TomoError TomoRecon::project(float * projections, float * reconstruction) {
	cuda(BindTexture2D(NULL, textError, projections, cudaCreateChannelDesc<float>(), Sys.Proj.Nx, Sys.Proj.Ny*Sys.Proj.NumViews, projPitch));
	KERNELCALL2(projectSlice, contBlocks, contThreads, reconstruction, distance, constants);
	cuda(UnbindTexture(textError));
}

TomoError TomoRecon::resetLight() {
	if (constants.dataDisplay == projections) {
		constants.baseXr = 7 * Sys.Proj.Nx / 8;
		constants.baseYr = 7 * Sys.Proj.Ny / 8;
		constants.currXr = Sys.Proj.Nx / 8;
		constants.currYr = Sys.Proj.Ny / 8;
	}
	else {
		constants.baseXr = 7 * Sys.Recon.Nx / 8;
		constants.baseYr = 7 * Sys.Recon.Ny / 8;
		constants.currXr = Sys.Recon.Nx / 8;
		constants.currYr = Sys.Recon.Ny / 8;
	}

	tomo_err_throw(autoLight());

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	return Tomo_OK;
}

TomoError TomoRecon::resetFocus() {
	if (constants.dataDisplay == projections) {
		constants.baseXr = 3 * Sys.Proj.Nx / 4;
		constants.baseYr = 3 * Sys.Proj.Ny / 4;
		constants.currXr = Sys.Proj.Nx / 4;
		constants.currYr = Sys.Proj.Ny / 4;
	}
	else {
		constants.baseXr = 3 * Sys.Recon.Nx / 4;
		constants.baseYr = 3 * Sys.Recon.Ny / 4;
		constants.currXr = Sys.Recon.Nx / 4;
		constants.currYr = Sys.Recon.Ny / 4;
	}

	tomo_err_throw(autoFocus(true));
	while (autoFocus(false) == Tomo_OK);

	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;

	return Tomo_OK;
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
		int sysWidth;
		if (constants.dataDisplay == projections) sysWidth = Sys.Proj.Nx;
		else sysWidth = Sys.Recon.Nx;
		float innerOffx = (width - sysWidth / scale) / 2.0f;

		return constants.orientation ? (int)((sysWidth - 1 - i - xOff) / scale + innerOffx) : (int)((i - xOff) / scale + innerOffx);
	}
	//else
	int sysHeight;
	if (constants.dataDisplay == projections) sysHeight = Sys.Proj.Ny;
	else sysHeight = Sys.Recon.Ny;
	float innerOffy = (height - sysHeight / scale) / 2.0f;

	return constants.flip ? (int)((sysHeight - 1 - i - yOff) / scale + innerOffy) : (int)((i - yOff) / scale + innerOffy);
}

//On-screen coordinates to image space
int TomoRecon::D2I(int d, bool xDir) {
	if (xDir) {
		int sysWidth;
		if (constants.dataDisplay == projections) sysWidth = Sys.Proj.Nx;
		else sysWidth = Sys.Recon.Nx;
		float innerOffx = (width - sysWidth / scale) / 2.0f;

		return (int)((d - innerOffx) * scale + xOff);
	}
	//else
	int sysHeight;
	if (constants.dataDisplay == projections) sysHeight = Sys.Proj.Ny;
	else sysHeight = Sys.Recon.Ny;
	float innerOffy = (height - sysHeight / scale) / 2.0f;

	return (int)((d - innerOffy) * scale + yOff);
}