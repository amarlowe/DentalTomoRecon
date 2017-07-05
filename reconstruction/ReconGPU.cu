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
#include "ReconGPUHeader.cuh"

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

//Macro for checking cuda errors following a cuda launch or api call
#define voidChkErr(x) {											\
	(x);														\
	cudaError_t e=cudaGetLastError();							\
	if(e!=cudaSuccess) {										\
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(e) << "\n";	\
		return Tomo_CUDA_err;									\
	}															\
}

/********************************************************************************************/
/* GPU Function specific functions															*/
/********************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to normalize the reconstruction
__global__ void ProjectionNorm(float * Norm, int view, float ex, float ey, float ez, params* constants){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//global memory local cache
	/*float PitchNz = constants->PitchNz;
	float Z_Offset = constants->Z_Offset;*/

	float PitchNz = d_PitchNz;
	float Z_Offset = d_Z_Offset;

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
		x1 += PitchNz * dx1 / ez * Z_Offset;
		x2 += PitchNz * dx2 / ez * Z_Offset;
		y1 += PitchNz * dy1 / ez * Z_Offset;
		y2 += PitchNz * dy2 / ez * Z_Offset;
			

		//Project by stepping through the image one slice at a time
		for (int z = 0; z < d_Nz; z++){
			//Get the next n and x
			x1 += PitchNz * dx1 / ez;
			x2 += PitchNz * dx2 / ez;
			y1 += PitchNz * dy1 / ez;
			y2 += PitchNz * dy2 / ez;

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

__global__ void AverageProjectionNorm(float * Norm, float * Norm2, params* constants)
{
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		float count = 0;

		//Get the value of the projection over all views
		for (int n = 0; n < d_Views; n++) count += Norm[(j + n*d_MPy)*d_MPx + i];

		//For each view calculate the precent contribution to the total
		for (int n = 0; n < d_Views; n++)
		{
			float val = Norm[(j + n*d_MPy)*d_MPx + i];

			float nVal = 0;

			if (count > 0) nVal = val / count;

			Norm2[(j + n*d_MPy)*d_MPx + i] = nVal;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
//Functions to do initial correction of raw data: log and scatter correction
__global__ void LogCorrectProj(float * Sino, int view, unsigned short *Proj, float MaxVal, params* constants)
{
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		//Log correct the sample size
		float sample = (float)Proj[j*d_Px + i];
		float val = logf(MaxVal) - logf(sample);
		if (sample > MaxVal) val = 0.0f;

		Sino[(j + view*d_MPy)*d_MPx + i] = val;
	}
}

__global__ void ApplyGaussianBlurX(float * Sino, float * BlurrX, int view, params* constants){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//global memory local cache
	/*int Px = constants->Px;
	int Py = constants->Py;
	int Nx = constants->Nx;
	int MPx = constants->MPx;
	int MPy = constants->MPy;*/

	int Px = d_Px;
	int Py = d_Py;
	int Nx = d_Nx;
	int MPx = d_MPx;
	int MPy = d_MPy;

	//Image is not a power of 2
	if ((i < Px) && (j < Py)){
		int N = 18;
		float sigma = -0.5f / (float)(6 * 6);
		float blur = 0;
		float norm = 0;
		//Use a neighborhood of 6 sigma
		for (int n = -N; n <= N; n++)
		{
			if (((n + i) >= 0) && (n + i < Nx))
			{
				float weight = __expf((float)(n*n)*sigma);
				norm += weight;
				blur += weight * Sino[(j + view*MPy)*MPx + i];
			}
		}
		if (norm == 0) norm = 1.0f;
		BlurrX[j*MPx + i] = blur / norm;
	}
}

__global__ void ApplyGaussianBlurY(float * BlurrX, float * BlurrY, params* constants){
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//global memory local cache
	/*int Px = constants->Px;
	int Py = constants->Py;
	int Ny = constants->Ny;
	int MPx = constants->MPx;
	int MPy = constants->MPy;*/

	int Px = d_Px;
	int Py = d_Py;
	int Ny = d_Ny;
	int MPx = d_MPx;
	int MPy = d_MPy;

	//Image is not a power of 2
	if ((i < Px) && (j < Py))
	{
		int N = 18;
		float sigma = -0.5f / (float)(6 * 6);
		float blur = 0;
		float norm = 0;
		//Use a neighborhood of 6 sigma
		for (int n = -N; n <= N; n++)
		{
			if (((n + j) >= 0) && (n + j < Ny))
			{
				float weight = __expf((float)(n*n)*sigma);
				norm += weight;
				blur += weight * BlurrX[j*MPx + i];
			}
		}
		if (norm == 0) norm = 1.0f;
		BlurrY[j*MPx + i] = blur / norm;
	}
}

__global__ void ScatterCorrect(float * Sino, unsigned short * Proj, float * BlurXY, int view, float MaxVal, params* constants)
{
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		//Log correct the sample size
		float sample = (float)Sino[(j + view*d_MPy)*d_MPx + i];
		float blur = (float)BlurXY[j*d_MPx + i];
		float val = sample +0.1f*__expf(blur);
		Sino[(j + view*d_MPy)*d_MPx + i] = val;
		Proj[j*d_Px + i] = (unsigned short)(val * 32768.0f / MaxVal);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//START part of the reconstuction code
__global__ void ProjectImage(float * Sino, float * Norm, float *Error, int view, float ex, float ey, float ez, params* constants){
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//shared memory local cache
	int sizex = d_Px;
	int sizey = d_Py;
	int Memx = d_MPx;
	int Memy = d_MPy;
	//int rMemx = d_MNx;//not used
	int rMemy = d_MNy;
	float Nz = (float)d_PitchNz;
	float zOff = (float)d_Z_Offset;

	//within image boudary
	if ((i < sizex) && (j < sizey)){
		//Check to make sure the ray passes through the image
		float NP = Norm[(j + view*Memy)*Memx + i];
		float err = 0;

		if (NP != 0){
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
			x1 += Nz * dx1 / ez * zOff;
			x2 += Nz * dx2 / ez * zOff;
			y1 += Nz * dy1 / ez * zOff;
			y2 += Nz * dy2 / ez * zOff;
			

			//Step through the image space by slice in z direction
			for (int z = 0; z < d_Nz; z++){
				//Get the next n and x
				x1 += Nz * dx1 / ez;
				x2 += Nz * dx2 / ez;
				y1 += Nz * dy1 / ez;
				y2 += Nz * dy2 / ez;

				//Get the first and last pixels in x and y the ray passes through
				int xx1 = (int)floorf(min(x1, x2));
				int xx2 = (int)ceilf(max(x1, x2));
				int yy1 = (int)floorf(min(y1, y2));
				int yy2 = (int)ceilf(max(y1, y2));

				//Get the length of the ray in the slice in x and y
				float dist = 1.0f / fabsf((x2 - x1)*(y2 - y1));

				//Set the first x value to the first pixel
				float xs = x1;

				//Cycle through pixels x and y and used to calculate projection
				for (int x = xx1; x < xx2; x++){
					float ys = y1;
					float xend = min((float)(x + 1), x2);

					for (int y = yy1; y < yy2; y++) {
						float yend = min((float)(y + 1), y2);

						//Calculate the weight as the overlap in x and y
						float weight = scale*((xend - xs)*(yend - ys)*dist);

						int nx = min(max(x, 0), d_Nx - 1);
						int ny = min(max(y, 0), d_Ny - 1);
						float test = tex2D(textImage, nx + 0.5f, ny + 0.5f + z*rMemy);
						Pro += test * weight;
						count += weight;

						ys = yend;
					}//y loop
					xs = xend;
				}//x loop
			}//z loop

			//If the ray passes through the image region get projection error
			err = (tex2D(textSino, (float)i + 0.5f, (float)j + 0.5f + view*Memy)
				- Pro*NP) / (max(count, 1.0f));
		}//Norm check

		//Add Calculated error to an error image to back project
		Error[j*Memx + i] = err;
	}//image boudary check
}

__global__ void BackProjectError(float * IM, float * IM2, float beta, int view, float ex, float ey, float ez, params* constants)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny))
	{
		//Define the direction in z to get r
		float r = (ez) / (((float)(k+ d_Z_Offset)*d_PitchNz) + ez);

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
		int xx1 = max((int)floorf(min(x1, x2)), 0);
		int xx2 = min((int)ceilf(max(x1, x2)), d_Px - 1);
		int yy1 = max((int)floorf(min(y1, y2)), 0);
		int yy2 = min((int)ceilf(max(y1, y2)), d_Py - 1);

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
		for (int x = xx1; x < xx2; x++)
		{
			float ys = y1;
			float xend = min((float)(x + 1), x2);
			for (int y = yy1; y < yy2; y++)
			{
				float yend = min((float)(y + 1), y2);

				//Calculate the weight as the overlap in x and y
				float weight = ((xend - xs))*((yend - ys))*dist;

				//Calculate the scaling of a ray from the center of the pixel
				//to the detector
				float cos_alpha = sqrtf(((float)x - xx)*((float)x - xx) + ez*ez);
				float cos_gamma = sqrtf(((float)y - yy)*((float)y - yy) + ez*ez);
				float scale = (cos_alpha*cos_gamma)*ezz * weight;

				//Update the value based on the error scaled and save the scake
				val += tex2D(textError, (float)x + 0.5f, (float)y + 0.5f) *scale;
				N += scale;
				ys = yend;
			}
			xs = xend;
		}
		//Get the current value of image
		float update = beta*val / N;

		if (N > 0) {
			float uval = IM[(j + k*d_MNy)*d_MNx + i];
			IM[(j + k*d_MNy)*d_MNx + i] = uval + update;
			IM2[(j + k*d_MNy)*d_MNx + i] = update;
		}
		else IM2[(j + k*d_MNy)*d_MNx + i] = -10.0f;
	}
	else IM2[(j + k*d_MNy)*d_MNx + i] = -10.0f;
}

__global__ void CorrectEdgesY(float * IM, float * IM2, params* constants)
{
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


__global__ void CorrectEdgesX(float * IM, float * IM2, params* constants)
{
	//Define pixel location in x, z
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int k = blockDim.y * blockIdx.y + threadIdx.y;
	if (j < d_Ny && k < d_Nz)
	{
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


__global__ void NormalizeImage(float * IM, params* constants)
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
__global__ void SobelEdgeDetection(float * IM, params* constants)
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


__global__ void SumSobelEdges(float * Image, int sizeIM, float * MaxVal, params* constants)
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
__global__ void DerivOfGradIm(float * Deriv, float ep, params* constants)
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

__global__ void GetGradIm(float * IM, params* constants)
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

__global__ void UpdateImageEstimate(float * IM, float * TV, float *TV2, params* constants)
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
__global__ void CreateSyntheticProjection(unsigned short * Proj, float * Win, float *Im, float MaxVal, params* constants)
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
		Proj[j*d_Px + i] = (unsigned short)(Pro * 32768.0f / (MaxV));
	}
}

__global__ void CreateSyntheticProjectionNew(unsigned short * Proj, float* Win, float * StIm, float *Im, float MaxVal, int cz, params* constants)
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

		Proj[j*d_Px + i] = (unsigned short)(update_val * 32768.0f);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to set the reconstructions to save and copy back to cpu
__global__ void GetMaxImageVal(float * Image, int sizeIM, float * MaxVal, params* constants)
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

__global__ void CopyImages(unsigned short * ImOut, float * ImIn, float maxVal, params* constants)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny))
	{
		float val = ImIn[(j + k*d_MNy)*d_MNx + i] - 0.015f;
		unsigned short val2 = (unsigned short)floorf(__saturatef(val / maxVal) * 32768.0f);
		ImOut[(j + k*d_Ny)*d_Nx + i] = val2;
	}
}


/********************************************************************************************/
/* Function to interface the CPU with the GPU:												*/
/********************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to Initialize the GPU and set up the reconstruction normalization

//Function to define the reconstruction structure
void DefineReconstructSpace(struct SystemControl * Sys){
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
TomoError SetUpGPUMemory(struct SystemControl * Sys){
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

	ChkErr(cudaDeviceReset());

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
	ChkErr(cudaMalloc((void**)&d_Image, sizeIM));
	ChkErr(cudaMalloc((void**)&d_Image2, sizeIM));
	ChkErr(cudaMalloc((void**)&d_GradIm, sizeIM));
	ChkErr(cudaMalloc((void**)&d_Proj, sizeProj));
	ChkErr(cudaMalloc((void**)&d_Error, sizeError));
	ChkErr(cudaMalloc((void**)&d_Sino, sizeSino));
	ChkErr(cudaMalloc((void**)&d_Norm, sizeSino));
	ChkErr(cudaMalloc((void**)&d_Pro, sizeSino));
	ChkErr(cudaMalloc((void**)&d_PriorIm, sizeIM));
	ChkErr(cudaMalloc((void**)&d_DerivGradIm, sizeIM));
	ChkErr(cudaMalloc((void**)&d_GradNorm, sizeSlice));
	ChkErr(cudaMalloc((void**)&d_dp, sizeSlice));
	ChkErr(cudaMalloc((void**)&d_dpp, sizeSlice));
	ChkErr(cudaMalloc((void**)&d_alpha, sizeSlice));

	//Set the values of the image and sinogram to all 0
	ChkErr(cudaMemset(d_Image, 0, sizeIM));
	ChkErr(cudaMemset(d_Image2, 0, sizeIM));
	ChkErr(cudaMemset(d_GradIm, 0, sizeIM));
	ChkErr(cudaMemset(d_Sino, 0, sizeSino));
	ChkErr(cudaMemset(d_Norm, 0, sizeSino));
	ChkErr(cudaMemset(d_Pro, 0, sizeProj));
	ChkErr(cudaMemset(d_PriorIm, 0, sizeIM));
	ChkErr(cudaMemset(d_DerivGradIm, 0, sizeIM));
	ChkErr(cudaMemset(d_GradNorm, 0, sizeSlice));
	ChkErr(cudaMemset(d_dp, 0, sizeSlice));
	ChkErr(cudaMemset(d_dpp, 0, sizeSlice));
	ChkErr(cudaMemset(d_alpha, 0, sizeSlice));

	//Define the textures
	textImage.filterMode = cudaFilterModePoint;
	textImage.addressMode[0] = cudaAddressModeClamp;
	textImage.addressMode[1] = cudaAddressModeClamp;

	textError.filterMode = cudaFilterModePoint;
	textError.addressMode[0] = cudaAddressModeClamp;
	textError.addressMode[1] = cudaAddressModeClamp;

	textSino.filterMode = cudaFilterModePoint;
	textSino.addressMode[0] = cudaAddressModeClamp;
	textSino.addressMode[1] = cudaAddressModeClamp;

	ChkErr(cudaMallocArray(&d_Sinogram, &textSino.channelDesc, MemP_Nx, MemP_Ny * Sys->Proj->NumViews));
	ChkErr(cudaBindTextureToArray(textSino, d_Sinogram));

	//Set the TV weighting value to start at a constant value less than 1
	float * AlphaSlice = new float[Sys->Recon->Nz];
	for (int slice = 0; slice < Sys->Recon->Nz; slice++) {
		AlphaSlice[slice] = 0.2f;
	}
	ChkErr(cudaMemcpy(d_alpha, AlphaSlice, sizeSlice, cudaMemcpyHostToDevice));
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

	//Copy the system geometry constants
	params constants;
	constants.Px = Sys->Proj->Nx;
	constants.Py = Sys->Proj->Ny;
	constants.Nx = Sys->Recon->Nx;
	constants.Ny = Sys->Recon->Ny;
	constants.Nz = Sys->Recon->Nz;
	constants.MPx = MemP_Nx;
	constants.MPy = MemP_Ny;
	constants.MNx = MemR_Nx;
	constants.MNy = MemR_Ny;

	ChkErr(cudaMemcpyToSymbol(d_Px, &Sys->Proj->Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Py, &Sys->Proj->Ny, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Nx, &Sys->Recon->Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Ny, &Sys->Recon->Ny, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MPx, &MemP_Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MPy, &MemP_Ny, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MNx, &MemR_Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MNy, &MemR_Ny, sizeof(int)));

	constants.HalfPx = HalfPx;
	constants.HalfPy = HalfPy;
	constants.HalfNx = HalfNx;
	constants.HalfNy = HalfNy;
	constants.PitchPx = Sys->Proj->Pitch_x;
	constants.PitchPy = Sys->Proj->Pitch_y;
	constants.PitchNx = Sys->Recon->Pitch_x;
	constants.PitchNy = Sys->Recon->Pitch_y;
	constants.PitchNz = Sys->Recon->Pitch_z;
	constants.alpharelax = alpha;
	constants.rmax = rmax;
	constants.Z_Offset = Sice_Offset;

	ChkErr(cudaMemcpyToSymbol(d_HalfPx2, &HalfPx, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_HalfPy2, &HalfPy, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_HalfNx2, &HalfNx, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_HalfNy2, &HalfNy, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Nz, &Sys->Recon->Nz, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPx, &Sys->Proj->Pitch_x, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPy, &Sys->Proj->Pitch_y, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPxInv, &PitchPxInv, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPyInv, &PitchPyInv, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNx, &Sys->Recon->Pitch_x, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNy, &Sys->Recon->Pitch_y, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNxInv, &PitchNxInv, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNyInv, &PitchNyInv, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNz, &Sys->Recon->Pitch_z, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Views, &Sys->Proj->NumViews, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_rmax, &rmax, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_alpharelax, &alpha, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Z_Offset, &Sice_Offset, sizeof(int)));

	ChkErr(cudaMalloc(&d_constants, sizeof(params)));
	ChkErr(cudaMemcpy(d_constants, &constants, sizeof(params), cudaMemcpyHostToDevice));

	return Tomo_OK;
}

TomoError GetReconNorm(struct SystemControl * Sys){
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Set up GPU kernel thread and block sizes based on image size
	dim3 dimBlockProj(16, 16);
	dim3 dimGridProj(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);

	float ex, ey, ez;

	//Calculate a the projection of a image of all ones
	for (int view = 0; view < Sys->Proj->NumViews; view++)
	{
		//these are the geometry values from the input file
		ex = Sys->SysGeo.EmitX[view];
		ey = Sys->SysGeo.EmitY[view];
		ez = Sys->SysGeo.EmitZ[view];

		//this is a GPU function call that causes the threads to perform the projection norm calculation
		voidChkErr(KERNELCALL2(ProjectionNorm, dimGridProj, dimBlockProj, d_Norm, view, ex, ey, ez, d_constants));
	}

	//Define the size of the sinogram space
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;

	size_t sizeSino = MemP_Nx * MemP_Ny * Sys->Proj->NumViews * sizeof(float);
	float  * d_NormCpy;
	ChkErr(cudaMalloc((void**)&d_NormCpy, sizeSino));

	//Calculate the contribution of each view to the total number of views
	voidChkErr(KERNELCALL2(AverageProjectionNorm, dimGridProj, dimBlockProj, d_Norm, d_NormCpy, d_constants));

	ChkErr(cudaMemcpy(d_Norm, d_NormCpy, sizeSino, cudaMemcpyDeviceToDevice));

	ChkErr(cudaFree(d_NormCpy));

	// Check the last error to make sure that norm calculations correction worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Calculate projection norm: " << cudaGetErrorString(error) << std::endl;

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions called to control the stages of reconstruction
TomoError LoadAndCorrectProjections(struct SystemControl * Sys){
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

	//Define the size of the sinogram space
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;
	int size_sino = MemP_Nx * MemP_Ny;
	size_t sizeSino = MemP_Nx  * MemP_Ny * Sys->Proj->NumViews * sizeof(float);

	//define the GPU kernel based on size of "ideal projection"
	dim3 dimBlockProj(32, 32);
	dim3 dimGridProj(Sys->Proj->Nx / 32 + Cx, Sys->Proj->Ny / 32 + Cy);

	float * d_SinoBlurX;
	float * d_SinoBlurXY;
	ChkErr(cudaMalloc((void**)&d_SinoBlurX, size_sino * sizeof(float)));
	ChkErr(cudaMalloc((void**)&d_SinoBlurXY, size_sino * sizeof(float)));

	ChkErr(cudaMalloc((void**)&d_Proj, sizeProj));
	ChkErr(cudaMalloc((void**)&d_Sino, sizeSino));

	//Cycle through each stream and do simple log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++){
		ChkErr(cudaMemcpyAsync(d_Proj, Sys->Proj->RawData + view*size_proj, sizeProj, cudaMemcpyHostToDevice));

		voidChkErr(KERNELCALL4(LogCorrectProj, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, view, d_Proj, 40000.0, d_constants));

		ChkErr(cudaMemset(d_SinoBlurX, 0, size_sino * sizeof(float)));
		ChkErr(cudaMalloc((void**)&d_SinoBlurX, size_sino * sizeof(float)));

		voidChkErr(KERNELCALL4(ApplyGaussianBlurX, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, d_SinoBlurX, view, d_constants));
		voidChkErr(KERNELCALL4(ApplyGaussianBlurY, dimGridProj, dimBlockProj, 0, streams[view], d_SinoBlurX, d_SinoBlurXY, d_constants));

		voidChkErr(KERNELCALL4(ScatterCorrect, dimGridProj, dimBlockProj, 0, streams[view], d_Sino, d_Proj, d_SinoBlurXY, view, log(40000.0f), d_constants));

		ChkErr(cudaMemcpyAsync(Sys->Proj->RawData + view*size_proj, d_Proj, sizeProj, cudaMemcpyDeviceToHost));
	}

	//Destroy the cuda streams used for log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) cudaStreamDestroy(streams[view]);

	ChkErr(cudaFree(d_SinoBlurX));
	ChkErr(cudaFree(d_SinoBlurXY));

	ChkErr(cudaMemcpyToArray(d_Sinogram, 0, 0, d_Sino, size_sino*Sys->Proj->NumViews * sizeof(float), cudaMemcpyDeviceToDevice));

	//Check the last error to make sure that log correction worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Load and Correct Projections: " << cudaGetErrorString(error) << std::endl;

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to control the SART and TV reconstruction
TomoError FindSliceOffset(struct SystemControl * Sys){
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
	ChkErr(cudaMemset(d_Image, 0, sizeIm));
	ChkErr(cudaMemset(d_Image2, 0, sizeIm));

	//Single SART Iteration with larger z_Pitch
	for (int iter = 0; iter < 1; iter++) {
		for (int view = 0; view < Sys->Proj->NumViews; view++){
			ex = Sys->SysGeo.EmitX[view];
			ey = Sys->SysGeo.EmitY[view];
			ez = Sys->SysGeo.EmitZ[view];

			ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

			voidChkErr(KERNELCALL2(ProjectImage, dimGridSino, dimBlockSino, d_Sino, d_Norm, d_Error, view, ex, ey, ez, d_constants));

			ChkErr(cudaBindTexture2D(NULL, textError, d_Error, textImage.channelDesc,MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

			voidChkErr(KERNELCALL2(BackProjectError, dimGridIm, dimBlockIm, d_Image, d_Image2, Beta, view, ex, ey, ez, d_constants));
		}//views
	}//iterations
	
	ChkErr(cudaMemset(d_Image2, 0, sizeIm));

	ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	voidChkErr(KERNELCALL2(SobelEdgeDetection, dimGridIm2, dimBlockIm2, d_Image2, d_constants));

	float * d_MaxVal;
	float * h_MaxVal = new float[Sys->Recon->Nz];
	ChkErr(cudaMalloc((void**)&d_MaxVal, Sys->Recon->Nz * sizeof(float)));
	voidChkErr(KERNELCALL3(SumSobelEdges, dimGridSum, dimBlockSum, sumSize, d_Image2, size_Im, d_MaxVal, d_constants));
	ChkErr(cudaMemcpy(h_MaxVal, d_MaxVal, Sys->Recon->Nz * sizeof(float), cudaMemcpyDeviceToHost));

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

	ChkErr(cudaMemcpyToSymbol(&d_Z_Offset, &Sice_Offset, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(&d_PitchNz, &Sys->Recon->Pitch_z, sizeof(float)));
	
	ChkErr(cudaMemset(d_Image2, 0, sizeIm));
	ChkErr(cudaMemset(d_Image, 0, sizeIm));

	tomo_err_throw(GetReconNorm(Sys));

	return Tomo_OK;
}

TomoError AddTVandTVSquared(struct SystemControl * Sys){
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

	ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	voidChkErr(KERNELCALL2(DerivOfGradIm, dimGridIm, dimBlockIm, d_DerivGradIm, eplison, d_constants));

	voidChkErr(KERNELCALL2(GetGradIm, dimGridIm, dimBlockIm, d_GradIm, d_constants));

	ChkErr(cudaBindTexture2D(NULL, textImage, d_GradIm, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	voidChkErr(KERNELCALL2(DerivOfGradIm, dimGridIm, dimBlockIm, d_Image2, eplison, d_constants));

	voidChkErr(KERNELCALL2(UpdateImageEstimate, dimGridIm, dimBlockIm, d_Image, d_DerivGradIm, d_Image2, d_constants));

	return Tomo_OK;
}

TomoError ReconUsingSARTandTV(struct SystemControl * Sys){
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
	ChkErr(cudaMemset(d_Image, 0, sizeIm));
	ChkErr(cudaMemset(d_Image2, 0, sizeIm));

	std::cout << "Reconstruction starting" << std::endl;

	//Do a set number of iterations
	for (int iter = 0; iter < 30; iter++){
		//Do one SART iteration by cycling through all views
		for (int view = 0; view <Sys->Proj->NumViews; view++){
			ex = Sys->SysGeo.EmitX[view];
			ey = Sys->SysGeo.EmitY[view];
			ez = Sys->SysGeo.EmitZ[view];

			ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc, MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

			voidChkErr(KERNELCALL2(ProjectImage, dimGridSino, dimBlockSino, d_Sino, d_Norm, d_Error, view, ex, ey, ez, d_constants));

			ChkErr(cudaBindTexture2D(NULL, textError, d_Error, textImage.channelDesc, MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

			voidChkErr(KERNELCALL2(BackProjectError, dimGridIm, dimBlockIm, d_Image, d_Image2, Beta, view, ex, ey, ez, d_constants));

			if(Sys->UsrIn->SmoothEdge == 1)
				KERNELCALL2(CorrectEdgesX, dimGridCorr, dimBlockCorr, d_Image, d_Image2, d_constants);
		}//views

		tomo_err_throw(AddTVandTVSquared(Sys));
		Beta = Beta*0.95f;
	}//iterations

	cudaDeviceSynchronize();
	
	std::cout << "Recon finised" << std::endl;

	//Code to create a sythnetic projection image
	/*	
	int size_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	size_t sizeProj = size_proj * sizeof(unsigned short);
	CreateSyntheticProjection << < dimGridSino, dimBlockSino >> >
			(d_Proj, d_Window, d_Image, log(40000.0f));

	ChkErr(cudaMemcpy(Sys->Proj->SyntData, d_Proj, sizeProj, cudaMemcpyDeviceToHost));
	*/

	// Check the last error to make sure that reconstruction functions  worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Reconstruction: " << cudaGetErrorString(error) << std::endl;

	return Tomo_OK;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to save the images
TomoError CopyAndSaveImages(struct SystemControl * Sys){
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
	ChkErr(cudaMalloc((void**)&d_ImCpy, sizeIM));
	ChkErr(cudaMemset(d_ImCpy, 0, sizeIM));

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
	ChkErr(cudaMalloc((void**)&d_MaxVal, Sys->Recon->Nz * sizeof(float)));
	voidChkErr(KERNELCALL3(GetMaxImageVal, dimGridSum, dimBlockSum, sumSize, d_Image, size_Im, d_MaxVal, d_constants));
	ChkErr(cudaMemcpy(h_MaxVal, d_MaxVal, Sys->Recon->Nz * sizeof(float), cudaMemcpyDeviceToHost));

	float MaxVal = 0;
	for (int slice = 0; slice < Sys->Recon->Nz; slice++)
		if (h_MaxVal[slice] > MaxVal) MaxVal = h_MaxVal[slice];

	std::cout << "The max reconstructed value is:" << MaxVal << std::endl;
	Sys->Recon->MaxVal = MaxVal;

	//Copy the image to smaller space
	voidChkErr(KERNELCALL2(CopyImages, dimGridIm, dimBlockIm, d_ImCpy, d_Image, MaxVal, d_constants));
	ChkErr(cudaMemcpy(Sys->Recon->ReconIm, d_ImCpy, sizeIM, cudaMemcpyDeviceToHost));

	//Remove temporary buffer
	ChkErr(cudaFree(d_ImCpy));

	return Tomo_OK;
}


//////////////////////////////////////////////////////////////////////////////////////////////
//Functions referenced from main
TomoError SetUpGPUForRecon(struct SystemControl * Sys){
	//Set up GPU Memory space
	DefineReconstructSpace(Sys);

	//Set up GPU memory space
	tomo_err_throw(SetUpGPUMemory(Sys));

	//Calulate the reconstruction Normalization for the SART
	tomo_err_throw(GetReconNorm(Sys));

	return Tomo_OK;
}

TomoError Reconstruct(struct SystemControl * Sys){
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
	tomo_err_throw(LoadAndCorrectProjections(Sys));
	std::cout << "Data Loaded" << std::endl;
	LONGLONG time1, time2;
	GetSystemTimeAsFileTime(&filetime4);
	time1 = (((ULONGLONG)filetime3.dwHighDateTime) << 32) + filetime3.dwLowDateTime;
	time2 = (((ULONGLONG)filetime4.dwHighDateTime) << 32) + filetime4.dwLowDateTime;
	std::cout << "Total LoadAndCorrectProjections time: " << (double)(time2 - time1) / 10000000 << " seconds";
	std::cout << std::endl;

	//Find Center Slice using increased Resolution
	if (Sys->UsrIn->CalOffset == 1)
		tomo_err_throw(FindSliceOffset(Sys));

	//Call the reconstruction function
	tomo_err_throw(ReconUsingSARTandTV(Sys));

	//Copy the reconstructed images to the CPU
	tomo_err_throw(CopyAndSaveImages(Sys));

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
TomoError FreeGPUMemory(void){
	//Free memory allocated on the GPU
	ChkErr(cudaFree(d_Proj));
	ChkErr(cudaFree(d_Norm));
	ChkErr(cudaFree(d_Image));
	ChkErr(cudaFree(d_Image2));
	ChkErr(cudaFree(d_GradIm));
	ChkErr(cudaFree(d_Error));
	ChkErr(cudaFree(d_Sino));
	ChkErr(cudaFree(d_Pro));
	ChkErr(cudaFree(d_PriorIm));
	ChkErr(cudaFree(d_dp));
	ChkErr(cudaFree(d_dpp));
	ChkErr(cudaFree(d_alpha));
	ChkErr(cudaFree(d_DerivGradIm));
	ChkErr(cudaFree(d_GradNorm));

	ChkErr(cudaFree(d_Image));
	ChkErr(cudaFree(d_Image2));
	ChkErr(cudaFree(d_Sino));
	ChkErr(cudaFree(d_Norm));
	ChkErr(cudaFree(d_Pro));
	ChkErr(cudaFree(d_Proj));

	ChkErr(cudaFree(d_PriorIm));
	ChkErr(cudaFree(d_DerivGradIm));
	ChkErr(cudaFree(d_GradNorm));
	ChkErr(cudaFree(d_dp));
	ChkErr(cudaFree(d_dpp));
	ChkErr(cudaFree(d_alpha));

	//Unbind the texture array and free the cuda array 
	ChkErr(cudaFreeArray(d_Sinogram));
	ChkErr(cudaUnbindTexture(textSino));

	ChkErr(cudaDeviceReset());

	return Tomo_OK;
}