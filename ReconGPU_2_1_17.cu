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
/* Helper functions used inline																*/
/********************************************************************************************/
inline void ChkErr(cudaError_t Error)
{
	if (Error != cudaSuccess)
	{
		std::cout << cudaGetErrorString(Error) << std::endl;
//		exit(1);
	}
}

/********************************************************************************************/
/* GPU Function specific functions															*/
/********************************************************************************************/

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to normalize the reconstruction
__global__ void ProjectionNorm(float * Error, float * Norm, int view,
	float ex, float ey, float ez)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		//Define edges of detector in x and y
		float dx1 = ((float)i - d_Px2) * d_PitchPx;
		float dy1 = ((float)j - d_Py2) * d_PitchPy;
		float dx2 = ((float)i - d_Px2 - 1.0f) * d_PitchPx;
		float dy2 = ((float)j - d_Py2 - 1.0f) * d_PitchPy;

		//Define the distance from source x and y to detector edges
		dx1 = dx1 - ex;
		dx2 = dx2 - ex;
		dy1 = dy1 - ey;
		dy2 = dy2 - ey;

		//Define step size in x and y
		float stepx1 = (dx1 / ez)*d_PitchNz;
		float stepx2 = (dx2 / ez)*d_PitchNz;
		float stepy1 = (dy1 / ez)*d_PitchNz;
		float stepy2 = (dy2 / ez)*d_PitchNz;

		//Define starting location in x and y
		float x1 = ex + (dx1 / ez) * (ez - (float)d_Z_Offset);
		float x2 = ex + (dx2 / ez) * (ez - (float)d_Z_Offset);
		float y1 = ey + (dy1 / ez) * (ez - (float)d_Z_Offset);
		float y2 = ey + (dy2 / ez) * (ez - (float)d_Z_Offset);

		//define projection variable;
		float Pro = 0;
		
		//Step through the image space by slice in z direction
		for (int z = 0; z < d_Nz; z++)
		{
			//convert x and y to image space pixels)
			float Nx1 = ((x1 - (d_Nx2 - 0.5f)) * d_PitchNx2);
			float Nx2 = ((x2 - (d_Nx2 - 0.5f)) * d_PitchNx2);
			float Ny1 = ((y1 - (d_Ny2 - 0.5f)) * d_PitchNy2);
			float Ny2 = ((y2 - (d_Ny2 - 0.5f)) * d_PitchNy2);

			//Get the first and last pixels in x and y the ray passes through
			int xx1 = (int)floorf(min(Nx1, Nx2));
			int xx2 = (int)ceilf(max(Nx1, Nx2));
			int yy1 = (int)floorf(min(Ny1, Ny2));
			int yy2 = (int)ceilf(max(Ny1, Ny2));

			//Get the length of the ray in the slice in x and y
			float dist = 1.0f / ((Nx2 - Nx1)*(Ny2 - Ny1));

			//Set the first x value to the first pixel
			float xs = Nx1;

			//Cycle through pixels x and y and used to calculate projection
			for (int x = xx1; x < xx2; x++)
			{
				float ys = Ny1;
				float xend = min((float)(x + 1), Nx2);

				for (int y = yy1; y < yy2; y++) {
					float yend = min((float)(y + 1), Ny2);

					//Calculate the weight as the overlap in x and y
					float weight = ((xend - xs)*(yend - ys)*dist);
					Pro += weight;

					ys = yend;
				}
				xs = xend;
			}
			x1 += stepx1;
			x2 += stepx2;
			y1 += stepy1;
			y2 += stepy2;
		}

		Norm[(j + view*d_MPy)*d_MPx + i] = Pro;
		if (Pro > 0) {
			Error[j*d_MPx + i] = 1.0f;
		}
	}
}


__global__ void BackProjectNorm(float * IM,  
	int view, float ex, float ey, float ez)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny))
	{
		//Define the direction in z to get r
		float r = ez / (ez - (d_Z_Offset - (float)k*d_PitchNz));

		//Use r to get detecor x and y
		float dx1 = ex + r * (((float)i - (d_Nx2))*d_PitchNx - ex);
		float dy1 = ey + r * (((float)j - (d_Ny2))*d_PitchNy - ey);
		float dx2 = ex + r * (((float)i - (d_Nx2 - 1.0f))*d_PitchNx - ex);
		float dy2 = ey + r * (((float)j - (d_Ny2 - 1.0f))*d_PitchNy - ey);

		//Use detector x and y to get pixels
		float x1 = dx1 * d_PitchPx2 + (d_Px2 - 0.5f);
		float x2 = dx2 * d_PitchPx2 + (d_Px2 - 0.5f);
		float y1 = dy1 * d_PitchPy2 + (d_Py2 - 0.5f);
		float y2 = dy2 * d_PitchPy2 + (d_Py2 - 0.5f);

		//Get the first and last pixels in x and y the ray passes through
		int xx1 = max((int)floorf(min(x1, x2)), 0);
		int xx2 = min((int)ceilf(max(x1, x2)), d_Px - 1);
		int yy1 = max((int)floorf(min(y1, y2)), 0);
		int yy2 = min((int)ceilf(max(y1, y2)), d_Py - 1);

		//Get the length of the ray in the slice in x and y
		float dist = 1.0f / fabsf((x2 - x1)*(y2 - y1));

		//Set a normalization and pixel value to 0
		float val = 0.0f;

		//Set the first x value to the first pixel
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

				//Update the value based on the error scaled and save the scake
				val += tex2D(textError, (float)x + 0.5f, (float)y + 0.5f) *weight;
				ys = yend;
			}
			xs = xend;
		}
		IM[(j + k*d_MNy)*d_MNx + i] +=  val;
	}
}

__global__ void AverageProjectionNorm(float * Norm, float * Norm2)
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

			Norm2[(j + n*d_MPy)*d_MPx + i] =  nVal;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////////////
//Functions to do initial correction of raw data: log and scatter correction
__global__ void LogCorrectProj(float * Sino, int view, unsigned short *Proj, float MaxVal)
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
__global__ void ApplyGaussianBlurX(float * Sino, float * BlurrX, int view)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		int N = 18;
		float sigma = -0.5f / (float)(6 * 6);
		float blur = 0;
		float norm = 0;
		//Use a neighborhood of 6 sigma
		for (int n = -N; n <= N; n++)
		{
			if (((n + i) >= 0) && (n + i < d_Px))
			{
				float weight = __expf((float)(n*n)*sigma);
				norm += weight;
				blur += weight * Sino[(j + view*d_MPy)*d_MPx + i];
			}
		}
		if (norm == 0) norm = 1.0f;
		BlurrX[j*d_MPx + i] = blur / norm;
	}
}

__global__ void ApplyGaussianBlurY(float * BlurrX, float * BlurrY)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		int N = 18;
		float sigma = -0.5f / (float)(6 * 6);
		float blur = 0;
		float norm = 0;
		//Use a neighborhood of 6 sigma
		for (int n = -N; n <= N; n++)
		{
			if (((n + j) >= 0) && (n + j < d_Ny))
			{
				float weight = __expf((float)(n*n)*sigma);
				norm += weight;
				blur += weight * BlurrX[j*d_MPx + i];
			}
		}
		if (norm == 0) norm = 1.0f;
		BlurrY[j*d_MPx + i] = blur / norm;
	}
}

__global__ void ScatterCorrect(float * Sino, unsigned short * Proj,
	float * BlurXY, int view, float MaxVal)
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
		float val = sample +0.25f*__expf(blur);
		Sino[(j + view*d_MPy)*d_MPx + i] = val;
		Proj[j*d_Px + i] = (unsigned short)(val * 32768.0f / MaxVal);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//SART part of the reconstuction code
__global__ void ProjectImage(float * Norm, float *Error, 
	int view, float ex, float ey, float ez)
{
	//Define pixel location in x and y
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	//Image is not a power of 2
	if ((i < d_Px) && (j < d_Py))
	{
		//Check to make sure the ray passes through the image
		float NP = Norm[(j + view*d_MPy)*d_MPx + i];
		float err = 0;

		if (NP != 0)
		{
			//Define edges of detector in x and y
			float dx1 = ((float)i - d_Px2) * d_PitchPx;
			float dy1 = ((float)j - d_Py2) * d_PitchPy;
			float dx2 = ((float)i - d_Px2 - 1.0f) * d_PitchPx;
			float dy2 = ((float)j - d_Py2 - 1.0f) * d_PitchPy;

	
			//Define step size in x and y
			float stepx1 = ((dx1 - ex) / ez)*d_PitchNz;
			float stepx2 = ((dx2 - ex) / ez)*d_PitchNz;
			float stepy1 = ((dy1 - ey) / ez)*d_PitchNz;
			float stepy2 = ((dy2 - ey) / ez)*d_PitchNz;

			//Define starting location in x and y
			float x1 = ex + ((dx1 - ex) / ez) * (ez - (float)d_Z_Offset);
			float x2 = ex + ((dx2 - ex) / ez) * (ez - (float)d_Z_Offset);
			float y1 = ey + ((dy1 - ey) / ez) * (ez - (float)d_Z_Offset);
			float y2 = ey + ((dy2 - ey) / ez) * (ez - (float)d_Z_Offset);

			//define projection variable;
			float Pro = 0;
			float count = 0.0f;

			//Step through the image space by slice in z direction
			for (int z = 0; z < d_Nz; z++)
			{
				//convert x and y to image space pixels)
				float Nx1 = ((x1 - (d_Nx2 - 0.5f)) * d_PitchNx2);
				float Nx2 = ((x2 - (d_Nx2 - 0.5f)) * d_PitchNx2);
				float Ny1 = ((y1 - (d_Ny2 - 0.5f)) * d_PitchNy2);
				float Ny2 = ((y2 - (d_Ny2 - 0.5f)) * d_PitchNy2);

				//Get the first and last pixels in x and y the ray passes through
				int xx1 = (int)floorf(min(Nx1, Nx2));
				int xx2 = (int)ceilf(max(Nx1, Nx2));
				int yy1 = (int)floorf(min(Ny1, Ny2));
				int yy2 = (int)ceilf(max(Ny1, Ny2));

				//Get the length of the ray in the slice in x and y
				float dist = 1.0f / ((Nx2 - Nx1)*(Ny2 - Ny1));

				//Set the first x value to the first pixel
				float xs = Nx1;

				//Cycle through pixels x and y and used to calculate projection
				for (int x = xx1; x < xx2; x++)
				{
					float ys = Ny1;
					float xend = min((float)(x + 1), Nx2);

					for (int y = yy1; y < yy2; y++) {
						float yend = min((float)(y + 1), Ny2);

						//Calculate the weight as the overlap in x and y
						float weight = ((xend - xs)*(yend - ys)*dist);

						int nx = min(max(x, 0), d_Nx - 1);
						int ny = min(max(y, 0), d_Ny - 1);

						Pro += tex2D(textImage, nx + 0.5f, ny + 0.5f + z*d_MNy) * weight;
						count += weight;

						ys = yend;
					}
					xs = xend;
				}
				x1 += stepx1;
				x2 += stepx2;
				y1 += stepy1;
				y2 += stepy2;
			}
			//If the ray passes through the image region get propjection error
			err = (tex2D(textSino, (float)i + 0.5f, (float)j + 0.5f + view*d_MPy)
				- Pro) / (max(count, 1.0f));

		}

		//Add Calculated error to an error image to back project
		Error[j*d_MPx + i] = err;
	}
}

__global__ void BackProjectError(float * IM, float * IM2,
	float beta, int view, float ex, float ey, float ez)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny))
	{
		//Define the direction in z to get r
		float r = ez / (ez - (d_Z_Offset - (float)k*d_PitchNz));

		//Use r to get detecor x and y
		float dx1 = ex + r * (((float)i - (d_Nx2))*d_PitchNx - ex);
		float dy1 = ey + r * (((float)j - (d_Ny2))*d_PitchNy - ey);
		float dx2 = ex + r * (((float)i - (d_Nx2 - 1.0f))*d_PitchNx - ex);
		float dy2 = ey + r * (((float)j - (d_Ny2 - 1.0f))*d_PitchNy - ey);

		//Use detector x and y to get pixels
		float x1 = dx1 * d_PitchPx2 + (d_Px2 - 0.5f);
		float x2 = dx2 * d_PitchPx2 + (d_Px2 - 0.5f);
		float y1 = dy1 * d_PitchPy2 + (d_Py2 - 0.5f);
		float y2 = dy2 * d_PitchPy2 + (d_Py2 - 0.5f);

		//Get the first and last pixels in x and y the ray passes through
		int xx1 = max((int)floorf(min(x1, x2)), 0);
		int xx2 = min((int)ceilf(max(x1, x2)), d_Px - 1);
		int yy1 = max((int)floorf(min(y1, y2)), 0);
		int yy2 = min((int)ceilf(max(y1, y2)), d_Py - 1);

		//Get the length of the ray in the slice in x and y
		float dist = 1.0f / fabsf((x2 - x1)*(y2 - y1));

		//Set a normalization and pixel value to 0
		float N = 0.0f;
		float val = 0.0f;

		//Set the first x value to the first pixel
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

				//Update the value based on the error scaled and save the scake
				val += tex2D(textError, (float)x + 0.5f, (float)y + 0.5f) *weight;
				N += weight;
				ys = yend;
			}
			xs = xend;
		}
		//Get the current value of image
		float update = beta*(val / N);
		
		if (N > 0.0) {
			float uval = IM[(j + k*d_MNy)*d_MNx + i];
			IM[(j + k*d_MNy)*d_MNx + i] = uval + update;
			IM2[(j + k*d_MNy)*d_MNx + i] = update;
		}
		else IM2[(j + k*d_MNy)*d_MNx + i] = -10.0f;
	}
	else IM2[(j + k*d_MNy)*d_MNx + i] = -10.0f;
}

__global__ void CorrectEdgesY(float * IM, float * IM2)
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

		if (j1 > 0) {
			float avgVal = 0;
			int count = min(20, j1);
			for (int j = j1; j < j1+count; j++)
				avgVal += max(0.0,IM2[(1 + j + k*d_MNy)*d_MNx + i]);

			avgVal = avgVal / (float)(count);

			for (int j = 0; j <= j1; j++) {
				float val = IM[(j + k*d_MNy)*d_MNx + i] + avgVal;
				IM[(j + k*d_MNy)*d_MNx + i] = val;
			}
		}
		if (j2 > 0) {
			float avgVal = 0;
			int count = min(20, j2);
			for (int j = j2; j < j2+count; j++)
				avgVal += max(0.0,IM2[(d_MNy - (j + 1) + k*d_MNy)*d_MNx + i]);

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


__global__ void CorrectEdgesX(float * IM, float * IM2)
{
	//Define pixel location in x, z
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	int k = blockDim.y * blockIdx.y + threadIdx.y;
	if (j < d_Ny && k < d_Nz)
	{
		float count1 = 0, count2 = 0;
		int i1 = 0, i2 = 0;

#pragma unroll
		for (int i = 0; i < d_Nx / 4; i++)
		{
			if (IM2[(j + k*d_MNy)*d_MNx + i] != -10.0f) break;
			count1++;
			i1++;
		}

#pragma unroll
		for (int i = 1; i <= d_Nx / 4; i++)
		{
			if (IM2[(j + k*d_MNy)*d_MNx + d_MNx - i] != -10.0f) break;
			count2++;
			i2++;
		}

		if(i1 > 0){
			float avgVal = 0;
			int count = min(20,i1);
			for (int i = i1; i <i1+count; i++)
				avgVal += max(0.0,IM2[(j + k*d_MNy)*d_MNx + i + 1]);

			avgVal = avgVal / (float)(count);

			for (int i = 0; i < i1; i++) {
				float val = IM[(j + k*d_MNy)*d_MNx + i] + avgVal;
				IM[(j + k*d_MNy)*d_MNx + i] = val;
			}
		}
	
		if (i2 > 0) {
			float avgVal = 0;
			int count = min(20, i2);
			for (int i = i2; i <= i2+count; i++)
				avgVal += max(0.0,IM2[(j + k*d_MNy)*d_MNx + d_MNx - (i+1)]);

			avgVal = avgVal / (float)(count);
			for (int i = 1; i <= i2; i++) {
				float val = IM[(j + k*d_MNy)*d_MNx + d_MNx - i] + avgVal;
				IM[(j + k*d_MNy)*d_MNx + d_MNx - i] = val;
			}
		}

		//smooth over edge
		float data[7] = { 0, 0, 0, 0, 0, 0, 0 };
		int nn = 0;
		if (i1 < d_Nx / 4 && i1 > 0) {
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
		if (i2 < d_Nx / 4 && i2 > 0) {
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
//		if (val1 < 1.0f)
//			val1 = 0.0f;
//		if (val2 < 1.0f)
//			val2 = 0.0f;
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

__global__ void DerivOfGradIm2(float * IM, float * TV, float ep)
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
		float sumTV = TV1 - TV2 - TV3 - TV4;
		
		float val1 = IM[(j + k*d_MNy)*d_MNx + i] -
			0.0025f* (TV[(j + k*d_MNy)*d_MNx + i] + sumTV);
		
		//Enforce positivity constraint
		if (val1 < 0) val1 = 0.0f;
		IM[(j + k*d_MNy)*d_MNx + i] = val1;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to create synthetic projections from reconstructed data
__global__ void CreateSyntheticProjection(unsigned short * Proj,
	float * Win, float *Im, float MaxVal)
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

__global__ void CreateSyntheticProjectionNew(unsigned short * Proj,
	float* Win, float * StIm, float *Im, float MaxVal, int cz)
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

__global__ void CopyImages(unsigned short * ImOut, float * ImIn, float maxVal)
{
	//Define pixel location in x, y, and z
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	int k = blockDim.z * blockIdx.z + threadIdx.z;

	//Image is not a power of 2
	if ((i < d_Nx) && (j < d_Ny))
	{
		float val = ImIn[(j + k*d_MNy)*d_MNx + i];// -0.015f;
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
void DefineReconstructSpace(struct SystemControl * Sys)
{
	//Define a new recon data pointer and define size
	Sys->Recon = new ReconGeometry;

	Sys->Recon->Pitch_z = Sys->SysGeo.ZPitch;
	Sys->Recon->Pitch_x = Sys->Proj->Pitch_x;
	Sys->Recon->Pitch_y = Sys->Proj->Pitch_y;
	Sys->Recon->Slice_0_z = Sys->SysGeo.ZDist;
	Sys->Recon->Nx = Sys->Proj->Nx;
	Sys->Recon->Ny = Sys->Proj->Ny;
//	Sys->Recon->Nx = 0.95*Sys->Proj->Nx;
//	Sys->Recon->Ny = 0.95*Sys->Proj->Ny;
	Sys->Recon->Nz = Sys->Proj->Nz;
	Sys->Recon->Mean = 210;
	Sys->Recon->Width = 410;
	Sys->Recon->MaxVal = 1.0f;

	if (Sys->UsrIn->CalOffset == 1) {
		Sys->Recon->Pitch_z = 1.0;
		Sys->Recon->Slice_0_z = 0;
	}

	//Define the size of the reconstruction memory and allocate a buffer in memory
	int size_slice = Sys->Recon->Nx * Sys->Recon->Ny;
	int size_image = Sys->Recon->Nz * size_slice;
	Sys->Recon->ReconIm = new unsigned short[size_image];
//	Sys->Recon->testval = new float[1824*1360];

}

//Function to set up the memory on the GPU
void SetUpGPUMemory(struct SystemControl * Sys)
{
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

	cudaDeviceReset();
	
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
	ChkErr(cudaMalloc((void**)&d_Image3, sizeIM));
	ChkErr(cudaMalloc((void**)&d_Error, sizeError));
	ChkErr(cudaMalloc((void**)&d_Sino, sizeSino));
	ChkErr(cudaMalloc((void**)&d_Norm, sizeSino));
	ChkErr(cudaMalloc((void**)&d_Proj, sizeProj));
	ChkErr(cudaMalloc((void**)&d_GradNorm, sizeSlice));
	ChkErr(cudaMalloc((void**)&d_dp, sizeSlice));
	ChkErr(cudaMalloc((void**)&d_dpp, sizeSlice));
	ChkErr(cudaMalloc((void**)&d_alpha, sizeSlice));

	//Set the values of the image and sinogram to all 0
	ChkErr(cudaMemset(d_Image, 0, sizeIM));
	ChkErr(cudaMemset(d_Image2, 0, sizeIM));
	ChkErr(cudaMemset(d_Image3, 0, sizeIM));
	ChkErr(cudaMemset(d_Sino, 0, sizeSino));
	ChkErr(cudaMemset(d_Proj, 0, sizeProj));
	ChkErr(cudaMemset(d_Norm, 0, sizeSino));
	ChkErr(cudaMemset(d_Error, 0, sizeError));
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

	ChkErr(cudaMallocArray(&d_Sinogram, &textSino.channelDesc,
		MemP_Nx, MemP_Ny * Sys->Proj->NumViews));
	cudaBindTextureToArray(textSino, d_Sinogram);

	//Set the TV weighting value to start at a constant value less than 1
	float * AlphaSlice = new float[Sys->Recon->Nz];
	for (int slice = 0; slice < Sys->Recon->Nz; slice++) {
		AlphaSlice[slice] = 0.2f;
	}
	ChkErr(cudaMemcpy(d_alpha, AlphaSlice, sizeSlice, cudaMemcpyHostToDevice));
	delete[] AlphaSlice;

	float Px2 = (float)Sys->Proj->Nx / 2.0f;
	float Py2 = (float)Sys->Proj->Ny / 2.0f;
	float Nx2 = (float)Sys->Recon->Nx / 2.0f;
	float Ny2 = (float)Sys->Recon->Ny / 2.0f;
	float PitchPx2 = 1.0f / Sys->Proj->Pitch_x;
	float PitchPy2 = 1.0f / Sys->Proj->Pitch_y;
	float PitchNx2 = 1.0f / Sys->Recon->Pitch_x;
	float PitchNy2 = 1.0f / Sys->Recon->Pitch_y;
	int Sice_Offset =(int)(((float)Sys->Recon->Nz)
		*(float)(Sys->Recon->Pitch_z)+
		(float)Sys->Recon->Slice_0_z);

	//Copy the system geometry constants
	ChkErr(cudaMemcpyToSymbol(d_Px, &Sys->Proj->Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Py, &Sys->Proj->Ny, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Nx, &Sys->Recon->Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Ny, &Sys->Recon->Ny, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MPx, &MemP_Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MPy, &MemP_Ny, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MNx, &MemR_Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_MNy, &MemR_Ny, sizeof(int)));

	ChkErr(cudaMemcpyToSymbol(d_Px2, &Px2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Py2, &Py2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Nx, &Sys->Recon->Nx, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Ny, &Sys->Recon->Ny, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_Nx2, &Nx2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Ny2, &Ny2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Nz, &Sys->Recon->Nz, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPx, &Sys->Proj->Pitch_x, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPy, &Sys->Proj->Pitch_y, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPx2, &PitchPx2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchPy2, &PitchPy2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNx, &Sys->Recon->Pitch_x, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNy, &Sys->Recon->Pitch_y, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNx2, &PitchNx2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNy2, &PitchNy2, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNz, &Sys->Recon->Pitch_z, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Views, &Sys->Proj->NumViews, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_rmax, &rmax, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_alpharelax, &alpha, sizeof(float)));
	ChkErr(cudaMemcpyToSymbol(d_Z_Offset, &Sice_Offset, sizeof(int)));
}

void GetReconNorm(struct SystemControl * Sys)
{
	int Cx = 1;
	int Cy = 1;
	if (Sys->Proj->Nx % 16 == 0) Cx = 0;
	if (Sys->Proj->Ny % 16 == 0) Cy = 0;

	//Set up GPU kernel thread and block sizes based on image size
	//Define the size of the sinogram space
	int MemP_Nx = (Sys->Proj->Nx / 16 + Cx) * 16;
	int MemP_Ny = (Sys->Proj->Ny / 16 + Cy) * 16;
	
	dim3 dimBlockProj(16, 16);
	dim3 dimGridProj(Sys->Proj->Nx / 16 + Cx, Sys->Proj->Ny / 16 + Cy);

	dim3 dimBlockIm(16, 8);
	dim3 dimGridIm(Sys->Recon->Nx / 16 + Cx, Sys->Recon->Ny / 8 + Cy, Sys->Recon->Nz);

	float ex, ey, ez;

	//Calculate a the projection of a image of all ones
	for (int view = 0; view < Sys->Proj->NumViews; view++)
	{
		ex = Sys->SysGeo.EmitX[view];
		ey = Sys->SysGeo.EmitY[view];
		ez = Sys->SysGeo.EmitZ[view];

		ProjectionNorm << < dimGridProj, dimBlockProj >> > (d_Error,d_Norm, view, ex, ey, ez);

	//	ChkErr(cudaBindTexture2D(NULL, textError, d_Error, textImage.channelDesc,
	//		MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

//		BackProjectNorm<< < dimGridIm, dimBlockIm >> >
	//		(d_Image3, view, ex, ey, ez);

	}

	// Check the last error to make sure that norm calculations correction worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Calculate projection norm: " << cudaGetErrorString(error) << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions called to control the stages of reconstruction
void LoadAndCorrectProjections(struct SystemControl * Sys)
{
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

	//define the GPU kernel based on size of "ideal projection"
	dim3 dimBlockProj(32, 32);
	dim3 dimGridProj(Sys->Proj->Nx / 32 + Cx, Sys->Proj->Ny / 32 + Cy);

	float * d_SinoBlurX;
	float * d_SinoBlurY;

	ChkErr(cudaMalloc((void**)&d_SinoBlurX, size_sino * sizeof(float)));
	ChkErr(cudaMalloc((void**)&d_SinoBlurY, size_sino * sizeof(float)));

	//Cycle through each stream and do simple log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++)
	{
		ChkErr(cudaMemcpyAsync(d_Proj, Sys->Proj->RawData + view*size_proj,
			sizeProj, cudaMemcpyHostToDevice));

		LogCorrectProj << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_Sino, view, d_Proj, 32768.0f);

		ChkErr(cudaMemset(d_SinoBlurX, 0, size_sino * sizeof(float)));
		ChkErr(cudaMemset(d_SinoBlurY, 0, size_sino * sizeof(float)));

		ApplyGaussianBlurX << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_Sino, d_SinoBlurX, view);
		ApplyGaussianBlurY << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_SinoBlurX, d_SinoBlurY);

		ScatterCorrect << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_Sino, d_Proj, d_SinoBlurY, view, log(32768.0f));

		ChkErr(cudaMemcpyAsync(Sys->Proj->RawData + view*size_proj, d_Proj,
			sizeProj, cudaMemcpyDeviceToHost));
	}

	//Destroy the cuda streams used for log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) cudaStreamDestroy(streams[view]);

	cudaFree(d_SinoBlurX);
	cudaFree(d_SinoBlurY);

	ChkErr(cudaMemcpyToArray(d_Sinogram, 0, 0, d_Sino,
		size_sino*Sys->Proj->NumViews * sizeof(float), cudaMemcpyDeviceToDevice));

	//Check the last error to make sure that log correction worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Load and Correct Projections: " << cudaGetErrorString(error) << std::endl;
}

void LoadAndCorrectProjectionsCenter(struct SystemControl * Sys)
{
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

	//define the GPU kernel based on size of "ideal projection"
	dim3 dimBlockProj(32, 32);
	dim3 dimGridProj(Sys->Proj->Nx / 32 + Cx, Sys->Proj->Ny / 32 + Cy);

	float * d_SinoBlurX;
	float * d_SinoBlurY;

	ChkErr(cudaMalloc((void**)&d_SinoBlurX, size_sino * sizeof(float)));
	ChkErr(cudaMalloc((void**)&d_SinoBlurY, size_sino * sizeof(float)));

	//Cycle through each stream and do simple log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++)
	{
		ChkErr(cudaMemcpyAsync(d_Proj, Sys->Proj->RawDataThresh + view*size_proj,
			sizeProj, cudaMemcpyHostToDevice));

		LogCorrectProj << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_Sino, view, d_Proj, 32768.0f);

		ChkErr(cudaMemset(d_SinoBlurX, 0, size_sino * sizeof(float)));
		ChkErr(cudaMemset(d_SinoBlurY, 0, size_sino * sizeof(float)));

		ApplyGaussianBlurX << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_Sino, d_SinoBlurX, view);
		ApplyGaussianBlurY << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_SinoBlurX, d_SinoBlurY);

		ScatterCorrect << < dimGridProj, dimBlockProj, 0, streams[view] >> >
			(d_Sino, d_Proj, d_SinoBlurY, view, log(32768.0f));

//		ChkErr(cudaMemcpyAsync(Sys->Proj->RawDataThresh + view*size_proj, d_Proj,
//			sizeProj, cudaMemcpyDeviceToHost));
	}

	//Destroy the cuda streams used for log correction
	for (int view = 0; view < Sys->Proj->NumViews; view++) cudaStreamDestroy(streams[view]);

	cudaFree(d_SinoBlurX);
	cudaFree(d_SinoBlurY);

	ChkErr(cudaMemcpyToArray(d_Sinogram, 0, 0, d_Sino,
		size_sino*Sys->Proj->NumViews * sizeof(float), cudaMemcpyDeviceToDevice));

	//Check the last error to make sure that log correction worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Load and Correct Projections: " << cudaGetErrorString(error) << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to control the SART and TV reconstruction
void FindSliceOffset(struct SystemControl * Sys)
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
	cudaMemset(d_Image, 0, sizeIm);
	cudaMemset(d_Image2, 0, sizeIm);

	//Single SART Iteration with larger z_Pitch
	for (int iter = 0; iter < 1; iter++) {

		for (int view = 0; view < Sys->Proj->NumViews; view++)
		{
			ex = Sys->SysGeo.EmitX[view];
			ey = Sys->SysGeo.EmitY[view];
			ez = Sys->SysGeo.EmitZ[view];

			ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc,
				MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

			ProjectImage << < dimGridSino, dimBlockSino >> >
				(d_Norm, d_Error, view, ex, ey, ez);

			ChkErr(cudaBindTexture2D(NULL, textError, d_Error, textImage.channelDesc,
				MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

			BackProjectError << < dimGridIm, dimBlockIm >> >
				(d_Image, d_Image2, Beta, view, ex, ey, ez);
		}
	}
	
	cudaMemset(d_Image2, 0, sizeIm);

	ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc,
			MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	SobelEdgeDetection << <dimGridIm2, dimBlockIm2 >> >(d_Image2);

//	float* testval = new float[MemR_Nx*MemR_Ny*Sys->Recon->Nz];

//	ChkErr(cudaMemcpy(testval, d_Image2, sizeIm, cudaMemcpyDeviceToHost));

//	int howmany = 0;
//	for (int cntnum = 0; cntnum < MemR_Nx*MemR_Ny*Sys->Recon->Nz; cntnum++)
//	{
//		if (testval[cntnum] > 1.0f)
//		{
//			testval[cntnum] = 0.0f;
//		}
//	}

//	ChkErr(cudaMemcpy(d_Image2, testval, sizeIm, cudaMemcpyHostToDevice));

	float * d_MaxVal;
	float * h_MaxVal = new float[Sys->Recon->Nz];
	cudaMalloc((void**)&d_MaxVal, Sys->Recon->Nz * sizeof(float));
	SumSobelEdges << < dimGridSum, dimBlockSum, sumSize >> > (d_Image2, size_Im, d_MaxVal);
	cudaMemcpy(h_MaxVal, d_MaxVal, Sys->Recon->Nz * sizeof(float), cudaMemcpyDeviceToHost);

	int centerSlice = 0;
	int MaxSum = 0;
	for (int n = 0; n < Sys->Recon->Nz; n++)
	{
		if (h_MaxVal[n] > MaxSum) {
			MaxSum = (int)h_MaxVal[n];
			centerSlice = n;
		}
	}
	//add 5.5 offset to account for error 
	centerSlice = Sys->Recon->Nz - (centerSlice+3);
	centerSlice -= 5;

	if (centerSlice < 0)
		centerSlice = 20;

	std::cout << "Center slice is:" << centerSlice << std::endl;
	float slice_offset = (float)centerSlice * (float)(Sys->Recon->Pitch_z);
	std::cout << "New Offset is:" << slice_offset << std::endl;

	Sys->Recon->Pitch_z = Sys->SysGeo.ZPitch;

	int Sice_Offset = (int)(((float)Sys->Recon->Nz)
		*(float)(Sys->Recon->Pitch_z) + slice_offset);

	std::cout << "New top of image is:" << Sice_Offset << std::endl;
	std::cout << "New Pitch is:" << Sys->Recon->Pitch_z << std::endl;

	ChkErr(cudaMemcpyToSymbol(d_Z_Offset, &Sice_Offset, sizeof(int)));
	ChkErr(cudaMemcpyToSymbol(d_PitchNz, &Sys->Recon->Pitch_z, sizeof(float)));
	
	cudaMemset(d_Image2, 0, sizeIm);
	cudaMemset(d_Image, 0, sizeIm);

	GetReconNorm(Sys);

	size_t sizeSino = MemP_Nx  * MemP_Ny * Sys->Proj->NumViews * sizeof(float);
	ChkErr(cudaMemset(d_Sino, 0, sizeSino));
}

void AddTVandTVSquared(struct SystemControl * Sys)
{
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

	ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc,
		MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	DerivOfGradIm << < dimGridIm, dimBlockIm >> >	(d_Image3, eplison);

	GetGradIm << < dimGridIm, dimBlockIm >> > (d_Image2);

	ChkErr(cudaBindTexture2D(NULL, textImage, d_Image2, textImage.channelDesc,
		MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

	DerivOfGradIm2 << < dimGridIm, dimBlockIm >> >	(d_Image, d_Image3, eplison);
	
}

void ReconUsingSARTandTV(struct SystemControl * Sys)
{
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
	dim3 dimGridCorrX((int)floor(Sys->Recon->Ny / 32), (Sys->Recon->Nz / 4) +1);
	dim3 dimGridCorrY((int)floor(Sys->Recon->Nx / 32), (Sys->Recon->Nz / 4) + 1);

	size_t sizeIm = MemR_Nx*MemR_Ny*Sys->Recon->Nz * sizeof(float);

	dim3 dimGridSum(1, Sys->Recon->Nz);
	dim3 dimBlockSum(1024, 1);

	float Beta = 1.0f;
	float ex, ey, ez;

	cudaMemset(d_Image, 0, sizeIm);
	cudaMemset(d_Image2, 0, sizeIm);
	
	std::cout << "Reconstruction starting" << std::endl;

	//Do a set number of iterations (used to be 30)
	for (int iter = 0; iter < 30; iter++)
	{
		//Do one SART iteration by cycling through all views
		for (int view = 0; view <Sys->Proj->NumViews; view++)
		{
			ex = Sys->SysGeo.EmitX[view];
			ey = Sys->SysGeo.EmitY[view];
			ez = Sys->SysGeo.EmitZ[view];

			ChkErr(cudaBindTexture2D(NULL, textImage, d_Image, textImage.channelDesc,
				MemR_Nx, MemR_Ny*Sys->Recon->Nz, MemR_Nx * sizeof(float)));

			ProjectImage << < dimGridSino, dimBlockSino >> >
				(d_Norm, d_Error, view, ex, ey, ez);

			ChkErr(cudaBindTexture2D(NULL, textError, d_Error, textImage.channelDesc,
				MemP_Nx, MemP_Ny, MemP_Nx * sizeof(float)));

			BackProjectError<< < dimGridIm, dimBlockIm >> >
				(d_Image, d_Image2,Beta, view, ex, ey, ez);
			
			if (Sys->UsrIn->SmoothEdge == 1 && iter < 29) {
				CorrectEdgesX << <dimGridCorrX, dimBlockCorr >> > (d_Image, d_Image2);
				CorrectEdgesY << <dimGridCorrY, dimBlockCorr >> > (d_Image, d_Image2);
			}
		}

		if (Sys->UsrIn->UseTV) {
			AddTVandTVSquared(Sys);
			Beta = Beta*0.975f;
		}
	}
	cudaDeviceSynchronize();

	std::cout << "Recon finised" << std::endl;

	cudaFree(d_Image2);
	cudaFree(d_Image3);

	// Check the last error to make sure that reconstruction functions  worked properly
	cudaError_t error = cudaGetLastError();
	std::cout << "Reconstruction: " << cudaGetErrorString(error) << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Functions to save the images
void CopyAndSaveImages(struct SystemControl * Sys)
{
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
	cudaMalloc((void**)&d_MaxVal, Sys->Recon->Nz * sizeof(float));
	GetMaxImageVal << < dimGridSum, dimBlockSum, sumSize >> > (d_Image, size_Im, d_MaxVal);
	cudaMemcpy(h_MaxVal, d_MaxVal, Sys->Recon->Nz * sizeof(float), cudaMemcpyDeviceToHost);

	float MaxVal = 0;
	for (int slice = 0; slice < Sys->Recon->Nz; slice++) {
		if (h_MaxVal[slice] > MaxVal) MaxVal = h_MaxVal[slice];
	}
	std::cout << "The max reconstructed value is:" << MaxVal << std::endl;
	Sys->Recon->MaxVal = MaxVal;

	//Copy the image to smaller space
	CopyImages << < dimGridIm, dimBlockIm >> > (d_ImCpy, d_Image, MaxVal);
	ChkErr(cudaMemcpy(Sys->Recon->ReconIm, d_ImCpy, sizeIM, cudaMemcpyDeviceToHost));

	//Remove temporary buffer
	cudaFree(d_ImCpy);
}


//////////////////////////////////////////////////////////////////////////////////////////////
//Functions referenced from main
void SetUpGPUForRecon(struct SystemControl * Sys)
{
	//Set up GPU Memory space
	DefineReconstructSpace(Sys);

	//Set up GPU memory space
	SetUpGPUMemory(Sys);

	//Calulate the reconstruction Normalization for the SART
	GetReconNorm(Sys);
}

void Reconstruct(struct SystemControl * Sys)
{
	//Get the time and start before projections are corrected
	FILETIME filetime, filetime2;
	GetSystemTimeAsFileTime(&filetime);

	int size_slice = Sys->Recon->Nx * Sys->Recon->Ny;
	int size_image = Sys->Recon->Nz * size_slice;
	memset(Sys->Recon->ReconIm, 0, size_image * sizeof(unsigned short));
	std::cout << "Memory Ready" << std::endl;

	//correct data for finding center of image
	if (Sys->UsrIn->CalOffset)
	{
		LoadAndCorrectProjectionsCenter(Sys);
		FindSliceOffset(Sys);
	}

	//Correct data and read onto GPU
	LoadAndCorrectProjections(Sys);
	std::cout << "Data Loaded" << std::endl;

	//Find Center Slice using increased Resolution
//	if (Sys->UsrIn->CalOffset == 1) {
//		FindSliceOffset(Sys);
//	}

	//Call the reconstruction function
	ReconUsingSARTandTV(Sys);

	//Copy the reconstructed images to the CPU
	CopyAndSaveImages(Sys);

	//Get and display the total ellasped time for the reconstruction
	GetSystemTimeAsFileTime(&filetime2);
	LONGLONG time1, time2;
	time1 = (((ULONGLONG)filetime.dwHighDateTime) << 32) + filetime.dwLowDateTime;
	time2 = (((ULONGLONG)filetime2.dwHighDateTime) << 32) + filetime2.dwLowDateTime;
	std::cout << "Total Recon time: " << (double)(time2 - time1) / 10000000 << " seconds";
	std::cout << std::endl;

	std::cout << "Reconstruction finished successfully." << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////
//Fucntion to free the gpu memory after program finishes
void FreeGPUMemory(void)
{
	//Free memory allocated on the GPU
	cudaFree(d_Proj);
	cudaFree(d_Norm);
	cudaFree(d_Image);
	cudaFree(d_Error);
	cudaFree(d_dp);
	cudaFree(d_dpp);
	cudaFree(d_alpha);
	cudaFree(d_GradNorm);
	cudaFree(d_Sino);


	//Unbind the texture array and free the cuda array 
	cudaFreeArray(d_Sinogram);
	cudaUnbindTexture(textSino);

	ChkErr(cudaDeviceReset());
}