/********************************************************************************************/
/* ReconGPUHeader.cuh																		*/
/* Copyright 2015, Xintek Inc., All rights reserved											*/
/********************************************************************************************/
#ifndef _RECONGPUHEADER_CUH_
#define _RECONGPUHEADER_CUH_

// CUDA runtime
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <iomanip>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <crtdbg.h>
#include "Shlwapi.h"

#include <Windows.h>
#include <WinBase.h>

#include "../UI/interop.h"

#pragma comment(lib, "Shlwapi.lib")

#define NUMVIEWS 7
#define MAXZOOM 30
#define ZOOMFACTOR 1.1f
#define LINEWIDTH 3
#define BARHEIGHT 40

//Projection correction parameters
#define LOWTHRESH 80.0f
#define HIGHTHRESH 0.98f

//Autofocus parameters
#define STARTSTEP 1.0f
#define LASTSTEP 0.001f
#define GEOSTART 5.0f
#define GEOLAST 0.01f
#define MINDIS 0
#define MAXDIS 20

//Phantom reader parameters
#define LINEPAIRS 5
#define INTENSITYTHRESH 300
#define UPPERBOUND 20.0f
#define LOWERBOUND 4.0f

//cuda constants
#define WARPSIZE 32
#define MAXTHREADS 1024

//Kernel options
#define SIGMA 1.5f
#define KERNELRADIUS 5
#define KERNELSIZE (2*KERNELRADIUS + 1)

//Maps to single instruction in cuda
#define MUL_ADD(a, b, c) ( __mul24((a), (b)) + (c) )
#define UMUL(a, b) ( (a) * (b) )

//Autoscale parameters
#define AUTOTHRESHOLD 50000
#define HISTLIMIT 10
#define HIST_BIN_COUNT 256

//Code use parameters
//#define PROFILER
//#define CHAMBOLLE
#define PRINTSCANCORRECTIONS

//Macro for checking cuda errors following a cuda launch or api call
#define voidChkErr(...) {										\
	(__VA_ARGS__);												\
	cudaError_t e=cudaGetLastError();							\
	if(e!=cudaSuccess) {										\
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(e) << "\n";	\
		return Tomo_CUDA_err;									\
	}															\
}

/********************************************************************************************/
/* Error type, used to pass errors to the caller											*/
/********************************************************************************************/
typedef enum {
	Tomo_OK,
	Tomo_input_err,
	Tomo_invalid_arg,
	Tomo_file_err,
	Tomo_DICOM_err,
	Tomo_CUDA_err,
	Tomo_Done
} TomoError;

typedef enum {
	projections,
	reconstruction
} sourceData;

typedef enum {
	no_der,
	x_mag_enhance,
	y_mag_enhance,
	mag_enhance,
	x_enhance,
	y_enhance,
	both_enhance,
	der_x,
	der_y,
	der2_x,
	der2_y,
	der3_x,
	der3_y,
	square_mag,
	slice_diff,
	orientation,
	der,
	der2,
	der3,
	der_all
} derivative_t;

typedef enum {
	dir_x,
	dir_y,
	dir_z
} direction;

#define tomo_err_throw(x) {TomoError err = x; if(err != Tomo_OK) return err;}

#ifdef __INTELLISENSE__
#include "intellisense.h"
#define KERNELCALL2(function, threads, blocks, ...) voidChkErr(function(__VA_ARGS__))
#define KERNELCALL3(function, threads, blocks, sharedMem, ...) voidChkErr(function(__VA_ARGS__))
#define KERNELCALL4(function, threads, blocks, sharedMem, stream, ...) voidChkErr(function(__VA_ARGS__))
#else
#define KERNELCALL2(function, threads, blocks, ...) voidChkErr(function <<< threads, blocks >>> (__VA_ARGS__))
#define KERNELCALL3(function, threads, blocks, sharedMem, ...) voidChkErr(function <<< threads, blocks, sharedMem >>> (__VA_ARGS__))
#define KERNELCALL4(function, threads, blocks, sharedMem, stream, ...) voidChkErr(function <<< threads, blocks, sharedMem, stream >>> (__VA_ARGS__))
#endif

/********************************************************************************************/
/* System Structures (All units are in millimeters)											*/
/********************************************************************************************/
struct Proj_Data {
	int NumViews;							//The number of projection views the recon uses
	float Pitch_x;							//The detector pixel pitch in the x direction
	float Pitch_y;							//The detector pixel pithc in the y direction
	int Nx;									//The number of detector pixels in x direction
	int Ny;									//The number of detector pixels in y direction
	int Flip;								//Flip the orientation of the detector
};

struct SysGeometry {
	float * EmitX;							//Location of the x-ray focal spot in x direction
	float * EmitY;							//Location of the x-ray focal spot in y direction
	float * EmitZ;							//Location of the x-ray focal spot in z direction
	float IsoX;								//Location of the system isocenter in x direction
	float IsoY;								//Location of the system isocenter in y direction
	float IsoZ;								//Location of the system isocenter in z direction
	float ZPitch;							//The distance between slices
};

struct ReconGeometry {
	float Pitch_x;							//Recon Image pixel pitch in the x direction
	float Pitch_y;							//Recon Image pixel pitch in the y direction
	int Nx;									//The number of recon image pixels in x direction
	int Ny;									//The number of recon image pixels in y direction
};

//Declare a structure containing other system description structures to pass info
struct SystemControl {
	struct Proj_Data Proj;
	struct ReconGeometry Recon;
	struct SysGeometry Geo;
};

//Define a number of constants
struct params {
	int Px;
	int Py;
	int Rx;
	int Ry;
	int Views;
	float PitchPx;
	float PitchPy;
	float PitchRx;
	float PitchRy;
	float * d_Beamx;
	float * d_Beamy;
	float * d_Beamz;
	int ReconPitchNum;
	int ProjPitchNum;
	bool orientation;
	bool flip;
	bool log;

	//Display parameters
	float minVal = 0.0;
	float maxVal = USHRT_MAX;

	//selection box parameters
	int baseXr = -1;
	int baseYr = -1;
	int currXr = -1;
	int currYr = -1;

	//User parameters
	float ratio = 0.0;
	bool useMaxNoise;
	int maxNoise = 30;
};

struct CPUParams {
	bool scanVertEnable;
	bool scanHorEnable;
	float vertTau = 0.25;
	float horTau = 0.1;
	int iterations = 40;
};

struct toleranceData {
	std::string name = "views ";
	int numViewsChanged;
	int viewsChanged;
	direction thisDir = dir_x;
	float offset;
	float phantomData;
};

class TomoRecon : public interop {
public:
	//Functions
	//constructor/destructor
	TomoRecon(int x, int y, struct SystemControl * Sys);
	~TomoRecon();

	TomoError init(const char * gainFile, const char * mainFile);

	//High level functions for command line call
	TomoError ReadProjections(const char * gainFile, const char * mainFile);
	TomoError FreeGPUMemory(void);

	TomoError setReconBox(int index);
	//TODO: make setters instead of converters public
	int P2R(int p, int view, bool xDir);
	int R2P(int r, int view, bool xDir);
	int I2D(int i, bool xDir);
	int D2I(int d, bool xDir);

	//Lower level functions for user interactive debugging
	template<typename T>
	TomoError getHistogram(T * image, unsigned int byteSize, unsigned int *histogram);
	TomoError singleFrame();
	TomoError autoFocus(bool firstRun);
	TomoError autoGeo(bool firstRun);
	TomoError autoLight(unsigned int histogram[HIST_BIN_COUNT], int threshold, float * minVal, float * maxVal);
	TomoError readPhantom(float * resolution);
	TomoError initTolerances(std::vector<toleranceData> &data, int numTests, std::vector<float> offsets);
	TomoError testTolerances(std::vector<toleranceData> &data, bool firstRun);

	//interop extension
	TomoError draw(int x, int y);

	//Getters and setters
	TomoError getLight(unsigned int * minVal, unsigned int * maxVal);
	TomoError setLight(unsigned int minVal, unsigned int maxVal);
	TomoError appendMaxLight(int amount);
	TomoError appendMinLight(int amount);
	float getDistance();
	TomoError setDistance(float distance);
	TomoError stepDistance(int steps);
	TomoError resetLight();
	TomoError resetFocus();
	TomoError setLogView(bool useLog);
	bool getLogView();
	TomoError setInputVeritcal(bool vertical);
	TomoError setActiveProjection(int index);
	TomoError setOffsets(int xOff, int yOff);
	TomoError appendOffsets(int xOff, int yOff);
	void getOffsets(int * xOff, int * yOff);
	TomoError setSelBoxStart(int x, int y);
	TomoError setSelBoxEnd(int x, int y);
	TomoError resetSelBox();
	bool selBoxReady();
	TomoError appendZoom(int amount);
	TomoError setZoom(int value);
	TomoError resetZoom();
	derivative_t getDisplay();
	TomoError setDisplay(derivative_t type);
	TomoError setEnhanceRatio(float ratio);
	TomoError enableNoiseMaxFilter(bool enable);
	TomoError setNoiseMaxVal(int max);
	TomoError enableScanVert(bool enable);
	TomoError setScanVertVal(float tau);
	TomoError enableScanHor(bool enable);
	TomoError setScanHorVal(float tau);

	/* Input Functions to read data into the program                      */	BOOL CheckFilePathForRepeatScans(std::string BasePathIn);
	int GetNumberOfScans(std::string BasePathIn);
	int GetNumOfProjectionsPerView(std::string BasePathIn);
	int GetNumProjectionViews(std::string BasePathIn);
	TomoError GetGainAverages(const char * gainFile);

	/********************************************************************************************/
	/* Variables																				*/
	/********************************************************************************************/
	struct SystemControl Sys;
	int NumViews;

	int iteration = 0;
	bool continuousMode = false;
	int sliceIndex = 0;
	int zoom = 0;
	int light = 0;
	int lightOff = 0;
	int xOff = 0;
	int yOff = 0;
	float scale = 1.5;
	float distance = 0.0;

	//Cuda constants
	params constants;

	//Selection variables

	//box
	int baseX = -1;
	int baseY = -1;
	int currX = -1;
	int currY = -1;

	//lower tick
	int lowX = -1;
	int lowY = -1;

	//upper tick
	int upX = -1;
	int upY = -1;

	//lower tick
	int lowXr = -1;
	int lowYr = -1;

	//upper tick
	int upXr = -1;
	int upYr = -1;

	bool vertical;
	derivative_t derDisplay = no_der;
	sourceData dataDisplay = reconstruction;

private:
	/********************************************************************************************/
	/* Function to interface the CPU with the GPU:												*/
	/********************************************************************************************/

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Functions to Initialize the GPU and set up the reconstruction normalization
	TomoError initGPU(const char * gainFile, const char * mainFile);
	TomoError setNOOP(float kernel[KERNELSIZE]);
	TomoError setGauss(float kernel[KERNELSIZE]);
	TomoError setGaussDer(float kernel[KERNELSIZE]);
	TomoError setGaussDer2(float kernel[KERNELSIZE]);
	TomoError setGaussDer3(float kernel[KERNELSIZE]);
	void diver(float * z, float * d, int n);
	void nabla(float * u, float * g, int n);
	void lapla(float * a, float * b, int n);

	//Get and set helpers
	TomoError checkOffsets(int * xOff, int * yOff);
	TomoError checkBoundaries(int * x, int * y);

	//Kernel call helpers
	float focusHelper();
	TomoError imageKernel(float xK[KERNELSIZE], float yK[KERNELSIZE], float * output);
	TomoError scanLineDetect(int view, float * d_sum, float * sum, float * offset, bool vert, bool enable);
	float graphCost(float * vertGraph, float * horGraph, int view = -1, float offset = 0.0, float lightScale = 1.0, float rsq = 0.0);
	float getMax(float * d_Image);

	//Coordinate conversions
	TomoError P2R(int* rX, int* rY, int pX, int pY, int view);
	TomoError R2P(int* pX, int* pY, int rX, int rY, int view);
	TomoError I2D(int* dX, int* dY, int iX, int iY);

	//Define data buffer
	float * d_Image;
	float * d_Error;
	float * d_Sino;
	float * d_MaxVal;
	float * d_MinVal;

	//Kernel memory
	float * d_noop;
	float * d_gauss;
	float * d_gaussDer;
	float * d_gaussDer2;
	float * d_gaussDer3;

	//Derivative buffers
	float * buff1;
	float * buff2;

	//Kernel call parameters
	size_t sizeIM;
	size_t sizeProj;
	size_t sizeSino;
	size_t sizeError;

	dim3 contThreads;
	dim3 contBlocks;
	dim3 reductionBlocks;
	dim3 reductionThreads;
	int reductionSize = MAXTHREADS * sizeof(float);

	cudaStream_t stream;

	//cuda pitch variables generated from 2d mallocs
	size_t reconPitch;
	size_t projPitch;
	int reconPitchNum;

	//Parameters for 2d geometry search
	int diffSlice = 0;

	//Constants for program set by outside caller
	CPUParams cConstants;
	//params constants;
};

/********************************************************************************************/
/* Library assertions																		*/
/********************************************************************************************/
TomoError cuda_assert(const cudaError_t code, const char* const file, const int line);
TomoError cuda_assert_void(const char* const file, const int line);

#define cudav(...)  cuda##__VA_ARGS__; cuda_assert_void(__FILE__, __LINE__);
#define cuda(...)  cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__);

#endif
