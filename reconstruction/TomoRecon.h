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
//#define PROFILER
#define ITERATIONS 7
#define DECAY 0.8f
#define MAXZOOM 30
#define ZOOMFACTOR 1.1f
#define LIGHTFACTOR 1.1f
#define LIGHTOFFFACTOR 3
#define LINEWIDTH 3
#define BARHEIGHT 40

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
#define SIGMA 1.0f
#define KERNELRADIUS 5
#define KERNELSIZE (2*KERNELRADIUS + 1)

//Maps to single instruction in cuda
#define MUL_ADD(a, b, c) ( __mul24((a), (b)) + (c) )
#define UMUL(a, b) ( (a) * (b) )

//Autoscale parameters
#define AUTOTHRESHOLD 5000
#define HISTLIMIT 10

//defines from cuda example
//TODO: merge into our architecture
#define PARTIAL_HISTOGRAM256_COUNT 240
#define LOG2_WARP_SIZE 5U
#define UINT_BITS 32
#define HIST_BIN_COUNT 256
#define MERGE_THREADBLOCK_SIZE 256
#define SHARED_MEMORY_BANKS 16
#define HISTOGRAM64_THREADBLOCK_SIZE (4 * SHARED_MEMORY_BANKS)
#define WARP_COUNT 6
#define HISTOGRAM256_THREADBLOCK_SIZE (WARP_COUNT * WARPSIZE)
#define HISTOGRAM256_THREADBLOCK_MEMORY (WARP_COUNT * HIST_BIN_COUNT)

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
	Tomo_file_err,
	Tomo_DICOM_err,
	Tomo_CUDA_err,
	Tomo_Done
} TomoError;

typedef enum {
	raw_images,
	recon_images
} display_t;

typedef enum {
	no_der,
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
	unsigned short * RawData;				//Pointer to a buffer containing the raw data
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
	std::string Name;						//A name of the emitter file
};


struct NormData {
	unsigned short * GainData;				//A buffer to contain the gain images
	unsigned short * DarkData;				//A buffer to contain the dark images
	unsigned short * ProjBuf;				//A tempory buffer to read projections into
	float * CorrBuf;						//A buffer to correct the and sum projections
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
	struct NormData Norm;
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
	float * h_Beamx;
	float * h_Beamy;
	float * h_Beamz;
	float * d_Beamx;
	float * d_Beamy;
	float * d_Beamz;
	int ReconPitchNum;
	int ProjPitchNum;
	bool orientation;
	bool flip;
	bool log;

	//Display parameters
	float minVal;
	float maxVal;
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
	//////////////////////////////////////////////////////////////////////////////////////////////
	//constructor/destructor
	TomoRecon(int x, int y, struct SystemControl * Sys);
	~TomoRecon();

	TomoError init(const char * gainFile, const char * darkFile, const char * mainFile);

	//////////////////////////////////////////////////////////////////////////////////////////////
	//High level functions for command line call
	TomoError TomoLoad(const char* file);
	TomoError FreeGPUMemory(void);

	TomoError setReconBox(int index);
	//TODO: make setters instead of converters public
	int P2R(int p, int view, bool xDir);
	int R2P(int r, int view, bool xDir);
	int I2D(int i, bool xDir);
	int D2I(int d, bool xDir);

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Lower level functions for user interactive debugging
	TomoError correctProjections();
	template<typename T>
	TomoError getHistogram(T * image, unsigned int byteSize, unsigned int *histogram);
	TomoError singleFrame();
	float getDistance();
	TomoError autoFocus(bool firstRun);
	TomoError autoGeo(bool firstRun);
	TomoError readPhantom(float * resolution);
	TomoError initTolerances(std::vector<toleranceData> &data, int numTests, std::vector<float> offsets);
	TomoError testTolerances(std::vector<toleranceData> &data, bool firstRun);

	//////////////////////////////////////////////////////////////////////////////////////////////
	//interop extensions
	void map() { interop::map(stream); }
	void unmap() { interop::unmap(stream); }
	TomoError test(int index);

	/********************************************************************************************/
	/* Variables																				*/
	/********************************************************************************************/
	bool initialized = false;

#ifdef PROFILER
	display_t currentDisplay = recon_images;
#else
	display_t currentDisplay = raw_images;
#endif

	struct SystemControl * Sys;
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

	//coordinates w.r.t. recon
	//box
	int baseXr = -1;
	int baseYr = -1;
	int currXr = -1;
	int currYr = -1;

	//lower tick
	int lowXr = -1;
	int lowYr = -1;

	//upper tick
	int upXr = -1;
	int upYr = -1;

	bool vertical;
	derivative_t derDisplay = no_der;

private:
	/********************************************************************************************/
	/* Function to interface the CPU with the GPU:												*/
	/********************************************************************************************/

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Functions to Initialize the GPU and set up the reconstruction normalization
	TomoError initGPU(const char * gainFile, const char * darkFile, const char * mainFile);
	TomoError setNOOP(float kernel[KERNELSIZE]);
	TomoError setGauss(float kernel[KERNELSIZE]);
	TomoError setGaussDer(float kernel[KERNELSIZE]);
	TomoError setGaussDer2(float kernel[KERNELSIZE]);
	TomoError setGaussDer3(float kernel[KERNELSIZE]);

	//Kernel call helpers
	float focusHelper();
	TomoError imageKernel(float xK[KERNELSIZE], float yK[KERNELSIZE], float * output);

	//Coordinate conversions
	TomoError P2R(int* rX, int* rY, int pX, int pY, int view);
	TomoError R2P(int* pX, int* pY, int rX, int rY, int view);
	TomoError I2D(int* dX, int* dY, int iX, int iY);

	size_t avail_mem;
	size_t total_mem;

	/********************************************************************************************/
	/* Input Functions to read data into the program											*/
	/********************************************************************************************/
	BOOL CheckFilePathForRepeatScans(std::string BasePathIn);
	int GetNumberOfScans(std::string BasePathIn);
	int GetNumOfProjectionsPerView(std::string BasePathIn);
	int GetNumProjectionViews(std::string BasePathIn);
	TomoError ReadDarkImages(const char * darkFile);
	TomoError ReadGainImages(const char * gainFile);

	//Define data buffer
	unsigned short * d_Proj;
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
	float * xDer;

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

	//Cuda constants
	params constants;

	cudaStream_t stream;

	//cuda pitch variables generated from 2d mallocs
	size_t reconPitch;
	size_t projPitch;
	int reconPitchNum;

	//Parameters for 2d geometry search
	int diffSlice = 0;
};

/********************************************************************************************/
/* Library assertions																		*/
/********************************************************************************************/
TomoError cuda_assert(const cudaError_t code, const char* const file, const int line);
TomoError cuda_assert_void(const char* const file, const int line);

#define cudav(...)  cuda##__VA_ARGS__; cuda_assert_void(__FILE__, __LINE__);
#define cuda(...)  cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__);

#endif
