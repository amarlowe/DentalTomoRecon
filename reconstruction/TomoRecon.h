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
#include <math.h>
#include <ctime>
#include <crtdbg.h>
#include "Shlwapi.h"

#include <Windows.h>
#include <WinBase.h>

#include "../UI/interop.h"

#pragma comment(lib, "Shlwapi.lib")

#define NUMVIEWS 7
#define PROFILER
#define ARRAY_SIZE 10
#define STRIDE 4

//Macro for checking cuda errors following a cuda launch or api call
#define voidChkErr(...) {											\
	(__VA_ARGS__);														\
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
	Tomo_CUDA_err
} TomoError;

typedef enum {
	raw_images,
	sino_images,
	raw_images2,
	norm_images,
	recon_images
} display_t;

typedef enum {

};

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
	unsigned short * RawDataThresh = NULL;	//Pointed to a buffer containing the raw data thresholded to eliminate metal for finding center of tooth more accurately
	unsigned short * SyntData;				//Pointer to buffer containing synthetic projection
	int * Views;							//Pointer to the view numbers
	int NumViews;							//The number of projection views the recon uses
	float Pitch_x;							//The detector pixel pitch in the x direction
	float Pitch_y;							//The detector pixel pithc in the y direction
	int Nx;									//The number of detector pixels in x direction
	int Ny;									//The number of detector pixels in y direction
	int Nz;									//The number of detector pixels in z direction
	int Mean;								//The display window mean
	int Width;								//The display window width
	int Flip;								//Flip the orientation of the detector
};

struct SysGeometry {
	float * EmitX;							//Location of the x-ray focal spot in x direction
	float * EmitY;							//Location of the x-ray focal spot in y direction
	float * EmitZ;							//Location of the x-ray focal spot in z direction
	float IsoX;								//Location of the system isocenter in x direction
	float IsoY;								//Location of the system isocenter in y direction
	float IsoZ;								//Location of the system isocenter in z direction
	float ZDist;							//Estimaged Distance to center of teeth
	float ZPitch;							//The distance between slices
	std::string Name;						//A name of the emitter file
};


struct NormData {
	unsigned short * GainData;				//A buffer to contain the gain images
	unsigned short * DarkData;				//A buffer to contain the dark images
	unsigned short * ProjBuf;				//A tempory buffer to read projections into
	float * CorrBuf;						//A buffer to correct the and sum projections
};

struct FileNames {
	std::string StudyName;					//The name of the entire study folder
	std::string ScanName;					//The name of the scans in the folder 
};

struct ReconGeometry {
	unsigned short * ReconIm;				//Pointer to a buffer containing the recon images
											//	float* testval;
	float Slice_0_z;						//Location of the first slice in the z direction
	float Pitch_x;							//Recon Image pixel pitch in the x direction
	float Pitch_y;							//Recon Image pixel pitch in the y direction
	float Pitch_z;							//Recon Image pixel pitch in the z direction
	int Nx;									//The number of recon image pixels in x direction
	int Ny;									//The number of recon image pixels in y direction
	int Nz;									//The number of recon image pixels in z direction
	int Mean;								//The display window mean
	int Width;								//The display window width
	float MaxVal;							//The max reconstruction floating point val
};

struct CommandLine {
	int PhantomNum;							//The phantom number choosen by user (defualt 1)
	int Dose;								//choose the dose (defualt 1 (Pspeed))
	int Sharpen;							//Apply a post processing sharpening filter
	int Bilateral;							//Apply a post processing bilateral filter
	int Normalize;							//Normalize Histogram and set to 2^15 scale
	int Synthetic;							//Create Synthetic projection images

	CommandLine() {
		PhantomNum = 1;
		Dose = 1;
		Sharpen = 0;
		Bilateral = 0;
		Normalize = 0;
		Synthetic = 0;
	}
};

struct UserInput {
	int CalOffset;				//variable to control automated offset calculation
	int SmoothEdge;				//variable to control smoothing of edges
	int UseTV;					//variable to control using TV
	int Orientation;			//left/right orientation, right=0, left=1, left is default

	UserInput() {
		CalOffset = 0;
		SmoothEdge = 0;
		UseTV = 0;
		Orientation = 1;
	}

};

//Declare a structure containing other system description structures to pass info
struct SystemControl {
	struct Proj_Data * Proj;
	struct NormData * Norm;
	struct ReconGeometry * Recon;
	struct SysGeometry SysGeo;
	struct CommandLine * CmdLine;
	struct FileNames * Name;
	struct UserInput * UsrIn;

};

//Define a number of constants
struct params {
	int Px;
	int Py;
	int Nx;
	int Ny;
	int Nz;
	int MPx;
	int MPy;
	int MNx;
	int MNy;
	float HalfPx;
	float HalfPy;
	float HalfNx;
	float HalfNy;
	int Views;
	float PitchPx;
	float PitchPy;
	float PitchNx;
	float PitchNy;
	float PitchNz;
	float alpharelax;
	float rmax;
	int Z_Offset;
};

class TomoRecon : public interop {
public:
	//Functions
	//////////////////////////////////////////////////////////////////////////////////////////////
	//constructor/destructor
	TomoRecon(int x, int y, struct SystemControl * Sys);
	~TomoRecon();

	TomoError init(const char * gainFile, const char * darkFile);

	//////////////////////////////////////////////////////////////////////////////////////////////
	//High level functions for command line call
	TomoError TomoSave();
	TomoError SetUpGPUForRecon();
	TomoError Reconstruct();
	TomoError FreeGPUMemory(void);

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Lower level functions for user interactive debugging
	TomoError LoadProjections(int index);
	TomoError correctProjections();
	TomoError reconInit();
	TomoError reconStep();

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

private:
	/********************************************************************************************/
	/* Function to interface the CPU with the GPU:												*/
	/********************************************************************************************/

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Functions to Initialize the GPU and set up the reconstruction normalization
	void DefineReconstructSpace();
	TomoError SetUpGPUMemory();
	TomoError GetReconNorm();

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Functions called to control the stages of reconstruction
	TomoError LoadAndCorrectProjections();

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Functions to control the SART and TV reconstruction
	TomoError FindSliceOffset();
	TomoError AddTVandTVSquared();
	TomoError ReconUsingSARTandTV();

	//////////////////////////////////////////////////////////////////////////////////////////////
	//Functions to save the images
	TomoError CopyAndSaveImages();
	template<typename T>
	TomoError resizeImage(T* in, int wIn, int hIn, cudaArray_t out, int wOut, int hOut, double maxVar);

	/********************************************************************************************/
	/* Input Functions to read data into the program											*/
	/********************************************************************************************/
	BOOL CheckFilePathForRepeatScans(std::string BasePathIn);
	int GetNumberOfScans(std::string BasePathIn);
	int GetNumOfProjectionsPerView(std::string BasePathIn);
	int GetNumProjectionViews(std::string BasePathIn);
	TomoError ReadDarkandGainImages(char * gainFile, char * darkFile);
	TomoError ReadDarkImages(const char * darkFile);
	TomoError ReadGainImages(const char * gainFile);
	TomoError ReadRawProjectionData(std::string BaseFileIn, std::string FileName);

	/********************************************************************************************/
	/* DICOM functions																			*/
	/********************************************************************************************/
	TomoError SaveDataAsDICOM(std::string BaseFileIn);
	TomoError SaveCorrectedProjections(std::string BaseFileIn);
	TomoError SaveSyntheticProjections(int PhantomNum, std::string BaseFileIn);

	//Define data buffer
	unsigned short * d_Proj;
	float * d_Norm;
	float * d_Image;
	float * d_Image2;
	float * d_GradIm;
	float * d_Error;
	float * d_Sino;
	float * d_Pro;
	float * d_PriorIm;
	float * d_dp;
	float * d_dpp;
	float * d_alpha;
	float * d_DerivGradIm;
	float * d_GradNorm;

	//Decay constant for recon
	float Beta = 1.0f;

	//Define Cuda arrays
	cudaArray * d_Sinogram;

	cudaStream_t stream;

	//cuda pitch variables generated from 2d mallocs
	size_t imagePitch;
	size_t image2Pitch;
	size_t errorPitch;

	std::string savefilename;
};

/********************************************************************************************/
/* Library assertions																		*/
/********************************************************************************************/
TomoError cuda_assert(const cudaError_t code, const char* const file, const int line);
TomoError cuda_assert_void(const char* const file, const int line);

#define cudav(...)  cuda##__VA_ARGS__; cuda_assert_void(__FILE__, __LINE__);
#define cuda(...)  cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__);

#endif
