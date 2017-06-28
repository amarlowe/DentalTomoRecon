/********************************************************************************************/
/* Recon.h																					*/
/* Copyright 2016, XinRay Inc., All Rights Reserved											*/
/********************************************************************************************/

/********************************************************************************************/
/* General C and C++ libraries included in the program										*/
/********************************************************************************************/
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

#pragma comment(lib, "Shlwapi.lib")

/********************************************************************************************/
/* Filepaths																				*/
/********************************************************************************************/
#define GEOMETRYFILE	"C:\\Users\\jdean\\Desktop\\recon_files_test\\geometry_files\\FocalSpotGeometry.txt"
#define GAINFILE	"C:\\Users\\jdean\\Desktop\\recon_files_test\\calibration_files\\Blank"
#define DARKFILE	"C:\\Users\\jdean\\Desktop\\recon_files_test\\calibration_files\\Dark"
//#define GEOMETRYFILE	"C:\\Users\\jdean\\Google Drive\\software\\Xinvivo_software\\Recon\\recon_files\\geometry_files\\FocalSpotGeometry.txt"
//#define GAINFILE	"C:\\Users\\jdean\\Google Drive\\software\\Xinvivo_software\\Recon\recon_files\\calibration_files\\Blank"
//#define DARKFILE	"C:\\Users\\jdean\\Google Drive\\software\\Xinvivo_software\\Recon\\recon_files\\calibration_files\\Dark"

#define NUMVIEWS 7

//#define PROFILER


/********************************************************************************************/
/* System Structures (All units are in millimeters)											*/
/********************************************************************************************/
struct Proj_Data
{
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

struct SysGeometry
{
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


struct NormData
{
	unsigned short * GainData;				//A buffer to contain the gain images
	unsigned short * DarkData;				//A buffer to contain the dark images
	unsigned short * ProjBuf;				//A tempory buffer to read projections into
	float * CorrBuf;						//A buffer to correct the and sum projections
};

struct FileNames
{
	std::string StudyName;					//The name of the entire study folder
	std::string ScanName;					//The name of the scans in the folder 
};

struct ReconGeometry
{
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

struct CommandLine
{
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

struct UserInput
{
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
struct SystemControl
{
	struct Proj_Data * Proj;
	struct NormData * Norm;
	struct ReconGeometry * Recon;
	struct SysGeometry SysGeo;
	struct CommandLine * CmdLine;
	struct FileNames * Name;
	struct UserInput * UsrIn;

};

/********************************************************************************************/
/* Input Functions to read data into the program											*/
/********************************************************************************************/
BOOL CheckFilePathForRepeatScans(std::string BasePathIn);

int GetNumberOfScans(std::string BasePathIn);

int GetNumOfProjectionsPerView(std::string BasePathIn);

int GetNumProjectionViews(std::string BasePathIn);

void SetUpSystemAndReadGeometry(struct SystemControl * Sys, int NumViews,
	 std::string BasePathIn);

void ReadDarkandGainImages(struct SystemControl * Sys, int NumViews, std::string BasePathIn);

void ReadDarkImages(struct SystemControl * Sys, int NumViews);
void ReadGainImages(struct SystemControl * Sys, int NumViews);

void ReadRawProjectionData(struct SystemControl * Sys,
			int NumViews, std::string BaseFileIn, std::string FileName);

/********************************************************************************************/
/* Function to interact with GPU portion of the program to reconstruct the image			*/
/********************************************************************************************/
void SetUpGPUForRecon(struct SystemControl * Sys);

void Reconstruct(struct SystemControl * Sys);

void FreeGPUMemory(void);

/********************************************************************************************/
/* DICOM functions																			*/
/********************************************************************************************/
void SaveDataAsDICOM(struct SystemControl * Sys, std::string BaseFileIn);

void SaveCorrectedProjections(struct SystemControl * Sys, std::string BaseFileIn);

void SaveSyntheticProjections(struct SystemControl * Sys,
	int PhantomNum, std::string BaseFileIn);
