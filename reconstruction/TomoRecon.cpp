/********************************************************************************************/
/* TomoRecon.cpp																			*/
/* Copyright 2016, XinRay Inc., All rights reserved											*/
/********************************************************************************************/

/********************************************************************************************/
/* Version: 1.2																				*/
/* Date: October 27, 2016																	*/
/* Author: Brian Gonzales																	*/
/* Project: TomoD IntraOral Tomosynthesis													*/
/********************************************************************************************/

/********************************************************************************************/
/* Include a general header																	*/
/********************************************************************************************/
#include "TomoRecon.h"

/********************************************************************************************/
/* Constructor and destructor																*/
/********************************************************************************************/

TomoRecon::TomoRecon(int *argc, char **argv, int x, int y, bool first) : interop(argc, argv, x, y, first){
}

TomoRecon::~TomoRecon() {
	FreeGPUMemory();
	if (Sys->Proj->RawDataThresh != NULL)
		delete[] Sys->Proj->RawDataThresh;
	delete[] Sys->Proj->RawData;
	delete[] Sys->Proj->SyntData;
	delete[] Sys->SysGeo.EmitX;
	delete[] Sys->SysGeo.EmitY;
	delete[] Sys->SysGeo.EmitZ;
	delete[] Sys->Norm->DarkData;
	delete[] Sys->Norm->GainData;
	delete[] Sys->Norm->ProjBuf;
	delete[] Sys->Norm->CorrBuf;
	delete[] Sys->Recon->ReconIm;
	//delete Sys->Proj;
	delete Sys;
}

TomoError TomoRecon::init() {
	//Step 1: Get and example file for get the path
#ifdef PROFILER
	char filename[] = "C:\\Users\\jdean\\Desktop\\Patient471\\Series1 20161118\\AcquiredImage1_0.raw";
#else
	char filename[MAX_PATH];

	OPENFILENAME ofn;
	ZeroMemory(&filename, sizeof(filename));
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
	ofn.lpstrFilter = "Raw File\0*.raw\0Any File\0*.*\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = "Select one raw image file";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	GetOpenFileNameA(&ofn);
#endif

	//Seperate base path from the example file path
	char * GetFilePath;
	std::string BasePath;
	std::string FilePath;
	std::string FileName;
	std::string savefilename = filename;
	GetFilePath = filename;
	PathRemoveFileSpec(GetFilePath);
	FileName = PathFindFileName(GetFilePath);
	FilePath = GetFilePath;
	PathRemoveFileSpec(GetFilePath);

	//Define Base Path
	BasePath = GetFilePath;
	if (CheckFilePathForRepeatScans(BasePath)) {
		FileName = PathFindFileName(GetFilePath);
		FilePath = GetFilePath;
		PathRemoveFileSpec(GetFilePath);
		BasePath = GetFilePath;
	}

	//Output FilePaths
	std::cout << "Reconstructing image set entitled: " << FileName << std::endl;

	//Step 2. Initialize structure and read emitter geometry
	const int NumViews = NUMVIEWS;
	Sys = new SystemControl;
	tomo_err_throw(SetUpSystemAndReadGeometry(Sys, NumViews, BasePath));

	//Step 3. Read the normalizaton data (dark and gain)
	PathRemoveFileSpec(GetFilePath);
	std::string GainPath = GetFilePath;
	//	ReadDarkandGainImages(Sys, NumViews, GainPath);
	tomo_err_throw(ReadDarkImages(Sys, NumViews));
	tomo_err_throw(ReadGainImages(Sys, NumViews));

	//Step 5. Read Raw Data
	tomo_err_throw(ReadRawProjectionData(Sys, NumViews, FilePath, savefilename));

	//Step 4. Set up the GPU for Reconstruction
	tomo_err_throw(SetUpGPUForRecon(Sys));
	std::cout << "GPU Ready" << std::endl;

	

	LoadProjections(0);
	initialized = true;
}

/********************************************************************************************/
/* use main right now to operate from command line											*/
/********************************************************************************************/
TomoError TomoRecon::TomoMain(){
	//Step 1: Get and example file for get the path
#ifdef PROFILER
	char filename[] = "C:\\Users\\jdean\\Desktop\\Patient471\\Series1 20161118\\AcquiredImage1_0.raw";
#else
	char filename[MAX_PATH];

	OPENFILENAME ofn;
	ZeroMemory(&filename, sizeof(filename));
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
	ofn.lpstrFilter = "Raw File\0*.raw\0Any File\0*.*\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = "Select one raw image file";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	GetOpenFileNameA(&ofn);
#endif

	//Seperate base path from the example file path
	char * GetFilePath;
	std::string BasePath; 
	std::string FilePath; 
	std::string FileName;
	std::string savefilename = filename;
	GetFilePath = filename;
	PathRemoveFileSpec(GetFilePath);
	FileName = PathFindFileName(GetFilePath);
	FilePath = GetFilePath;
	PathRemoveFileSpec(GetFilePath);

	//Define Base Path
	BasePath = GetFilePath;
	if (CheckFilePathForRepeatScans(BasePath)) {
		FileName = PathFindFileName(GetFilePath);
		FilePath = GetFilePath;
		PathRemoveFileSpec(GetFilePath);
		BasePath = GetFilePath;
	}

	//Output FilePaths
	std::cout << "Reconstructing image set entitled: "<< FileName << std::endl;

	//Step 2. Initialize structure and read emitter geometry
	const int NumViews = NUMVIEWS;
	struct SystemControl * Sys = new SystemControl;
	tomo_err_throw(SetUpSystemAndReadGeometry(Sys, NumViews,BasePath));

	//Step 3. Read the normalizaton data (dark and gain)
	PathRemoveFileSpec(GetFilePath);
	std::string GainPath = GetFilePath;
//	ReadDarkandGainImages(Sys, NumViews, GainPath);
	tomo_err_throw(ReadDarkImages(Sys, NumViews));
	tomo_err_throw(ReadGainImages(Sys, NumViews));

	//Step 4. Set up the GPU for Reconstruction
	tomo_err_throw(SetUpGPUForRecon(Sys));
	std::cout << "GPU Ready" << std::endl;

	//Step 5. Read Raw Data
	tomo_err_throw(ReadRawProjectionData(Sys, NumViews,FilePath,savefilename));
	std::cout << "Add Data has been read" << std::endl;
	
	//Step 6: Reconstruct Images
	tomo_err_throw(Reconstruct(Sys));

	std::string whichsubstr = "AcquiredImage";

	std::size_t pos = savefilename.find(whichsubstr);
	 
	std::string str2 = savefilename.substr(0, pos + whichsubstr.length() + 1);
	str2 += "_Recon.dcm";

	//Step 7: SaVE Images
	tomo_err_throw(SaveDataAsDICOM(Sys, str2));
//	SaveDataAsDICOM(Sys, BasePath);
//	SaveCorrectedProjections(Sys, BasePath);

	//Step 9. Clear memory and end program
	tomo_err_throw(FreeGPUMemory());
	if (Sys->Proj->RawDataThresh != NULL)
		delete[] Sys->Proj->RawDataThresh;
	delete[] Sys->Proj->RawData;
	delete[] Sys->Proj->SyntData;
	delete[] Sys->SysGeo.EmitX;
	delete[] Sys->SysGeo.EmitY;
	delete[] Sys->SysGeo.EmitZ;
	delete[] Sys->Norm->DarkData;
	delete[] Sys->Norm->GainData;
	delete[] Sys->Norm->ProjBuf;
	delete[] Sys->Norm->CorrBuf;
	delete[] Sys->Recon->ReconIm;
	//delete Sys->Proj;
	delete Sys;
	
	return Tomo_OK;
}