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

TomoRecon::TomoRecon(int x, int y, struct SystemControl * Sys) : interop(x, y), Sys(Sys){
	cuda(StreamCreate(&stream));
}

TomoRecon::~TomoRecon() {
	cuda(StreamDestroy(stream));
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

TomoError TomoRecon::init(const char * gainFile, const char * darkFile, const char * mainFile) {
	//Seperate base path from the example file path
	char GetFilePath[MAX_PATH];
	std::string BasePath;
	std::string FilePath;
	std::string FileName;
	savefilename = mainFile;
	strcpy(GetFilePath, mainFile);
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
	NumViews = NUMVIEWS;

	//Step 3. Read the normalizaton data (dark and gain)
	PathRemoveFileSpec(GetFilePath);
	std::string GainPath = GetFilePath;
	//	ReadDarkandGainImages(Sys, NumViews, GainPath);
	tomo_err_throw(ReadDarkImages(darkFile));
	tomo_err_throw(ReadGainImages(gainFile));

	//Step 5. Read Raw Data
	tomo_err_throw(ReadRawProjectionData(FilePath, savefilename));

	//Step 4. Set up the GPU for Reconstruction
	tomo_err_throw(SetUpGPUForRecon());
	std::cout << "GPU Ready" << std::endl;

	initialized = true;

	zoom = 0;
	xOff = 0;
	yOff = 0;
}

TomoError TomoRecon::TomoSave() {
	std::string whichsubstr = "AcquiredImage";

	std::size_t pos = savefilename.find(whichsubstr);

	std::string str2 = savefilename.substr(0, pos + whichsubstr.length() + 1);
	str2 += "_Recon.dcm";

	//Copy the reconstructed images to the CPU
	tomo_err_throw(CopyAndSaveImages());

	//Save Images
	tomo_err_throw(SaveDataAsDICOM(str2));
}