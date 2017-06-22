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
/* use main right now to operate from command line											*/
/********************************************************************************************/
int main(int argc, char ** argv)
{
	//Step 1: Get and example file for get the path
	char filename[] = "C:\\Users\\jdean\\Desktop\\Patient471\\Series1 20161118\\AcquiredImage1_0.raw";
	/*char filename[MAX_PATH];

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

	GetOpenFileNameA(&ofn);*/

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
	SetUpSystemAndReadGeometry(Sys, NumViews,BasePath);

	//Step 3. Read the normalizaton data (dark and gain)
	PathRemoveFileSpec(GetFilePath);
	std::string GainPath = GetFilePath;
//	ReadDarkandGainImages(Sys, NumViews, GainPath);
	ReadDarkImages(Sys, NumViews);
	ReadGainImages(Sys, NumViews);

	//Step 4. Set up the GPU for Reconstruction
	SetUpGPUForRecon(Sys);
	std::cout << "GPU Ready" << std::endl;

	//Step 5. Read Raw Data
	ReadRawProjectionData(Sys, NumViews,FilePath,savefilename);
	std::cout << "Add Data has been read" << std::endl;
	
	//Step 6: Reconstruct Images
	Reconstruct(Sys);

	std::string whichsubstr = "AcquiredImage";

	std::size_t pos = savefilename.find(whichsubstr);
	 
	std::string str2 = savefilename.substr(0, pos + whichsubstr.length() + 1);
	str2 += "_Recon.dcm";

	//Step 7: SaVE Images
	SaveDataAsDICOM(Sys, str2);
//	SaveDataAsDICOM(Sys, BasePath);
//	SaveCorrectedProjections(Sys, BasePath);

	//Step 9. Clear memory and end program
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
	
	return (0);
}