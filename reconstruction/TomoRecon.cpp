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

TomoError TomoRecon::setGauss(float kernel[KERNELSIZE]) {
	float factor = 1 / (sqrt(2 * M_PI) * SIGMA);
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = factor * exp(-pow(i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}

TomoError TomoRecon::setGaussDer(float kernel[KERNELSIZE]) {
	float factor = 1 / (sqrt(2 * M_PI) * pow(SIGMA,3));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = -i * factor * exp(-pow(i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}

TomoError TomoRecon::setGaussDer2(float kernel[KERNELSIZE]) {
	float factor1 = 1 / (sqrt(2 * M_PI) * pow(SIGMA, 3));
	float factor2 = 1 / (sqrt(2 * M_PI) * pow(SIGMA, 5));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = (pow(i,2) * factor2 - factor1) * exp(-pow(i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}

TomoError TomoRecon::setGaussDer3(float kernel[KERNELSIZE]) {
	float factor1 = 1 / pow(SIGMA, 2);
	float factor2 = 1 / (sqrt(2 * M_PI) * pow(SIGMA, 5));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = i * factor2*(3 - pow(i, 2)*factor1) * exp(-pow(i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}