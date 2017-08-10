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
	delete[] Sys->Proj->RawData;
	delete[] Sys->Geo.EmitX;
	delete[] Sys->Geo.EmitY;
	delete[] Sys->Geo.EmitZ;
	delete Sys->Recon;
	delete Sys->Proj;
	delete Sys;
}

TomoError TomoRecon::init(const char * gainFile, const char * darkFile, const char * mainFile) {
	//Seperate base path from the example file path
	char GetFilePath[MAX_PATH];
	std::string BasePath;
	std::string FilePath;
	std::string FileName;
	strcpy_s(GetFilePath, mainFile);
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
	tomo_err_throw(ReadRawProjectionData(FilePath, mainFile));

	//Step 4. Set up the GPU for Reconstruction
	tomo_err_throw(initGPU());
	std::cout << "GPU Ready" << std::endl;

	initialized = true;

	zoom = 0;
	xOff = 0;
	yOff = 0;

	return Tomo_OK;
}

TomoError TomoRecon::TomoLoad(const char* file) {
	char GetFilePath[MAX_PATH];
	std::string BasePath;
	std::string FilePath;
	std::string FileName;
	strcpy_s(GetFilePath, file);
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

	tomo_err_throw(ReadRawProjectionData(FilePath, file));
	correctProjections();
	singleFrame();

	return Tomo_OK;
}

TomoError TomoRecon::setNOOP(float kernel[KERNELSIZE]) {
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		kernel[i + KERNELRADIUS] = 0.0;
	}
	kernel[KERNELRADIUS] = 1.0;

	return Tomo_OK;
}

TomoError TomoRecon::setGauss(float kernel[KERNELSIZE]) {
	float factor = 1.0f / ((float)sqrt(2.0 * M_PI) * SIGMA);
	float denom = 2.0f * pow(SIGMA, 2);
	float sum = 0.0f;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = factor * exp(-pow((float)i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}
	sum--;//Make it sum to 1, not 0

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}

TomoError TomoRecon::setGaussDer(float kernel[KERNELSIZE]) {
	float factor = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA,3));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = -i * factor * exp(-pow((float)i, 2) / denom);
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
	float factor1 = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA, 3));
	float factor2 = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA, 5));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = (pow((float)i,2) * factor2 - factor1) * exp(-pow((float)i, 2) / denom);
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
	float factor2 = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA, 5));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = i * factor2*(3 - pow((float)i, 2)*factor1) * exp(-pow((float)i, 2) / denom);
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