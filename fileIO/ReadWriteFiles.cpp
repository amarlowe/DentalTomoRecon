/********************************************************************************************/
/* ReadWriteFiles.cpp																		*/
/* Copyright 2016, XinRay Inc., All Rights Reserved											*/
/********************************************************************************************/

/********************************************************************************************/
/* Version: 1.3																				*/
/* Date: October 27, 2016																	*/
/* Author: Brian Gonzales																	*/
/* Project: TomoD IntraOral Tomosynthesis													*/
/********************************************************************************************/

/********************************************************************************************/
/* Change of path to new data																*/
/********************************************************************************************/

#include "../reconstruction/TomoRecon.h"
#include <filesystem>

/********************************************************************************************/
/* Functions to get the size and location of the data										*/
/********************************************************************************************/
//Simple function to ensure the file name is valid
bool CheckFileName(std::string Name)
{
	bool ValidName = false;
	if (Name != "." && Name != ".." && Name != "blank"
		&& Name != "dark.dcm" && Name != "Source_Sequence.txt" && Name != "AVG") {
		ValidName = true;
	}
	return (ValidName);
}

bool CheckDirName(std::string Name)
{
	bool ValidName = false;
	std::string endstr;
	endstr = Name;
	endstr.erase(0, endstr.length() - 4);
	if (endstr != ".raw" && Name != "." && Name != ".." && Name != "blank"
		&& Name != "dark.dcm" && Name != "Source_Sequence.txt" && Name != "AVG") {
		ValidName = true;
	}
	return (ValidName);
}
/*
BOOL TomoRecon::CheckFilePathForRepeatScans(std::string BasePathIn) {
	std::string BasePath;
	BasePath = BasePathIn + "/*";
	WIN32_FIND_DATA FindFile;
	HANDLE hfind;
	bool MoreViews = true;
	bool RepeatScans = FALSE;

	hfind = FindFirstFile(BasePath.c_str(), &FindFile);
	if (hfind != INVALID_HANDLE_VALUE) {
		if (CheckFileName(std::string(FindFile.cFileName))) {
			if (isdigit(FindFile.cFileName[0])) {
				RepeatScans = TRUE;
			}
		}
	}
	while (MoreViews) {
		if (FindNextFile(hfind, &FindFile)) {
			if (CheckFileName(std::string(FindFile.cFileName))) {
				if (isdigit(FindFile.cFileName[0])) {
					RepeatScans = TRUE;
					MoreViews = false;
					break;
				}
			}
		}
		else {
			MoreViews = false;
			break;
		}
	}
	return (RepeatScans);
}


//Simple Function to count the number example images
int TomoRecon::GetNumberOfScans(std::string BasePathIn) {
	int NumScans = 0;
	bool MoreViews = true;

	std::string BasePath;
	BasePath = BasePathIn + "/*";
	WIN32_FIND_DATA FindFile;
	HANDLE hfind;

	hfind = FindFirstFile(BasePath.c_str(), &FindFile);
	if (hfind != INVALID_HANDLE_VALUE) {
		if (CheckDirName(std::string(FindFile.cFileName))) NumScans++;
	}
	while (MoreViews) {
		if (FindNextFile(hfind, &FindFile)) {
			if (CheckDirName(std::string(FindFile.cFileName))) NumScans++;
		}
		else {
			MoreViews = false;
			break;
		}
	}

	return (NumScans);
}

//Function to determine how many files exist to be averaged together
int TomoRecon::GetNumOfProjectionsPerView(std::string BasePathIn) {
	int NumProjections = 0;
	bool MoreFolders = true;
	bool MoreProjections = true;
	std::string BasePath;
	BasePath = BasePathIn + "/";
	std::string FilePath = BasePath + "*";

	WIN32_FIND_DATA FindFile, FindScan;
	HANDLE hfind, hfindSub;

	hfind = FindFirstFile(FilePath.c_str(), &FindFile);

	while (MoreFolders) {
		if (FindNextFile(hfind, &FindFile)) {
			if (CheckFileName(std::string(FindFile.cFileName)))
			{
				BasePath += std::string(FindFile.cFileName) + "/*";
				hfindSub = FindFirstFile(BasePath.c_str(), &FindScan);
				while (MoreProjections) {
					if (FindNextFile(hfindSub, &FindScan)) {
						if (CheckFileName(std::string(FindScan.cFileName)))
						{
							NumProjections++;
						}
					}
					else {
						MoreProjections = false;
						break;
					}
				}
				MoreFolders = false;
				break;
			}
		}
		else {
			MoreFolders = false;
			break;
		}
	}

	return (NumProjections);
}

//Get the number of views to use for the reconstruction
int TomoRecon::GetNumProjectionViews(std::string BasePathIn) {
	int NumViews = 0;
	bool MoreFolders = true;
	bool MoreProjections = true;
	bool MoreViews = true;
	std::string BasePath;
	BasePath = BasePathIn + "/";
	std::string FilePath = BasePath + "*";

	WIN32_FIND_DATA FindFile, FindScan, FindView;
	HANDLE hfind, hfindSub, hfindView;

	hfind = FindFirstFile(FilePath.c_str(), &FindFile);

	while (MoreFolders) {
		if (FindNextFile(hfind, &FindFile)) {
			if (CheckFileName(std::string(FindFile.cFileName)))
			{
				BasePath += std::string(FindFile.cFileName) + "/";
				std::string ScanPath = BasePath + "*";
				hfindSub = FindFirstFile(ScanPath.c_str(), &FindScan);
				while (MoreProjections) {
					if (FindNextFile(hfindSub, &FindScan)) {
						if (CheckFileName(std::string(FindScan.cFileName)))
						{
							BasePath += std::string(FindScan.cFileName) + "/*";
							hfindView = FindFirstFile(BasePath.c_str(), &FindView);
							while (MoreViews) {
								if (FindNextFile(hfindView, &FindView)) {
									if (CheckFileName(std::string(FindView.cFileName)))
									{
										NumViews++;
									}
								}
								else {
									MoreViews = false;
									break;
								}
							}
						}
					}
					else {
						MoreProjections = false;
						break;
					}
				}
				MoreFolders = false;
				break;
			}
		}
		else {
			MoreFolders = false;
			break;
		}
	}
	return (NumViews);
}

//Read a subset of the total views
TomoError ReadSubSetViews(struct SystemControl * Sys, int NumViews, std::string BasePathIn) {
	std::string FilePath = BasePathIn + "/Source_Sequence.txt";

	//Open fstream to text file
	std::ifstream file(FilePath.c_str());

	if (!file.is_open()) {
		std::cout << "Error opening file: " << FilePath.c_str() << std::endl;
		std::cout << "Please check and re-run program." << std::endl;
		return Tomo_file_err;
	}

	//Define two character arrays to read values
	char data[1024], data_in[4];

	//skip the first line
	file.getline(data, 1024);
	int vnum = 0;
	int * ViewNum = new int[NumViews];
	int * ProjNum = new int[NumViews];

	for (int view = 0; view < NumViews; view++)
	{
		file.getline(data, 1024);
		int count = 0, num = 0;

		//Read first colomun: View Number	
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' &&  data[count] != '\n' && num < 4);

		ViewNum[vnum] = atoi(data_in);
		for (int i = 0; i < 4; i++) data_in[i] = '\0';
		count++; num = 0;

		//Read Second column: Projection Number
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' &&  data[count] != '\n' && num < 4);

		ProjNum[vnum] = atoi(data_in);
		for (int i = 0; i < 4; i++) data_in[i] = '\0';
		count++; num = 0;

		vnum++;
		if (file.eof()) break;
	}

	Sys->Proj.NumViews = vnum;

	delete[] ViewNum;
	delete[] ProjNum;

	return Tomo_OK;
}

TomoError TomoRecon::GetGainAverages(const char * gainFile) {
	int size_single_proj = Sys->Proj.Nx * Sys->Proj.Ny;
	int size_single_proj_bytes = size_single_proj * sizeof(unsigned short);

	char temp[MAX_PATH];
	strncpy(temp, gainFile, MAX_PATH - 1);
	PathRemoveFileSpec(temp);
	std::string GainPath = temp;

	std::string avgFilePath;

	FILE* fileptr = NULL;

	std::string gainSearchPath;

	avgFilePath = GainPath;

	int size_gain_buf = size_single_proj * NumViews;

	//Allocate memory to each buffer
	unsigned short * GainData = new unsigned short[size_gain_buf];

	std::string Path;
	std::string tempPath;
	if (!constants.orientation) {
		Path = avgFilePath + "\\right\\average_";
		gainSearchPath = GainPath + "\\right";
	}
	else {
		Path = avgFilePath + "\\left\\average_";
		gainSearchPath = GainPath + "\\left";
	}

	int NumGainSamples = GetNumberOfScans(gainSearchPath);

	int size_gain_buf_temp = size_single_proj * NumViews * NumGainSamples;

	USHORT* GainBuf = new USHORT[size_gain_buf_temp];

	gainSearchPath = gainSearchPath + "\\*";

	//Set up the find files variables
	WIN32_FIND_DATA FindFile;
	HANDLE hfind;

	std::string FileName;
	//Cycle through the number of samples and read the blank images
	WIN32_FIND_DATA FindDir;
	int NumProjections = 0;
	HANDLE hfindDir = FindFirstFile(gainSearchPath.c_str(), &FindDir);
	bool MoreFolders = true;
	std::string ProjPath;

	while (MoreFolders) {
		if (FindNextFile(hfindDir, &FindDir)) {
			if (CheckFileName(FindDir.cFileName)) {
				ProjPath = gainSearchPath;
				FileName = FindDir.cFileName;
				ProjPath.replace(ProjPath.length() - 1, 1, FileName.c_str(), FileName.length());
				ProjPath += "/*.raw";

				hfind = FindFirstFile(ProjPath.c_str(), &FindFile);
				if (hfind != INVALID_HANDLE_VALUE) {
					tempPath = ProjPath;
					FileName = FindFile.cFileName;
					tempPath.replace(tempPath.length() - 5, 5, FileName.c_str(), FileName.length());
					fopen_s(&fileptr, tempPath.c_str(), "rb");

					if (fileptr == NULL)
						return Tomo_file_err;

					fread(GainBuf + NumProjections*size_single_proj*NumViews,
						sizeof(USHORT), size_single_proj, fileptr);
					fclose(fileptr);
				}
				else continue;

				for (int view = 1; view < NumViews; view++) {
					if (FindNextFile(hfind, &FindFile)) {
						tempPath = ProjPath;
						FileName = FindFile.cFileName;
						tempPath.replace(tempPath.length() - 5, 5, FileName.c_str(), FileName.length());

						fopen_s(&fileptr, tempPath.c_str(), "rb");
						if (fileptr == NULL)
							return Tomo_file_err;

						fread(GainBuf + NumProjections*size_single_proj*NumViews + view * size_single_proj,
							sizeof(unsigned short), size_single_proj, fileptr);
						fclose(fileptr);
					}
					else
						return Tomo_file_err;
				}
				NumProjections++;
			}
		}

		if (NumProjections >= NumGainSamples) {
			MoreFolders = false;
			break;
		}
	}

	float GainPro = 0.0;
	//Average the gain and blank images
	for (int i = 0; i < Sys->Proj.Nx; i++) {
		for (int j = 0; j < Sys->Proj.Ny; j++) {
			for (int view = 0; view < NumViews; view++) {
				GainPro = 0.0;
				for (int n = 0; n < NumGainSamples; n++) {
					GainPro +=
						(float)(GainBuf[n*size_single_proj*NumViews + view*size_single_proj + i + j*Sys->Proj.Nx]) /
						((float)NumGainSamples);
				}

				GainData[i + j*Sys->Proj.Nx + view*size_single_proj] = (unsigned short)GainPro;
			}
		}
	}

	//write average file
	for (int view = 0; view < NumViews; view++) {
		tempPath = Path + std::to_string(view) + ".raw";
		fopen_s(&fileptr, tempPath.c_str(), "wb");

		if (fileptr == NULL)
			return Tomo_file_err;

		fwrite(GainData + view * size_single_proj, sizeof(USHORT), size_single_proj, fileptr);
		fclose(fileptr);
	}

	delete GainBuf;

	delete[] GainData;

	return Tomo_OK;
}
*/