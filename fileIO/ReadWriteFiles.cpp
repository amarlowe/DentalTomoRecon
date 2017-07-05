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

/********************************************************************************************/
/* Functions to get the size and location of the data										*/
/********************************************************************************************/
//Simple function to ensure the file name is valid
bool CheckFileName(std::string Name)
{
	bool ValidName = false;
	if (Name != "." && Name != ".." && Name != "blank"
		&& Name != "dark.dcm" && Name != "Source_Sequence.txt" && Name!="AVG") {
		ValidName = true;
	}
	return (ValidName);
}

bool CheckDirName(std::string Name)
{
	bool ValidName = false;
	std::string endstr;
	endstr = Name;
	endstr.erase(0,endstr.length() - 4);
	if(endstr != ".raw" && Name != "." && Name != ".." && Name != "blank"
		&& Name != "dark.dcm" && Name != "Source_Sequence.txt" && Name != "AVG") {
		ValidName = true;
	}
	return (ValidName);
}

BOOL CheckFilePathForRepeatScans(std::string BasePathIn)
{
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
int GetNumberOfScans(std::string BasePathIn){
	int NumScans = 0;
	bool MoreViews = true;

	std::string BasePath;
	BasePath = BasePathIn + "/*";
	WIN32_FIND_DATA FindFile;
	HANDLE hfind;

	hfind = FindFirstFile(BasePath.c_str(), &FindFile);
	if (hfind != INVALID_HANDLE_VALUE) {
		if (CheckDirName(std::string(FindFile.cFileName))) NumScans++;
		//if (CheckFileName(std::string(FindFile.cFileName))) NumScans++;
	}
	while (MoreViews) {
		if (FindNextFile(hfind, &FindFile)) {
			if (CheckDirName(std::string(FindFile.cFileName))) NumScans++;
			//if (CheckFileName(std::string(FindFile.cFileName))) {
			//	NumScans++;
		//	}
		}
		else {
			MoreViews = false;
			break;
		}
	}

	return (NumScans);
}

//Function to determine how many files exist to be averaged together
int GetNumOfProjectionsPerView(std::string BasePathIn){
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
int GetNumProjectionViews(std::string BasePathIn){
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
TomoError ReadSubSetViews(struct SystemControl * Sys, int NumViews, std::string BasePathIn){
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

	Sys->Proj->NumViews = vnum;
	Sys->Proj->Views = new int[vnum];
	for (int view = 0; view < vnum; view++) {
		int n = ViewNum[view];
		Sys->Proj->Views[n] = ProjNum[view];
	}

	delete[] ViewNum;
	delete[] ProjNum;

	return Tomo_OK;
}

//Function to read the system geoemtry file
TomoError SetUpSystemAndReadGeometry(struct SystemControl * Sys, int NumViews, std::string BasePathIn){
	//Define a new projection data pointer and define the projection geometry constants
	Sys->Proj = new Proj_Data;
	Sys->UsrIn = new UserInput;

	Sys->Proj->NumViews = NumViews;
	Sys->Proj->Views = new int[NumViews];
	for (int n = 0; n < NumViews; n++) {
		Sys->Proj->Views[n] = n;
	}
	
	//Define new buffers to store the x,y,z locations of the x-ray focal spot array
	Sys->SysGeo.EmitX = new float[Sys->Proj->NumViews];
	Sys->SysGeo.EmitY = new float[Sys->Proj->NumViews];
	Sys->SysGeo.EmitZ = new float[Sys->Proj->NumViews];

	//Set the isocenter to the center of the detector array
	Sys->SysGeo.IsoX = 0;
	Sys->SysGeo.IsoY = 0;
	Sys->SysGeo.IsoZ = 0;
	
	std::string FilePath = GEOMETRYFILE;

	//Open fstream to text file
	std::ifstream file(FilePath.c_str());

	if (!file.is_open()) {
		std::cout << "Error opening file: " << FilePath.c_str() << std::endl;
		std::cout << "Please check and re-run program." << std::endl;
		return Tomo_file_err;
	}

	//Define two character arrays to read values
	char data[1024], data_in[12];

	//skip the first line
	file.getline(data, 1024);
	int vnum = 0;
	bool useview = false;
	int count = 0, num = 0;

	//Cycle through the views and read geometry
	for (int view = 0; view < NumViews; view++)
	{
		file.getline(data, 1024);
		if (view == (int)Sys->Proj->Views[vnum]) useview = true;

		//Read first colomun: Beam Number	
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);

		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count++; num = 0;

		//Read second colomn: emitter x location
		do { data_in[num] = data[count];	count++; num++; } while (data[count] != '\t' && num < 12);

		if (useview == true) Sys->SysGeo.EmitX[vnum] = (float)atof(data_in);

		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count++; num = 0;

		//Read third colomn: emitter y location
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);

		if (useview == true) Sys->SysGeo.EmitY[vnum] = (float)atof(data_in);

		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count++; num = 0;

		//Read fourth colomn: emitter z location
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);

		if (useview == true) Sys->SysGeo.EmitZ[vnum] = (float)atof(data_in);

		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count = 0; num = 0;

		if (useview == true)vnum += 1;
		useview = false;

	}

	//Skip the next 2 lines and read the third to get estimated center of tooth
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);

	Sys->SysGeo.ZDist = (float)atof(data_in);

	//skip the next 2 lines and read third to get slice thickness
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);

	Sys->SysGeo.ZPitch = (float)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	//Read four values defining the detector size
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->Proj->Nx = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count ++; num = 0;

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->Proj->Ny = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count ++; num = 0;

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->Proj->Pitch_x = (float)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count ++; num = 0;

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->Proj->Pitch_y = (float)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third to read number of slices to reconstruct
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->Proj->Nz = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third to see direction of data
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->Proj->Flip = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third to see if automatic offset calculation
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->UsrIn->CalOffset = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	// Skip the next two lines and read the third to see if automatic offset calculation
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->UsrIn->SmoothEdge = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	// Skip the next two lines and read the third to see if use TV reconstruction
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->UsrIn->UseTV = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	// Skip the next two lines and read the third to see if use TV reconstruction
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	Sys->UsrIn->Orientation = (int)atof(data_in);

	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	file.close();

	//Define Final Image Buffers 
	Sys->Proj->RawData = new unsigned short[Sys->Proj->Nx*Sys->Proj->Ny * Sys->Proj->NumViews];
	Sys->Proj->SyntData = new unsigned short[Sys->Proj->Nx*Sys->Proj->Ny];

	if (Sys->UsrIn->CalOffset)
		Sys->Proj->RawDataThresh = new unsigned short[Sys->Proj->Nx*Sys->Proj->Ny * Sys->Proj->NumViews];

	return Tomo_OK;
}

//Functions to read the dark and gain images
TomoError ReadDarkandGainImages(struct SystemControl * Sys, int NumViews, std::string BasePathIn){
	//Define two paths to gain and dark data
	std::string GainPath;
	std::string DarkPath;

	GainPath = BasePathIn;
	DarkPath = BasePathIn;

	GainPath += "/Blank";
	DarkPath += "/Dark";

	int NumDarkSamples = GetNumberOfScans(DarkPath);
	int NumGainSamples = GetNumberOfScans(GainPath);

	GainPath += "/*";
	DarkPath += "/*";

	//Set up the basic path to the raw projection dark and gain data
	FILE * ProjData = NULL;

	//Define the size of the data buffers 
	int size_single_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	int size_dark_buf_temp = size_single_proj * 5 * NumDarkSamples;
	int size_gain_buf_temp = size_single_proj * NumViews * NumGainSamples;
	int size_dark_buf = size_single_proj;
	int size_gain_buf = size_single_proj * NumViews;

	//Allocate memory to each buffer
	Sys->Norm = new NormData();
	Sys->Norm->DarkData = new unsigned short[size_dark_buf];
	Sys->Norm->GainData = new unsigned short[size_gain_buf];
	unsigned short * DarkBuf = new unsigned short[size_dark_buf_temp];
	unsigned short * GainBuf = new unsigned short[size_gain_buf_temp];


	//Set up the find files variables
	WIN32_FIND_DATA FindFile;
	HANDLE hfind;
	bool MoreViews = false;
	std::string Path;

	//Cycle through the number of samples and read the blank images
	WIN32_FIND_DATA FindDarkDir;
	int NumDarkIM = 0;
	HANDLE hfindDarkDir = FindFirstFile(DarkPath.c_str(), &FindDarkDir);
	bool MoreDarkFolders = true;

	//Skip first three files
	while (MoreDarkFolders) {
		if (FindNextFile(hfindDarkDir, &FindDarkDir)) {
			if (CheckFileName(std::string(FindDarkDir.cFileName)))
			{
				std::string DarkIMPath;
				DarkIMPath = DarkPath;
				std::string FileName = FindDarkDir.cFileName;
				DarkIMPath.replace(DarkIMPath.length() - 1, 1, FileName.c_str(), FileName.length());
				DarkIMPath += "/*.raw";

				hfind = FindFirstFile(DarkIMPath.c_str(), &FindFile);
				if (hfind != INVALID_HANDLE_VALUE) {
					Path = DarkIMPath;
					std::string FileName = FindFile.cFileName;

					Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());
					fopen_s(&ProjData, Path.c_str(), "rb");
					if (ProjData == NULL)
					{
						std::cout << "Error opening the file: " << Path.c_str() << std::endl;
						std::cout << "Please check the path and re-run the program." << std::endl;
						return Tomo_file_err;
					}

					//Write the reconstructed data into the predefine memory location
					fread(DarkBuf + NumDarkIM*size_single_proj*5,
						sizeof(unsigned short), size_single_proj, ProjData);
					fclose(ProjData);
				}
				else {
					std::cout << "Error: Dark data path is invalid." << std::endl;
					std::cout << "Make sure dark data is located at: " << DarkIMPath.c_str();
					std::cout << std::endl;
					return Tomo_file_err;
				}

				for (int view = 1; view < 5; view++)
				{
					if (FindNextFile(hfind, &FindFile)) {
						Path = DarkIMPath;
						std::string FileName = FindFile.cFileName;
						Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());

						fopen_s(&ProjData, Path.c_str(), "rb");
						if (ProjData == NULL)
						{
							std::cout << "Vew: " << view << std::endl;
							std::cout << "Error opening the file: " << Path.c_str() << std::endl;
							std::cout << "Please check the path and re-run the program." << std::endl;
							return Tomo_file_err;
						}

						//Write the reconstructed data into the predefine memory location
						fread(DarkBuf + NumDarkIM*size_single_proj*5
							+ view * size_single_proj,sizeof(unsigned short), size_single_proj, ProjData);
						fclose(ProjData);

					}
					else {
						std::cout << "Error: Not enough dark images."<< std::endl;
						return Tomo_file_err;
					}
				}
				NumDarkIM++;
			}
		}
		if (NumDarkIM >= NumDarkSamples) {
			MoreDarkFolders = false;
			break;
		}
	}

	//Cycle through the number of samples and read the blank images
	WIN32_FIND_DATA FindDir;
	int NumProjections = 0;
	HANDLE hfindDir = FindFirstFile(GainPath.c_str(), &FindDir);
	bool MoreFolders = true;

	while (MoreFolders) {
		if (FindNextFile(hfindDir, &FindDir)) {
			if (CheckFileName(std::string(FindDir.cFileName)))
			{
				std::string ProjPath;
				ProjPath = GainPath;
				std::string FileName = FindDir.cFileName;
				ProjPath.replace(ProjPath.length() - 1, 1, FileName.c_str(), FileName.length());
				ProjPath += "/*.raw";

				hfind = FindFirstFile(ProjPath.c_str(), &FindFile);
				if (hfind != INVALID_HANDLE_VALUE) {
					Path = ProjPath;
					std::string FileName = FindFile.cFileName;
					Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());

					fopen_s(&ProjData, Path.c_str(), "rb");
					if (ProjData == NULL)
					{
						std::cout << "Error opening the file: " << Path.c_str() << std::endl;
						std::cout << "Please check the path and re-run the program." << std::endl;
						return Tomo_file_err;
					}

					//Write the reconstructed data into the predefine memory location
					fread(GainBuf + NumProjections*size_single_proj*NumViews,
						sizeof(unsigned short), size_single_proj, ProjData);
					fclose(ProjData);
				}
				else {
					std::cout << "Error: Gain data path is invalid." << std::endl;
					std::cout << "Make sure dark data is located at: " << ProjPath.c_str();
					std::cout << std::endl;
					return Tomo_file_err;
				}
				for (int view = 1; view < NumViews; view++)
				{
					if (FindNextFile(hfind, &FindFile)) {
						Path = ProjPath;
						std::string FileName = FindFile.cFileName;
						Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());

						fopen_s(&ProjData, Path.c_str(), "rb");
						if (ProjData == NULL)
						{
							std::cout << "Error opening the file: " << ProjPath.c_str() << std::endl;
							std::cout << "Please check the path and re-run the program." << std::endl;
							return Tomo_file_err;
						}

						//Write the reconstructed data into the predefine memory location
						fread(GainBuf + NumProjections*size_single_proj*NumViews
							+ view * size_single_proj,
							sizeof(unsigned short), size_single_proj, ProjData);
						fclose(ProjData);

					}
					else {
						std::cout << "Error: Not enough gain images. Need one gain image";
						std::cout << " for each projection view." << std::endl;
						return Tomo_file_err;
					}
				}
				NumProjections++;
			}
		}

		if (NumProjections >= NumGainSamples) {
			MoreFolders = false;
			break;
		}
	}
	//Average the gain and blank images
	for (int i = 0; i < Sys->Proj->Nx; i++) {
		for (int j = 0; j < Sys->Proj->Ny; j++) {
			float Dark = 0;
			for (int n = 0; n < NumDarkSamples; n++) {
				Dark +=	(float)(DarkBuf[n*size_single_proj * 5 + i + j*Sys->Proj->Nx]
						+ DarkBuf[n*size_single_proj * 5 + i + j*Sys->Proj->Nx + size_single_proj]
						+ DarkBuf[n*size_single_proj * 5 + i + j*Sys->Proj->Nx + 2 * size_single_proj]
						+ DarkBuf[n*size_single_proj * 5 + i + j*Sys->Proj->Nx + 3 * size_single_proj]
						+ DarkBuf[n*size_single_proj * 5 + i + j*Sys->Proj->Nx + 4 * size_single_proj])
					/ ((float)5 * (float)NumDarkSamples);
			}
			Sys->Norm->DarkData[i + j*Sys->Proj->Nx] = (unsigned short)Dark;
			for(int view = 0; view < NumViews; view++){
				float GainPro = 0;
				for (int n = 0; n < NumGainSamples; n++) {
					GainPro +=
						(float)(GainBuf[n*size_single_proj*NumViews + view*size_single_proj + i + j*Sys->Proj->Nx]) /
						((float)NumGainSamples);
				}
				Sys->Norm->GainData[i + j*Sys->Proj->Nx + view*size_single_proj] = (unsigned short)GainPro;
			}

		}
	}

	delete GainBuf;
	delete DarkBuf;

	return Tomo_OK;
}

TomoError ReadDarkImages(struct  SystemControl * Sys, int NumViews)
{
	int size_single_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	int size_single_proj_bytes = size_single_proj * 2;

	//Allocate memory to each buffer
	Sys->Norm = new NormData();
	Sys->Norm->DarkData = new unsigned short[size_single_proj];

	//Define paths to dark data
	std::string DarkPath = DARKFILE;
	std::string darkSearchPath = DarkPath + "/*";

	std::string avgFilePath;
	avgFilePath = DarkPath + "\\average.raw";

	int status = 0;

	FILE* fileptr = NULL;

	fopen_s(&fileptr, avgFilePath.c_str(), "rb");

	if (fileptr == NULL)
	{
		std::cout << "Error opening the file: " << avgFilePath.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		status = 1;
		return Tomo_file_err;
	}

	fread(Sys->Norm->DarkData, sizeof(USHORT), size_single_proj, fileptr);

	fclose(fileptr);

	return Tomo_OK;
}

TomoError ReadGainImages(struct  SystemControl * Sys, int NumViews)
{
	int size_single_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	int size_single_proj_bytes = size_single_proj * 2;

	std::string GainPath = GAINFILE;

	std::string avgFilePath;

	FILE* fileptr = NULL;

	std::string gainSearchPath;

	avgFilePath = GainPath;

	int size_gain_buf = size_single_proj * NumViews;

	//Allocate memory to each buffer
	Sys->Norm->GainData = new unsigned short[size_gain_buf];

	bool foundAvgFile = false;

	std::string Path;
	std::string tempfilename;
	//write average file
	for (int view = 0; view < NumViews; view++)
	{
		if (!Sys->UsrIn->Orientation)
			Path = avgFilePath + "\\right\\average_";
		else
			Path = avgFilePath + "\\left\\average_";

		tempfilename = std::to_string(view);
		Path = Path + tempfilename + ".raw";

		fopen_s(&fileptr, Path.c_str(), "rb");
		if (fileptr != NULL)
		{
			foundAvgFile = true;

			fread(&(Sys->Norm->GainData[view*size_single_proj]), sizeof(USHORT), size_single_proj, fileptr);
			fclose(fileptr);
		}
		else
			foundAvgFile = false;
	}

	if (!foundAvgFile)
	{
		if (!Sys->UsrIn->Orientation)
			gainSearchPath = GainPath + "\\right";
		else
			gainSearchPath = GainPath + "\\left";

		int NumGainSamples = GetNumberOfScans(gainSearchPath);

		int size_gain_buf_temp = size_single_proj * NumViews * NumGainSamples;

		USHORT* GainBuf = new USHORT[size_gain_buf_temp];

		gainSearchPath = gainSearchPath + "\\*";

		//Set up the find files variables
		WIN32_FIND_DATA FindFile;
		HANDLE hfind;
		bool MoreViews = false;

		std::string FileName;
		//Cycle through the number of samples and read the blank images
		WIN32_FIND_DATA FindDir;
		int NumProjections = 0;
		HANDLE hfindDir = FindFirstFile(gainSearchPath.c_str(), &FindDir);
		bool MoreFolders = true;
		std::string ProjPath;

		while (MoreFolders) {
			if (FindNextFile(hfindDir, &FindDir)) {
				if (CheckFileName(FindDir.cFileName))
				{
					ProjPath = gainSearchPath;
					FileName = FindDir.cFileName;
					ProjPath.replace(ProjPath.length() - 1, 1, FileName.c_str(), FileName.length());
					ProjPath += "/*.raw";

					hfind = FindFirstFile(ProjPath.c_str(), &FindFile);
					if (hfind != INVALID_HANDLE_VALUE) {
						Path = ProjPath;
						FileName = FindFile.cFileName;
						Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());

						fopen_s(&fileptr, Path.c_str(), "rb");
						if (fileptr == NULL)
						{
							//							status = 1;
							//							return status;
							std::cout << "Error opening the file: " << avgFilePath.c_str() << std::endl;
							std::cout << "Please check the path and re-run the program." << std::endl;
							return Tomo_file_err;
						}

						fread(GainBuf + NumProjections*size_single_proj*NumViews,
							sizeof(USHORT), size_single_proj, fileptr);
						fclose(fileptr);
					}
					else
					{
						//						status = 1;
						//						return status;
						return Tomo_file_err;
					}
					for (int view = 1; view < NumViews; view++)
					{
						if (FindNextFile(hfind, &FindFile)) {
							Path = ProjPath;
							FileName = FindFile.cFileName;
							Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());

							fopen_s(&fileptr, Path.c_str(), "rb");
							if (fileptr == NULL)
							{
								//								status = 1;
								//								return status;
								std::cout << "Error opening the file: " << avgFilePath.c_str() << std::endl;
								std::cout << "Please check the path and re-run the program." << std::endl;
								return Tomo_file_err;
							}

							fread(GainBuf + NumProjections*size_single_proj*NumViews
								+ view * size_single_proj,
								sizeof(unsigned short), size_single_proj, fileptr);
							fclose(fileptr);

						}
						else
						{
							//							status = 1;
							//							return status;
							std::cout << "Error opening the file: " << avgFilePath.c_str() << std::endl;
							std::cout << "Please check the path and re-run the program." << std::endl;
							return Tomo_file_err;
						}
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
		for (int i = 0; i < Sys->Proj->Nx; i++)
		{
			for (int j = 0; j < Sys->Proj->Ny; j++)
			{
				for (int view = 0; view < NumViews; view++)
				{
					GainPro = 0.0;
					for (int n = 0; n < NumGainSamples; n++)
					{
						GainPro +=
							(float)(GainBuf[n*size_single_proj*NumViews + view*size_single_proj + i + j*Sys->Proj->Nx]) /
							((float)NumGainSamples);
					}

					Sys->Norm->GainData[i + j*Sys->Proj->Nx + view*size_single_proj] = (unsigned short)GainPro;
				}
			}
		}

		//write average file
		for (int view = 0; view < NumViews; view++)
		{
			if (!Sys->UsrIn->Orientation)
				Path = avgFilePath + "\\right\\average_";
			else
				Path = avgFilePath + "\\left\\average_";

			tempfilename = std::to_string(view);
			Path = Path + tempfilename + ".raw";

			fopen_s(&fileptr, Path.c_str(), "wb");
			if (fileptr == NULL)
			{
				//				status = 1;
				//				return status;
				std::cout << "Error opening the file: " << avgFilePath.c_str() << std::endl;
				std::cout << "Please check the path and re-run the program." << std::endl;
				return Tomo_file_err;
			}
			//			fwrite(&(GainData[view*size_single_proj]), sizeof(USHORT), numPerProj, fileptr);
			fwrite(Sys->Norm->GainData + view * size_single_proj, sizeof(USHORT), size_single_proj, fileptr);
			fclose(fileptr);
		}

		delete GainBuf;
	}
	return Tomo_OK;
	//	return status;
}

TomoError ReadRawProjectionData(struct SystemControl * Sys, int NumViews, std::string BaseFileIn, std::string FileName){
	//Set up the basic path to the raw projection dark and gain data
	FILE * ProjData = NULL;

	//Define the size of the raw projection buffer and set points
	int size_single_proj = Sys->Proj->Nx * Sys->Proj->Ny;
	int size_raw_proj = size_single_proj * NumViews;
	int size_raw_subproj = size_single_proj * Sys->Proj->NumViews;

	//Define the temp buffer to read and correct data
	Sys->Name = new FileNames;
	Sys->Norm->CorrBuf = new float[size_raw_subproj];
	memset(Sys->Norm->CorrBuf, 0, size_raw_subproj * sizeof(unsigned short));

	std::string BasePath, ProjPath;
//	BasePath = BaseFileIn;
//	ProjPath = BaseFileIn;
	ProjPath = FileName;
//	int NumProjSamples = max(GetNumberOfScans(BasePath), 1);
	int NumProjSamples = 1;
	Sys->Name->ScanName = FileName;
	Sys->Name->StudyName = FileName;

	//Define a projection buffer to read all data
	Sys->Norm->ProjBuf = new unsigned short[size_raw_proj * NumProjSamples];
	memset(Sys->Norm->ProjBuf, 0, size_raw_subproj * NumProjSamples * sizeof(unsigned short));

	WIN32_FIND_DATA FindFile;
	HANDLE hfind;
	bool MoreViews = false;
	std::string Path, FullPath;

	for (int sample = 0; sample < NumProjSamples; sample++)
	{
		if (NumProjSamples > 1) {
			std::stringstream SampleNumStream;
			SampleNumStream << sample + 1;
			std::string SampleNum = SampleNumStream.str();

			FullPath = ProjPath + "/";
			FullPath.replace(FullPath.length(), 1, SampleNum.c_str(), SampleNum.length());
			FullPath += "/*.raw";
		}
		else {
			FullPath = ProjPath;
			FullPath += "/*.raw";
		}
//		hfind = FindFirstFile(FullPath.c_str(), &FindFile);
//		if (hfind != INVALID_HANDLE_VALUE) {
//			Path = FullPath;
//			std::string FileName = FindFile.cFileName;
//			Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());

			fopen_s(&ProjData, ProjPath.c_str(), "rb");
			if (ProjData == NULL)
			{
				std::cout << "Error opening the file: " << Path.c_str() << std::endl;
				std::cout << "Please check the path and re-run the program." << std::endl;
				return Tomo_file_err;
			}

			//Write the reconstructed data into the predefine memory location
			fread(Sys->Norm->ProjBuf + (sample)*size_raw_proj,
				sizeof(unsigned short), size_single_proj, ProjData);
			fclose(ProjData);

//		}

		//Read the rest of the blank images for given projection sample set 
		for (int view = 1; view < NumViews; view++)
		{
			//if (FindNextFile(hfind, &FindFile)) {
//				Path = FullPath;
//				std::string FileName = FindFile.cFileName;
//				Path.replace(Path.length() - 5, 5, FileName.c_str(), FileName.length());
				Path = ProjPath.substr(0, ProjPath.length() - 5);
				Path += std::to_string(view) + ".raw";
//				ProjPath.replace(ProjPath.length() - 5, 5, , FileName.length());
				fopen_s(&ProjData, Path.c_str(), "rb");

				if (ProjData == NULL)
				{
					std::cout << "Error opening the file: " << Path.c_str() << std::endl;
					std::cout << "Please check the path and re-run the program.";
					std::cout << std::endl;
					return Tomo_file_err;
				}

				//Write the reconstructed data into the predefine memory location
				fread(Sys->Norm->ProjBuf + (sample)*size_raw_proj + view * size_single_proj,
					sizeof(unsigned short), size_single_proj, ProjData);
				fclose(ProjData);

//			}
//			else {
//				std::cout << "Error: Not enough proj images. Need 7 images";
//				std::cout << " for each projection view." << std::endl;
//				exit(1);
	//		}
		}
	}

	int loc1;
	int GainLoc;
	int DarkLoc;
	int vnum = 0;
	//Correct the Raw projections with the gain and dark images
	for (int sample = 0; sample < NumProjSamples; sample++) {
		vnum = 0;
		for (int view = 0; view < NumViews; view++) {
			for (int x = 0; x < Sys->Proj->Nx; x++) {
				for (int y = 0; y < Sys->Proj->Ny; y++)
				{
					loc1 = (y + view*Sys->Proj->Ny)*Sys->Proj->Nx + x
						+ sample*Sys->Proj->Nx*Sys->Proj->Ny*NumViews;
					GainLoc = y*Sys->Proj->Nx + x + view*Sys->Proj->Nx*Sys->Proj->Ny;
					DarkLoc = y*Sys->Proj->Nx + x;

					unsigned short val = Sys->Norm->ProjBuf[loc1];
					unsigned short gval = Sys->Norm->GainData[GainLoc];
//					unsigned short dval = Sys->Norm->DarkData[DarkLoc];


//					float C_val = (float)(val)-(float)dval;
//					if (C_val < 1) C_val = 1;

//					float C_gval = (float)(gval)-(float)dval;

					float C_val = (float)(val);
					float C_gval = (float)(gval);

					float n_val = C_val / C_gval;
					if (C_val > C_gval) n_val = 1.0;

					int loc3 = (y + view*Sys->Proj->Ny)*Sys->Proj->Nx
						+ Sys->Proj->Nx - (x + 1);

					if (Sys->Proj->Flip == 1) {
						loc3 = (y + view*Sys->Proj->Ny)*Sys->Proj->Nx + x;
					}
					Sys->Norm->CorrBuf[loc3] += (float)n_val;

				}
			}
		}
	}

	int howmany = 0;

	//Allocate memory for the projection data buffer and move data into the buffer
	for (int view = 0; view < Sys->Proj->NumViews; view++) {
		for (int x = 0; x < Sys->Proj->Nx; x++) {
			for (int y = 0; y < Sys->Proj->Ny; y++)
			{
				int loc = (y + view*Sys->Proj->Ny)*Sys->Proj->Nx + x;
				float val = Sys->Norm->CorrBuf[loc];
				val = val / (float)NumProjSamples;
				Sys->Proj->RawData[loc] = (unsigned short)(val *32768.0f);
				if (Sys->UsrIn->CalOffset)
				{
					if (Sys->Proj->RawData[loc] > 3000)
					{
						Sys->Proj->RawDataThresh[loc] = 3000;
						howmany++;
					}
					else
						Sys->Proj->RawDataThresh[loc] = Sys->Proj->RawData[loc];
				}
			}
		}
	}

	std::cout << "threshold #: " << howmany << std::endl;

	//if orientation = 0, then right orientation and need to rotate the image
	if (!Sys->UsrIn->Orientation)
	{
		int i, y;
		int oldPos, newPos;

		int m_imageSizeAdj = Sys->Proj->Ny*Sys->Proj->Nx;

		USHORT* tempstore = new USHORT[m_imageSizeAdj];
		memset(tempstore, 0, m_imageSizeAdj);

		for (i = 0; i < NumViews; i++)
		{
			for (y = 0; y < m_imageSizeAdj; y++)
			{
				oldPos = y + i*m_imageSizeAdj;

				newPos = (m_imageSizeAdj - 1) - y;

				tempstore[newPos] = Sys->Proj->RawData[oldPos];
			}

			for (y = 0; y < m_imageSizeAdj; y++)
			{
				newPos = y + i*m_imageSizeAdj;

				Sys->Proj->RawData[newPos] = tempstore[y];
			}
		}

		delete[] tempstore;
	}

	return Tomo_OK;

/*	ProjData = NULL;
	fopen_s(&ProjData, "D:\\Patients\\testcenter.raw", "wb");
	if (ProjData != NULL)
	{
		fwrite(Sys->Proj->RawData, sizeof(USHORT), size_single_proj, ProjData);
		fclose(ProjData);
	}
	*/
}