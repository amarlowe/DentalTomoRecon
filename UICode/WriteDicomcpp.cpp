/*********************************************************************************************
* WriteDicom.cpp
* Copyright 2015, XinVivo Inc., All Rights Reserved
******************************************************************************************** /

/*********************************************************************************************
* Version: 1.1
* Date: February 15, 2015
* Author: Brian Gonzales
********************************************************************************************/
#include "../MainCode/TomoRecon.h"
#include "DICOM_IM_Settings.h"
#include "DICOM_TAGS.h"
#include "DICOM_Functions.h"

/*********************************************************************************************
* Code to control the wrting of DICOM data
********************************************************************************************/
void WriteDICOMHeader(struct SystemControl * Sys, struct SystemSettings * Set,
	struct PatientInfo * Patient, struct ExamInstitution * Inst, std::string Path, int Nz, int slice)
{
	//Open a fstream to the file location to write the header
	std::ofstream FILE;
	FILE.open(Path.c_str(), std::ios::binary);
	if (!FILE.is_open())
	{
		std::cout << "Error opening the file: " << Path.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
	}

	//Get the current time and date
	struct DateAndTimeStamp * TM = new DateAndTimeStamp();
	std::stringstream DateStream, TimeStream;
	DateStream << "20" << TM->Year.str() << TM->Month.str() << TM->Day.str();
	TimeStream << TM->Hour.str() << TM->Min.str() << TM->Sec.str() << ".000000";

	//Set the initial header to describe the system
	FILE << StandardInitialHeader().str();
	FILE << SetCTFileMetaInfo(TM).str();

	std::string company_name = "XinVivo, Inc";

	//Define the group 8 scan setting information
	DICOM_Header_Tags<TAG_8> * TAG8 = new DICOM_Header_Tags<TAG_8>();
	FILE << TAG8->Write_DICOM_Header_Tag(CHAR_SET, 10, "ISO_IR 100").str();
	FILE << TAG8->Write_DICOM_Header_Tag(IMAGE_TYPE, 18, "ORIGINAL\\TOMO_SCAN").str();
	FILE << TAG8->Write_DICOM_Header_Tag(INSTANCE_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(INSTANCE_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SOP_CLASS_UID, 32, DefineUID(2, TM).str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SOP_INST_UID, 60, DefineUID(0, TM).str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(STUDY_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SERIES_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(ACQUIRE_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(IMAGE_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(STUDY_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SERIES_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(ACQUIRE_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(IMAGE_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(MODALITY, 2, "CT").str();
	FILE << TAG8->Write_DICOM_Header_Tag(MANUFACTURER, company_name.length(), company_name).str();
	FILE << TAG8->Write_DICOM_Header_Tag(INSTITUTION, Inst->Name.length(), Inst->Name).str();
	FILE << TAG8->Write_DICOM_Header_Tag(REF_PHYSICIAN, Inst->Ref_Physican.length(), Inst->Ref_Physican).str();
	FILE << TAG8->Write_DICOM_Header_Tag(STATION, Inst->Station.length(), Inst->Station).str();
	FILE << TAG8->Write_DICOM_Header_Tag(DESCRIPTION, 38, "Tomosynthesis scan of extracted teeth.").str();
	FILE << TAG8->Write_DICOM_Header_Tag(PHYSICIAN, Inst->Physicain.length(), Inst->Physicain).str();
	FILE << TAG8->Write_DICOM_Header_Tag(READER, Inst->Reader.length(), Inst->Reader).str();
	FILE << TAG8->Write_DICOM_Header_Tag(OPERATOR, Inst->Operator.length(), Inst->Operator).str();
	FILE << TAG8->Write_DICOM_Header_Tag(MODEL_NUM, 6, "000001").str();
	free(TAG8);

	//Define the group 10: patient information
	DICOM_Header_Tags<TAG_10> * TAG10 = new DICOM_Header_Tags<TAG_10>();
	FILE << TAG10->Write_DICOM_Header_Tag(NAME, Patient->Name.length(), Patient->Name).str();
	FILE << TAG10->Write_DICOM_Header_Tag(ID_NUM, Patient->IDNum.length(), Patient->IDNum).str();
	FILE << TAG10->Write_DICOM_Header_Tag(BIRTHDAY, 8, Patient->Birthday).str();
	FILE << TAG10->Write_DICOM_Header_Tag(ALERTS, Patient->Alerts.length(), Patient->Alerts).str();
	FILE << TAG10->Write_DICOM_Header_Tag(ALLERGIES, Patient->Allergies.length(), Patient->Allergies).str();
	FILE << TAG10->Write_DICOM_Header_Tag(COMMENTS, Patient->Comments.length(), Patient->Comments).str();
	free(TAG10);

	//Define group 18: system settings
	DICOM_Header_Tags<TAG_18> * TAG18 = new DICOM_Header_Tags<TAG_18>();
	FILE << TAG18->Write_DICOM_Header_Tag(CONTRAST_AGENT, 4, "None").str();
	FILE << TAG18->Write_DICOM_Header_Tag(SLICE_THICKNESS, NumToStr<float>(Sys->Recon->Pitch_z).length(), NumToStr<float>(Sys->Recon->Pitch_z)).str();
	FILE << TAG18->Write_DICOM_Header_Tag(kVp, NumToStr<int>(Set->AnodeVolt).length(), NumToStr<int>(Set->AnodeVolt)).str();
	FILE << TAG18->Write_DICOM_Header_Tag(SERIAL_NUM, 4, "0001").str();
	FILE << TAG18->Write_DICOM_Header_Tag(PROTOCOL, 11, "DENTAL TOMO").str();
	FILE << TAG18->Write_DICOM_Header_Tag(DETECT_TILT, NumToStr<float>(Set->Tilt).length(), NumToStr<float>(Set->Tilt)).str();
	FILE << TAG18->Write_DICOM_Header_Tag(EXPOSURE_TIME, NumToStr<int>(Set->Exposure).length(), NumToStr<int>(Set->Exposure)).str();
	FILE << TAG18->Write_DICOM_Header_Tag(XRAY_CURRENT, NumToStr<int>(Set->Current).length(), NumToStr<int>(Set->Current)).str();
	FILE << TAG18->Write_DICOM_Header_Tag(FILTER_TYPE, 1, "0").str();
	FILE << TAG18->Write_DICOM_Header_Tag(PATIENT_POSITION, 4, "None").str();
	free(TAG18);

	//Define group 20: series settings
	std::stringstream SliceNum;
	SliceNum << std::setw(4) << slice;
	std::string slicestr = SliceNum.str();

	int sliceLoc = slice - 1;
	if (Nz == Sys->Recon->Nz) sliceLoc = Nz / 2;

	DICOM_Header_Tags<TAG_20> * TAG20 = new DICOM_Header_Tags<TAG_20>();
	FILE << TAG20->Write_DICOM_Header_Tag(STUDY_ID, 48, DefineUID(3, TM).str()).str();
	FILE << TAG20->Write_DICOM_Header_Tag(SERIES_ID, 59, DefineUID(4, TM).str()).str();
	FILE << TAG20->Write_DICOM_Header_Tag(SERIES_NUM, 4, "0001").str();
	FILE << TAG20->Write_DICOM_Header_Tag(ACQUISTION_NUM, 4, slicestr).str();
	FILE << TAG20->Write_DICOM_Header_Tag(IMAGE_NUM, 4, slicestr).str();
	FILE << TAG20->Write_DICOM_Header_Tag(SLICE_LOC, NumToStr<int>(sliceLoc).length(), NumToStr<int>(sliceLoc)).str();
	free(TAG20);

	//Define the 28: Image settings
	DICOM_Header_Tags<TAG_28> * TAG28 = new DICOM_Header_Tags<TAG_28>();
	FILE << TAG28->Write_DICOM_Header_Tag(SAMPLE_PER_PIXEL, 2, ConvertIntToHex(1, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(PHOTOMETRIC, 11, "MONOCHROME2").str();
	FILE << TAG28->Write_DICOM_Header_Tag(NUM_FRAMES, NumToStr<int>(Nz).length(), NumToStr<int>(Nz)).str();
	FILE << TAG28->Write_DICOM_Header_Tag(ROWS, 2, ConvertIntToHex(Sys->Recon->Ny, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(COLUMNS, 2, ConvertIntToHex(Sys->Recon->Nx, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(PIXEL_SPACING, 15, "0.03300\\0.03300").str();
	FILE << TAG28->Write_DICOM_Header_Tag(BITS_ALLOCATED, 2, ConvertIntToHex(16, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(BITS_STORED, 2, ConvertIntToHex(16, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(HIGH_BIT, 2, ConvertIntToHex(15, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(PIXEL_REP, 2, ConvertIntToHex(1, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(WIN_CENTER, NumToStr<int>(Sys->Recon->Mean).length(), NumToStr<int>(Sys->Recon->Mean)).str();
	FILE << TAG28->Write_DICOM_Header_Tag(WIN_WIDTH, NumToStr<int>(Sys->Recon->Width).length(), NumToStr<int>(Sys->Recon->Width)).str();
	FILE << TAG28->Write_DICOM_Header_Tag(RESCALE_INTERCEPT, 1, "0").str();
	FILE << TAG28->Write_DICOM_Header_Tag(RESCALE_SLOPE, 1, "1").str();
	free(TAG28);

	//Define the Pixel information 
	FILE << SetStartOfPixelInfo(Sys, Nz).str();

	FILE.close();
}

void WriteDICOMDentalHeader(struct SystemControl * Sys, struct SystemSettings * Set,
	struct PatientInfo * Patient, struct ExamInstitution * Inst, std::string Path)
{
	//Open a fstream to the file location to write the header
	std::ofstream FILE;
	FILE.open(Path.c_str(), std::ios::binary);
	if (!FILE.is_open())
	{
		std::cout << "Error opening the file: " << Path.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
	}

	//Get the current time and date
	struct DateAndTimeStamp * TM = new DateAndTimeStamp();
	std::stringstream DateStream, TimeStream;
	DateStream << "20" << TM->Year.str() << TM->Month.str() << TM->Day.str();
	TimeStream << TM->Hour.str() << TM->Min.str() << TM->Sec.str() << ".000000";

	//Set the initial header to describe the system
	FILE << StandardInitialHeader().str();
	FILE << SetDentalFileMetaInfo(TM).str();

	std::string company_name = "XinVivo, Inc";

	//Define the group 8 scan setting information
	DICOM_Header_Tags<TAG_8> * TAG8 = new DICOM_Header_Tags<TAG_8>();
	FILE << TAG8->Write_DICOM_Header_Tag(CHAR_SET, 10, "ISO_IR 100").str();
	FILE << TAG8->Write_DICOM_Header_Tag(IMAGE_TYPE, 18, "ORIGINAL\\TOMO_SCAN").str();
	FILE << TAG8->Write_DICOM_Header_Tag(INSTANCE_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(INSTANCE_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SOP_CLASS_UID, 32, DefineUID(2, TM).str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SOP_INST_UID, 60, DefineUID(0, TM).str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(STUDY_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SERIES_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(ACQUIRE_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(IMAGE_DATE, 8, DateStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(STUDY_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(SERIES_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(ACQUIRE_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(IMAGE_TIME, 13, TimeStream.str()).str();
	FILE << TAG8->Write_DICOM_Header_Tag(MODALITY, 2, "IO").str();
	FILE << TAG8->Write_DICOM_Header_Tag(MANUFACTURER, company_name.length(), company_name).str();
	FILE << TAG8->Write_DICOM_Header_Tag(INSTITUTION, Inst->Name.length(), Inst->Name).str();
	FILE << TAG8->Write_DICOM_Header_Tag(REF_PHYSICIAN, Inst->Ref_Physican.length(), Inst->Ref_Physican).str();
	FILE << TAG8->Write_DICOM_Header_Tag(STATION, Inst->Station.length(), Inst->Station).str();
	FILE << TAG8->Write_DICOM_Header_Tag(DESCRIPTION, 62, "Synthetic Projection of Tomosythense recon of extracted teeth.").str();
	FILE << TAG8->Write_DICOM_Header_Tag(PHYSICIAN, Inst->Physicain.length(), Inst->Physicain).str();
	FILE << TAG8->Write_DICOM_Header_Tag(READER, Inst->Reader.length(), Inst->Reader).str();
	FILE << TAG8->Write_DICOM_Header_Tag(OPERATOR, Inst->Operator.length(), Inst->Operator).str();
	FILE << TAG8->Write_DICOM_Header_Tag(MODEL_NUM, 6, "000001").str();
	free(TAG8);

	//Define the group 10: patient information
	DICOM_Header_Tags<TAG_10> * TAG10 = new DICOM_Header_Tags<TAG_10>();
	FILE << TAG10->Write_DICOM_Header_Tag(NAME, Patient->Name.length(), Patient->Name).str();
	FILE << TAG10->Write_DICOM_Header_Tag(ID_NUM, Patient->IDNum.length(), Patient->IDNum).str();
	FILE << TAG10->Write_DICOM_Header_Tag(BIRTHDAY, 8, Patient->Birthday).str();
	FILE << TAG10->Write_DICOM_Header_Tag(ALERTS, Patient->Alerts.length(), Patient->Alerts).str();
	FILE << TAG10->Write_DICOM_Header_Tag(ALLERGIES, Patient->Allergies.length(), Patient->Allergies).str();
	FILE << TAG10->Write_DICOM_Header_Tag(COMMENTS, Patient->Comments.length(), Patient->Comments).str();
	free(TAG10);

	//Define group 18: system settings
	DICOM_Header_Tags<TAG_18> * TAG18 = new DICOM_Header_Tags<TAG_18>();
	free(TAG18);

	//Define group 20: series settings
	DICOM_Header_Tags<TAG_20> * TAG20 = new DICOM_Header_Tags<TAG_20>();
	FILE << TAG20->Write_DICOM_Header_Tag(STUDY_ID, 48, DefineUID(3, TM).str()).str();
	FILE << TAG20->Write_DICOM_Header_Tag(SERIES_ID, 59, DefineUID(4, TM).str()).str();
	FILE << TAG20->Write_DICOM_Header_Tag(SERIES_NUM, 4, "0001").str();
	free(TAG20);

	//Define the 28: Image settings
	DICOM_Header_Tags<TAG_28> * TAG28 = new DICOM_Header_Tags<TAG_28>();
	FILE << TAG28->Write_DICOM_Header_Tag(SAMPLE_PER_PIXEL, 2, ConvertIntToHex(1, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(PHOTOMETRIC, 11, "MONOCHROME2").str();
	FILE << TAG28->Write_DICOM_Header_Tag(ROWS, 2, ConvertIntToHex(Sys->Recon->Nx, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(COLUMNS, 2, ConvertIntToHex(Sys->Recon->Ny, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(PIXEL_SPACING, 15, "0.03300\\0.03300").str();
	FILE << TAG28->Write_DICOM_Header_Tag(BITS_ALLOCATED, 2, ConvertIntToHex(16, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(BITS_STORED, 2, ConvertIntToHex(16, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(HIGH_BIT, 2, ConvertIntToHex(15, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(PIXEL_REP, 2, ConvertIntToHex(1, 2).str()).str();
	FILE << TAG28->Write_DICOM_Header_Tag(WIN_CENTER, NumToStr<int>(Sys->Proj->Mean).length(), NumToStr<int>(Sys->Proj->Mean)).str();
	FILE << TAG28->Write_DICOM_Header_Tag(WIN_WIDTH, NumToStr<int>(Sys->Proj->Width).length(), NumToStr<int>(Sys->Proj->Width)).str();
	FILE << TAG28->Write_DICOM_Header_Tag(RESCALE_INTERCEPT, 1, "0").str();
	FILE << TAG28->Write_DICOM_Header_Tag(RESCALE_SLOPE, 1, "1").str();
	free(TAG28);

	//Define the Pixel information 
	FILE << SetStartOfPixelInfo(Sys, 1).str();

	FILE.close();
}

void WriteDICOMFullData(struct SystemControl * Sys, std::string Path)
{
	//Set up the basic path to the raw projection data
	FILE * ReconData = NULL;

	//Open the path and read data to buffer
	fopen_s(&ReconData, Path.c_str(), "ab");
	if (ReconData == NULL)
	{
		std::cout << "Error opening the file: " << Path.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
	}

	//Write the reconstructed data into the predefine memory location
	fwrite(Sys->Recon->ReconIm, sizeof(unsigned short), Sys->Recon->Nx * Sys->Recon->Ny * Sys->Recon->Nz, ReconData);
	fclose(ReconData);
}

void WriteDICOMFullDataDental(struct SystemControl * Sys, std::string Path)
{
	//Set up the basic path to the raw projection data
	FILE * ProjData = NULL;

	//Open the path and read data to buffer
	fopen_s(&ProjData, Path.c_str(), "ab");
	if (ProjData == NULL)
	{
		std::cout << "Error opening the file: " << Path.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
	}

	//Write the reconstructed data into the predefine memory location
	fwrite(Sys->Proj->SyntData, sizeof(unsigned short), Sys->Proj->Nx * Sys->Proj->Ny, ProjData);
	fclose(ProjData);
}

void WriteDICOMSingleData(struct SystemControl * Sys, std::string Path, int slice)
{
	//Set up the basic path to the raw projection data
	FILE * ReconData = NULL;

	//Open the path and read data to buffer
	fopen_s(&ReconData, Path.c_str(), "ab");
	if (ReconData == NULL)
	{
		std::cout << "Error opening the file: " << Path.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
	}
	int sizeIm = Sys->Recon->Nx * Sys->Recon->Ny;

	//Write the reconstructed data into the predefine memory location
	fwrite(Sys->Recon->ReconIm + slice * sizeIm, sizeof(unsigned short), sizeIm, ReconData);
	fclose(ReconData);
}

void WriteRawData(struct SystemControl * Sys, std::string Path, int slice)
{
	//Set up the basic path to the raw projection data
	FILE * ReconData = NULL;

	//Open the path and read data to buffer
	fopen_s(&ReconData, Path.c_str(), "wb");
	if (ReconData == NULL)
	{
		std::cout << "Error opening the file: " << Path.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
	}
	int sizeIm = Sys->Recon->Nx * Sys->Recon->Ny;

	//Write the reconstructed data into the predefine memory location
	fwrite(Sys->Recon->ReconIm + slice * sizeIm, sizeof(unsigned short), sizeIm, ReconData);
	fclose(ReconData);
}


void WriteRawProj(struct SystemControl * Sys, std::string Path)
{
	//Set up the basic path to the raw projection data
	FILE * ProjData = NULL;

	//Open the path and read data to buffer
	fopen_s(&ProjData, Path.c_str(), "wb");
	if (ProjData == NULL)
	{
		std::cout << "Error opening the file: " << Path.c_str() << std::endl;
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
	}
	int sizeProj = Sys->Proj->Nx * Sys->Proj->Ny;

	//Write the reconstructed data into the predefine memory location
	fwrite(Sys->Proj->SyntData, sizeof(unsigned short), sizeProj, ProjData);
	fclose(ProjData);
}


/*********************************************************************************************
* Define Fictional Patient, system, and institution
********************************************************************************************/
void CreatePatientandSetSystem(struct PatientInfo * Patient,
	struct SystemSettings * Set, struct SystemControl * Sys)
{
	//Simple example using progam author
	std::stringstream PhanNumStream;
	PhanNumStream << std::setw(4) << 1.0;

	Patient->Name = "John Doe";
	Patient->IDNum = PhanNumStream.str();
	Patient->Birthday = "20141010";
	Patient->Sex = 0;
	Patient->Alerts = "None";
	Patient->Allergies = "None";
	Patient->Comments = "This is a scan of extracted teeth defined by phantom number.";

	//Description of the system using example phantom data
	Set->AnodeVolt = 70;
	Set->Exposure = 1;
	Set->Current = 7;
	Set->EmitX = Sys->SysGeo.EmitX[0];
	Set->EmitY = Sys->SysGeo.EmitY[0];
	Set->EmitZ = Sys->SysGeo.EmitZ[0];
	Set->NumEmit = Sys->Proj->NumViews;
	Set->Tilt = 0.0;
}

void CreateScanInsitution(struct ExamInstitution * Institute)
{
	//Set the physicans based on people working on the dental project
	Institute->Name = "UNC School of Dentistry";
	Institute->Ref_Physican = "Platin, Rick";
	Institute->Physicain = "Platin, Rick";
	Institute->Reader = "Platin, Rick";
	Institute->Operator = "Platin, Rick";
	Institute->Station = "XinVivo Dental Tomo";
}

void FreeStrings(struct PatientInfo * Patient, struct ExamInstitution * Institute)
{
	Patient->Name.clear();
	Patient->IDNum.clear();
	Patient->Birthday.clear();
	Patient->Alerts.clear();
	Patient->Allergies.clear();
	Patient->Comments.clear();

	Institute->Name.clear();
	Institute->Ref_Physican.clear();
	Institute->Physicain.clear();
	Institute->Reader.clear();
	Institute->Operator.clear();
	Institute->Station.clear();
}

/*********************************************************************************************
* Simple Function to conver the data to DICOM scale
********************************************************************************************/
void ConvertImage(struct SystemControl * Sys)
{
	//Create a temp image to store the data in
	int size_IM = Sys->Recon->Nx * Sys->Recon->Ny * Sys->Recon->Nz;
	unsigned short * TempIm = new unsigned short[size_IM];

	//Cycle through the image an convert scale and change ordering of data
	for (int z = 0; z < Sys->Recon->Nz; z++) {
		for (int x = 0; x < Sys->Recon->Ny; x++) {
			for (int y = 0; y < Sys->Recon->Nx; y++) {
				int loc1 = (x + z * Sys->Recon->Ny) * Sys->Recon->Nx + y;
				int loc2 = (Sys->Recon->Nx - y - 1 + z * Sys->Recon->Nx) * Sys->Recon->Ny + x;
				float val = (float)Sys->Recon->ReconIm[loc1];
				//float val2 = (val / 32768.0f)*Sys->Recon->MaxVal;
				//float nVal = (val2/0.25f)*(32768.0f / 100.0f);
//				TempIm[loc2] = (unsigned short)(val);
				TempIm[loc1] = (unsigned short)(val);
			}
		}
	}


	Sys->Recon->Mean = (int)32768 / 2;// (600.0f * Sys->Recon->MaxVal);
	Sys->Recon->Width = 2 * Sys->Recon->Mean;

	//copy the temp image to image setting data buffer
	memcpy(Sys->Recon->ReconIm, TempIm, size_IM * sizeof(unsigned short));
	delete[] TempIm;
}

void ConvertImageSynthetic(struct SystemControl * Sys)
{
	//Create a temp image to store the data in
	int size_IM = Sys->Proj->Nx * Sys->Proj->Ny;
	unsigned short * TempIm = new unsigned short[size_IM];

	//Cycle through the image an convert scale and change ordering of data
	float maxVal = 0;
	for (int x = 0; x < Sys->Proj->Ny; x++) {
		for (int y = 0; y < Sys->Proj->Nx; y++) {
			int loc1 = x*Sys->Proj->Nx + y;
			//			int loc2 = (Sys->Proj->Nx - y - 1) * Sys->Proj->Ny + x;
			float val = (float)Sys->Proj->SyntData[loc1];
			if (val > maxVal) maxVal = val;
			//		float val2 = ((val) / 32768.0f)*(log(40000.0f) / ((float)Sys->Recon->Nz));
			//		float nVal = (val2 / 0.25f)*(32768.0f / 100.0f);
			//		TempIm[loc2] = (unsigned short)(nVal);
		}
	}
	//	std::cout << maxVal << std::endl;
	for (int x = 0; x < Sys->Proj->Ny; x++) {
		for (int y = 0; y < Sys->Proj->Nx; y++) {
			int loc1 = x*Sys->Proj->Nx + y;
			int loc2 = (Sys->Proj->Nx - y - 1) * Sys->Proj->Ny + x;
			float val = (float)Sys->Proj->SyntData[loc1];
			//		float val2 = ((val) / 32768.0f)*(log(40000.0f) / ((float)Sys->Recon->Nz));
			//		float nVal = (val2 / 0.25f)*(32768.0f / 100.0f);
			float nVal = val / maxVal;
			if (nVal < 0) nVal = 0;
			else if (nVal > 1.0f) nVal = 1.0f;
			TempIm[loc2] = (unsigned short)(nVal * 32767.0f);
		}
	}


	Sys->Proj->Mean = (int)32768 / 2;// (int)(600.0f * (log(40000.0f) / ((float)Sys->Recon->Nz)));
	Sys->Proj->Width = 2 * Sys->Proj->Mean;

	//copy the temp image to image setting data buffer
	memcpy(Sys->Proj->SyntData, TempIm, size_IM * sizeof(unsigned short));
	delete[] TempIm;
}

/*********************************************************************************************
* Function to control the saving of the dicom images
********************************************************************************************/
void SaveDataAsDICOM(struct SystemControl * Sys, std::string BaseFileIn)
{
	//Set the patient and System settings
	struct SystemSettings * Set = new SystemSettings;
	struct PatientInfo * Patient = new PatientInfo;
	CreatePatientandSetSystem(Patient, Set, Sys);

	//Set the Exam Institution Physician Information
	struct ExamInstitution * Institute = new ExamInstitution;
	CreateScanInsitution(Institute);

	//Conver the image to DICOM scale
	ConvertImage(Sys);
//	char * BaseFilePath;
//	BaseFilePath = (char*)BaseFileIn.c_str();
//	PathRemoveFileSpec(BaseFilePath);
//	std::string Path;
//	Path = BaseFilePath;
//	Path += "/Reconstructions";
	//Path += "c:/Users/XinVivo/Documents/Reconstructions/";

//	int FileExists = PathFileExists(Path.c_str());
//	if (FileExists != 1)
//	{
//		CreateDirectory(Path.c_str(), NULL);
//	}
	
	std::string FullImagePath;
//	FullImagePath = Path + "/TomoRecon-";
//	FullImagePath += Sys->Name->ScanName;

	FullImagePath = BaseFileIn;

//	FullImagePath = "D:\\Patients\\Reconstructions\\TomoRecon";
	//int clip = Sys->Name->ScanName.length() - Sys->Name->ScanName.find("-");
	//FullImagePath.erase(FullImagePath.length() - clip, clip);

	//Write the DICOM Header based on the System Info
 	WriteDICOMHeader(Sys, Set, Patient, Institute, FullImagePath, Sys->Recon->Nz, 0);

	//Write dicom data
	WriteDICOMFullData(Sys, FullImagePath);
	/*
	//Save the full data as individual dicom images as well
	Path += "\\Slices";
	if (PathFileExists(Path.c_str()) != 1)
	{
		CreateDirectory(Path.c_str(), NULL);
	}

	for (int slice = 0; slice < Sys->Recon->Nz; slice++)
	{
		std::string SingleImagePath;
		std::stringstream slicestream;
		slicestream << slice + 1;
		SingleImagePath = Path + "/TomoRecon-";
		SingleImagePath += Sys->Name->ScanName;
		int clip = Sys->Name->ScanName.length() - Sys->Name->ScanName.find("-");
		SingleImagePath.erase(SingleImagePath.length() - clip, clip);
		SingleImagePath += "-" + slicestream.str();

		WriteDICOMHeader(Sys, Set, Patient, Institute, SingleImagePath, 1, slice + 1);
		//		WriteRawData(Sys, SingleImagePath,slice);
		WriteDICOMSingleData(Sys, SingleImagePath, slice);
	}
	*/
	//Delete structure pointers, can't just delete the structures with strings
	FreeStrings(Patient, Institute);
	free(Patient);
	free(Institute);

	delete[] Set;
	
}

void SaveCorrectedProjections(struct SystemControl * Sys, std::string BaseFileIn)
{
	//Set up the basic path to the raw projection dark and gain data
	FILE * ProjData = NULL;
	int size_single_image = Sys->Proj->Nx * Sys->Proj->Ny;

	char * BaseFilePath;
	BaseFilePath = (char*)BaseFileIn.c_str();
	PathRemoveFileSpec(BaseFilePath);
	std::string Path;
	Path = BaseFilePath;
	//Path += "c:/Users/XinVivo/Documents/Reconstructions/";
	Path += "/Projections";
	if (PathFileExists(Path.c_str()) != 1)
	{
		CreateDirectory(Path.c_str(), NULL);
	}

	Path += "/Corrected";
	if (PathFileExists(Path.c_str()) != 1)
	{
		CreateDirectory(Path.c_str(), NULL);
	}

	Path += "/" + Sys->Name->StudyName;

	if (PathFileExists(Path.c_str()) != 1)
	{
		CreateDirectory(Path.c_str(), NULL);
	}

	for (int view = 0; view < Sys->Proj->NumViews; view++)
	{
		std::string ViewPath;
		std::stringstream viewstream;
		viewstream << Sys->Proj->Views[view];
		ViewPath = Path + "/" + Sys->Name->ScanName;
		int clip = Sys->Name->ScanName.length() - Sys->Name->ScanName.find("-");
		ViewPath.erase(ViewPath.length() - clip, clip);
		ViewPath += "/Projection-" + viewstream.str() + ".raw";

		//Open the path and read data to buffer
		fopen_s(&ProjData, ViewPath.c_str(), "wb");
		if (ProjData == NULL)
		{
			std::cout << "Error opening the file: " << ViewPath.c_str() << std::endl;
			std::cout << "Please check the path and re-run the program." << std::endl;
			exit(1);
		}

		//Write the reconstructed data into the predefine memory location
		fwrite(Sys->Proj->RawData + view * size_single_image,
			sizeof(unsigned short), size_single_image, ProjData);
		fclose(ProjData);
	}
}

void SaveSyntheticProjections(struct SystemControl * Sys,
	int PhantomNum, std::string BaseFileIn)
{
	//Set the patient and System settings
	struct SystemSettings * Set = new SystemSettings;
	struct PatientInfo * Patient = new PatientInfo;
	CreatePatientandSetSystem(Patient, Set, Sys);

	//Set the Exam Institution Physician Information
	struct ExamInstitution * Institute = new ExamInstitution;
	CreateScanInsitution(Institute);

	//Conver the image to DICOM scale
	ConvertImageSynthetic(Sys);

	//Define a path to save the data
	char * BaseFilePath;
	BaseFilePath = (char*)BaseFileIn.c_str();
	PathRemoveFileSpec(BaseFilePath);
	std::string Path;
	Path = BaseFilePath;
	Path += "\\Projections\\Synthetic\\";

	if (PathFileExists(Path.c_str()) != 1)
	{
		CreateDirectory(Path.c_str(), NULL);
	}

	//std::string Path = "C:/Users/TomoD/Desktop/NewData/Projections/Synthetic";
	Path += Sys->Name->StudyName;

	//	Path += ".raw";
	//Write the DICOM Header based on the System Info
	WriteDICOMDentalHeader(Sys, Set, Patient, Institute, Path);

	WriteDICOMFullDataDental(Sys, Path);
	//	WriteRawProj(Sys, Path);

	//Delete structure pointers, can't just delete the structures with strings
	FreeStrings(Patient, Institute);
	free(Patient);
	free(Institute);

	delete[] Set;
}