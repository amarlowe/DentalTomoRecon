/********************************************************************************************/
/* DICOM_TAGS.cpp																			*/
/* Copyright 2015, Xintek Inc., All Rights Reserved											*/
/********************************************************************************************/

/********************************************************************************************/
/* Version: 1.0																				*/
/* Date: Septmeber 28, 2015																	*/
/* Author: Brian Gonzales																	*/
/********************************************************************************************/
#include "TomoRecon.h"
#include "DICOM_TAGS.h"

/********************************************************************************************/
/* Function to convert decimal integer into hex string										*/
/********************************************************************************************/
std::stringstream ConvertIntToHex(int num, int size)
{
	//Convert number to hex string with (size) bytes
	std::stringstream HexStream;
	HexStream << std::setfill('0') << std::setw(size * 2) << std::hex << num;
	std::string str = HexStream.str();

	//Parse the string into bytes
	unsigned short * HexByte = new unsigned short[size];
	for (int n = 0; n < size; n++) {
		std::string parse_str = str.substr(n * 2, 2);
		HexByte[n] = (unsigned short)strtoul(parse_str.c_str(), 0, 16);
	}

	//Clear the string stream to reuse as return
	HexStream.clear();
	HexStream.str(std::string());

	//Reverse byte order and right as characters to string stream
	for (int n = 1; n <= size; n++)
	{
		HexStream << char(HexByte[size - n]);
	}

	//Delete the pointer to the bytes
	delete[] HexByte;
	HexByte = NULL;

	return(HexStream);
}
/********************************************************************************************/
/* Functions to read hex strings to integers and string										*/
/********************************************************************************************/
int ConvertHexToInt(std::ifstream& FILE, int size)
{

	char * DataVal = new char[size];
	int * Val = new int[size];
	for (int n = 0; n < size; n++) {
		FILE >> std::noskipws >> DataVal[n];
		Val[n] = DataVal[n] & 0xff;
	}

	std::stringstream HexStream;
	for (int n = size - 1; n >= 0; n--) {
		HexStream << std::hex << std::setw(2) << std::setfill('0') << Val[n];

	}
	int RtrVal;
	HexStream >> std::hex >> RtrVal;

	delete[] DataVal;
	delete[] Val;

	return (RtrVal);
}

std::string ReadTagVR(std::ifstream& FILE, int skip)
{
	char * DataVal = new char[2];
	if (skip == 2) {
		FILE >> std::noskipws >> DataVal[0];
		FILE >> std::noskipws >> DataVal[1];
	}

	FILE >> std::noskipws >> DataVal[0];
	FILE >> std::noskipws >> DataVal[1];

	std::stringstream VRStream;
	VRStream << DataVal[0] << DataVal[1];

	if (VRStream.str() == "SQ") {
		FILE >> std::noskipws >> DataVal[0];
		FILE >> std::noskipws >> DataVal[1];
	}

	return (VRStream.str());

}
/*********************************************************************************************
* Private functions for swithing the size and defining the VR
********************************************************************************************/
std::stringstream  DICOM_Header_Tags<TAG_8>::SetVR(TAG_8 Tag)
{
	std::stringstream VRStream;
	switch (Tag)
	{
	case CHAR_SET:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case IMAGE_TYPE:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case INSTANCE_DATE:
		VRStream << 'D' << 'A'; Mem_Size = 2; break;
	case INSTANCE_TIME:
		VRStream << 'T' << 'M'; Mem_Size = 2; break;
	case SOP_CLASS_UID:
		VRStream << 'U' << 'I'; Mem_Size = 2; break;
	case SOP_INST_UID:
		VRStream << 'U' << 'I'; Mem_Size = 2; break;
	case STUDY_DATE:
		VRStream << 'D' << 'A'; Mem_Size = 2; break;
	case SERIES_DATE:
		VRStream << 'D' << 'A'; Mem_Size = 2; break;
	case ACQUIRE_DATE:
		VRStream << 'D' << 'A'; Mem_Size = 2; break;
	case IMAGE_DATE:
		VRStream << 'D' << 'A'; Mem_Size = 2; break;
	case STUDY_TIME:
		VRStream << 'T' << 'M'; Mem_Size = 2; break;
	case SERIES_TIME:
		VRStream << 'T' << 'M'; Mem_Size = 2; break;
	case ACQUIRE_TIME:
		VRStream << 'T' << 'M'; Mem_Size = 2; break;
	case IMAGE_TIME:
		VRStream << 'T' << 'M'; Mem_Size = 2; break;
	case MODALITY:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case MANUFACTURER:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case INSTITUTION:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case REF_PHYSICIAN:
		VRStream << 'P' << 'N'; Mem_Size = 2; break;
	case STATION:
		VRStream << 'S' << 'H'; Mem_Size = 2; break;
	case DESCRIPTION:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case PHYSICIAN:
		VRStream << 'P' << 'N'; Mem_Size = 2; break;
	case READER:
		VRStream << 'P' << 'N'; Mem_Size = 2; break;
	case OPERATOR:
		VRStream << 'P' << 'N'; Mem_Size = 2; break;
	case MODEL_NUM:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case ANATOMIC_REGION_SEQ:
		VRStream << 'S' << 'Q' << char(0) << char(0); Mem_Size = 4; break;
	case ANATOMIC_REGION_MOD_SEQ:
		VRStream << 'S' << 'Q' << char(0) << char(0); Mem_Size = 4; break;
	case ANATOMIC_STRUCTURE_SEQ:
		VRStream << 'S' << 'Q' << char(0) << char(0); Mem_Size = 4; break;
	default:
		std::cout << "Error invalid Tag. ";
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
		break;
	}

	return(VRStream);
}

std::stringstream  DICOM_Header_Tags<TAG_10>::SetVR(TAG_10 Tag)
{
	std::stringstream VRStream;
	switch (Tag)
	{
	case NAME:
		VRStream << 'P' << 'N'; Mem_Size = 2; break;
	case ID_NUM:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case BIRTHDAY:
		VRStream << 'D' << 'A'; Mem_Size = 2; break;
	case SEX:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case ALERTS:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case ALLERGIES:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case COMMENTS:
		VRStream << 'L' << 'T'; Mem_Size = 2; break;
	default:
		std::cout << "Error invalid Tag. ";
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
		break;
	}

	return(VRStream);
}

std::stringstream  DICOM_Header_Tags<TAG_18>::SetVR(TAG_18 Tag)
{
	std::stringstream VRStream;
	switch (Tag)
	{
	case CONTRAST_AGENT:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case SLICE_THICKNESS:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case kVp:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case SERIAL_NUM:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case SOFTWARE:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case PROTOCOL:
		VRStream << 'L' << 'O'; Mem_Size = 2; break;
	case SOURCE_TO_DETECT:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case DETECT_TILT:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case EXPOSURE_TIME:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case XRAY_CURRENT:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case FILTER_TYPE:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case CALIBRATION_DATE:
		VRStream << 'D' << 'A'; Mem_Size = 2; break;
	case CALIBRATION_TIME:
		VRStream << 'T' << 'M'; Mem_Size = 2; break;
	case TOMO_LAYER_HEIGHT:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case TOMO_ANGLE:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case TOMO_TIME:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case TOMO_TYPE:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case TOMO_CLASS:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case TOMO_NUM_IMAGES:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case POSITIONER_TYPE:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case PATIENT_POSITION:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case XRAY_GEOMETRY_SEQ:
		VRStream << 'S' << 'Q' << char(0) << char(0); Mem_Size = 4; break;
	default:
		std::cout << "Error invalid Tag. ";
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
		break;
	}

	return(VRStream);
}

std::stringstream  DICOM_Header_Tags<TAG_20>::SetVR(TAG_20 Tag)
{
	std::stringstream VRStream;
	switch (Tag)
	{
	case STUDY_ID:
		VRStream << 'U' << 'I'; Mem_Size = 2; break;
	case SERIES_ID:
		VRStream << 'U' << 'I'; Mem_Size = 2; break;
	case SERIES_NUM:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case ACQUISTION_NUM:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case IMAGE_NUM:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case PATIENT_ORIENTATION:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case IMAGE_POSITION:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case IMAGE_ORIENTATION:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case REF_FRAME_UID:
		VRStream << 'U' << 'I'; Mem_Size = 2; break;
	case SLICE_LOC:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	default:
		std::cout << "Error invalid Tag. ";
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
		break;
	}

	return(VRStream);
}

std::stringstream  DICOM_Header_Tags<TAG_28>::SetVR(TAG_28 Tag)
{
	std::stringstream VRStream;
	switch (Tag)
	{
	case SAMPLE_PER_PIXEL:
		VRStream << 'U' << 'S'; Mem_Size = 2; break;
	case PHOTOMETRIC:
		VRStream << 'C' << 'S'; Mem_Size = 2; break;
	case NUM_FRAMES:
		VRStream << 'I' << 'S'; Mem_Size = 2; break;
	case ROWS:
		VRStream << 'U' << 'S'; Mem_Size = 2; break;
	case COLUMNS:
		VRStream << 'U' << 'S'; Mem_Size = 2; break;
	case PIXEL_SPACING:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case BITS_ALLOCATED:
		VRStream << 'U' << 'S'; Mem_Size = 2; break;
	case BITS_STORED:
		VRStream << 'U' << 'S'; Mem_Size = 2; break;
	case HIGH_BIT:
		VRStream << 'U' << 'S'; Mem_Size = 2; break;
	case PIXEL_REP:
		VRStream << 'U' << 'S'; Mem_Size = 2; break;
	case WIN_CENTER:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case WIN_WIDTH:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case RESCALE_INTERCEPT:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	case RESCALE_SLOPE:
		VRStream << 'D' << 'S'; Mem_Size = 2; break;
	default:
		std::cout << "Error invalid Tag. ";
		std::cout << "Please check the path and re-run the program." << std::endl;
		exit(1);
		break;
	}

	return(VRStream);
}

/*********************************************************************************************
* Simple Functions to define the return the group and element numbers
********************************************************************************************/
int  DICOM_Header_Tags<TAG_8>::GetTagGroupNum(void) { return (8); }
int  DICOM_Header_Tags<TAG_10>::GetTagGroupNum(void) { return (16); }
int  DICOM_Header_Tags<TAG_18>::GetTagGroupNum(void) { return (24); }
int  DICOM_Header_Tags<TAG_20>::GetTagGroupNum(void) { return (32); }
int  DICOM_Header_Tags<TAG_28>::GetTagGroupNum(void) { return (40); }

int DICOM_Header_Tags<TAG_8>::GetTagElementNum(TAG_8 Tag) { return (Tag); }
int DICOM_Header_Tags<TAG_10>::GetTagElementNum(TAG_10 Tag) { return (Tag); }
int DICOM_Header_Tags<TAG_18>::GetTagElementNum(TAG_18 Tag) { return (Tag); }
int DICOM_Header_Tags<TAG_20>::GetTagElementNum(TAG_20 Tag) { return (Tag); }
int DICOM_Header_Tags<TAG_28>::GetTagElementNum(TAG_28 Tag) { return (Tag); }

/*********************************************************************************************
* Function to read the DICOM element information based on VR
********************************************************************************************/
std::string ReadHeaderTagDataBytes(std::string VR, char* Data, int MemSize)
{
	std::stringstream DataStream;
	if (VR == "US") {
		int * DataInt = new int[MemSize];
		for (int n = 0; n < MemSize; n++) {
			DataInt[n] = Data[n] & 0xff;
		}
		int IntVal;
		std::stringstream HexStream;
		for (int n = MemSize - 1; n >= 0; n--) {
			HexStream << std::hex << std::setw(2) << std::setfill('0') << DataInt[n];
		}
		HexStream >> std::hex >> IntVal;
		DataStream << IntVal;
	}
	else {
		for (int n = 0; n < MemSize; n++) {
			DataStream << Data[n];
		}
	}
	return (DataStream.str());
}
/*********************************************************************************************
* Functions to write the dicom header element
********************************************************************************************/
std::stringstream DICOM_Header_Tags<TAG_8>::Write_DICOM_Header_Tag
(TAG_8 Tag, int Memsize, std::string Data)
{
	std::stringstream Group8Stream;
	if (Memsize % 2 == 0) {
		Group8Stream << ConvertIntToHex(8, 2).str();
		Group8Stream << ConvertIntToHex(Tag, 2).str();
		Group8Stream << SetVR(Tag).str();
		Group8Stream << ConvertIntToHex(Memsize, Mem_Size).str();
		Group8Stream << Data;
	}
	else {
		Group8Stream << ConvertIntToHex(8, 2).str();
		Group8Stream << ConvertIntToHex(Tag, 2).str();
		Group8Stream << SetVR(Tag).str();
		Group8Stream << ConvertIntToHex(Memsize + 1, Mem_Size).str();
		Group8Stream << Data;
		Group8Stream << char(0);
	}
	return (Group8Stream);
}

std::stringstream DICOM_Header_Tags<TAG_10>::Write_DICOM_Header_Tag
(TAG_10 Tag, int Memsize, std::string Data)
{
	std::stringstream Group10Stream;

	if (Memsize % 2 == 0) {
		Group10Stream << ConvertIntToHex(16, 2).str();
		Group10Stream << ConvertIntToHex(Tag, 2).str();
		Group10Stream << SetVR(Tag).str();
		Group10Stream << ConvertIntToHex(Memsize, Mem_Size).str();
		Group10Stream << Data;
	}
	else {
		Group10Stream << ConvertIntToHex(16, 2).str();
		Group10Stream << ConvertIntToHex(Tag, 2).str();
		Group10Stream << SetVR(Tag).str();
		Group10Stream << ConvertIntToHex(Memsize + 1, Mem_Size).str();
		Group10Stream << Data;
		Group10Stream << char(0);
	}
	return (Group10Stream);
}


std::stringstream DICOM_Header_Tags<TAG_18>::Write_DICOM_Header_Tag
(TAG_18 Tag, int Memsize, std::string Data)
{
	std::stringstream Group18Stream;

	if (Memsize % 2 == 0) {
		Group18Stream << ConvertIntToHex(24, 2).str();
		Group18Stream << ConvertIntToHex(Tag, 2).str();
		Group18Stream << SetVR(Tag).str();
		Group18Stream << ConvertIntToHex(Memsize, Mem_Size).str();
		Group18Stream << Data;
	}
	else {
		Group18Stream << ConvertIntToHex(24, 2).str();
		Group18Stream << ConvertIntToHex(Tag, 2).str();
		Group18Stream << SetVR(Tag).str();
		Group18Stream << ConvertIntToHex(Memsize + 1, Mem_Size).str();
		Group18Stream << Data;
		Group18Stream << char(0);
	}
	return (Group18Stream);
}

std::stringstream DICOM_Header_Tags<TAG_20>::Write_DICOM_Header_Tag
(TAG_20 Tag, int Memsize, std::string Data)
{
	std::stringstream Group20Stream;
	if (Memsize % 2 == 0) {
		Group20Stream << ConvertIntToHex(32, 2).str();
		Group20Stream << ConvertIntToHex(Tag, 2).str();
		Group20Stream << SetVR(Tag).str();
		Group20Stream << ConvertIntToHex(Memsize, Mem_Size).str();
		Group20Stream << Data;
	}
	else {
		Group20Stream << ConvertIntToHex(32, 2).str();
		Group20Stream << ConvertIntToHex(Tag, 2).str();
		Group20Stream << SetVR(Tag).str();
		Group20Stream << ConvertIntToHex(Memsize + 1, Mem_Size).str();
		Group20Stream << Data;
		Group20Stream << char(0);
	}
	return (Group20Stream);
}

std::stringstream DICOM_Header_Tags<TAG_28>::Write_DICOM_Header_Tag
(TAG_28 Tag, int Memsize, std::string Data)
{
	std::stringstream Group28Stream;
	if (Memsize % 2 == 0) {
		Group28Stream << ConvertIntToHex(40, 2).str();
		Group28Stream << ConvertIntToHex(Tag, 2).str();
		Group28Stream << SetVR(Tag).str();
		Group28Stream << ConvertIntToHex(Memsize, Mem_Size).str();
		Group28Stream << Data;
	}
	else {
		Group28Stream << ConvertIntToHex(40, 2).str();
		Group28Stream << ConvertIntToHex(Tag, 2).str();
		Group28Stream << SetVR(Tag).str();
		Group28Stream << ConvertIntToHex(Memsize + 1, Mem_Size).str();
		Group28Stream << Data;
		Group28Stream << char(0);
	}

	return (Group28Stream);
}

/*********************************************************************************************
* Functions to determine the offset in bytes to the start of each group number
********************************************************************************************/
int  SeekTagOffset(std::ifstream& FILE, int Tag, int offset)
{
	FILE.seekg(0, FILE.end);
	int length = (int)FILE.tellg();
	FILE.seekg(0, FILE.beg);

	char * DataVal = new char[2];
	int val;

	for (int n = offset; n < length; n++) {
		FILE.seekg(n, FILE.beg);
		val = ConvertHexToInt(FILE, 2);

		if (val != Tag) {
			std::string VR = ReadTagVR(FILE, 2);
			if (VR != "SQ") {
				int tempOffset = ConvertHexToInt(FILE, 2);
				n += tempOffset + 7;
			}
			else {
				int tempOffset = ConvertHexToInt(FILE, 8);
				n += tempOffset + 11;
			}
		}
		else {
			offset = n;
			n = length + 1;
			break;
		}
	}
	return offset;
}

std::string Read_DICOM_Header_Tag(std::ifstream& FILE, int Group, int Element, int offset)
{
	FILE.seekg(0, FILE.end);
	int length = (int)FILE.tellg();
	FILE.seekg(0, FILE.beg);

	char * DataVal = new char[2];
	int val;
	std::string RtrVal;

	for (int n = offset; n < length; n++) {
		FILE.seekg(n, FILE.beg);
		val = ConvertHexToInt(FILE, 2);

		if (val != Group) {
			n = length + 1;
			RtrVal = "0";
			break;
		}
		else {
			int Index = ConvertHexToInt(FILE, 2);
			if (Index == Element)
			{
				std::string VR = ReadTagVR(FILE, 0);
				if (VR != "SQ") {
					int SizeMem = ConvertHexToInt(FILE, 2);

					//I have to make options for reading depending on what type of data it is
					char * ReadData = new char[SizeMem];
					for (int m = 0; m < SizeMem; m++) {
						FILE >> std::noskipws >> ReadData[m];
					}
					RtrVal = ReadHeaderTagDataBytes(VR, ReadData, SizeMem);

					delete[] ReadData;
					ReadData = NULL;
				}
				n = length + 1;
				break;

			}
			else {
				std::string VR = ReadTagVR(FILE, 0);
				if (VR != "SQ") {
					int tempOffset = ConvertHexToInt(FILE, 2);
					n += tempOffset + 7;
				}
				else {
					int tempOffset = ConvertHexToInt(FILE, 8);
					n += tempOffset + 11;
				}
			}
		}
	}
	return(RtrVal);
}