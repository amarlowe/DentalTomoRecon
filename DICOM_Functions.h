/*********************************************************************************************
* Dicom_Functions.h
* Copyright 2014-2015, XinRay Systems Inc, All Rights Reserved
********************************************************************************************/
//Set the time and date
struct DateAndTimeStamp
{
	std::stringstream Year, Month, Day, Hour, Min, Sec;

	DateAndTimeStamp(void)
	{
		time_t tval = time(0);
		struct tm * t = new tm;
		localtime_s(t, &tval);

		Year << t->tm_year - 100;

		if (t->tm_mon < 9) Month << "0" << t->tm_mon + 1;
		else Month << t->tm_mon + 1;

		if (t->tm_mday < 10) Day << "0" << t->tm_mday;
		else Day << t->tm_mday;

		if (t->tm_hour < 10) Hour << "0" << t->tm_hour;
		else Hour << t->tm_hour;

		if (t->tm_min < 10) Min << "0" << t->tm_min;
		else Min << t->tm_min;

		if (t->tm_sec < 10) Sec << "0" << t->tm_sec;
		else Sec << t->tm_sec;

		delete[] t;
	}
};

//A simple function to generate a random number for the UID
std::stringstream RndNumStr(int size)
{
	std::stringstream RandNum;

	for (int n = 0; n < size; n++) {
		RandNum << rand() % 10;
	}

	return(RandNum);
}

//Define the DICOM Unique Identifer (UID)
std::stringstream DefineUID(int state, struct DateAndTimeStamp * TM)
{
	//Initial part of the stream is Hex (with letters) for XinTek
	std::stringstream UIDStream;
	UIDStream << "1.585969546569.";

	//Three different cased for how the UID is defined
	if (state == 0) {
		UIDStream << TM->Year.str() << ".";
		UIDStream << TM->Month.str() << "." << TM->Day.str() << ".";
		UIDStream << TM->Hour.str() << "." << TM->Min.str() << ".";
		UIDStream << TM->Sec.str() << ".0000000001.";
		UIDStream << RndNumStr(16).str();
	}
	else if (state == 1) {
		UIDStream << "1.1";
	}
	else if (state == 2) {
		UIDStream << TM->Year.str() << "." << TM->Month.str() << "." << TM->Day.str();
		UIDStream << "." << TM->Hour.str() << "." << TM->Min.str() << "." << TM->Sec.str();
	}
	else if (state == 3) {
		UIDStream << TM->Year.str() << ".";
		UIDStream << TM->Month.str() << "." << TM->Day.str() << ".";
		UIDStream << TM->Hour.str() << "." << TM->Min.str() << ".";
		UIDStream << TM->Sec.str() << ".000001.";
		UIDStream << RndNumStr(8).str();
	}
	else if (state == 4) {
		UIDStream << TM->Year.str() << ".";
		UIDStream << TM->Month.str() << "." << TM->Day.str() << ".";
		UIDStream << TM->Hour.str() << "." << TM->Min.str() << ".";
		UIDStream << TM->Sec.str() << ".000001.";
		UIDStream << RndNumStr(8).str() << "." << RndNumStr(5).str();// "12345678.12345.1234";
		UIDStream << "." << RndNumStr(4).str();
	}
	else if (state == 5) {
		UIDStream << TM->Year.str() << ".";
		UIDStream << TM->Month.str() << "." << TM->Day.str() << ".";
		UIDStream << TM->Hour.str() << "." << TM->Min.str() << ".";
		UIDStream << TM->Sec.str() << ".000001.";
		UIDStream << RndNumStr(8).str();
	}
	return (UIDStream);
}

template <typename Type>
std::string NumToStr(Type Num)
{
	std::stringstream NumStream;
	NumStream << Num;
	std::string NumString = NumStream.str();
	return (NumString);
}

//Write a standard header to stringstream
std::stringstream StandardInitialHeader(void)
{
	//Define a string stream to read the initial header into
	std::stringstream HeaderString;

	//set first 128 bytes to null character
	for (int n = 0; n < 128; n++) HeaderString << char(0);

	//Set the next 4 bytes to the characters DICM
	HeaderString << 'D' << 'I' << 'C' << 'M';

	return (HeaderString);
}

//Write the 0002 meta data portion of the header
std::stringstream SetCTFileMetaInfo(struct DateAndTimeStamp * TimeStamp)
{
	std::stringstream MetaDataStream;

	//0002 0000 is the size of the meta data (set to 188 bytes)
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(0, 2).str();
	MetaDataStream << 'U' << 'L';
	MetaDataStream << ConvertIntToHex(4, 2).str();
	MetaDataStream << ConvertIntToHex(188, 4).str();

	//0002 0001 is the information version set to 256
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(1, 2).str();
	MetaDataStream << 'O' << 'B';
	MetaDataStream << ConvertIntToHex(0, 2).str();
	MetaDataStream << ConvertIntToHex(2, 4).str();
	MetaDataStream << ConvertIntToHex(256, 2).str();

	//0002 0002 is the type of information: CT Image storage
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(26, 2).str();
	MetaDataStream << "1.2.840.10008.5.1.4.1.1.2";
	MetaDataStream << char(0);

	//0002 0003 the UID tag 
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(3, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(60, 2).str();
	MetaDataStream << DefineUID(0, TimeStamp).str();

	//0002 0004 the string to indicate how to read data: Explicit VR little Endian 
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(16, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(20, 2).str();
	MetaDataStream << "1.2.840.10008.1.2.1";
	MetaDataStream << char(0);

	//0002 0012 gives the setting of the UID (a number correlated to XinTek in hex)
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(18, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(18, 2).str();
	MetaDataStream << DefineUID(1, TimeStamp).str();

	//0002 0016 gives the DICOM writer version number and name
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(22, 2).str();
	MetaDataStream << 'A' << 'E';
	MetaDataStream << ConvertIntToHex(10, 2).str();
	MetaDataStream << "XINTEKV1r1";

	return (MetaDataStream);
}

//Write the 0002 meta data portion of the header
std::stringstream SetDentalFileMetaInfo(struct DateAndTimeStamp * TimeStamp)
{
	std::stringstream MetaDataStream;

	//0002 0000 is the size of the meta data (set to 188 bytes)
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(0, 2).str();
	MetaDataStream << 'U' << 'L';
	MetaDataStream << ConvertIntToHex(4, 2).str();
	MetaDataStream << ConvertIntToHex(190, 4).str();

	//0002 0001 is the information version set to 256
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(1, 2).str();
	MetaDataStream << 'O' << 'B';
	MetaDataStream << ConvertIntToHex(0, 2).str();
	MetaDataStream << ConvertIntToHex(2, 4).str();
	MetaDataStream << ConvertIntToHex(256, 2).str();

	//0002 0002 is the type of information: CT Image storage
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(28, 2).str();
	MetaDataStream << "1.2.840.10008.5.1.4.1.1.1.3";
	MetaDataStream << char(0);

	//0002 0003 the UID tag 
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(3, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(60, 2).str();
	MetaDataStream << DefineUID(0, TimeStamp).str();

	//0002 0004 the string to indicate how to read data: Explicit VR little Endian 
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(16, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(20, 2).str();
	MetaDataStream << "1.2.840.10008.1.2.1";
	MetaDataStream << char(0);

	//0002 0012 gives the setting of the UID (a number correlated to XinTek in hex)
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(18, 2).str();
	MetaDataStream << 'U' << 'I';
	MetaDataStream << ConvertIntToHex(18, 2).str();
	MetaDataStream << DefineUID(1, TimeStamp).str();

	//0002 0016 gives the DICOM writer version number and name
	MetaDataStream << ConvertIntToHex(2, 2).str();
	MetaDataStream << ConvertIntToHex(22, 2).str();
	MetaDataStream << 'A' << 'E';
	MetaDataStream << ConvertIntToHex(10, 2).str();
	MetaDataStream << "XINTEKV1r1";

	return (MetaDataStream);
}

std::stringstream SetStartOfPixelInfo(struct SystemControl * Sys, int Nz)
{
	std::stringstream PixelStream;

	int SizeIM = Sys->Recon->Nx * Sys->Recon->Ny * Nz * sizeof(unsigned short);

	PixelStream << ConvertIntToHex(32736, 2).str();
	PixelStream << ConvertIntToHex(16, 2).str();
	PixelStream << 'O' << 'W';
	PixelStream << ConvertIntToHex(0, 2).str();
	PixelStream << ConvertIntToHex(SizeIM, 4).str();

	return (PixelStream);
}

int CheckDICOMHeader(std::ifstream& FILE)
{
	//Initialize character array to NULL
	char * Header = new char[4];
	for (int n = 0; n < 4; n++) Header[n] = '\0';

	int offset = 0;

	//Skip first 128 locations and read next four values
	FILE.seekg(128, FILE.beg);
	FILE >> std::noskipws >> Header[0];
	FILE >> std::noskipws >> Header[1];
	FILE >> std::noskipws >> Header[2];
	FILE >> std::noskipws >> Header[3];

	if (Header[0] == 'D' && Header[1] == 'I' && Header[2] == 'C' && Header[3] == 'M')
	{
		std::cout << "Image is a DICOM image" << std::endl;

		FILE.seekg(132, FILE.beg);
		FILE >> std::noskipws >> Header[0];
		FILE >> std::noskipws >> Header[1];
		FILE >> std::noskipws >> Header[2];
		FILE >> std::noskipws >> Header[3];
		int val1 = Header[0] & 0xff;
		int val2 = Header[1] & 0xff;
		int val3 = Header[2] & 0xff;
		int val4 = Header[3] & 0xff;

		if (val1 == 2 && val3 == 0) {
			FILE.seekg(140, FILE.beg);
			FILE >> std::noskipws >> Header[0];
			FILE >> std::noskipws >> Header[1];
			FILE >> std::noskipws >> Header[2];
			FILE >> std::noskipws >> Header[3];
			val1 = Header[0] & 0xff;
			val2 = Header[1] & 0xff;
			val3 = Header[2] & 0xff;
			val4 = Header[3] & 0xff;

			std::stringstream HexNum;
			HexNum << std::hex << val4 << val3 << val2 << val1;
			HexNum >> std::hex >> offset;
		}
		else {
			std::cout << "Image is not recongized as DICOM image." << std::endl;
			FILE.close();
			exit(1);
		}

		if ((FILE.rdstate() != 0 && std::ifstream::failbit != 0)) {
			std::cerr << "Error reading bit" << std::endl;
			FILE.close();
			exit(1);
		}
		offset += 144;

	}
	else {
		std::cout << "Image is not recongized as DICOM image." << std::endl;
		FILE.close();
		exit(1);
	}

	return offset;
}