/********************************************************************************************/
/* DICOM_TAGS.h																				*/
/* Copyright 2015, Xintek Inc., All Rights Reserved											*/
/********************************************************************************************/
#ifndef DICOM_HEADER_TAGS
#define DICOM_HEADER_TAGS

template<typename TAG>
class DICOM_Header_Tags
{
public:
	DICOM_Header_Tags() : Mem_Size(2) {}

	std::stringstream Write_DICOM_Header_Tag(TAG Tag, int Memsize, std::string Data);

	int GetTagGroupNum(void);
	int GetTagElementNum(TAG Tag);

	~DICOM_Header_Tags();

private:
	int Mem_Size;

	std::stringstream SetVR(TAG Tag);
	int GetTag(void);
};

//State 0008
enum TAG_8
{
	CHAR_SET = 5,
	IMAGE_TYPE = 8,
	INSTANCE_DATE = 18,
	INSTANCE_TIME = 19,
	SOP_CLASS_UID = 22,
	SOP_INST_UID = 24,
	STUDY_DATE = 32,
	SERIES_DATE = 33,
	ACQUIRE_DATE = 34,
	IMAGE_DATE = 35,
	STUDY_TIME = 48,
	SERIES_TIME = 49,
	ACQUIRE_TIME = 50,
	IMAGE_TIME = 51,
	MODALITY = 96,
	MANUFACTURER = 112,
	INSTITUTION = 128,
	REF_PHYSICIAN = 144,
	STATION = 4112,
	DESCRIPTION = 4144,
	PHYSICIAN = 4176,
	READER = 4192,
	OPERATOR = 4208,
	MODEL_NUM = 4240,
	ANATOMIC_REGION_SEQ = 8728,
	ANATOMIC_REGION_MOD_SEQ = 8736,
	ANATOMIC_STRUCTURE_SEQ = 8744,
};

//State 0010 (PATIENT Information)
enum TAG_10
{
	NAME = 16,
	ID_NUM = 32,
	BIRTHDAY = 48,
	SEX = 64,
	ALERTS = 8192,
	ALLERGIES = 8464,
	COMMENTS = 16384,
};

//Stat 0018 (System Settings)
enum TAG_18
{
	CONTRAST_AGENT = 16,
	SLICE_THICKNESS = 80,
	kVp = 96,
	SERIAL_NUM = 4096,
	SOFTWARE = 4128,
	PROTOCOL = 4144,
	SOURCE_TO_DETECT = 4368,
	DETECT_TILT = 4384,
	EXPOSURE_TIME = 4432,
	XRAY_CURRENT = 4433,
	FILTER_TYPE = 4448,
	CALIBRATION_DATE = 4608,
	CALIBRATION_TIME = 4609,
	TOMO_LAYER_HEIGHT = 5216,
	TOMO_ANGLE = 5232,
	TOMO_TIME = 5248,
	TOMO_TYPE = 5264,
	TOMO_CLASS = 5265,
	TOMO_NUM_IMAGES = 5269,
	POSITIONER_TYPE = 5384,
	PATIENT_POSITION = 20736,
	XRAY_GEOMETRY_SEQ = 38006,
};

//State 0020 (STUDY ID)
enum TAG_20
{
	STUDY_ID = 13,
	SERIES_ID = 14,
	SERIES_NUM = 17,
	ACQUISTION_NUM = 18,
	IMAGE_NUM = 19,
	PATIENT_ORIENTATION = 32,
	IMAGE_POSITION = 50,
	IMAGE_ORIENTATION = 55,
	REF_FRAME_UID = 82,
	SLICE_LOC = 4161,
};

//State 0028 (Image settings
enum TAG_28
{
	SAMPLE_PER_PIXEL = 2,
	PHOTOMETRIC = 4,
	NUM_FRAMES = 8,
	ROWS = 16,
	COLUMNS = 17,
	PIXEL_SPACING = 48,
	BITS_ALLOCATED = 256,
	BITS_STORED = 257,
	HIGH_BIT = 258,
	PIXEL_REP = 259,
	WIN_CENTER = 4176,
	WIN_WIDTH = 4177,
	RESCALE_INTERCEPT = 4178,
	RESCALE_SLOPE = 4179,
};

std::stringstream ConvertIntToHex(int num, int size);

int  SeekTagOffset(std::ifstream& FILE, int Tag, int offset);

std::string Read_DICOM_Header_Tag(std::ifstream& FILE, int Group, int Element, int offset);

#endif