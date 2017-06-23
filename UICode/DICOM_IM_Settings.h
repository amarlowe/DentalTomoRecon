/********************************************************************************************/
/* DICOM_IM_Settings.h																		*/
/* Copyright 2014-2015, XinRay Systems Inc, All Rights Reserved								*/
/********************************************************************************************/
//The key system setting used to describe the physical system
struct SystemSettings
{
	int AnodeVolt;							//The anode volate in kVp
	int Exposure;							//The on time in milliseconds
	int Current;							//The x-ray current used
	float EmitX;							//The location of the frist source in x
	float EmitY;							//The location of the first source in y
	float EmitZ;							//The location of the first source in z
	int NumEmit;							//The number of emitters in the sequence
	float Tilt;								//The Detector tilt
};

//The key patient settings
struct PatientInfo
{
	std::string Name;						//Name is a simple string
	std::string IDNum;						//ID Num is a simple string
	std::string Birthday;					//Birthday is string
	int Sex;								//Sex is int (male = 0, female = 1, other = 2)
	std::string Alerts;						//ID Num is a simple string
	std::string Allergies;					//Birthday is string
	std::string Comments;					//Birthday is string
};

//The key information about the institution
struct ExamInstitution
{
	std::string Name;						//The name of the institution
	std::string Ref_Physican;				//The name of the referering physician
	std::string Station;					//The name of the station used
	std::string Physicain;					//The name of the physician overseeing scan
	std::string Reader;						//The name of the physician reading the scan
	std::string Operator;					//The name of the operator of the system
};