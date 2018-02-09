///Class for creating CUDA accelerated reconstructions through and OpenGL window.
///Created by Jesse Dean at Xinvivio Inc.

#ifndef _RECONGPUHEADER_CUH_
#define _RECONGPUHEADER_CUH_

// CUDA runtime
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <iomanip>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <crtdbg.h>
#include "Shlwapi.h"

#include <Windows.h>
#include <WinBase.h>

#include "../UI/interop.h"

#pragma comment(lib, "Shlwapi.lib")

#define NUMVIEWS 7
#define MAXZOOM 20
#define ZOOMFACTOR 1.1f
#define LINEWIDTH 3
#define BARHEIGHT 40

//Projection correction parameters
#define HIGHTHRESH 0.95f

//Autofocus parameters
#define STARTSTEP 1.0f
#define LASTSTEP 0.05f
#define GEOSTART 1.0f
#define GEOLAST 0.001f
#define MINDIS 0
#define MAXDIS 40

//Phantom reader parameters
#define LINEPAIRS 5
#define INTENSITYTHRESH 150
#define UPPERBOUND 20.0f
#define LOWERBOUND 4.0f

//cuda constants
#define WARPSIZE 32
#define MAXTHREADS 1024

//Maps to single instruction in cuda
#define MUL_ADD(a, b, c) ( __mul24((a), (b)) + (c) )
#define UMUL(a, b) ( (a) * (b) )

//Autoscale parameters
#define AUTOTHRESHOLD 5000
#define HISTLIMIT 10
#define HIST_BIN_COUNT 256

//Code use parameters
//#define PROFILER
//#define PRINTSCANCORRECTIONS
//#define ENABLEZDER
//#define ENABLESOLVER
#define PRINTMEMORYUSAGE
#define USEITERATIVE
//#define SHOWERROR

#ifdef ENABLEZDER
//Kernel options
#define SIGMA 1.0f
#define KERNELRADIUS 3
#define KERNELSIZE (2*KERNELRADIUS + 1)
#else
#define SIGMA 1.0f
#define KERNELRADIUS 10
#define KERNELSIZE (2*KERNELRADIUS + 1)
#endif

//Defaults
#define EXPOSUREDEFAULT 75
#define VOLTAGEDEFAULT 70
#define EXPOSUREBASE 50
#define METALDEFAULT 8000
#define ENHANCEDEFAULT 1.0f
#define SCANVERTDEFAULT 0.25f
#define SCANHORDEFAULT 0.1f
#define NOISEMAXDEFAULT 700
#define LAMBDADEFAULT 2
#define ITERDEFAULT 20

#define RECONSLICES 30
#define RECONDIS 6.0f

//RPROP parameters
#define DELTA0 0.0f
#define MINDELTA 0.01f
#define MAXDELTA 1000.0f
#define DELTAGROWTH 1.0f
#define DELTADECAY 0.99f

#define SKIPITERTV false
#define TVX 0.02f
#define TVY 0.02f
#define TVZ 0.00f
#define TVITERATIONS 0
//#define USELOGITER
//#define INVERSEITER
#define MEDIANFAC 0.0f
#define TAPERSIZE 200.0f

//#define RECONDERIVATIVE
//#define SQUAREMAGINX

#define ITERATIONS 70

//Macro for checking cuda errors following a cuda launch or api call
#define voidChkErr(...) {										\
	(__VA_ARGS__);												\
	cudaError_t e=cudaGetLastError();							\
	if(e!=cudaSuccess) {										\
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(e) << "\n";	\
		return Tomo_CUDA_err;									\
	}															\
}

///Error type, used to pass errors to the caller
typedef enum {
	Tomo_OK,
	Tomo_input_err,
	Tomo_invalid_arg,
	Tomo_file_err,
	Tomo_DICOM_err,
	Tomo_CUDA_err,
	Tomo_Done,
	Tomo_proj_file,
	Tomo_image_stack,
	Tomo_cancelled
} TomoError;

///Possible types of data that could be displayed through each of the possible filters
typedef enum {
	iterRecon = 0,
	reconstruction,
	projections,
	error
} sourceData;

///Filters through which one can look at the data
typedef enum {
	no_der,
	x_mag_enhance,
	y_mag_enhance,
	mag_enhance,
	x_enhance,
	y_enhance,
	both_enhance,
	der_x,
	der_y,
	der2_x,
	der2_y,
	der3_x,
	der3_y,
	square_mag,
	slice_diff,
	orientation,
	der,
	der2,
	der3,
	der_all,
	mag_der,
	z_der_mag
} derivative_t;

///The catesian directions
typedef enum {
	dir_x,
	dir_y,
	dir_z
} direction;

#define tomo_err_throw(x) {TomoError err = x; if(err != Tomo_OK) return err;}

#ifdef __INTELLISENSE__
#include "intellisense.h"
#define KERNELCALL2(function, threads, blocks, ...) voidChkErr(function(__VA_ARGS__))
#define KERNELCALL3(function, threads, blocks, sharedMem, ...) voidChkErr(function(__VA_ARGS__))
#define KERNELCALL4(function, threads, blocks, sharedMem, stream, ...) voidChkErr(function(__VA_ARGS__))
#else
#define KERNELCALL2(function, threads, blocks, ...) voidChkErr(function <<< threads, blocks >>> (__VA_ARGS__))
#define KERNELCALL3(function, threads, blocks, sharedMem, ...) voidChkErr(function <<< threads, blocks, sharedMem >>> (__VA_ARGS__))
#define KERNELCALL4(function, threads, blocks, sharedMem, stream, ...) voidChkErr(function <<< threads, blocks, sharedMem, stream >>> (__VA_ARGS__))
#endif

///Projection input parameters
struct Proj_Data {
	int NumViews;							//The number of projection views the recon uses
	float Pitch_x;							//The detector pixel pitch in the x direction
	float Pitch_y;							//The detector pixel pithc in the y direction
	int Nx;									//The number of detector pixels in x direction
	int Ny;									//The number of detector pixels in y direction
	int Flip;								//Flip the orientation of the detector
};

///Parameters for the particular x-ray system
struct SysGeometry {
	float * EmitX;							//Location of the x-ray focal spot in x direction
	float * EmitY;							//Location of the x-ray focal spot in y direction
	float * EmitZ;							//Location of the x-ray focal spot in z direction
	float IsoX;								//Location of the system isocenter in x direction
	float IsoY;								//Location of the system isocenter in y direction
	float IsoZ;								//Location of the system isocenter in z direction
	float ZPitch;							//The distance between slices
	bool raw;
};

///Desired system outputs
struct ReconGeometry {
	float Pitch_x;							//Recon Image pixel pitch in the x direction
	float Pitch_y;							//Recon Image pixel pitch in the y direction
	int Nx;									//The number of recon image pixels in x direction
	int Ny;									//The number of recon image pixels in y direction
	int Nz = RECONSLICES;
};

///Meta variable containing all parameters
struct SystemControl {
	struct Proj_Data Proj;
	struct ReconGeometry Recon;
	struct SysGeometry Geo;
};

///Parameters that will be frequently called inside the CUDA kernels
struct params {
	int Px;
	int Py;
	int Rx;
	int Ry;
	int Views;
	float PitchPx;
	float PitchPy;
	float PitchRx;
	float PitchRy;
	float pitchZ;
	float * d_Beamx;
	float * d_Beamy;
	float * d_Beamz;
	int ReconPitchNum;
	int ProjPitchNum;
	bool orientation;
	bool flip;
	bool log;

	//Internal display variable
	bool isReconstructing = false;

	//Display parameters
	float minVal = 0.0;
	float maxVal = USHRT_MAX;
	bool showNegative = false;

	//selection box parameters
	int baseXr = -1;
	int baseYr = -1;
	int currXr = -1;
	int currYr = -1;

	//User parameters
	float ratio = ENHANCEDEFAULT;
	bool useMaxNoise = true;
	int maxNoise = NOISEMAXDEFAULT;

	sourceData dataDisplay = reconstruction;

	bool useGain = true;
	bool useMetal = false;
	int exposure = EXPOSUREDEFAULT;
	int voltage = VOLTAGEDEFAULT;
	int metalThresh = METALDEFAULT;

	float startDis = RECONDIS;
	int slices = RECONSLICES;
};

///Parameters used in CPU control systems, but not kernel calls
struct CPUParams {
	bool scanVertEnable = true;
	bool scanHorEnable = false;
	float vertTau = SCANVERTDEFAULT;
	float horTau = SCANHORDEFAULT;
	int iterations = 40;
};

///Input and ouptuts of the geometry test CSV
struct toleranceData {
	std::string name = "views ";
	int numViewsChanged;
	int viewsChanged;
	direction thisDir = dir_x;
	float offset;
	float phantomData;
};

///Reconstruction main class, extending CUDA interop for display through an OpenGL window
class TomoRecon : public interop {
public:
	//Functions
	///System constructor
	TomoRecon(
		///width of the OpenGL canvas.
		int x,
		///height of the OpenGL canvas.
		int y,
		///System parameters.
		struct SystemControl * Sys);

	///System destructor
	~TomoRecon();

	///Initialization code for CPU and GPU memory.
	TomoError init();

	///Load in gain and primary data files for display.
	TomoError ReadProjectionsFromFile(
		///Images taken of the detector with no subject.
		const char * gainFile,
		///Main data file to be reconstructed.
		const char * mainFile,
		///Raw file directly from detector
		bool raw);

	///Cleanup function that should be called in a destructor.
	TomoError FreeGPUMemory();

	///Conversion tool to transfer projection selection boxes to the reconstruction space.

	///Values are automatically clamped to reconstruction dimensions.
	TomoError setReconBox(
		///Projection number that the selection box is from.
		int index);

	///Get a histogram of brightness value accross an entire image. Limited to values between 0 and USHRT_MAX.
	template<typename T>
	TomoError getHistogram(
		///The dataset to be histogrammed.
		T * image, 
		///The size of the dataset in bytes.
		unsigned int byteSize, 
		///Pointer to an array in which the histogram will be returned. Must be of size 256.
		unsigned int *histogram);

	///Render a single frame using predefined datasets and derivative filters
	TomoError singleFrame(bool outputFrame = false, float** output = NULL, unsigned int * histogram = NULL);

	///Iterative method used to find the distance the selection is from the detector. 

	///Must be called in a loop until Tomo_Done is returned (useful if the caller wishes to paint each frame.
	TomoError autoFocus(
		///Set to true before the loop, false in the loop. Used for initialization.
		bool firstRun);

	///Experimental function used to automatically detect system geometries.

	///Function must be called in a loop until Tomo_Done is returned.
	TomoError autoGeo(
		///True for initialization before the loop, false while in the loop.
		bool firstRun);

	///Function used to automattically find the window and level of the current selection.
	TomoError autoLight(
		///Optional parameter for calling a custom histogram instead of generating one from the current selection.
		unsigned int histogram[HIST_BIN_COUNT] = NULL, 
		///Optional parameter for the required threshold of counts to be considered meaningful. Threshold is automatically set based on selection side if blank.
		int threshold = NULL, 
		///Optional parameter to specify an output for the minimum value. Minval of the system will be set for display if left blank.
		float * minVal = NULL, 
		///Optional parameter to specify and output for the maximum value. Maxval of the system will be set for display if left blank.
		float * maxVal = NULL);

	///Read a resolution phantom based on the current image and set selection paraemters. 
	TomoError readPhantom(
		///Ouput of the function, units in line pairs.
		float * resolution);

	///Initialize dataset to be tested by the geometry test function.
	TomoError initTolerances(
		///The datastructure to be initializaed.
		std::vector<toleranceData> &data,
		///Total number of tests that will be performed.
		int numTests,
		///Desired offsets to be run.
		std::vector<float> offsets);

	///High level function to test all combinations initialized in initTolerances. 

	///Function is iterative and should be called in a loop until Tomo_Done is returned.
	TomoError testTolerances(
		///Input datastructure outlining all tests to be performed
		std::vector<toleranceData> &data,
		///Set to true outside the loop, false inside. Used for initialization.
		bool firstRun);

	///Render the image and place it into the buffer used for display.

	///After this, a swapbuffers() must be called by the parent for the associated openGL canvas to display the rendering.
	TomoError draw(
		///OpenGL canvas width.
		int x,
		///OpenGL canvas height.
		int y);

	///Function used to ouptut the current reconstruction to file.

	///Reconstruction will save based on current position; half the slices will be in front of the current postion, half will be behind.
	TomoError SaveDataAsDICOM(
		///Desired save file name (will be overwritten if it exists).
		std::string BaseFileIn);// ,
		///Number of reconstruction slices to save.
		//int slices);

	//Getters and setters

	///Get the window and level currently set for display.
	TomoError getLight(
		///Minimum value/level output.
		unsigned int * minVal,
		///Maximum value output, used to calculate window.
		unsigned int * maxVal);

	///Set custom minimum and maximum display values.

	///Can be used to set window and level as level=minimum, window = maximum - minimum.
	///Returns Tomo_invalid_arg if either is less than 0, if minVal > maxVal or if either is larger than max unsigned short.
	TomoError setLight(
		///minimum diplay value.
		unsigned int minVal,
		///maximum display value.
		unsigned int maxVal);

	///Setter used as an increment for maximum light, recommended use is scrolling or +/- controls.

	///Returns Tomo_invalid_arg if amount would cause max light to be less than min, or greather than max unsigned short.
	TomoError appendMaxLight(
		///amount to increment/decrement max light.
		int amount);

	///Setter used as an increment for minimum light, recommended use is scrolling or +/- controls.

	///Returns Tomo_invalid_arg if amount would cause min light to be less than 0 or greater than max.
	TomoError appendMinLight(
		///amount to increment/decrement min light.
		int amount);

	///Return set distance from detector used in reconstruction calcluation.

	///Useful for reading out values after an auto-focus is performed.
	float getDistance();

	///Set the internal distance variable for reconstruction calculations.

	///Returns Tomo_invalid_arg if value is less than 0.
	TomoError setDistance(
		///Distance variable to be set. Value will be clamped between min and max distance.
		float distance);

	///Increment the distance by the internal step amount.

	///Change step size with setStep.
	///Useful for calling from a mousewheel.
	///DIstance value will be clamped between min and max distance.
	TomoError stepDistance(
		///Number of steps to increment. 
		int steps);

	///Get the distnace increments used in stepDistance.
	float getStep();

	///Set the distnace increments used in stepDistance.
	TomoError setStep(
		///Value to increment distance by per step.
		float dis);

	///Auto-calibrate light from a set selection window.

	///The default selection window is the center quarter of the image.
	TomoError resetLight();

	///Auto-calibrate focus from a set selection window.

	///The default selection window is the center quarter of the image.
	TomoError resetFocus();

	///Returns whether or not the inverse log correction is currently in use.

	///Returns true if in use, false if not.
	bool getLogView();

	///Set or unset use of the inverse log scale for viewing.

	///Setting the view back to standard can sometimes show more contrast in higher ranges.
	TomoError setLogView(
		///Set or unset. True == use inverse log, false == use uncorrected intensities.
		bool useLog);

	///Returns whether or not the image is being flipped vertically.

	///Returns true if in use, false if not.
	bool getVertFlip();

	///Set or unset flipping the image vertically.
	TomoError setVertFlip(
		///True == flip vertically, false == original.
		bool flip);

	///Returns whether or not the image is being flipped vertically.

	///Returns true if in use, false if not.
	bool getHorFlip();

	///Set or unset flipping the image horizontally.
	TomoError setHorFlip(
		///True == flip horizontally, false == originial.
		bool flip);

	///Sets whether or not an input line pair phantom is vertical or horizontal.

	///Only used in the geometry calibration tests.
	TomoError setInputVeritcal(
		///True == vertical, false == horizontal.
		bool vertical);

	///Set the projection number that will be displayed or otherwise processed.

	///Generally not used if in reconstruction view.
	///Returns Tomo_invalid_arg if index is not between 0 and total projections - 1.
	TomoError setActiveProjection(
		///Value of the slice to be displayed. Must be between 0 and the total slices - 1.
		int index);

	///Set the image display offsets.

	///Only used when the image is zoomed in, otherwise they are set to 0 and fill up the available area.
	///Values will be clamped to stay within the available window.
	TomoError setOffsets(
		///Offset in the horizontal direciton. Positive is right.
		int xOff, 
		///Offset in the vertical direction. Positive is down.
		int yOff);

	///Apprend the image display offsets.

	///See setOffsets.
	///Values will be clamped to stay within the available window.
	TomoError appendOffsets(
		///Value to append to horizontal offset. Positive is right.
		int xOff,
		///Value to append to vertical offset. Positive is down.
		int yOff);

	///Return the values of the offsets set.

	///Values will be 0 if zoomed out.
	void getOffsets(
		///Offset in the horizontal direciton. Positive is right.
		int * xOff,
		///Offset in the vertical direction. Positive is down.
		int * yOff);

	///Set the first set of corrdinates for a selection box.

	///Sets are seperated for ease of input via click and drag.
	///Values will be clamped to reconstruction dimensions.
	TomoError setSelBoxStart(
		///Horizontal position, right is positive.
		int x, 
		///Vertical position, down is positive.
		int y);

	///Set the second set of corrdinates for a selection box.

	///Sets are seperated for ease of input via click and drag.
	///Values will be clamped to reconstruction dimensions.
	TomoError setSelBoxEnd(
		///Horizontal position, right is positive.
		int x, 
		///Vertical position, down is positive.
		int y);

	///Get the selection box in image coordinates.

	///Coordinates are converted from display (monitor pixel position) space to image space internally, and can be extracted here.
	TomoError getSelBoxRaw(
		///First horizontal position set through setSelBoxStart.
		int* x1, 
		///Second horizontal position set through setSelBoxEnd.
		int* x2, 
		///First vertical position set through setSelBoxStart.
		int* y1, 
		///Second vertical position set through setSelBoxEnd.
		int* y2);

	///Set selection box for a projection image.

	///Generally used just for geometry testing, normal setSelBox functions work for displaying projection images.
	///Values will be clamped to projection dimensions.
	TomoError setSelBoxProj(
		///First horizontal position, right is positive.
		int x1, 
		///Second horizontal position, right is positive.
		int x2,
		///First vertical position, down is positive.
		int y1, 
		///Second vertical position, down is positive.
		int y2);

	///Sets the coordinates for the upper value of a line pair phantom.

	///Only used for setting line pair phantom tests.
	///Should be aligned with the 20 pair mark.
	///Values will be clamped to reconstruction dimensions.
	TomoError setUpperTick(
		///Horizontal position, right is positive.
		int x, 
		///Vertical position, down is positive.
		int y);

	///Sets the coordinates for the lower value of a line pair phantom.

	///Only used for setting line pair phantom tests.
	///Should be aligned with the 4 pair mark.
	///Values will be clamped to reconstruction dimensions.
	TomoError setLowerTick(
		///Horizontal position, right is positive.
		int x, 
		///Vertical position, down is positive.
		int y);

	///Sets the coordinates for the upper value of a line pair phantom of a projection image.

	///Only used for setting line pair phantom tests.
	///Should be aligned with the 20 pair mark.
	///Values will be clamped to projection dimensions.
	TomoError setUpperTickProj(
		///Horizontal position, right is positive.
		int x, 
		///Vertical position, down is positive.
		int y);

	///Sets the coordinates for the lower value of a line pair phantom of a projection image.

	///Only used for setting line pair phantom tests.
	///Should be aligned with the 4 pair mark.
	///Values will be clamped to projection dimensions.
	TomoError setLowerTickProj(
		///Horizontal position, right is positive.
		int x, 
		///Vertical position, down is positive.
		int y);

	///Return the value previously set for the upper line pair value.

	///Will return -1 if the values were never set.
	TomoError getUpperTickRaw(
		///Horizontal position, right is positive.
		int* x, 
		///Vertical position, down is positive.
		int* y);

	///Return the value previously set for the lower line pair value.

	///Will return -1 if the values were never set.
	TomoError getLowerTickRaw(
		///Horizontal position, right is positive.
		int* x, 
		///Vertical position, down is positive.
		int* y);

	///Check is the upper tick value has been set.

	///Returns true if the value was set.
	bool upperTickReady();

	///Check is the upper tick value has been set.

	///Returns true if the value was set.
	bool lowerTickReady();

	///Clear the selection box.

	///Used to remove the red lines from the display.
	TomoError resetSelBox();

	///Check if the selection box has been set.

	///Returns true if the box is set.
	bool selBoxReady();

	///Increment the zoom value by an integer amount.

	///Zoom caculations are done on an exponential scale.
	TomoError appendZoom(
		///Amount to append the zoom. Value will be clamped if requested to go above a max, or below 0.
		int amount);

	///Return the current value set for zoom.

	///This value is the raw integer value, before exponentiation.
	int getZoom();

	///Set the zoom value directly.
	TomoError setZoom(
		///Raw integer value to be set. Value will be clamped if above a max, or below 0.
		int value);

	///Reset zoom to default value, 0.
	TomoError resetZoom();

	///Return the current derivative display type.
	derivative_t getDisplay();

	///Set the current derivative display filter.
	TomoError setDisplay(
		///The type of derivative filter to be displayed.
		derivative_t type);

	///Return the current data type being displayed.
	sourceData getDataDisplay();

	///Set the current source of data.
	TomoError setDataDisplay(
		///The data type to be displayed.
		sourceData data);

	///Returns the ratio for the edge enhancement filters.
	float getEnhanceRatio();

	///Set the ratio for the edge enhancement filters.

	///Values will be clamped between 0 and 1.
	TomoError setEnhanceRatio(
		///Value to be set. Values closer to 0 favor derivatives, values closer to 1 favor the original image.
		float ratio);

	///Returns if vertical scan line correction is in use.

	///See enableScanVert.
	bool scanVertIsEnabled();

	///Enable vertical scan line correction.

	///Corrects for vertical dark lines that tend to be a single pixel width. This filter can also produce artifacts if there are artifically strong vertically oriented features.
	///Recommended only if the detector in use produces vertical scan lines.
	TomoError enableScanVert(
		///Enable the filter (true).
		bool enable);

	///Return the strength of the vertical scan line correction.
	float getScanVertVal();

	///Set the strength of the vertical scan line correction.

	///Recommended to test different vaules for each detector system to find the minimum value that removes the lines.
	///Values below 0 will be clamped to 0.
	TomoError setScanVertVal(
		///Value to be set. Larger vaules remove the scan lines more effectively, but also may produce more artifacts.
		float tau);

	///Returns if horizontal scan line correction is in use.

	///See enableScanVert.
	bool scanHorIsEnabled();

	///Enable horizontal scan line correction.

	///Corrects for horizontal dark lines that tend to be a single pixel width. This filter can also produce artifacts if there are artifically strong horizontally oriented features.
	///Recommended only if the detector in use produces horizontal scan lines.
	TomoError enableScanHor(
		///Enable the filter (true).
		bool enable);

	///Return the strength of the vertical scan line correction.
	float getScanHorVal();

	///Set the strength of the horizontal scan line correction.

	///Recommended to test different vaules for each detector system to find the minimum value that removes the lines.
	///Values below 0 will be clamped to 0.
	TomoError setScanHorVal(
		///Value to be set. Larger vaules remove the scan lines more effectively, but also may produce more artifacts.
		float tau);

	///Show negative intensities as a blue color.

	///Standard images shouldn't produce negative vaules, but derivative images will. Edge enhancements may also, if the ratio favors derivatives enough.
	TomoError setShowNegative(
		///Value to be set. True == enable. 
		bool showNegative);

	///Return if the outlier removal noise filter is enabled.
	bool noiseMaxFilterIsEnabled();

	///Enable the outlier removal noise filter.
	TomoError enableNoiseMaxFilter(
		///Value to be set. True enables, false disables.
		bool enable);

	///Return the max value for the outlier removal noise filter.

	///See setNoiseMaxVal.
	int getNoiseMaxVal();

	///Set the max value for the outlier removal noise filter.

	///This value also deterimines similarity in neighboring pixels, so values that are too small may not produce any results.
	///Values below 0 will be clamped to 0.
	TomoError setNoiseMaxVal(
		///Value to be set. Must be greater than 0, recommended to be greater than 50.
		int max);

	///Returns whether or not TV is enabled.
	bool TVIsEnabled();

	///Enable the use of blanket total variation denoising accross the projection images.

	///Noise removal helps clear up images, but can remove small features.
	TomoError enableTV(
		///Value to be set. True == enable.
		bool useTV);

	///Return the strength of TV correction per iteration. 
	float getTVLambda();

	///Set the strength of TV correction per iteration. 

	///Values below 0 will be clamped to 0.
	TomoError setTVLambda(
		///Value to be set. Larger values remove more noise, but blur the image more.
		float lambda);

	///Return the number of TV iterations.
	float getTVIter();

	///Set the number of TV iterations.

	///Values below 0 will be clamped to 0.
	TomoError setTVIter(
		///Value to be set. Larger values remove more noise, but also blur the image and slow down read ins. 
		float iter);

	///Make a single step in the iteration process. Should be called in a loop after initIterative() or resetIterative().

	///Recommended to use in conjunctions with drawing functions to keep the reconstruction dispaly up to date. 
	///finalizeIter() should be called after this function is called the desired number of times.
	TomoError iterStep();

	///Cleans up unsused memory after all iterations have been completed, leaving the main reconstruction structure in tact.

	///It also uninverts the pixels values, so log correction can be re-enabled.
	TomoError finalizeIter();

	///Returns the index for currently active slice of projection or iterative reconstruction.

	///Note that both projection and iterative reconstruction share the same variable, so be sure to update active slice when switching between displayed data.
	int getActiveProjection();

	///Set the beginning and end distances to reconstruct within. These parameters along with stepsize are used to calculate how many slices are needed for the reconstruction.
	TomoError setBoundaries(
		///Start boundary of the iterative reconstruction (mm).
		float begin, 
		///End boundary of the iterative reconstruction (mm).
		float end);

	///Return the currently set start boundary of the iterative reconstruction in millimeters.
	float getStartBoundary();

	///Return the currently set end boundary of the iterative reconstruction in millimeters.
	float getEndBoundary();

	///Return the number of slices allocated in the iterative reconstruction, calculated from start distance, end distance and stepsize.

	///Useful for setting up a slider for scrolling along the reconstruction.
	int getNumSlices();

	///Enable gain correction for the reconstructions. 

	///Gain file is set seperately in ReadProjections(). 
	///Gain file only needs to be set once, this can be toggled as many times as necessary without reloading gain file.
	TomoError enableGain(bool enable);

	///Returns whether or not the gain correction is currently in use.
	bool gainIsEnabled();

	TomoError setExposure(int exposure);
	int getExposure();
	TomoError setVoltage(int voltage);
	int getVoltage();

	TomoError enableMetal(bool enable);
	bool metalIsEnabled();
	TomoError setMetalThreshold(int threshold);
	int getMetalThreshold();

	///Special use case for getting the histogram for the iterative reconstruction data for the current slice.

	///Usually used to adjust window and level.
	TomoError getHistogramRecon(unsigned int * histogram, bool useall, bool useLog);

	///Allocates the necessary memory for the iterative reconstruction and required tools.

	///Start and end distance should alread be set before this function is called. 
	TomoError initIterative();

	///Cleans up previous iterative reconstructions and internally calls initIterative().

	///Required for switching start/end distances or stepsize.
	TomoError resetIterative();

	///Reads in parsed data arrays into the GPU.

	///Parameters for the data like width and hieght must be set ahead of calling this function. 
	///Read in corrections like scanline reduction and TV denosing are done in this step.
	TomoError ReadProjections(unsigned short ** GainData, unsigned short ** RawData);

	///Return the number of viewpoints set for projections.
	int getNumViews();

	///Return the dimensions of the projections.
	void getProjectionDimensions(int * width, int * height);

	///Return the dimensions of the projections.
	void getReconstructionDimensions(int * width, int * height);

	bool hasRawInput();

	///Save the current iterative reconstruction directly to disk.

	///Modifications like edge filters and current lighting will be computed before saving to disk; it will save as currently displayed.
	TomoError exportRecon(unsigned short * exportData);

private:
	/********************************************************************************************/
	/* Function to interface the CPU with the GPU:												*/
	/********************************************************************************************/

	//Functions to Initialize the GPU and set up the reconstruction normalization
	TomoError initGPU();
	TomoError setNOOP(float kernel[KERNELSIZE]);
	TomoError setGauss(float kernel[KERNELSIZE]);
	TomoError setGaussDer(float kernel[KERNELSIZE]);
	TomoError setGaussDer2(float kernel[KERNELSIZE]);
	TomoError setGaussDer3(float kernel[KERNELSIZE]);
	void diver(float * z, float * d, int n);
	void nabla(float * u, float * g, int n);
	void lapla(float * a, float * b, int n);

	//Get and set helpers
	TomoError checkOffsets(int * xOff, int * yOff);
	TomoError checkBoundaries(int * x, int * y);

	//Kernel call helpers
	float focusHelper();
	TomoError imageKernel(float xK[KERNELSIZE], float yK[KERNELSIZE], float * output, bool projs);
	TomoError project(float * projections, float * reconstruction);
	TomoError scanLineDetect(int view, float * d_sum, float * sum, float * offset, bool vert, bool enable);
	float getMax(float * d_Image);

	//Coordinate conversions
	int P2R(int p, int view, bool xDir);
	int R2P(int r, int view, bool xDir);
	int I2D(int i, bool xDir);
	int D2I(int d, bool xDir);
	TomoError P2R(int* rX, int* rY, int pX, int pY, int view);
	TomoError R2P(int* pX, int* pY, int rX, int rY, int view);
	TomoError I2D(int* dX, int* dY, int iX, int iY);

	//DICOM writing helpers
	void ConvertImage();
	void CreatePatientandSetSystem(struct PatientInfo * Patient, struct SystemSettings * Set);
	void CreateScanInsitution(struct ExamInstitution * Institute);
	void FreeStrings(struct PatientInfo * Patient, struct ExamInstitution * Institute);

	//Variables

	struct SystemControl Sys;
	int NumViews;

	int iteration = 0;
	bool continuousMode = false;
	int sliceIndex = 0;
	int zoom = 0;
	int light = 0;
	int lightOff = 0;
	int xOff = 0;
	int yOff = 0;
	float scale = 1.5;
	float distance = 0.0;

	//Selection variables

	//box
	int baseX = -1;
	int baseY = -1;
	int currX = -1;
	int currY = -1;

	//lower tick
	int lowX = -1;
	int lowY = -1;

	//upper tick
	int upX = -1;
	int upY = -1;

	//lower tick
	int lowXr = -1;
	int lowYr = -1;

	//upper tick
	int upXr = -1;
	int upYr = -1;

	bool vertical;
	derivative_t derDisplay = mag_enhance;

	//Define data buffer
	float * d_Image;
	float * d_Error;
	float * d_Sino;
	float * d_Raw;
	float * d_Weights;
	cudaArray_t d_Recon2 = NULL;
	cudaArray_t d_ReconDelta = NULL;
	cudaArray_t d_ReconError = NULL;
	float * d_ReconOld = NULL;

	//Kernel memory
	float * d_MaxVal;
	float * d_MinVal;
	float * d_gauss;
	float * d_gaussDer;
	float * d_gaussDer2;
	float * d_gaussDer3;

	//Derivative buffers
	float * buff1;
	float * buff2;
	float * inXBuff;
	float * inYBuff;
	float * inZBuff;
	float ** zBuffs;
	float * maxZVal;
	float * maxZPos;

	//Kernel call parameters
	size_t sizeIM;
	size_t sizeProj;
	size_t sizeSino;
	size_t sizeError;

	dim3 contThreads;
	dim3 contBlocks;
	dim3 reductionBlocks;
	dim3 reductionThreads;
	int reductionSize = MAXTHREADS * sizeof(float);

	cudaStream_t stream;

	//cuda pitch variables generated from 2d mallocs
	size_t reconPitch;
	size_t projPitch;
	int reconPitchNum;

	//Parameters for 2d geometry search
	int diffSlice = 0;

	//Constants for program set by outside caller
	CPUParams cConstants;
	params constants;

	//TV variables
	bool useTV = true;
	float lambda = LAMBDADEFAULT * UCHAR_MAX;
	float iter = ITERDEFAULT;

	bool iterativeInitialized = false;
	cudaSurfaceObject_t surfReconObj = 0;
	cudaSurfaceObject_t surfErrorObj = 0;
	cudaSurfaceObject_t surfDeltaObj = 0;

	//Input histogram for end recon matching
	unsigned int inputHistogram[HIST_BIN_COUNT];
};

/********************************************************************************************/
/* Library assertions																		*/
/********************************************************************************************/
TomoError cuda_assert(const cudaError_t code, const char* const file, const int line);
TomoError cuda_assert_void(const char* const file, const int line);

#define cudav(...)  cuda##__VA_ARGS__; tomo_err_throw(cuda_assert_void(__FILE__, __LINE__));
#define cuda(...)  cuda_assert((cuda##__VA_ARGS__), __FILE__, __LINE__);

#endif
