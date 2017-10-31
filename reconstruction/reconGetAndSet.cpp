#include "TomoRecon.h"

/****************************************************************************/
/*								Getters and Setters							*/
/****************************************************************************/

TomoError TomoRecon::getLight(unsigned int * minVal, unsigned int * maxVal) {
	*minVal = constants.minVal;
	*maxVal = constants.maxVal;
	return Tomo_OK;
}

TomoError TomoRecon::setLight(unsigned int minVal, unsigned int maxVal) {
	if (minVal < 0 || maxVal < 0 || minVal > maxVal || maxVal > USHRT_MAX) return Tomo_invalid_arg;
	constants.minVal = minVal;
	constants.maxVal = maxVal;
	return Tomo_OK;
}

TomoError TomoRecon::appendMaxLight(int amount) {
	int append = UCHAR_MAX * amount;
	if (append > USHRT_MAX - constants.maxVal || append <= constants.minVal - constants.maxVal) return Tomo_invalid_arg;
	constants.maxVal += append;
	return Tomo_OK;
}

TomoError TomoRecon::appendMinLight(int amount) {
	int append = UCHAR_MAX * amount;
	if (append < -constants.minVal || append >= constants.maxVal - constants.minVal) return Tomo_invalid_arg;
	constants.minVal += append;
	return Tomo_OK;
}

bool TomoRecon::getLogView() {
	return constants.log;
}

TomoError TomoRecon::setLogView(bool useLog) {
	constants.log = useLog;
	return Tomo_OK;
}

bool TomoRecon::getVertFlip() {
	return constants.flip;
}

TomoError TomoRecon::setVertFlip(bool flip) {
	constants.flip = flip;
	return Tomo_OK;
}

bool TomoRecon::getHorFlip() {
	return constants.orientation;
}

TomoError TomoRecon::setHorFlip(bool flip) {
	constants.orientation = flip;
	return Tomo_OK;
}

float TomoRecon::getDistance() {
	if (constants.dataDisplay == iterRecon) return constants.startDis + constants.pitchZ * sliceIndex;
	return distance;
}

TomoError TomoRecon::setDistance(float dist) {
	distance = dist;
	float clampedDist = distance;
	//if (dist < MINDIS) distance = MINDIS;
	//if (dist > MAXDIS) distance = MAXDIS;
	
	if (dist < constants.startDis) clampedDist = constants.startDis;
	if (dist > constants.startDis + constants.pitchZ * (constants.slices - 1)) clampedDist = constants.startDis + constants.pitchZ * (constants.slices - 1);
	sliceIndex = round((clampedDist - constants.startDis) / constants.pitchZ);
	if (constants.dataDisplay == iterRecon) {
		distance = constants.startDis + constants.pitchZ * sliceIndex;
	}
	return Tomo_OK;
}

TomoError TomoRecon::stepDistance(int steps) {
	distance += Sys.Geo.ZPitch * steps;
	float clampedDist = distance;
	//if (dist < MINDIS) distance = MINDIS;
	//if (dist > MAXDIS) distance = MAXDIS;

	if (distance < constants.startDis) clampedDist = constants.startDis;
	if (distance > constants.startDis + constants.pitchZ * (constants.slices - 1)) clampedDist = constants.startDis + constants.pitchZ * (constants.slices - 1);
	sliceIndex = round((clampedDist - constants.startDis) / constants.pitchZ);
	return Tomo_OK;
}

float TomoRecon::getStep() {
	return Sys.Geo.ZPitch;
}

TomoError TomoRecon::setStep(float dis) {
	if (dis < 0) return Tomo_invalid_arg;
	constants.slices = round(constants.pitchZ / dis * constants.slices);
	Sys.Recon.Nz = constants.slices;
	Sys.Geo.ZPitch = dis;
	constants.pitchZ = dis;
	return Tomo_OK;
}

TomoError TomoRecon::setReconBox(int index) {
	P2R(&upXr, &upYr, upX, upY, index);
	P2R(&lowXr, &lowYr, lowX, lowY, index);
	P2R(&constants.currXr, &constants.currYr, currX, currY, index);
	P2R(&constants.baseXr, &constants.baseYr, baseX, baseY, index);

	return Tomo_OK;
}

TomoError TomoRecon::setInputVeritcal(bool vert) {
	vertical = vert;
	return Tomo_OK;
}

int TomoRecon::getActiveProjection() {
	return sliceIndex;
}

TomoError TomoRecon::setActiveProjection(int index) {
	if( index < 0 
		|| (constants.dataDisplay == projections && index >= constants.Views)
		|| (constants.dataDisplay == error && index >= constants.Views)
		|| (constants.dataDisplay == iterRecon && index >= constants.slices)) return Tomo_invalid_arg;
	sliceIndex = index;
	if (constants.dataDisplay == iterRecon) {
		distance = constants.startDis + constants.pitchZ * sliceIndex;
	}
	return Tomo_OK;
}

void TomoRecon::getOffsets(int * x, int * y) {
	*x = xOff / scale;
	*y = yOff / scale;
}

TomoError TomoRecon::setOffsets(int x, int y) {
	int this_x = x * scale;
	int this_y = y * scale;
	checkOffsets(&this_x, &this_y);
	xOff = this_x;
	yOff = this_y;
	return Tomo_OK;
}

TomoError TomoRecon::appendOffsets(int x, int y) {
	int this_x = x * scale + xOff;
	int this_y = y * scale + yOff;
	checkOffsets(&x, &y);
	xOff = this_x;
	yOff = this_y;
	return Tomo_OK;
}

TomoError TomoRecon::checkOffsets(int * x, int * y) {
	int xLim, yLim;
	if (constants.dataDisplay == reconstruction) {
		xLim = Sys.Recon.Nx;
		yLim = Sys.Recon.Ny;
	}
	else {
		xLim = Sys.Proj.Nx;
		yLim = Sys.Proj.Ny;
	}
	int maxX = (int)((xLim - width * scale) / 2.0f);
	int maxY = (int)((yLim - height * scale) / 2.0f);
	if (maxX < 0) maxX = 0;
	if (maxY < 0) maxY = 0;
	if (*x > maxX || *x < -maxX) *x = *x > 0 ? maxX : -maxX;
	if (*y > maxY || *y < -maxY) *y = *y > 0 ? maxY : -maxY;
	return Tomo_OK;
}

TomoError TomoRecon::setSelBoxStart(int x, int y) {
	int this_x = D2I(x, true);
	int this_y = D2I(y, false);
	checkBoundaries(&this_x, &this_y);
	constants.baseXr = this_x;
	constants.baseYr = this_y;
	return Tomo_OK;
}

TomoError TomoRecon::setSelBoxEnd(int x, int y) {
	int this_x = D2I(x, true);
	int this_y = D2I(y, false);
	checkBoundaries(&this_x, &this_y);
	constants.currXr = this_x;
	constants.currYr = this_y;
	return Tomo_OK;
}

//Get selection box with recon-space coordinates
TomoError TomoRecon::getSelBoxRaw(int* x1, int* x2, int* y1, int* y2) {
	*x1 = constants.baseXr;
	*x2 = constants.currXr;
	*y1 = constants.baseYr;
	*y2 = constants.currYr;

	return Tomo_OK;
}

//Set selection box directly to recon-space coordinates
TomoError TomoRecon::setSelBoxProj(int x1, int x2, int y1, int y2) {
	int this_x = x1;
	int this_y = y1;
	checkBoundaries(&this_x, &this_y);
	baseX = this_x;
	baseY = this_y;

	this_x = x2;
	this_y = y2;
	checkBoundaries(&this_x, &this_y);
	currX = this_x;
	currY = this_y;

	return Tomo_OK;
}

TomoError TomoRecon::setUpperTick(int x, int y) {
	int this_x = D2I(x, true);
	int this_y = D2I(y, false);
	checkBoundaries(&this_x, &this_y);
	upXr = this_x;
	upYr = this_y;
	return Tomo_OK;
}

TomoError TomoRecon::setLowerTick(int x, int y) {
	int this_x = D2I(x, true);
	int this_y = D2I(y, false);
	checkBoundaries(&this_x, &this_y);
	lowXr = this_x;
	lowYr = this_y;
	return Tomo_OK;
	return Tomo_OK;
}

TomoError TomoRecon::setUpperTickProj(int x, int y) {
	int this_x = x;
	int this_y = y;
	checkBoundaries(&this_x, &this_y);
	upX = this_x;
	upY = this_y;
	return Tomo_OK;
}

TomoError TomoRecon::setLowerTickProj(int x, int y) {
	int this_x = x;
	int this_y = y;
	checkBoundaries(&this_x, &this_y);
	lowX = this_x;
	lowY = this_y;
	return Tomo_OK;
	return Tomo_OK;
}

TomoError TomoRecon::getUpperTickRaw(int* x, int* y) {
	*x = upXr;
	*y = upYr;
	return Tomo_OK;
}

TomoError TomoRecon::getLowerTickRaw(int* x, int* y) {
	*x = lowXr;
	*y = lowYr;
	return Tomo_OK;
}

inline TomoError TomoRecon::checkBoundaries(int * x, int * y) {
	int xLim, yLim;
	if (constants.dataDisplay == reconstruction) {
		xLim = Sys.Recon.Nx;
		yLim = Sys.Recon.Ny;
	}
	else {
		xLim = Sys.Proj.Nx;
		yLim = Sys.Proj.Ny;
	}
	if (*x > xLim) *x = xLim;
	if (*x < 0) *x = 0;
	if (*y > yLim) *y = yLim;
	if (*y < 0) *y = 0;
	return Tomo_OK;
}

TomoError TomoRecon::resetSelBox() {
	constants.baseXr = -1;
	constants.baseYr = -1;
	constants.currXr = -1;
	constants.currYr = -1;
	return Tomo_OK;
}

bool TomoRecon::selBoxReady() {
	return constants.baseXr >= 0 && constants.currXr >= 0;
}

bool TomoRecon::upperTickReady() {
	return upXr >= 0 && upYr >= 0;
}

bool TomoRecon::lowerTickReady() {
	return lowXr >= 0 && lowYr >= 0;
}

TomoError TomoRecon::appendZoom(int amount) {
	zoom += amount;
	if (zoom < 0) zoom = 0;
	if (zoom > MAXZOOM) zoom = MAXZOOM;
	return Tomo_OK;
}

int TomoRecon::getZoom() {
	return zoom;
}

TomoError TomoRecon::setZoom(int value) {
	zoom = value;
	if (zoom < 0) zoom = 0;
	if (zoom > MAXZOOM) zoom = MAXZOOM;
	return Tomo_OK;
}

TomoError TomoRecon::resetZoom() {
	zoom = 0;
	return Tomo_OK;
}

derivative_t TomoRecon::getDisplay() {
	return derDisplay;
}

TomoError TomoRecon::setDisplay(derivative_t type) {
	derDisplay = type;
	return Tomo_OK;
}

sourceData TomoRecon::getDataDisplay() {
	return constants.dataDisplay;
}

TomoError TomoRecon::setDataDisplay(sourceData data) {
	constants.dataDisplay = data;
	return Tomo_OK;
}

float TomoRecon::getEnhanceRatio() {
	return constants.ratio;
}

TomoError TomoRecon::setEnhanceRatio(float ratio) {
	constants.ratio = ratio;
	if (ratio < 0.0f) constants.ratio = 0.0f;
	if (ratio > 1.0f) constants.ratio = 1.0f;
	return Tomo_OK;
}

bool TomoRecon::scanVertIsEnabled() {
	return cConstants.scanVertEnable;
}

TomoError TomoRecon::enableScanVert(bool enable) {
	cConstants.scanVertEnable = enable;
	return Tomo_OK;
}

float TomoRecon::getScanVertVal() {
	return cConstants.vertTau;
}

TomoError TomoRecon::setScanVertVal(float tau) {
	cConstants.vertTau = tau;
	if (tau < 0.0f) cConstants.vertTau = 0.0f;
	return Tomo_OK;
}

bool TomoRecon::scanHorIsEnabled() {
	return cConstants.scanHorEnable;
}

TomoError TomoRecon::enableScanHor(bool enable) {
	cConstants.scanHorEnable = enable;
	return Tomo_OK;
}

float TomoRecon::getScanHorVal() {
	return cConstants.horTau;
}

TomoError TomoRecon::setScanHorVal(float tau) {
	cConstants.horTau = tau;
	if (tau < 0.0f) cConstants.horTau = 0.0f;
	return Tomo_OK;
}

TomoError TomoRecon::setShowNegative(bool showNegative) {
	constants.showNegative = showNegative;
	return Tomo_OK;
}

bool TomoRecon::noiseMaxFilterIsEnabled() {
	return constants.useMaxNoise;
}

TomoError TomoRecon::enableNoiseMaxFilter(bool enable) {
	constants.useMaxNoise = enable;
	return Tomo_OK;
}

int TomoRecon::getNoiseMaxVal() {
	return constants.maxNoise;
}

TomoError TomoRecon::setNoiseMaxVal(int max) {
	constants.maxNoise = max;
	if (max < 0.0f) constants.maxNoise = 0.0f;
	return Tomo_OK;
}

bool TomoRecon::TVIsEnabled() {
	return useTV;
}

TomoError TomoRecon::enableTV(bool enable) {
	useTV = enable;
	return Tomo_OK;
}

float TomoRecon::getTVLambda() {
	return lambda / UCHAR_MAX;
}

TomoError TomoRecon::setTVLambda(float TVLambda) {
	lambda = TVLambda * UCHAR_MAX;
	if (TVLambda < 0.0f) lambda = 0.0f;
	return Tomo_OK;
}

float TomoRecon::getTVIter() {
	return iter;
}

TomoError TomoRecon::setTVIter(float TVIter) {
	iter = TVIter;
	if (TVIter < 0.0f) iter = 0.0f;
	return Tomo_OK;
}

TomoError TomoRecon::setBoundaries(float begin, float end) {
	float start = min(begin, end);
	float finish = max(begin, end);

	constants.startDis = start;
	constants.slices = round((finish - start) / constants.pitchZ) + 1;
	Sys.Recon.Nz = constants.slices;
	return Tomo_OK;
}

float TomoRecon::getStartBoundary() {
	return constants.startDis;
}

float TomoRecon::getEndBoundary() {
	return constants.startDis + (constants.slices - 1) * constants.pitchZ;
}

int TomoRecon::getNumSlices() {
	return constants.slices;
}

TomoError TomoRecon::enableGain(bool enable) {
	constants.useGain = enable;
	return Tomo_OK;
}

bool TomoRecon::gainIsEnabled() {
	return constants.useGain;
}

int TomoRecon::getNumViews() {
	return NumViews;
}

void TomoRecon::getProjectionDimensions(int* width, int* height) {
	*width = Sys.Proj.Nx;
	*height = Sys.Proj.Ny;
}

void TomoRecon::getReconstructionDimensions(int* width, int* height) {
	*width = Sys.Recon.Nx;
	*height = Sys.Recon.Ny;
}