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

float TomoRecon::getDistance() {
	return distance;
}

TomoError TomoRecon::setDistance(float dist) {
	distance = dist;
	return Tomo_OK;
}

TomoError TomoRecon::stepDistance(int steps) {
	distance += Sys.Geo.ZPitch * steps;
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

TomoError TomoRecon::setActiveProjection(int index) {
	if (index < 0 || index > constants.Views) return Tomo_invalid_arg;
	sliceIndex = index;
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
	int maxX = (int)((Sys.Recon.Nx - width * scale) / 2.0f);
	int maxY = (int)((Sys.Recon.Ny - height * scale) / 2.0f);
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

inline TomoError TomoRecon::checkBoundaries(int * x, int * y) {
	if (*x > Sys.Recon.Nx) *x = Sys.Recon.Nx;
	if (*x < 0) *x = 0;
	if (*y > Sys.Recon.Ny) *y = Sys.Recon.Ny;
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

TomoError TomoRecon::appendZoom(int amount) {
	zoom += amount;
	if (zoom < 0) zoom = 0;
	return Tomo_OK;
}

TomoError TomoRecon::setZoom(int value) {
	zoom = value;
	if (zoom < 0) zoom = 0;
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