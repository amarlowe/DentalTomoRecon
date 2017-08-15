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

TomoError TomoRecon::addMaxLight(int amount) {
	int append = UCHAR_MAX * amount;
	if (append > USHRT_MAX - constants.maxVal || append <= constants.minVal - constants.maxVal) return Tomo_invalid_arg;
	constants.maxVal += append;
	return Tomo_OK;
}

TomoError TomoRecon::addMinLight(int amount) {
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

TomoError TomoRecon::setReconBox(int index) {
	P2R(&upXr, &upYr, upX, upY, 0);
	P2R(&lowXr, &lowYr, lowX, lowY, 0);
	P2R(&constants.currXr, &constants.currYr, currX, currY, 0);
	P2R(&constants.baseXr, &constants.baseYr, baseX, baseY, 0);

	return Tomo_OK;
}