#include "TomoRecon.h"

/****************************************************************************/
/*								Getters and Setters							*/
/****************************************************************************/

inline TomoError TomoRecon::getLight(unsigned int * minVal, unsigned int * maxVal) {
	*minVal = constants.minVal;
	*maxVal = constants.maxVal;
	return Tomo_OK;
}

inline TomoError TomoRecon::setLight(unsigned int minVal, unsigned int maxVal) {
	constants.minVal = minVal;
	constants.maxVal = maxVal;
	return Tomo_OK;
}



bool TomoRecon::getLogView() {
	return constants.log;
}

TomoError TomoRecon::setLogView(bool useLog) {
	constants.log = useLog;
	return Tomo_OK;
}

inline float TomoRecon::getDistance() {
	return distance;
}

TomoError TomoRecon::setReconBox(int index) {
	P2R(&upXr, &upYr, upX, upY, 0);
	P2R(&lowXr, &lowYr, lowX, lowY, 0);
	P2R(&constants.currXr, &constants.currYr, currX, currY, 0);
	P2R(&constants.baseXr, &constants.baseYr, baseX, baseY, 0);

	return Tomo_OK;
}