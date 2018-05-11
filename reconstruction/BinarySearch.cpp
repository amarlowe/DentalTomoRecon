#include "BinarySearch.h"
#include <algorithm>

using namespace std;

BinarySearch::BinarySearch(float * var, int dimensions, float startStep, float limit)
	: var(var), dimensions(dimensions), startStep(startStep), limit(limit) {
	if (dimensions > 1) {
		nest = new BinarySearch(var + 1, dimensions - 1);
	}
}

BinarySearch::BinarySearch(const BinarySearch& other) {

}

BinarySearch::BinarySearch(BinarySearch&& other) {

}

BinarySearch& BinarySearch::operator=(BinarySearch other) {
	swap(*this, other);
	return *this;
}

BinarySearch::~BinarySearch() {
	if (dimensions > 1) delete nest;
}

void BinarySearch::run() {
	float bestVar = *var;
	float bestErr = getError();
	float startVar = *var;
	for (*var -= limit; *var < startVar + limit; *var += startStep) {
		float newErr = getError();
		update();
		if (newErr < bestErr) {
			bestErr = newErr;
			bestVar = *var;
		}
	}

}
