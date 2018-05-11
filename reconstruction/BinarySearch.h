#pragma once
class BinarySearch
{
public:
	float startStep;
	float limit;
	int dimensions;
	float * var;

	void (*update)();
	float (*getError)();

	BinarySearch * nest;

	BinarySearch(float * var, int dimensions = 1, float startStep = 1.0f, float limit = 3.0f);
	BinarySearch(const BinarySearch& other);
	BinarySearch(BinarySearch&& other);
	BinarySearch& operator=(BinarySearch other);
	~BinarySearch();

	void run();
};

