#pragma once
#include <vector>

void findGeometry(float* input, int length, int width, int threshold, float * leastSqX, float * leastSqY);
void getCenterPoints(float* input, std::vector<std::vector<float> > &sortedPairs, int length, int width, int threshold);
void findGeometryCircle(float* input, std::vector<float> &h1, std::vector<float> &h2, std::vector<std::vector<float> > baseline, int length, int width, int threshold);