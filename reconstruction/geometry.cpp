#include "geometry.h"

#define MAX_NUM_OBJECTS		100

#include <opencv2/core/core.hpp>
#include <opencv2\highgui.hpp>
#include "opencv2\imgproc.hpp"
#include <opencv\cv.h>

using namespace cv;

void findGeometry(float* input, int length, int width, int threshold, float * leastSqX, float * leastSqY) {
	//valid component array
	bool m_validcomponent[MAX_NUM_OBJECTS];

	Vec4f m_componentline[MAX_NUM_OBJECTS];
	int m_actualobjcnt;

	//left point array
	float m_lefty[MAX_NUM_OBJECTS];

	// initialize valid component array
	// and left point array
	for (int initcntr = 0; initcntr < MAX_NUM_OBJECTS; initcntr++)
	{
		m_validcomponent[initcntr] = false;
		m_lefty[initcntr] = 0.0;
	}

	bool hitLast = false;
	for (int i = 0; i < length; i++)
		for (int j = 0; j < width; j++) {
			float val = input[i*width + j];
			if (val > threshold) {
				input[i*width + j] = 0.0f;
				hitLast = false;
			}
			else {
				if (val == 0.0f) {
					if(hitLast) input[i*width + j] = 1.0f;
					else input[i*width + j] = 0.0f;
				}
				else {
					input[i*width + j] = 1.0f;
					hitLast = true;
				}
			}
		}

	
	Mat img(length, width, CV_32F, input);

	//convert the image to 8bits
	//image manipulation in OpenCV is on 8 bit data
	Mat img8bit(length, width, CV_8U);
	img.convertTo(img8bit, CV_8U);

	//use connectedComponents to divide objects in image
	Mat labels;
	int num_objects = connectedComponents(img8bit, labels);

	//create matrix for each object
	Mat eachobject = Mat::zeros(img8bit.rows, img8bit.cols, CV_8U);

	//create a black output image to put colored objects on top of
	Mat output = Mat::zeros(img8bit.rows, img8bit.cols, CV_8UC3);

	m_actualobjcnt = num_objects;

	int validcnt = 0, cntr = 0;
	Mat nonZeroCoordinates;

	//loop over objects found skipping the first one since it is the background
	for (int i = 1; i < m_actualobjcnt; i++)
	{
		//grabs one object at a time
		compare(labels, i, eachobject, CMP_EQ);

		cntr = 0;
		int tempx, tempy;

		//find the coordinates of the object
		//count # of pixels in the object
		//found the rods are relatively repeatable sizes
		//tried to limit the size of the object
		//again this is for prototypes
		//hopefully as process is perfected, this will become very repeatable
		//and only objects in the image will be the rods which we want
		//and the rods will be angled correctly such that they don't
		//end up making a line through the wrong axis
		findNonZero(eachobject, nonZeroCoordinates);
		for (int izero = 0; izero < nonZeroCoordinates.total(); izero++)
		{
			tempx = nonZeroCoordinates.at<Point>(izero).x;
			tempy = nonZeroCoordinates.at<Point>(izero).y;
			cntr++;
		}

		//integrated geo phantom has thicker rods so larger area of pixels
		//if the # of pixels in the object falls outside of this range,
		//do not count the object as valid
		if (cntr < 2300 || cntr > 50000)
			m_validcomponent[i] = false;
		else
		{
			m_validcomponent[i] = true;
			validcnt++;
		}

		//find a line fit through the long axis of the rod
		//void fitLine(InputArray points, OutputArray line, int distType, double param, double reps, double aeps)
		//line – Output line parameters.  In case of 2D fitting, it should be a vector of 4 elements(like Vec4f) - (vx, vy, x0, y0), 
		//where(vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line.
		//Y - Y0 = M(X - X0)
		//Y0 = FitedLine[3]; X0 = FitedLine[2]; m = FitedLine[1] / FitedLine[0];
		if (m_validcomponent[i])
		{
			fitLine(nonZeroCoordinates, m_componentline[i], 2, 0, 0.01, 0.01);
			m_lefty[i] = ((-m_componentline[i][2] * m_componentline[i][1] / m_componentline[i][0]) + m_componentline[i][3]);
		}
	}

	float normx, normy;
	float normxsqr;
	float normysqr;
	float normxy;
	int loopcnt;
	float Sxx = 0.0;
	float Syy = 0.0;
	float Sxy = 0.0;
	float Cx = 0.0;
	float Cy = 0.0;
	float x0, y0;

	//as one last check, find the least squares distance between lines recursively
	//eliminate the ones with intersection points too far from the other 
	//intersection points

	//In order to find the intersection point of a set of lines, 
	//we calculate the point with minimum distance to them.
	//Each line is defined by an origin ai
	// and a unit direction vector, ni.
	//The square of the distance from a point p
	//to one of the lines given by Pythagoras
	//least squares application
	//results in matrix form S * p = C
	//where p is the least squares intersection solution

	//find solution recursively
	bool solveDone = false;

	while (!solveDone)
	{
		//initialize least squares variables
		Sxx = 0.0;
		Syy = 0.0;
		Sxy = 0.0;
		Cx = 0.0;
		Cy = 0.0;

		//have line equations in vector form from fitline operation
		for (loopcnt = 1; loopcnt < m_actualobjcnt; loopcnt++)
		{
			if (m_validcomponent[loopcnt])
			{
				//(normx,normy) represents a normalized vector colinear to the line
				normx = m_componentline[loopcnt][0];
				normy = m_componentline[loopcnt][1];
				x0 = m_componentline[loopcnt][2];
				y0 = m_componentline[loopcnt][3];

				normxsqr = normx * normx;
				normysqr = normy * normy;
				normxy = normx * normy;

				//sum of squares of the norm
				Sxx += normxsqr - 1.0f;
				Syy += normysqr - 1.0f;
				Sxy += normxy;

				//CX = sum(PA(:, 1).*(nx. ^ 2 - 1) + PA(:, 2).*(nx.*ny));
				Cx += x0*(normxsqr - 1.0f) + y0*normxy;
				//CY  = sum(PA(:,1).*(nx.*ny)  + PA(:,2).*(ny.^2-1));
				Cy += x0*normxy + y0*(normysqr - 1.0f);
			}
		}

		//get inverted matrix needed for least squares solution
		float a[] = { Sxx,  Sxy,  Sxy,  Syy };
		Mat S = Mat(2, 2, CV_32FC1, a);
		Mat inverted = Mat(2, 2, CV_32FC1);
		invert(S, inverted, cv::DECOMP_SVD);

		//this is how to obtain the data from the OpenCV matrix type
		float* input = (float*)(inverted.data);

		//find inverted*C
		*leastSqX = input[0] * Cx + input[1] * Cy;
		*leastSqY = input[2] * Cx + input[3] * Cy;

		float line1_x1, line1_x2, line1_y1, line1_y2;
		float line2_x1, line2_x2, line2_y1, line2_y2;
		float A1, B1, C1;
		float A2, B2, C2;

		int intersectcnt = 0;
		float det;
		int innerloop;

		float maxX = 0.0;
		float maxY = 0.0;

		float intersectX;
		float intersectY;

		float maxdist = 0.0;
		float distance = 0.0;
		float diffx, diffy;

		//determine if lines intersect
		//if line in eqn form Ax + By = C then can solve set of equations to get intersection
		//have point on line from fitline in x0=m_componentline[i][2] and y0=m_componentline[i][3]
		//have 2nd point from lefty that is not restricted to the image and can use x=0 for 2nd x coord

		//line(cdst, Point((int)leastSqrx, showintersection.rows - 1), Point((int)leastSqrx, 0), Scalar(0, 0, 255), 2);
		//need to get lines that intersect with least squares vertical line through x point.
		line1_x1 = *leastSqX;
		line1_x2 = *leastSqX;
		line1_y1 = *leastSqY;
		line1_y2 = 0;

		//assume solved but if distance is greater than 100 pixels
		//try again
		solveDone = true;

		for (innerloop = 1; innerloop < m_actualobjcnt; innerloop++)
		{
			if (m_validcomponent[innerloop] && solveDone)
			{
				line2_x1 = 0.0;
				line2_x2 = m_componentline[innerloop][2];  //x0 of line# 7
				line2_y1 = m_lefty[innerloop]; //some other y point on line# 7
				line2_y2 = m_componentline[innerloop][3];  //y0 on line# 7

				A1 = line1_y2 - line1_y1;
				B1 = line1_x1 - line1_x2;
				C1 = B1 * line1_y1 + A1 * line1_x1;

				A2 = line2_y2 - line2_y1;
				B2 = line2_x1 - line2_x2;
				C2 = B2 * line2_y1 + A2 * line2_x1;

				det = A1*B2 - A2*B1;

				//if det = 0, lines are parallel and thus do not intersect
				//compare it to a very small # since 0 won't happen on a computer
				//FLT_EPSILON = 1.0e-0.5 or smaller
				if (fabs(det) > FLT_EPSILON)
				{
					//lines intersect so continue
					intersectX = (B2*C1 - B1*C2) / det;
					intersectY = (A1*C2 - A2*C1) / det;

					//find the interection of the least squares line and line from object
					//find distance from that intersection point to the least squares solution point

					diffx = intersectX - *leastSqX;
					diffy = intersectY - *leastSqY;
					distance = sqrt((diffx * diffx) + (diffy * diffy));

					//if the distance is > than this value, this intersection is too far away to be useful
					//don't include it
					if (distance > 120.0)
					{
						m_validcomponent[innerloop] = false;
						solveDone = false;
					}
				}
			}
		}
	}

	//Size size(width / 2, length / 2);
	//resize(img, img, size);
	//imshow("output", img);
}