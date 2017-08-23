/********************************************************************************************/
/* Include a general header																	*/
/********************************************************************************************/
#include "TomoRecon.h"

/********************************************************************************************/
/* Constructor and destructor																*/
/********************************************************************************************/

TomoRecon::TomoRecon(int x, int y, struct SystemControl * Sys) : interop(x, y), Sys(*Sys){
	cuda(StreamCreate(&stream));
}

TomoRecon::~TomoRecon() {
	cudaDeviceSynchronize();
	cuda(StreamDestroy(stream));
	FreeGPUMemory();
	delete[] Sys.Geo.EmitX;
	delete[] Sys.Geo.EmitY;
	delete[] Sys.Geo.EmitZ;
}

TomoError TomoRecon::init(const char * gainFile, const char * darkFile, const char * mainFile) {
	NumViews = NUMVIEWS;

	//Step 4. Set up the GPU for Reconstruction
	tomo_err_throw(initGPU(gainFile, darkFile, mainFile));

	zoom = 0;
	xOff = 0;
	yOff = 0;

	return Tomo_OK;
}

TomoError TomoRecon::setNOOP(float kernel[KERNELSIZE]) {
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		kernel[i + KERNELRADIUS] = 0.0;
	}
	kernel[KERNELRADIUS] = 1.0;

	return Tomo_OK;
}

TomoError TomoRecon::setGauss(float kernel[KERNELSIZE]) {
	float factor = 1.0f / ((float)sqrt(2.0 * M_PI) * SIGMA);
	float denom = 2.0f * pow(SIGMA, 2);
	float sum = 0.0f;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = factor * exp(-pow((float)i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}
	sum--;//Make it sum to 1, not 0

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}

TomoError TomoRecon::setGaussDer(float kernel[KERNELSIZE]) {
	float factor = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA,3));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = -i * factor * exp(-pow((float)i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		if(i < 0) sum += temp;
	}

	//must make sum = 1
	sum -= 1;
	sum /= KERNELRADIUS;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		if (i < 0) kernel[i + KERNELRADIUS] -= sum;
		else if(i > 0) kernel[i + KERNELRADIUS] += sum;
	}


	return Tomo_OK;
}

TomoError TomoRecon::setGaussDer2(float kernel[KERNELSIZE]) {
	float factor1 = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA, 3));
	float factor2 = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA, 5));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = (pow((float)i,2) * factor2 - factor1) * exp(-pow((float)i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}

TomoError TomoRecon::setGaussDer3(float kernel[KERNELSIZE]) {
	float factor1 = 1 / pow(SIGMA, 2);
	float factor2 = 1 / ((float)sqrt(2.0 * M_PI) * pow(SIGMA, 5));
	float denom = 2 * pow(SIGMA, 2);
	float sum = 0;
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++) {
		float temp = i * factor2*(3 - pow((float)i, 2)*factor1) * exp(-pow((float)i, 2) / denom);
		kernel[i + KERNELRADIUS] = temp;
		sum += temp;
	}

	//must make sum = 0
	sum /= KERNELSIZE;

	//subtracting sum/variables is constrained optimization of gaussian
	for (int i = -KERNELRADIUS; i <= KERNELRADIUS; i++)
		kernel[i + KERNELRADIUS] -= sum;

	return Tomo_OK;
}

//difference of differences in x and y (normalizes for derivative)
void TomoRecon::diver(float *z, float * d, int n){
	double a, b;
	int i, j, adr;

	adr = 0;
	for (i = 0; i < n; i++) {
		if (i == 0) d[adr] = z[adr];
		else if (i == n - 1) d[adr] = -z[adr - 1];
		else d[adr] = z[adr] - z[adr - 1];
		adr++;
	}
}

//difference with right/upper neighbor
void TomoRecon::nabla(float *u, float *g, int n){
	int i, j, adr;

	adr = 0;
	for (i = 0; i < n; i++) {
		if (i == (n - 1)) g[adr] = 0;
		else g[adr] = u[adr + 1] - u[adr];
		adr++;
	}
}

//average difference on either side
void TomoRecon::lapla(float *a, float *b, int n){
	int x, y, idx = 0;
	for (x = 0; x<n; x++){
		float AX = 0, BX = 0;
		if (x>0) { BX += a[idx - 1]; AX++; }
		if (x<n - 1) { BX += a[idx + 1]; AX++; }
		b[idx] = -AX*a[idx] + BX;
		idx++;
	}
}