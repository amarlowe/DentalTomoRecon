#pragma once
#include <cuda_runtime.h>

void reconGlutInit(int *argc, char **argv);

class interop {
public:
	//Functions

	interop(int x, int y);
	void resize(int x, int y);
	~interop();

	void display(int x, int y);
	void map(cudaStream_t stream);
	void unmap(cudaStream_t stream);
	void blit();

	//Variables

	// w x h
	int						width = 0;
	int						height = 0;

	//fbo info
	int						index;

	// CUDA resources
	cudaArray_t*			ca;
private:
	//Functions

	//Variables

	cudaStream_t ourstream;
	cudaEvent_t  ourevent;

	// split GPUs?
	bool					multi_gpu;

	// number of fbo's
	int						count;

	// GL buffers
	unsigned int*			fb;
	unsigned int*			rb;

	// CUDA resources
	cudaGraphicsResource_t*	cgr;
};