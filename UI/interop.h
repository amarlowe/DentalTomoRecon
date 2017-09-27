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
	int						width = 0;
	int						height = 0;

	// CUDA resources
	cudaArray_t				ca = NULL;
private:
	//Variables

	// GL buffers
	unsigned int			fb;
	unsigned int			rb;

	// CUDA resources
	cudaGraphicsResource_t	cgr = NULL;
};