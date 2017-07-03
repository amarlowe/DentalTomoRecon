#pragma once
#include <cuda_runtime.h>

class interop {
public:
	interop(int *argc, char **argv, int x, int y, bool first);
	~interop();

	void display(int x, int y);
private:
	bool runTest(int argc, char **argv, char *ref_file);
	void cleanup();

	// GL functionality
	bool initGL(int *argc, char **argv);
	void createVBO(unsigned int *vbo, struct cudaGraphicsResource **vbo_res,
		unsigned int vbo_res_flags);
	void deleteVBO(unsigned int *vbo, struct cudaGraphicsResource *vbo_res);

	// Cuda functionality
	void runCuda(struct cudaGraphicsResource **vbo_resource);
	void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);
	void runAutoTest(int devID, char **argv, char *ref_file);
	void checkResultCuda(int argc, char **argv, const unsigned int &vbo);

	cudaError_t pxl_kernel_launcher(cudaArray_const_t array,
		const int         width,
		const int         height,
		cudaEvent_t       event,
		cudaStream_t      stream);

	//Variables

	// vbo variables
	unsigned int vbo;
	struct cudaGraphicsResource *cuda_vbo_resource;
	void *d_vbo_buffer = NULL;

	float g_fAnim = 0.0;

	// mouse controls
	int mouse_old_x, mouse_old_y;
	int mouse_buttons = 0;
	float rotate_x = 0.0, rotate_y = 0.0;
	float translate_z = -3.0;

	// Auto-Verification Code
	int fpsCount = 0;        // FPS count for averaging
	int fpsLimit = 1;        // FPS limit for sampling
	int g_Index = 0;
	float avgFPS = 0.0f;
	unsigned int frameCount = 0;
	unsigned int g_TotalErrors = 0;
	bool g_bQAReadback = false;
	int width;
	int height;

	int *pArgc = NULL;
	char **pArgv = NULL;

};