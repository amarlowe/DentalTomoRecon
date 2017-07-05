#pragma once
#include <cuda_runtime.h>

class interop {
public:
	interop(int *argc, char **argv, int x, int y, bool first);
	~interop();

	void display(int x, int y);
private:
	// GL functionality
	void createVBO(unsigned int *vbo, struct cudaGraphicsResource **vbo_res,
		unsigned int vbo_res_flags);
	void deleteVBO(unsigned int *vbo, struct cudaGraphicsResource *vbo_res);

	// Cuda functionality
	void runCuda(struct cudaGraphicsResource **vbo_resource);
	void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);

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
	int g_Index = 0;
	unsigned int g_TotalErrors = 0;
	bool g_bQAReadback = false;
	int width;
	int height;
};