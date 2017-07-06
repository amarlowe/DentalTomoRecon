#include "interop.h"
#include "../reconstruction/TomoRecon.h"

//#include <helper_gl.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#include <stdlib.h>
#include <stdio.h>

TomoError gl_assert_void();

#define gl(...)  gl##__VA_ARGS__; gl_assert_void();

TomoError gl_assert_void() {
	GLenum gl_error = glGetError(); 
	if (gl_error != GL_NO_ERROR) {
			std::cout << "GL failure " << __FILE__ << ":" << __LINE__ << ": " << glErrorToString(gl_error) << "\n";
			return Tomo_CUDA_err;
	}
	else return Tomo_OK;
}

#define PXL_KERNEL_THREADS_PER_BLOCK  256 // enough for 4Kx2 monitor

surface<void, cudaSurfaceType2D> surf;

union pxl_rgbx_24
{
	uint1       b32;

	struct {
		unsigned  r : 8;
		unsigned  g : 8;
		unsigned  b : 8;
		unsigned  na : 8;
	};
};

//
//
//

extern "C"
__global__
void
pxl_kernel(const int width, const int height)
{
	// pixel coordinates
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
	const int x = idx % width;
	const int y = idx / width;

#if 0

	// pixel color
	const int          t = (unsigned int)clock() / 1100000; // 1.1 GHz
	const int          xt = (idx + t) % width;
	const unsigned int ramp = (unsigned int)(((float)xt / (float)(width - 1)) * 255.0f + 0.5f);
	const unsigned int bar = ((y + t) / 32) & 3;

	union pxl_rgbx_24  rgbx;

	rgbx.r = (bar == 0) || (bar == 1) ? ramp : 0;
	rgbx.g = (bar == 0) || (bar == 2) ? ramp : 0;
	rgbx.b = (bar == 0) || (bar == 3) ? ramp : 0;
	rgbx.na = 255;

#else // DRAW A RED BORDER TO VALIDATE FLIPPED BLIT

	const bool        border = (x < 5) || (x > width - 5) || (y < 5) || (y > height - 5);
	//const bool        border = true;
	union pxl_rgbx_24 rgbx = { border ? 0xFF0000FF : 0xFF000000 };

#endif

	surf2Dwrite(rgbx.b32, // even simpler: (unsigned int)clock()
		surf,
		x * sizeof(rgbx),
		y,
		cudaBoundaryModeZero); // squelches out-of-bound writes
}

cudaError_t pxl_kernel_launcher(cudaArray_t array,
	const int         width,
	const int         height,
	cudaEvent_t       event,
	cudaStream_t      stream){

	cuda(BindSurfaceToArray(surf, array));

	const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

	if (blocks > 0)
		pxl_kernel <<<blocks, PXL_KERNEL_THREADS_PER_BLOCK, 0, stream >>>(width, height);

	return cudaSuccess;
}

interop::interop(int *argc, char **argv, int x, int y, bool first) {
	if (first) glutInit(argc, argv);
	glewInit();

	gl(ColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE));

	int fbo_count = 2;
	multi_gpu = true;
	count = fbo_count;
	index = 0;

	// allocate arrays
	fb = (GLuint*)calloc(fbo_count, sizeof(*fb));
	rb = (GLuint*)calloc(fbo_count, sizeof(*rb));
	cgr = (cudaGraphicsResource_t*)calloc(fbo_count, sizeof(*cgr));
	ca = (cudaArray_t*)calloc(fbo_count, sizeof(*ca));

	// render buffer object w/a color buffer
	gl(GenRenderbuffers(fbo_count, rb));

	// frame buffer object
	gl(GenFramebuffers(fbo_count, fb));

	// attach rbo to fbo
	for (int index = 0; index<fbo_count; index++){
		gl(BindRenderbuffer(GL_RENDERBUFFER, rb[index]));
		gl(BindFramebuffer(GL_FRAMEBUFFER, fb[index]));
		gl(NamedFramebufferRenderbuffer((GLuint)fb[index], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, (GLuint)rb[index]));
	}
}

interop::~interop() {
	// unregister CUDA resources
	for (int index = 0; index < count; index++){
		if (cgr[index] != NULL)
			cuda(GraphicsUnregisterResource(cgr[index]));
	}

	// delete rbo's
	gl(DeleteRenderbuffers(count, rb));

	// delete fbo's
	gl(DeleteFramebuffers(count, fb));

	// free buffers and resources
	free(fb);
	free(rb);
	free(cgr);
	free(ca);
}

void interop::resize(int x, int y) {
	// save new size
	this->width = x;
	this->height = y;

	// resize color buffer
	for (int index = 0; index < count; index++) {
		// unregister resource
		if (cgr[index] != NULL)
			cuda(GraphicsUnregisterResource(cgr[index]));

		// resize rbo
		gl(NamedRenderbufferStorage((GLuint)rb[index], GL_RGBA8, width, height));

		//const char* test = glErrorToString(glGetError());
		SDK_CHECK_ERROR_GL();

		// register rbo
		cuda(GraphicsGLRegisterImage(&(cgr[index]), (GLuint)rb[index], GL_RENDERBUFFER, cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard));
	}

	// map graphics resources
	cuda(GraphicsMapResources(count, cgr, 0));

	// get CUDA Array refernces
	for (int index = 0; index < count; index++) {
		cuda(GraphicsSubResourceGetMappedArray(&ca[index], cgr[index], 0, 0));
	}

	// unmap graphics resources
	cuda(GraphicsUnmapResources(count, cgr, 0));
}

void interop::display(int x, int y){
	if (x != width || y != height) {
		resize(x, y);
		glViewport(0, 0, (GLint)x, (GLint)y);
	}
	/*width = x;
	height = y;
	glViewport(0, 0, (GLint)x, (GLint)y);

	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, width * height / 4);
	glDisableClientState(GL_VERTEX_ARRAY);

	g_fAnim += 0.01f;*/
}

void interop::map(cudaStream_t stream) {
	// map graphics resources
	cuda(GraphicsMapResources(1, &cgr[index], stream));
}

void interop::unmap(cudaStream_t stream){
	cuda(GraphicsUnmapResources(1, &cgr[index], stream));
}

void interop::swap() {
	index = (index + 1) % count;
}

void interop::clear() {
	GLfloat clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	gl(ClearNamedFramebufferfv(fb[index], GL_COLOR, 0, clear_color));
}

void interop::blit() {
	gl(BlitNamedFramebuffer(fb[index], 0,
		0, 0, width, height,
		0, height, width, 0,
		GL_COLOR_BUFFER_BIT,
		GL_NEAREST));
}