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

void reconGlutInit(int *argc, char **argv) {
	glutInit(argc, argv);
}

interop::interop(int x, int y) {
	glewInit();

	gl(ColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE));

	int fbo_count = 1;
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
	for (int i = 0; i<fbo_count; i++){
		gl(BindRenderbuffer(GL_RENDERBUFFER, rb[i]));
		gl(BindFramebuffer(GL_FRAMEBUFFER, fb[i]));
		gl(NamedFramebufferRenderbuffer((GLuint)fb[i], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, (GLuint)rb[i]));
	}
}

interop::~interop() {
	cudaDeviceSynchronize();

	// unregister CUDA resources
	//TODO: investigate necessity
	/*for (int i = 0; i < count; i++){
		if (cgr[i] != NULL)
			cuda(GraphicsUnregisterResource(cgr[i]));
	}*/

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
	width = x;
	height = y;

	// resize color buffer
	for (int i = 0; i < count; i++) {
		// unregister resource
		if (cgr[i] != NULL)
			cuda(GraphicsUnregisterResource(cgr[i]));

		// resize rbo
		gl(NamedRenderbufferStorage((GLuint)rb[i], GL_RGBA8, width, height));

		//const char* test = glErrorToString(glGetError());
		SDK_CHECK_ERROR_GL();

		// register rbo
		cuda(GraphicsGLRegisterImage(&(cgr[i]), (GLuint)rb[i], GL_RENDERBUFFER, cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard));
	}

	// map graphics resources
	cuda(GraphicsMapResources(count, cgr, 0));

	// get CUDA Array refernces
	for (int i = 0; i < count; i++) {
		cuda(GraphicsSubResourceGetMappedArray(&ca[i], cgr[i], 0, 0));
	}

	// unmap graphics resources
	cuda(GraphicsUnmapResources(count, cgr, 0));
}

void interop::display(int x, int y){
	if (x != width || y != height) {
		resize(x, y);
		glViewport(0, 0, (GLint)x, (GLint)y);
	}
}

void interop::map(cudaStream_t stream) {
	// map graphics resources
	cuda(GraphicsMapResources(1, &cgr[index], stream));
}

void interop::unmap(cudaStream_t stream){
	cuda(GraphicsUnmapResources(1, &cgr[index], stream));
}

void interop::blit() {
	gl(BlitNamedFramebuffer(fb[index], 0,
		0, 0, width, height,
		0, height, width, 0,
		GL_COLOR_BUFFER_BIT,
		GL_NEAREST));
}