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

#define gl(...)  gl##__VA_ARGS__; gl_assert_void(__FILE__, __LINE__);

TomoError gl_assert_void(const char* const file, const int line) {
	GLenum gl_error = glGetError(); 
	if (gl_error != GL_NO_ERROR) {
			std::cout << "GL failure " << file << ":" << line << ": " << glErrorToString(gl_error) << "\n";
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

	// render buffer object w/a color buffer
	gl(GenRenderbuffers(1, &rb));
	gl(BindRenderbuffer(GL_RENDERBUFFER, rb));

	// frame buffer object
	gl(GenFramebuffers(1, &fb));

	// attach rbo to fbo
	gl(BindFramebuffer(GL_FRAMEBUFFER, fb));
	gl(NamedFramebufferRenderbuffer((GLuint)fb, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, (GLuint)rb));

	resize(x, y);
}

interop::~interop() {
	// delete rbo's
	gl(DeleteRenderbuffers(1, &rb));

	// delete fbo's
	gl(DeleteFramebuffers(1, &fb));
}

void interop::resize(int x, int y) {
	// save new size
	width = x;
	height = y;

	if (cgr != NULL)
		cuda(GraphicsUnregisterResource(cgr));

	// resize rbo
	gl(NamedRenderbufferStorage((GLuint)rb, GL_RGBA8, width, height));

	// register rbo
	cuda(GraphicsGLRegisterImage(&cgr, (GLuint)rb, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard));

	// map graphics resources
	cuda(GraphicsMapResources(1, &cgr, 0));

	// get CUDA Array refernces
	cuda(GraphicsSubResourceGetMappedArray(&ca, cgr, 0, 0));

	// unmap graphics resources
	cuda(GraphicsUnmapResources(1, &cgr, 0));
}

void interop::display(int x, int y){
	if (x != width || y != height) {
		resize(x, y);
		glViewport(0, 0, (GLint)x, (GLint)y);
	}
}

void interop::map(cudaStream_t stream) {
	// map graphics resources
	cuda(GraphicsMapResources(1, &cgr, stream));
}

void interop::unmap(cudaStream_t stream){
	cuda(GraphicsUnmapResources(1, &cgr, stream));
}

void interop::blit() {
	gl(BlitNamedFramebuffer(fb, 0,
		0, 0, width, height,
		0, height, width, 0,
		GL_COLOR_BUFFER_BIT,
		GL_NEAREST));
}