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

#define MAX(a,b) ((a > b) ? a : b)

__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width + x] = make_float4(u, w, v, 1.0f);
}

interop::interop(int *argc, char **argv, int x, int y, bool first) {
	width = x;
	height = y;
	
	if (first) glutInit(argc, argv);
	glewInit();
	//glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	//glutInitWindowSize(x, y);

	/*glClearColor(0.0, 0.0, 0.0, 0.0);
	glDisable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT);
	glViewport(0, 0, (GLint)x, (GLint)y);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)x / (GLfloat)y, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();

	//use device with highest Gflops/s
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// run the cuda part
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		cuda_vbo_resource));

	launch_kernel(dptr, (GLfloat)x / 2, (GLfloat)y / 2, g_fAnim);*/

	int fbo_count = 2;
	this->multi_gpu = true;
	this->count = fbo_count;
	this->index = 0;

	// allocate arrays
	this->fb = (GLuint*)calloc(fbo_count, sizeof(*(this->fb)));
	this->rb = (GLuint*)calloc(fbo_count, sizeof(*(this->rb)));
	this->cgr = (cudaGraphicsResource_t*)calloc(fbo_count, sizeof(*(this->cgr)));
	this->ca = (cudaArray_t*)calloc(fbo_count, sizeof(*(this->ca)));

	// render buffer object w/a color buffer
	glGenRenderbuffers(fbo_count, this->rb);

	// frame buffer object
	glGenFramebuffers(fbo_count, this->fb);

	// attach rbo to fbo
	for (int index = 0; index<fbo_count; index++){
		glNamedFramebufferRenderbuffer(this->fb[index], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, this->rb[index]);
	}

	/*
	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

	//Step 1: Get and example file for get the path
#ifdef PROFILER
	char filename[] = "C:\\Users\\jdean\\Desktop\\Patient471\\Series1 20161118\\AcquiredImage1_0.raw";
#else
	char filename[MAX_PATH];

	OPENFILENAME ofn;
	ZeroMemory(&filename, sizeof(filename));
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = NULL;  // If you have a window to center over, put its HANDLE here
	ofn.lpstrFilter = "Raw File\0*.raw\0Any File\0*.*\0";
	ofn.lpstrFile = filename;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = "Select one raw image file";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	GetOpenFileNameA(&ofn);
#endif

	//Seperate base path from the example file path
	char * GetFilePath;
	std::string BasePath;
	std::string FilePath;
	std::string FileName;
	std::string savefilename = filename;
	GetFilePath = filename;
	PathRemoveFileSpec(GetFilePath);
	FileName = PathFindFileName(GetFilePath);
	FilePath = GetFilePath;
	PathRemoveFileSpec(GetFilePath);

	//Define Base Path
	BasePath = GetFilePath;
	if (CheckFilePathForRepeatScans(BasePath)) {
		FileName = PathFindFileName(GetFilePath);
		FilePath = GetFilePath;
		PathRemoveFileSpec(GetFilePath);
		BasePath = GetFilePath;
	}

	//Output FilePaths
	std::cout << "Reconstructing image set entitled: " << FileName << std::endl;

	//Step 2. Initialize structure and read emitter geometry
	const int NumViews = NUMVIEWS;
	struct SystemControl * Sys = new SystemControl;
	tomo_err_throw(SetUpSystemAndReadGeometry(Sys, NumViews, BasePath));

	//Step 3. Read the normalizaton data (dark and gain)
	PathRemoveFileSpec(GetFilePath);
	std::string GainPath = GetFilePath;
	//	ReadDarkandGainImages(Sys, NumViews, GainPath);
	tomo_err_throw(ReadDarkImages(Sys, NumViews));
	tomo_err_throw(ReadGainImages(Sys, NumViews));

	//Step 4. Set up the GPU for Reconstruction
	tomo_err_throw(SetUpGPUForRecon(Sys));
	std::cout << "GPU Ready" << std::endl;

	//Step 5. Read Raw Data
	tomo_err_throw(ReadRawProjectionData(Sys, NumViews, FilePath, savefilename));
	std::cout << "Add Data has been read" << std::endl;*/
}

interop::~interop() {
	if (vbo) deleteVBO(&vbo, cuda_vbo_resource);
}
/*
void interop::createVBO(unsigned int *vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags) {
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = width * height * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}
*/
/*
void interop::deleteVBO(unsigned int *vbo, struct cudaGraphicsResource *vbo_res){
	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}
*/

void interop::display(int x, int y){
	width = x;
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

	g_fAnim += 0.01f;
}

void interop::launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time){
	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel <<< grid, block >>>(pos, mesh_width, mesh_height, time);
}

void interop::runCuda(struct cudaGraphicsResource **vbo_resource){
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	launch_kernel(dptr, width/2, height/2, g_fAnim);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

/*void pxl_interop_destroy(struct pxl_interop* const interop){
  cudaError_t cuda_err;

  // unregister CUDA resources
  for (int index=0; index<interop->count; index++)
    {
      if (interop->cgr[index] != NULL)
        cuda_err = cudaGraphicsUnregisterResource(interop->cgr[index]);
    }

  // delete rbo's
  glDeleteRenderbuffers(interop->count,interop->rb);

  // delete fbo's
  glDeleteFramebuffers(interop->count,interop->fb);

  // free buffers and resources
  free(interop->fb);
  free(interop->rb);
  free(interop->cgr);
  free(interop->ca);

  // free interop
  free(interop);
}

cudaError_t pxl_interop_size_set(struct pxl_interop* const interop, const int width, const int height){
  cudaError_t cuda_err = cudaSuccess;

  // save new size
  interop->width  = width;
  interop->height = height;

  // resize color buffer
  for (int index=0; index<interop->count; index++)
    {
      // unregister resource
      if (interop->cgr[index] != NULL)
        cuda_err = cudaGraphicsUnregisterResource(interop->cgr[index]);

      // resize rbo
      glNamedRenderbufferStorage(interop->rb[index],GL_RGBA8,width,height);

      // probe fbo status
      // glCheckNamedFramebufferStatus(interop->fb[index],0);

      // register rbo
      cuda_err = cudaGraphicsGLRegisterImage(&interop->cgr[index],
					      interop->rb[index],
					      GL_RENDERBUFFER,
					      cudaGraphicsRegisterFlagsSurfaceLoadStore | 
					      cudaGraphicsRegisterFlagsWriteDiscard);
    }

  // map graphics resources
  cuda_err = cudaGraphicsMapResources(interop->count,interop->cgr,0);

  // get CUDA Array refernces
  for (int index=0; index<interop->count; index++)
    {
      cuda_err = cudaGraphicsSubResourceGetMappedArray(&interop->ca[index],
							interop->cgr[index],
							0,0);
    }

  // unmap graphics resources
  cuda_err = cudaGraphicsUnmapResources(interop->count,interop->cgr,0);
  
  return cuda_err;
}

void pxl_interop_size_get(struct pxl_interop* const interop, int* const width, int* const height){
  *width  = interop->width;
  *height = interop->height;
}

//
//
//

cudaError_t pxl_interop_map(struct pxl_interop* const interop, cudaStream_t stream){
  if (!interop->multi_gpu)
    return cudaSuccess;

  // map graphics resources
  return cudaGraphicsMapResources(1,&interop->cgr[interop->index],stream);
}
 
cudaError_t
pxl_interop_unmap(struct pxl_interop* const interop, cudaStream_t stream)
{
  if (!interop->multi_gpu)
    return cudaSuccess;

  return cudaGraphicsUnmapResources(1, &interop->cgr[interop->index], stream);
}

cudaError_t pxl_interop_array_map(struct pxl_interop* const interop){
  //
  // FIXME -- IS THIS EVEN NEEDED?
  //

  cudaError_t cuda_err;
  
  // get a CUDA Array
  cuda_err = cudaGraphicsSubResourceGetMappedArray(&interop->ca[interop->index],
						    interop->cgr[interop->index],
						    0,0);
  return cuda_err;
}

//
//
//

cudaArray_const_t pxl_interop_array_get(struct pxl_interop* const interop){
  return interop->ca[interop->index];
}

int pxl_interop_index_get(struct pxl_interop* const interop){
  return interop->index;
}

//
//
//

void pxl_interop_swap(struct pxl_interop* const interop){
  interop->index = (interop->index + 1) % interop->count;
}

void pxl_interop_clear(struct pxl_interop* const interop){

  const GLfloat clear_color[] = { 1.0f, 1.0f, 1.0f, 1.0f };
  glDeleteFrameBuffers()
  glClearNamedFramebufferfv(interop->fb[interop->index],GL_COLOR,0,clear_color);
}


void pxl_interop_blit(struct pxl_interop* const interop){
  glBlitNamedFramebuffer(interop->fb[interop->index],0,
                         0,0,              interop->width,interop->height,
                         0,interop->height,interop->width,0,
                         GL_COLOR_BUFFER_BIT,
                         GL_NEAREST);
}*/