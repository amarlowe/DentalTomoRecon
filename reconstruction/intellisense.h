template<typename type1, int type2, cudaTextureReadMode type3> class texture {
public:
	cudaTextureFilterMode filterMode;
	cudaTextureAddressMode* addressMode;
	cudaChannelFormatDesc channelDesc;
};
template<typename T, int t2, cudaTextureReadMode t3> typename texture;
typedef typename textureReference* texture;
float tex2D(typename texture, float, float);
void __syncthreads();
float __saturatef(float);
float __expf(float);
cudaError_t cudaMemcpyToSymbol(float, float*, int);
cudaError_t cudaMemcpyToSymbol(int, int*, int);
cudaError_t cudaBindTextureToArray(texture, cudaArray*);
void surf2Dwrite(uint1 data, cudaSurfaceObject_t srufObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
unsigned short surf2Dread(cudaSurfaceObject_t srufObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
template<class T> void surf2Dread(T* data, cudaSurfaceObject_t srufObj, int x, int y, cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
float atomicAdd(float*, float);
float __shfl(float var, int srcLane, int width = warpSize);
float __int_as_float(int);
int __float_as_int(float);
int atomicCAS(int*, int, int);
float __shfl_down(float, int);
cudaError_t cudaBindTexture2D(size_t*, const textureReference*, const void*, cudaChannelFormatDesc, size_t, size_t, size_t);
cudaError_t cudaBindSurfaceToArray(const surfaceReference, cudaArray_const_t);