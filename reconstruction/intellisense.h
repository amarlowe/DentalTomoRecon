template<typename type1, int type2, cudaTextureReadMode type3> class texture {
public:
	cudaTextureFilterMode filterMode;
	cudaTextureAddressMode* addressMode;
	cudaChannelFormatDesc channelDesc;
};
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