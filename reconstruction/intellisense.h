template<typename type1, int type2, int type3> class texture {
public:
	cudaTextureFilterMode filterMode;
	cudaTextureAddressMode* addressMode;
	cudaChannelFormatDesc channelDesc;
};
float tex2D(typename texture, float, float);
void __syncthreads();
float __saturatef(float);
float __expf(float);
cudaError_t cudaMemcpyToSymbol(float, float*, int);
cudaError_t cudaMemcpyToSymbol(int, int*, int);