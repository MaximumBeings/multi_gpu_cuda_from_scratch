// 61_gpudirect_rdma_query.cu
// GPUDirect RDMA capability API probe, not a NIC transfer test.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
 if(cudaSetDevice(0)!=cudaSuccess) return 2;
 int supported=-1,options=-1,ordering=-1,errors=0;
 if(cudaDeviceGetAttribute(&supported,cudaDevAttrGPUDirectRDMASupported,0)!=cudaSuccess) ++errors;
 if(cudaDeviceGetAttribute(&options,cudaDevAttrGPUDirectRDMAFlushWritesOptions,0)!=cudaSuccess) ++errors;
 if(cudaDeviceGetAttribute(&ordering,cudaDevAttrGPUDirectRDMAWritesOrdering,0)!=cudaSuccess) ++errors;
 cudaDeviceProp properties{};
 if(cudaGetDeviceProperties(&properties,0)!=cudaSuccess) ++errors;
 if(supported!=properties.gpuDirectRDMASupported || unsigned(options)!=properties.gpuDirectRDMAFlushWritesOptions || ordering!=properties.gpuDirectRDMAWritesOrdering) ++errors;
 printf("device 0 GPUDirectRDMA: supported=%d flushOptions=%d writesOrdering=%d\n",supported,options,ordering);
 if(supported && ((options & cudaFlushGPUDirectRDMAWritesOptionHost) || ordering>=cudaGPUDirectRDMAWritesOrderingOwner)) {
  cudaError_t result=cudaDeviceFlushGPUDirectRDMAWrites(cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,cudaFlushGPUDirectRDMAWritesToOwner);
  printf("owner-scope flush API: %s (no external DMA was submitted)\n",cudaGetErrorString(result));
  if(result!=cudaSuccess) ++errors;
 } else printf("owner-scope host flush not requested: unsupported/unneeded capability path\n");
 printf("API PROBE %s: queries and supported flush API only; no RDMA/NIC transfer validated, errors=%d\n",errors?"FAIL":"PASS",errors);
 return errors?1:0;
}
