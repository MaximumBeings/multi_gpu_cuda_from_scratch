// 16_overlap_query.cu
// Capability query only: no measured overlap or throughput claim.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
 int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count<1) return 2;
 int engines=-1,kernels=-1,errors=0;
 if(cudaDeviceGetAttribute(&engines,cudaDevAttrAsyncEngineCount,0)!=cudaSuccess) ++errors;
 if(cudaDeviceGetAttribute(&kernels,cudaDevAttrConcurrentKernels,0)!=cudaSuccess) ++errors;
 cudaDeviceProp properties{};
 if(cudaGetDeviceProperties(&properties,0)!=cudaSuccess) ++errors;
 if(engines<0 || engines!=properties.asyncEngineCount || kernels!=properties.concurrentKernels) ++errors;
 printf("device 0: asyncEngineCount=%d concurrentKernels=%d\n",engines,kernels);
 printf("API PROBE %s: capability queries only, no overlap benchmark, errors=%d\n",errors?"FAIL":"PASS",errors);
 return errors?1:0;
}
