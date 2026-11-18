// 130_cooperative_multi_device.cu
// Capability API probe; does not claim a deprecated multi-device kernel ran.
#include <cstdio>
#include <cuda_runtime.h>
int main() {
 int count=0,errors=0;
 if(cudaGetDeviceCount(&count)!=cudaSuccess || count<1) return 2;
 for(int device=0;device<count;++device) {
  int cooperative=0,multi=0;
  if(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,device)!=cudaSuccess) ++errors;
  if(cudaDeviceGetAttribute(&multi,cudaDevAttrCooperativeMultiDeviceLaunch,device)!=cudaSuccess) ++errors;
  if((cooperative!=0 && cooperative!=1)||(multi!=0 && multi!=1)) ++errors;
  printf("device %d: cooperative=%d deprecated multi-device capability=%d\n",device,cooperative,multi);
 }
 printf("API PROBE %s: capability queries only, errors=%d\n",errors?"FAIL":"PASS",errors);
 return errors?1:0;
}
