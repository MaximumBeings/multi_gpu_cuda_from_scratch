// 12_staged_host_transfer.cu
// Checked teaching implementation; serial host scheduling favors clarity, not speed.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CUDA(call) do { cudaError_t status = (call); if (status != cudaSuccess) { \
 fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(status)); \
 return status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver ? 2 : 1; } } while(0)
int main(){
 int count=0;CUDA(cudaGetDeviceCount(&count));if(count<2){printf("BLOCKED: requires 2 GPUs; found %d.\n",count);return 2;}
 const int n=4096;const size_t bytes=n*sizeof(float);std::vector<float> input(n),output(n);
 for(int i=0;i<n;++i)input[i]=float((i*17)%4093-2046)/8.0f;
 float *source=nullptr,*destination=nullptr,*staging=nullptr;
 CUDA(cudaSetDevice(0));CUDA(cudaMalloc(&source,bytes));CUDA(cudaMemcpy(source,input.data(),bytes,cudaMemcpyHostToDevice));CUDA(cudaMallocHost(&staging,bytes));
 CUDA(cudaMemcpy(staging,source,bytes,cudaMemcpyDeviceToHost));
 CUDA(cudaSetDevice(1));CUDA(cudaMalloc(&destination,bytes));CUDA(cudaMemset(destination,0,bytes));CUDA(cudaMemcpy(destination,staging,bytes,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(output.data(),destination,bytes,cudaMemcpyDeviceToHost));
 for(int i=0;i<n;++i)if(output[i]!=input[i]){fprintf(stderr,"FAIL staged transfer[%d]\n",i);return 1;}
 CUDA(cudaFree(destination));CUDA(cudaSetDevice(0));CUDA(cudaFree(source));CUDA(cudaFreeHost(staging));
 printf("PASS: 4096 nonuniform values verified through explicit pinned-host staging.\n");return 0;
}
