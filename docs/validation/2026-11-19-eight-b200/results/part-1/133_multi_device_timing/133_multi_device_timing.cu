// 133_multi_device_timing.cu
// Local GPU event durations and end-to-end host completion are different metrics.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
__global__ void increment(int* data,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)data[i]+=1;}
int main(){
    int count=0;CUDA_OK(cudaGetDeviceCount(&count));if(count<2){fprintf(stderr,"Requires 2 GPUs\n");return 2;}
    constexpr int n=1024;std::vector<int*> data(count,nullptr);std::vector<cudaStream_t> streams(count,nullptr);
    std::vector<cudaEvent_t> starts(count,nullptr),stops(count,nullptr);std::vector<float> ms(count);
    for(int d=0;d<count;++d){CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaMalloc(&data[d],n*sizeof(int)));CUDA_OK(cudaMemset(data[d],0,n*sizeof(int)));
        CUDA_OK(cudaStreamCreateWithFlags(&streams[d],cudaStreamNonBlocking));CUDA_OK(cudaEventCreate(&starts[d]));CUDA_OK(cudaEventCreate(&stops[d]));CUDA_OK(cudaDeviceSynchronize());}
    auto begin=std::chrono::steady_clock::now();
    for(int d=0;d<count;++d){CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaEventRecord(starts[d],streams[d]));
        increment<<<4,256,0,streams[d]>>>(data[d],n);CUDA_OK(cudaGetLastError());CUDA_OK(cudaEventRecord(stops[d],streams[d]));}
    for(int d=0;d<count;++d){CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaEventSynchronize(stops[d]));}
    auto end=std::chrono::steady_clock::now();int errors=0;int host[n];
    for(int d=0;d<count;++d){CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaEventElapsedTime(&ms[d],starts[d],stops[d]));
        CUDA_OK(cudaMemcpy(host,data[d],sizeof(host),cudaMemcpyDeviceToHost));for(int v:host)if(v!=1)++errors;
        printf("GPU %d local stream duration: %.6f ms\n",d,ms[d]);}
    printf("Maximum local duration: %.6f ms (GPU clocks are not a shared timeline)\n",*std::max_element(ms.begin(),ms.end()));
    printf("Host launch-to-all-complete: %.6f ms (includes submission and completion waits)\n",std::chrono::duration<double,std::milli>(end-begin).count());
    printf("Independent kernels only; this is not a collective bandwidth benchmark.\n");
    printf("VALIDATION %s: errors=%d\n",errors?"FAIL":"PASS",errors);
    for(int d=0;d<count;++d){CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaEventDestroy(starts[d]));CUDA_OK(cudaEventDestroy(stops[d]));CUDA_OK(cudaStreamDestroy(streams[d]));CUDA_OK(cudaFree(data[d]));}
    return errors?1:0;
}
