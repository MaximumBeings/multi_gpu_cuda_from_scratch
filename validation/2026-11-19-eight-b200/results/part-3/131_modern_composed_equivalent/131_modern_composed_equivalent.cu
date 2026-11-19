// 131_modern_composed_equivalent.cu
// A cooperative grid on one device plus a captured cross-device kernel dependency.
// The event edge is between kernels; it is NOT an in-kernel multi-GPU barrier.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
__global__ void first(int* p,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]*=2;cg::this_grid().sync();if(i<n)p[i]+=1;}
__global__ void second(int* p,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]+=100;}
int main(){
    int count=0;CUDA_OK(cudaGetDeviceCount(&count));if(count<2){fprintf(stderr,"Requires 2 GPUs\n");return 2;}
    constexpr int n=64;int host[n],got[n],*p[2]={};cudaStream_t streams[2]={};cudaEvent_t events[2]={};
    for(int i=0;i<n;++i)host[i]=i;
    for(int d=0;d<2;++d){
        CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaMalloc(&p[d],sizeof(host)));
        CUDA_OK(cudaMemcpy(p[d],host,sizeof(host),cudaMemcpyHostToDevice));
        CUDA_OK(cudaStreamCreateWithFlags(&streams[d],cudaStreamNonBlocking));
        CUDA_OK(cudaEventCreateWithFlags(&events[d],cudaEventDisableTiming));
    }
    int cooperative=0;CUDA_OK(cudaDeviceGetAttribute(&cooperative,cudaDevAttrCooperativeLaunch,0));
    if(!cooperative){fprintf(stderr,"Cooperative launch unsupported\n");return 2;}
    CUDA_OK(cudaSetDevice(0));CUDA_OK(cudaStreamBeginCapture(streams[0],cudaStreamCaptureModeGlobal));
    cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeCooperative;attr.val.cooperative=1;
    cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(2);cfg.blockDim=dim3(32);cfg.stream=streams[0];cfg.attrs=&attr;cfg.numAttrs=1;
    CUDA_OK(cudaLaunchKernelEx(&cfg,first,p[0],n));CUDA_OK(cudaEventRecord(events[0],streams[0]));
    CUDA_OK(cudaSetDevice(1));CUDA_OK(cudaStreamWaitEvent(streams[1],events[0],0));
    second<<<2,32,0,streams[1]>>>(p[1],n);CUDA_OK(cudaGetLastError());CUDA_OK(cudaEventRecord(events[1],streams[1]));
    CUDA_OK(cudaSetDevice(0));CUDA_OK(cudaStreamWaitEvent(streams[0],events[1],0));
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;CUDA_OK(cudaStreamEndCapture(streams[0],&graph));
    CUDA_OK(cudaGraphInstantiate(&exec,graph,0));CUDA_OK(cudaGraphLaunch(exec,streams[0]));CUDA_OK(cudaStreamSynchronize(streams[0]));
    int errors=0;
    for(int d=0;d<2;++d){CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaMemcpy(got,p[d],sizeof(got),cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i)if(got[i]!=(d==0?2*i+1:i+100))++errors;}
    printf("Two cooperative blocks on GPU0; whole-kernel event dependency to GPU1. Separate buffers, no data transfer.\n");
    printf("VALIDATION %s: errors=%d\n",errors?"FAIL":"PASS",errors);
    CUDA_OK(cudaSetDevice(0));CUDA_OK(cudaGraphExecDestroy(exec));CUDA_OK(cudaGraphDestroy(graph));
    for(int d=0;d<2;++d){CUDA_OK(cudaSetDevice(d));CUDA_OK(cudaEventDestroy(events[d]));CUDA_OK(cudaStreamDestroy(streams[d]));CUDA_OK(cudaFree(p[d]));}
    return errors?1:0;
}
