// 66_cross_device_graph_capture.cu
// Initialized two-device capture with a fork AND a join back to the origin.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
 fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
__global__ void incrementKernel(int *data,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) ++data[i];
}
__global__ void peerIncrementKernel(const int *source,int *destination,int n) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n) destination[i]=source[i]+1; // UVA peer load from device 0
}
int main() {
 int count; CUDA_OK(cudaGetDeviceCount(&count));
 if(count<2) { fprintf(stderr,"Requires two GPUs\n"); return 2; }
 constexpr int n=256,replays=3;
 cudaStream_t s0,s1; cudaEvent_t forkEvent,joinEvent; int *a,*b;
 CUDA_OK(cudaSetDevice(0)); CUDA_OK(cudaStreamCreate(&s0));
 CUDA_OK(cudaEventCreateWithFlags(&forkEvent,cudaEventDisableTiming));
 CUDA_OK(cudaMalloc(&a,n*sizeof(int))); CUDA_OK(cudaMemset(a,0,n*sizeof(int)));
 CUDA_OK(cudaSetDevice(1));
 int peerAccess=0; CUDA_OK(cudaDeviceCanAccessPeer(&peerAccess,1,0));
 if(!peerAccess) { fprintf(stderr,"Requires device 1 peer access to device 0\n"); return 2; }
 CUDA_OK(cudaDeviceEnablePeerAccess(0,0));
 CUDA_OK(cudaStreamCreate(&s1));
 CUDA_OK(cudaEventCreateWithFlags(&joinEvent,cudaEventDisableTiming));
 CUDA_OK(cudaMalloc(&b,n*sizeof(int))); CUDA_OK(cudaMemset(b,0,n*sizeof(int)));
 CUDA_OK(cudaSetDevice(0));
 CUDA_OK(cudaStreamBeginCapture(s0,cudaStreamCaptureModeGlobal));
 incrementKernel<<<1,n,0,s0>>>(a,n); CUDA_OK(cudaGetLastError());
 CUDA_OK(cudaEventRecord(forkEvent,s0));
 CUDA_OK(cudaSetDevice(1)); CUDA_OK(cudaStreamWaitEvent(s1,forkEvent,0));
 peerIncrementKernel<<<1,n,0,s1>>>(a,b,n); CUDA_OK(cudaGetLastError());
 CUDA_OK(cudaEventRecord(joinEvent,s1));
 CUDA_OK(cudaSetDevice(0)); CUDA_OK(cudaStreamWaitEvent(s0,joinEvent,0));
 cudaGraph_t graph; CUDA_OK(cudaStreamEndCapture(s0,&graph));
 cudaGraphExec_t exec; CUDA_OK(cudaGraphInstantiate(&exec,graph,0));
 for(int i=0;i<replays;++i) CUDA_OK(cudaGraphLaunch(exec,s0));
 CUDA_OK(cudaStreamSynchronize(s0));
 int host[n],errors=0;
 CUDA_OK(cudaMemcpy(host,a,sizeof(host),cudaMemcpyDeviceToHost));
 for(int v:host) if(v!=replays) ++errors;
 CUDA_OK(cudaSetDevice(1)); CUDA_OK(cudaMemcpy(host,b,sizeof(host),cudaMemcpyDeviceToHost));
 for(int v:host) if(v!=replays+1) ++errors;
 CUDA_OK(cudaFree(b)); CUDA_OK(cudaEventDestroy(joinEvent)); CUDA_OK(cudaStreamDestroy(s1));
 CUDA_OK(cudaDeviceDisablePeerAccess(0));
 CUDA_OK(cudaSetDevice(0)); CUDA_OK(cudaGraphExecDestroy(exec)); CUDA_OK(cudaGraphDestroy(graph));
 CUDA_OK(cudaFree(a)); CUDA_OK(cudaEventDestroy(forkEvent)); CUDA_OK(cudaStreamDestroy(s0));
 printf("VALIDATION %s: %d replays, %d checked values, errors=%d\n",errors?"FAIL":"PASS",replays,2*n,errors);
 return errors?1:0;
}
