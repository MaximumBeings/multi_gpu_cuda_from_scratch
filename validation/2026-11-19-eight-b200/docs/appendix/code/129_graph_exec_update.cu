// 129_graph_exec_update.cu
// Capture compatible graphs, update an existing executable, and check both buffers.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
__global__ void add(int* p,int n,int amount){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]+=amount;}
cudaGraph_t capture(cudaStream_t s,int* p,int n,int amount){
    cudaGraph_t g=nullptr;
    CUDA_OK(cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal));
    add<<<1,32,0,s>>>(p,n,amount); CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaStreamEndCapture(s,&g)); return g;
}
int main(){
    constexpr int n=8; int input[n],got[n],*a=nullptr,*b=nullptr;
    for(int i=0;i<n;++i)input[i]=i;
    CUDA_OK(cudaSetDevice(0)); CUDA_OK(cudaMalloc(&a,sizeof(input))); CUDA_OK(cudaMalloc(&b,sizeof(input)));
    CUDA_OK(cudaMemcpy(a,input,sizeof(input),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(b,input,sizeof(input),cudaMemcpyHostToDevice));
    cudaStream_t s=nullptr; CUDA_OK(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    cudaGraph_t first=capture(s,a,n,1); cudaGraphExec_t exec=nullptr;
    CUDA_OK(cudaGraphInstantiate(&exec,first,0)); CUDA_OK(cudaGraphLaunch(exec,s)); CUDA_OK(cudaStreamSynchronize(s));
    CUDA_OK(cudaMemcpy(got,a,sizeof(got),cudaMemcpyDeviceToHost)); int errors=0;
    for(int i=0;i<n;++i)if(got[i]!=i+1)++errors;
    cudaGraph_t second=capture(s,b,n,3); cudaGraphExecUpdateResultInfo info={};
    CUDA_OK(cudaGraphExecUpdate(exec,second,&info));
    if(info.result!=cudaGraphExecUpdateSuccess){fprintf(stderr,"Update rejected: %d\n",(int)info.result);return 1;}
    CUDA_OK(cudaGraphLaunch(exec,s)); CUDA_OK(cudaStreamSynchronize(s));
    CUDA_OK(cudaMemcpy(got,b,sizeof(got),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i)if(got[i]!=i+3)++errors;
    CUDA_OK(cudaMemcpy(got,a,sizeof(got),cudaMemcpyDeviceToHost));
    for(int i=0;i<n;++i)if(got[i]!=i+1)++errors;
    printf("Compatible update: new pointer and scalar argument; original buffer unchanged.\n");
    printf("VALIDATION %s: errors=%d\n",errors?"FAIL":"PASS",errors);
    CUDA_OK(cudaGraphExecDestroy(exec)); CUDA_OK(cudaGraphDestroy(first)); CUDA_OK(cudaGraphDestroy(second));
    CUDA_OK(cudaStreamDestroy(s)); CUDA_OK(cudaFree(a)); CUDA_OK(cudaFree(b)); return errors?1:0;
}
