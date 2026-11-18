// 127_nccl_point_to_point.cu
// Two same-peer messages use matching issue order on sender and receiver.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#define CUDA_OK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);}}while(0)
#define NCCL_OK(x) do {ncclResult_t e=(x);if(e!=ncclSuccess){fprintf(stderr,"%s: %s\n",#x,ncclGetErrorString(e));exit(1);}}while(0)

int main(){
    int count=0;CUDA_OK(cudaGetDeviceCount(&count));if(count<2){fprintf(stderr,"Requires 2 GPUs\n");return 2;}
    int devices[2]={0,1};ncclComm_t comm[2]={};cudaStream_t streams[2]={};int* buffers[2]={};NCCL_OK(ncclCommInitAll(comm,2,devices));
    for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaMalloc(&buffers[r],2*sizeof(int)));CUDA_OK(cudaStreamCreate(&streams[r]));}
    int input[2]={100,101},got[2]={};CUDA_OK(cudaSetDevice(0));CUDA_OK(cudaMemcpy(buffers[0],input,sizeof(input),cudaMemcpyHostToDevice));
    CUDA_OK(cudaSetDevice(1));CUDA_OK(cudaMemset(buffers[1],0xff,sizeof(got)));
    NCCL_OK(ncclGroupStart());CUDA_OK(cudaSetDevice(0));NCCL_OK(ncclSend(buffers[0],1,ncclInt,1,comm[0],streams[0]));NCCL_OK(ncclSend(buffers[0]+1,1,ncclInt,1,comm[0],streams[0]));
    CUDA_OK(cudaSetDevice(1));NCCL_OK(ncclRecv(buffers[1],1,ncclInt,0,comm[1],streams[1]));NCCL_OK(ncclRecv(buffers[1]+1,1,ncclInt,0,comm[1],streams[1]));NCCL_OK(ncclGroupEnd());
    for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaStreamSynchronize(streams[r]));}
    CUDA_OK(cudaSetDevice(1));CUDA_OK(cudaMemcpy(got,buffers[1],sizeof(got),cudaMemcpyDeviceToHost));int errors=(got[0]!=100)+(got[1]!=101);
    printf("Same-peer message results: %d %d. This two-GPU test does not test scheduling among different remote peers.\n",got[0],got[1]);
    for(int r=0;r<2;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaFree(buffers[r]));CUDA_OK(cudaStreamDestroy(streams[r]));NCCL_OK(ncclCommDestroy(comm[r]));}
    printf("VALIDATION %s: errors=%d\n",errors?"FAIL":"PASS",errors);return errors?1:0;
}
