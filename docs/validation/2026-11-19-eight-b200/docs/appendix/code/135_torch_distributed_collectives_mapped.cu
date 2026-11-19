// 135_torch_distributed_collectives_mapped.cu
// Semantic reference table plus a checked C++ NCCL initialization probe.
// This program does not execute or trace PyTorch itself.
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>
int main(){
    const char* python[]={"dist.broadcast","dist.reduce","dist.all_reduce","dist.all_gather","dist.reduce_scatter","dist.isend / dist.irecv"};
    const char* nccl[]={"ncclBroadcast","ncclReduce","ncclAllReduce","ncclAllGather","ncclReduceScatter","ncclSend / ncclRecv"};
    for(int i=0;i<6;++i)printf("%-24s -> %s (CUDA/NCCL backend)\n",python[i],nccl[i]);
    int version=0;ncclResult_t r=ncclGetVersion(&version);if(r!=ncclSuccess)return 1;
    printf("C++ linked NCCL: %d.%d.%d. A PyTorch installation may load a different NCCL build.\n",version/10000,(version/100)%100,version%100);
    cudaError_t e=cudaSetDevice(0);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));return 1;}
    ncclUniqueId id;ncclComm_t comm=nullptr;r=ncclGetUniqueId(&id);
    if(r==ncclSuccess)r=ncclCommInitRank(&comm,1,id,0);
    if(r==ncclSuccess)r=ncclCommDestroy(comm);
    if(r!=ncclSuccess){fprintf(stderr,"NCCL: %s\n",ncclGetErrorString(r));return 1;}
    printf("API PROBE PASS: errors=0; semantic mapping and one-rank C++ setup, not a PyTorch execution test.\n");return 0;
}
