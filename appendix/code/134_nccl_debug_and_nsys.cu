// 134_nccl_debug_and_nsys.cu
// Run once normally and once with NCCL_DEBUG=INFO to compare library diagnostics.
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>
int main(){
    cudaError_t e=cudaSetDevice(0);if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));return 1;}
    ncclUniqueId id;ncclComm_t comm=nullptr;ncclResult_t r=ncclGetUniqueId(&id);
    if(r==ncclSuccess)r=ncclCommInitRank(&comm,1,id,0);
    if(r!=ncclSuccess){fprintf(stderr,"NCCL initialization: %s\n",ncclGetErrorString(r));return 1;}
    r=ncclCommDestroy(comm);if(r!=ncclSuccess){fprintf(stderr,"NCCL destroy: %s\n",ncclGetErrorString(r));return 1;}
    printf("API PROBE PASS: errors=0; one-rank initialization and cleanup, no collective throughput measured.\n");
    printf("Set NCCL_DEBUG=INFO before starting the process; library logging is separate from application output.\n");return 0;
}
