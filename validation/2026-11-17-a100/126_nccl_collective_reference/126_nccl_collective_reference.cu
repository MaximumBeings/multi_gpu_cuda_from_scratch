// 126_nccl_collective_reference.cu
// Six collective recipes, checked against exact integer results on two GPUs.
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
    constexpr int ranks=2;int devices[ranks]={0,1};ncclComm_t comm[ranks]={};cudaStream_t stream[ranks]={};int *send[ranks]={},*recv[ranks]={};
    NCCL_OK(ncclCommInitAll(comm,ranks,devices));
    for(int r=0;r<ranks;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaStreamCreate(&stream[r]));CUDA_OK(cudaMalloc(&send[r],2*sizeof(int)));CUDA_OK(cudaMalloc(&recv[r],2*sizeof(int)));}
    int errors=0;
    for(int op=0;op<6;++op){
        for(int r=0;r<ranks;++r){CUDA_OK(cudaSetDevice(r));int input[2]={r+1,r+1};if(op==5){input[0]=100*r;input[1]=100*r+1;}
            CUDA_OK(cudaMemcpy(send[r],input,sizeof(input),cudaMemcpyHostToDevice));CUDA_OK(cudaMemset(recv[r],0xff,sizeof(input)));}
        NCCL_OK(ncclGroupStart());
        for(int r=0;r<ranks;++r){CUDA_OK(cudaSetDevice(r));
            if(op==0)NCCL_OK(ncclBroadcast(send[r],recv[r],1,ncclInt,0,comm[r],stream[r]));
            if(op==1)NCCL_OK(ncclReduce(send[r],recv[r],1,ncclInt,ncclSum,0,comm[r],stream[r]));
            if(op==2)NCCL_OK(ncclAllReduce(send[r],recv[r],1,ncclInt,ncclSum,comm[r],stream[r]));
            if(op==3)NCCL_OK(ncclAllGather(send[r],recv[r],1,ncclInt,comm[r],stream[r]));
            if(op==4)NCCL_OK(ncclReduceScatter(send[r],recv[r],1,ncclInt,ncclSum,comm[r],stream[r]));
            if(op==5)for(int peer=0;peer<ranks;++peer){NCCL_OK(ncclSend(send[r]+peer,1,ncclInt,peer,comm[r],stream[r]));NCCL_OK(ncclRecv(recv[r]+peer,1,ncclInt,peer,comm[r],stream[r]));}
        }NCCL_OK(ncclGroupEnd());
        for(int r=0;r<ranks;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaStreamSynchronize(stream[r]));int got[2];CUDA_OK(cudaMemcpy(got,recv[r],sizeof(got),cudaMemcpyDeviceToHost));
            if(op==0)errors+=got[0]!=1;if(op==1&&r==0)errors+=got[0]!=3;if(op==2||op==4)errors+=got[0]!=3;
            if(op==3)errors+=(got[0]!=1)+(got[1]!=2);if(op==5)errors+=(got[0]!=r)+(got[1]!=100+r);}
        const char* names[]={"Broadcast","Reduce","AllReduce","AllGather","ReduceScatter","Grouped Send/Recv all-to-all"};printf("%s checked; cumulative errors=%d\n",names[op],errors);
    }
    for(int r=0;r<ranks;++r){CUDA_OK(cudaSetDevice(r));CUDA_OK(cudaFree(send[r]));CUDA_OK(cudaFree(recv[r]));CUDA_OK(cudaStreamDestroy(stream[r]));NCCL_OK(ncclCommDestroy(comm[r]));}
    printf("VALIDATION %s: errors=%d\n",errors?"FAIL":"PASS",errors);return errors?1:0;
}
