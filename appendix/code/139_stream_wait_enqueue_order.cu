// 139_stream_wait_enqueue_order.cu
// A deterministic missed-dependency example: no intentional concurrent data race.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));exit(1);} } while(0)
__global__ void produce(int* p){*p=42;}
int main(){
    int count=0;CUDA_OK(cudaGetDeviceCount(&count));if(count<2){fprintf(stderr,"Requires 2 GPUs\n");return 2;}
    int *source=nullptr,*destination=nullptr;cudaStream_t producer=nullptr,consumer=nullptr;cudaEvent_t ready=nullptr;
    CUDA_OK(cudaSetDevice(0));CUDA_OK(cudaMalloc(&source,sizeof(int)));CUDA_OK(cudaMemset(source,0,sizeof(int)));
    CUDA_OK(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking));CUDA_OK(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaSetDevice(1));CUDA_OK(cudaMalloc(&destination,sizeof(int)));CUDA_OK(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking));
    // Consumer queued and completed before production: a later wait cannot repair it.
    CUDA_OK(cudaMemcpyPeerAsync(destination,1,source,0,sizeof(int),consumer));CUDA_OK(cudaStreamSynchronize(consumer));
    int early=-1;CUDA_OK(cudaMemcpy(&early,destination,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaSetDevice(0));produce<<<1,1,0,producer>>>(source);CUDA_OK(cudaGetLastError());CUDA_OK(cudaEventRecord(ready,producer));
    CUDA_OK(cudaSetDevice(1));CUDA_OK(cudaStreamWaitEvent(consumer,ready,0));CUDA_OK(cudaStreamSynchronize(consumer));
    int after_late_wait=-1;CUDA_OK(cudaMemcpy(&after_late_wait,destination,sizeof(int),cudaMemcpyDeviceToHost));
    // The corrected consumer transfer is queued AFTER the recorded-event wait.
    CUDA_OK(cudaStreamWaitEvent(consumer,ready,0));CUDA_OK(cudaMemcpyPeerAsync(destination,1,source,0,sizeof(int),consumer));CUDA_OK(cudaStreamSynchronize(consumer));
    int corrected=-1;CUDA_OK(cudaMemcpy(&corrected,destination,sizeof(int),cudaMemcpyDeviceToHost));
    int errors=(early!=0)+(after_late_wait!=0)+(corrected!=42);
    printf("Consumer before production: %d; after a too-late wait: %d; correctly ordered copy: %d\n",early,after_late_wait,corrected);
    printf("The early copy is deliberately completed first, avoiding an intentional data race.\n");
    printf("VALIDATION %s: errors=%d\n",errors?"FAIL":"PASS",errors);
    CUDA_OK(cudaFree(destination));CUDA_OK(cudaStreamDestroy(consumer));CUDA_OK(cudaSetDevice(0));CUDA_OK(cudaEventDestroy(ready));CUDA_OK(cudaStreamDestroy(producer));CUDA_OK(cudaFree(source));
    return errors?1:0;
}
