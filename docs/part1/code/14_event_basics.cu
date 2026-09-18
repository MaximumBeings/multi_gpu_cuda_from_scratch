// Chapter 6: Streams, Events, and Cross-Device Synchronization
// 14_event_basics.cu
//
// The real event lifecycle: create, record (asynchronously, into a
// stream), check without blocking (cudaEventQuery), and check while
// blocking (cudaEventSynchronize). Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaStream_t stream;
    cudaError_t eStream = cudaStreamCreate(&stream);
    printf("cudaStreamCreate: %s (code %d)\n", cudaGetErrorString(eStream), (int)eStream);

    cudaEvent_t event;
    cudaError_t eEvent = cudaEventCreate(&event);
    printf("cudaEventCreate: %s (code %d)\n", cudaGetErrorString(eEvent), (int)eEvent);

    cudaError_t eRecord = cudaEventRecord(event, stream);
    printf("cudaEventRecord(event, stream): %s (code %d)\n", cudaGetErrorString(eRecord), (int)eRecord);

    cudaError_t eQuery = cudaEventQuery(event);
    printf("cudaEventQuery(event) [non-blocking check]: %s (code %d)\n",
           cudaGetErrorString(eQuery), (int)eQuery);

    cudaError_t eSync = cudaEventSynchronize(event);
    printf("cudaEventSynchronize(event) [blocking wait]: %s (code %d)\n",
           cudaGetErrorString(eSync), (int)eSync);

    if (eEvent == cudaSuccess) cudaEventDestroy(event);
    if (eStream == cudaSuccess) cudaStreamDestroy(stream);
    return 0;
}
