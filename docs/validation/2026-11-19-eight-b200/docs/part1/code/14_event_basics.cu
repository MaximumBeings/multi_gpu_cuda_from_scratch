// Chapter 6: Streams, Events, and Cross-Device Synchronization
// 14_event_basics.cu
// Guard each lifecycle step; no-device diagnostics are not a GPU pass.
#include <cstdio>
#include <cuda_runtime.h>

static int failure(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ? 2 : 1;
}

int main() {
    cudaStream_t stream = nullptr;
    cudaError_t status = cudaStreamCreate(&stream);
    printf("cudaStreamCreate: %s (code %d)\n", cudaGetErrorString(status), (int)status);
    if (status != cudaSuccess) return failure(status);
    cudaEvent_t event = nullptr;
    status = cudaEventCreate(&event);
    printf("cudaEventCreate: %s (code %d)\n", cudaGetErrorString(status), (int)status);
    if (status != cudaSuccess) { cudaStreamDestroy(stream); return failure(status); }
    status = cudaEventRecord(event, stream);
    printf("cudaEventRecord(event, stream): %s (code %d)\n", cudaGetErrorString(status), (int)status);
    int result = status == cudaSuccess ? 0 : failure(status);
    if (result == 0) {
        status = cudaEventQuery(event);
        printf("cudaEventQuery(event) [non-blocking check]: %s (code %d)\n", cudaGetErrorString(status), (int)status);
        if (status != cudaSuccess && status != cudaErrorNotReady) result = failure(status);
    }
    if (result == 0) {
        status = cudaEventSynchronize(event);
        printf("cudaEventSynchronize(event) [blocking wait]: %s (code %d)\n", cudaGetErrorString(status), (int)status);
        if (status != cudaSuccess) result = failure(status);
    }
    if (cudaEventDestroy(event) != cudaSuccess) result = 1;
    if (cudaStreamDestroy(stream) != cudaSuccess) result = 1;
    return result;
}
