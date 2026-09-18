// Chapter 6: Streams, Events, and Cross-Device Synchronization
// 15_cross_device_wait.cu
//
// The real cross-device dependency mechanism: an event recorded into
// one device's stream, then handed to cudaStreamWaitEvent() on a
// DIFFERENT device's stream -- no cudaDeviceSynchronize() anywhere,
// and no host blocking at all. Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    // Device 0 side: produce work, mark its completion point with an event.
    cudaError_t eSet0 = cudaSetDevice(0);
    printf("cudaSetDevice(0): %s (code %d)\n", cudaGetErrorString(eSet0), (int)eSet0);

    cudaStream_t stream0;
    cudaError_t eStream0 = cudaStreamCreate(&stream0);
    printf("cudaStreamCreate(stream0, on device 0): %s (code %d)\n",
           cudaGetErrorString(eStream0), (int)eStream0);

    cudaEvent_t event0;
    cudaError_t eEvent0 = cudaEventCreate(&event0);
    printf("cudaEventCreate(event0, on device 0): %s (code %d)\n",
           cudaGetErrorString(eEvent0), (int)eEvent0);

    cudaError_t eRecord0 = cudaEventRecord(event0, stream0);
    printf("cudaEventRecord(event0, stream0): %s (code %d)\n",
           cudaGetErrorString(eRecord0), (int)eRecord0);

    // Device 1 side: create its own stream, then make it wait on
    // device 0's event -- WITHOUT calling cudaDeviceSynchronize()
    // anywhere, and without the host blocking at all.
    cudaError_t eSet1 = cudaSetDevice(1);
    printf("cudaSetDevice(1): %s (code %d)\n", cudaGetErrorString(eSet1), (int)eSet1);

    cudaStream_t stream1;
    cudaError_t eStream1 = cudaStreamCreate(&stream1);
    printf("cudaStreamCreate(stream1, on device 1): %s (code %d)\n",
           cudaGetErrorString(eStream1), (int)eStream1);

    cudaError_t eWait = cudaStreamWaitEvent(stream1, event0, 0);
    printf("cudaStreamWaitEvent(stream1 waits on event0 from device 0): %s (code %d)\n",
           cudaGetErrorString(eWait), (int)eWait);

    printf("Host reached this line without blocking on any of the above --\n"
           "cudaStreamWaitEvent only ever inserts a dependency for FUTURE\n"
           "work on stream1; it does not itself wait for anything.\n");

    if (eEvent0 == cudaSuccess) cudaEventDestroy(event0);
    if (eStream0 == cudaSuccess) cudaStreamDestroy(stream0);
    if (eStream1 == cudaSuccess) cudaStreamDestroy(stream1);
    return 0;
}
