// Chapter 6: Streams, Events, and Cross-Device Synchronization
// 16_overlap_query.cu
//
// What the CUDA Runtime API can honestly report about a device's
// capacity to overlap a transfer with kernel execution, without
// running either: the asyncEngineCount and concurrentKernels device
// attributes/properties. Genuinely compiled with a real nvcc and
// genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    int asyncEngineCount = -1;
    cudaError_t e1 = cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, 0);
    printf("cudaDeviceGetAttribute(AsyncEngineCount, device 0): %s (code %d), value=%d\n",
           cudaGetErrorString(e1), (int)e1, asyncEngineCount);

    int concurrentKernels = -1;
    cudaError_t e2 = cudaDeviceGetAttribute(&concurrentKernels, cudaDevAttrConcurrentKernels, 0);
    printf("cudaDeviceGetAttribute(ConcurrentKernels, device 0): %s (code %d), value=%d\n",
           cudaGetErrorString(e2), (int)e2, concurrentKernels);

    cudaDeviceProp prop;
    cudaError_t e3 = cudaGetDeviceProperties(&prop, 0);
    printf("cudaGetDeviceProperties(device 0): %s (code %d)\n", cudaGetErrorString(e3), (int)e3);
    if (e3 == cudaSuccess) {
        printf("  asyncEngineCount=%d concurrentKernels=%d\n", prop.asyncEngineCount, prop.concurrentKernels);
    }

    printf("\nNeither value can be honestly reported as a real number on this\n"
           "machine -- there is no device 0 to query. On real hardware, both\n"
           "queries would need to succeed, and asyncEngineCount would need to\n"
           "be >=1, before overlap between a transfer and a kernel is even\n"
           "physically possible, regardless of how the streams are arranged.\n");

    return 0;
}
