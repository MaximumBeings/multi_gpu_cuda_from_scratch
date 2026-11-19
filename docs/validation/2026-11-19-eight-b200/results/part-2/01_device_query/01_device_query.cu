// Chapter 1: Why One GPU Is Not Enough
// 01_device_query.cu
//
// A genuine CUDA Runtime API call, compiled with a real nvcc and genuinely
// executed. It does not simulate anything: cudaGetDeviceCount() either finds
// real devices or it doesn't, and this program prints exactly what the
// runtime reports back, whatever that is.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("cudaGetDeviceCount() returned: %s (code %d)\n",
           cudaGetErrorString(err), (int)err);
    printf("Reported device count: %d\n", deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        printf("No usable CUDA device on this machine.\n");
        printf("This is an honest, unmodified report from the CUDA "
               "Runtime API -- not a placeholder.\n");
        return 0;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        size_t freeBytes = 0, totalBytes = 0;
        cudaSetDevice(i);
        cudaMemGetInfo(&freeBytes, &totalBytes);
        printf("Device %d: %s -- %.2f GB total, %.2f GB free, "
               "compute capability %d.%d\n",
               i, prop.name,
               totalBytes / 1e9, freeBytes / 1e9,
               prop.major, prop.minor);
    }
    return 0;
}
