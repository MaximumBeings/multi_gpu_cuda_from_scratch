// Chapter 4: Peer-to-Peer Memory Access and Unified Virtual Addressing
// 08_pointer_attributes.cu
//
// A genuine cudaPointerGetAttributes() call, compiled with a real nvcc
// and genuinely run. Under Unified Virtual Addressing, a single flat
// address space covers the host and every device, so this call is what
// answers "whose memory does this pointer actually point into" -- the
// exact mechanism cudaMemcpyDefault relies on to auto-detect a copy's
// direction without being told explicitly.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int hostArr[4] = {1, 2, 3, 4};
    cudaPointerAttributes attr;
    cudaError_t e = cudaPointerGetAttributes(&attr, hostArr);
    printf("cudaPointerGetAttributes(ordinary stack array): %s (code %d)\n",
           cudaGetErrorString(e), (int)e);
    if (e == cudaSuccess) {
        printf("  type=%d device=%d\n", (int)attr.type, attr.device);
    }

    void* devPtr = nullptr;
    cudaError_t e2 = cudaMalloc(&devPtr, 16);
    printf("cudaMalloc: %s (code %d)\n", cudaGetErrorString(e2), (int)e2);

    if (e2 == cudaSuccess) {
        cudaPointerAttributes attr2;
        cudaError_t e3 = cudaPointerGetAttributes(&attr2, devPtr);
        printf("cudaPointerGetAttributes(device ptr): %s\n", cudaGetErrorString(e3));
        cudaFree(devPtr);
    }
    return 0;
}
