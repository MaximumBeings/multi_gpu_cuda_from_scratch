// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 24_ring_step_by_hand.cu
//
// ONE ring step: every device, symmetrically, sends its current chunk
// to its ring-neighbor (i+1)%N. Unlike Section 9.1's reduce/broadcast
// loops, there is no root here -- every device runs the identical
// operation. Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    int stepTransfers = 0;
    for (int i = 0; i < deviceCount; ++i) {
        int next = (i + 1) % deviceCount;
        void* srcPtr = nullptr;
        void* dstPtr = nullptr;
        cudaError_t eAllocSrc = cudaMalloc(&srcPtr, CHUNK_BYTES);
        cudaError_t eAllocDst = cudaMalloc(&dstPtr, CHUNK_BYTES);
        cudaError_t eCopy = cudaMemcpyPeer(dstPtr, next, srcPtr, i, CHUNK_BYTES);
        printf("Ring step, cudaMemcpyPeer(dst=dev%d, src=dev%d): %s (code %d)\n",
               next, i, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) ++stepTransfers;
        if (eAllocSrc == cudaSuccess) cudaFree(srcPtr);
        if (eAllocDst == cudaSuccess) cudaFree(dstPtr);
    }

    printf("\n%d of %d device(s) completed this ring step. On real hardware,\n"
           "every device performs exactly one cudaMemcpyPeer of exactly one\n"
           "chunk per step -- the same amount of work, every step, no matter\n"
           "how many devices are in the ring, and no device singled out.\n",
           stepTransfers, deviceCount);

    return 0;
}
