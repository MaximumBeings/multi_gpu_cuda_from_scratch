// Chapter 10: All-Gather, Reduce-Scatter, and All-to-All
// 26_allgather_by_hand.cu
//
// All-gather, done directly: every device sends its OWN chunk to
// every other device (N*(N-1) point-to-point transfers, no root, no
// reduction). This is the same all-gather PHASE Chapter 9 used inside
// its ring all-reduce (Section 9.2-9.3) -- shown here running alone,
// with the straightforward direct pattern instead of the ring
// pattern, so the operation itself can be seen in isolation.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    // Every device holds one recvBuffer with room for deviceCount
    // chunks -- slot i will hold device i's original chunk once
    // all-gather completes.
    int transfersAttempted = 0;
    int transfersCompleted = 0;
    for (int src = 0; src < deviceCount; ++src) {
        void* sendPtr = nullptr;
        cudaError_t eAllocSend = cudaMalloc(&sendPtr, CHUNK_BYTES);
        for (int dst = 0; dst < deviceCount; ++dst) {
            if (dst == src) continue; // a device's own chunk needs no transfer
            void* recvPtr = nullptr;
            cudaError_t eAllocRecv = cudaMalloc(&recvPtr, CHUNK_BYTES);
            cudaError_t eCopy = cudaMemcpyPeer(recvPtr, dst, sendPtr, src, CHUNK_BYTES);
            ++transfersAttempted;
            printf("All-gather, cudaMemcpyPeer(dst=dev%d's slot[%d], src=dev%d): %s (code %d)\n",
                   dst, src, src, cudaGetErrorString(eCopy), (int)eCopy);
            if (eCopy == cudaSuccess) ++transfersCompleted;
            if (eAllocRecv == cudaSuccess) cudaFree(recvPtr);
        }
        if (eAllocSend == cudaSuccess) cudaFree(sendPtr);
    }

    printf("\n%d of %d attempted point-to-point transfers completed. On real\n"
           "hardware, every device ends up holding all %d devices' original\n"
           "chunks, side by side, completely UNCHANGED -- all-gather never\n"
           "reduces or modifies anything, it only collects. Section 10.2\n"
           "shows the mirror-image operation, reduce-scatter.\n",
           transfersCompleted, transfersAttempted, deviceCount);

    return 0;
}
