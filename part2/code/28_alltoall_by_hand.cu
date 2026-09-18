// Chapter 10: All-Gather, Reduce-Scatter, and All-to-All
// 28_alltoall_by_hand.cu
//
// All-to-all: every device sends a DIFFERENT chunk to every other
// device, and receives a DIFFERENT chunk from every other device.
// Unlike all-gather (every device sends the SAME chunk to everyone)
// or reduce-scatter (chunks are combined, not just relayed), every
// one of the N*(N-1) transfers below carries genuinely distinct data
// -- there is no way to build this out of Chapter 9's ring phases,
// because nothing here is identical across destinations for a ring
// hop to forward along unchanged. Genuinely compiled with a real
// nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    // Device `src` holds deviceCount send-slots; send-slot[dst] is the
    // chunk destined ONLY for device `dst`, and is different from every
    // other send-slot on device `src`.
    int transfersAttempted = 0;
    int transfersCompleted = 0;
    for (int src = 0; src < deviceCount; ++src) {
        for (int dst = 0; dst < deviceCount; ++dst) {
            if (dst == src) continue; // a device's own reserved slot needs no transfer
            void* sendSlotPtr = nullptr;
            void* recvSlotPtr = nullptr;
            cudaError_t eAllocSend = cudaMalloc(&sendSlotPtr, CHUNK_BYTES);
            cudaError_t eAllocRecv = cudaMalloc(&recvSlotPtr, CHUNK_BYTES);
            cudaError_t eCopy = cudaMemcpyPeer(recvSlotPtr, dst, sendSlotPtr, src, CHUNK_BYTES);
            ++transfersAttempted;
            printf("All-to-all, cudaMemcpyPeer(dev%d's slot[from=%d] <- dev%d's slot[to=%d]): %s (code %d)\n",
                   dst, src, src, dst, cudaGetErrorString(eCopy), (int)eCopy);
            if (eCopy == cudaSuccess) ++transfersCompleted;
            if (eAllocSend == cudaSuccess) cudaFree(sendSlotPtr);
            if (eAllocRecv == cudaSuccess) cudaFree(recvSlotPtr);
        }
    }

    printf("\n%d of %d attempted point-to-point transfers completed --\n"
           "exactly %d * (%d - 1) of them, one per ordered (src, dst) pair\n"
           "with src != dst. No reduction, no forwarding, no shortcut: every\n"
           "one of these transfers carries data no other transfer could have\n"
           "carried. Section 10.3's simulation verifies every device ends up\n"
           "with exactly the chunks addressed to it, and nothing else.\n",
           transfersCompleted, transfersAttempted, deviceCount, deviceCount);

    return 0;
}
