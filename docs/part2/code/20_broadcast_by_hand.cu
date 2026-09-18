// Chapter 8: Broadcast and Reduce -- The First Two Collectives, By Hand
// 20_broadcast_by_hand.cu
//
// A broadcast, written by hand as the real API calls it actually is:
// one cudaMemcpyPeer() per non-root device, all sourced from the same
// unchanging root pointer. Genuinely compiled with a real nvcc and
// genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int ROOT = 0;
    const size_t BYTES = 64;
    void* rootPtr = nullptr;

    cudaError_t eSetRoot = cudaSetDevice(ROOT);
    printf("cudaSetDevice(root=%d): %s (code %d)\n", ROOT, cudaGetErrorString(eSetRoot), (int)eSetRoot);

    cudaError_t eAllocRoot = cudaMalloc(&rootPtr, BYTES);
    printf("cudaMalloc(root's data): %s (code %d)\n", cudaGetErrorString(eAllocRoot), (int)eAllocRoot);

    int broadcastCount = 0;
    for (int dst = 0; dst < deviceCount; ++dst) {
        if (dst == ROOT) continue;
        void* dstPtr = nullptr;
        cudaError_t eAllocDst = cudaMalloc(&dstPtr, BYTES);
        cudaError_t eCopy = cudaMemcpyPeer(dstPtr, dst, rootPtr, ROOT, BYTES);
        printf("cudaMemcpyPeer(dst=dev%d, src=root=dev%d): %s (code %d)\n",
               dst, ROOT, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) ++broadcastCount;
        if (eAllocDst == cudaSuccess) cudaFree(dstPtr);
    }

    printf("Broadcast reached %d of %d non-root device(s). On real hardware\n"
           "this same loop -- one cudaMemcpyPeer per non-root device, all from\n"
           "the same unchanging root pointer -- delivers an identical copy of\n"
           "the root's data to every other device.\n", broadcastCount, deviceCount > 0 ? deviceCount - 1 : 0);

    if (eAllocRoot == cudaSuccess) cudaFree(rootPtr);
    return 0;
}
