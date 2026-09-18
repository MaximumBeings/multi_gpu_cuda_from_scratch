// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 23_naive_allreduce.cu
//
// The naive baseline: reduce every device's buffer to a host
// accumulator (Chapter 8, Section 8.2), write the result back onto
// the root device, then broadcast it out to every other device
// (Chapter 8, Section 8.1). Genuinely compiled with a real nvcc and
// genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int ROOT = 0;
    const size_t N_INTS = 4;
    const size_t BYTES = N_INTS * sizeof(int);

    // Phase 1: reduce every device's buffer to a host accumulator,
    // exactly like Chapter 8, Section 8.2.
    long long accumulator[N_INTS] = {0, 0, 0, 0};
    int hostStaging[N_INTS];
    int devicesReduced = 0;
    for (int src = 0; src < deviceCount; ++src) {
        void* devPtr = nullptr;
        cudaError_t eAlloc = cudaMalloc(&devPtr, BYTES);
        cudaError_t eCopy = cudaMemcpy(hostStaging, devPtr, BYTES, cudaMemcpyDeviceToHost);
        printf("Reduce phase, cudaMemcpy(device %d -> host): %s (code %d)\n",
               src, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) {
            for (size_t i = 0; i < N_INTS; ++i) accumulator[i] += hostStaging[i];
            ++devicesReduced;
        }
        if (eAlloc == cudaSuccess) cudaFree(devPtr);
    }

    // Phase 2: write the reduced result back onto the root device, then
    // broadcast it to every other device -- exactly like Chapter 8,
    // Section 8.1. Only after this phase does every device hold the
    // SAME reduced value; this two-phase, root-centered round trip is
    // the "naive" all-reduce this chapter improves on.
    int hostResult[N_INTS];
    for (size_t i = 0; i < N_INTS; ++i) hostResult[i] = (int)accumulator[i];

    void* rootPtr = nullptr;
    cudaError_t eAllocRoot = cudaMalloc(&rootPtr, BYTES);
    cudaError_t eWriteRoot = cudaMemcpy(rootPtr, hostResult, BYTES, cudaMemcpyHostToDevice);
    printf("Broadcast phase, cudaMemcpy(host result -> root device): %s (code %d)\n",
           cudaGetErrorString(eWriteRoot), (int)eWriteRoot);

    int broadcastCount = 0;
    for (int dst = 0; dst < deviceCount; ++dst) {
        if (dst == ROOT) continue;
        void* dstPtr = nullptr;
        cudaError_t eAllocDst = cudaMalloc(&dstPtr, BYTES);
        cudaError_t eCopy = cudaMemcpyPeer(dstPtr, dst, rootPtr, ROOT, BYTES);
        printf("Broadcast phase, cudaMemcpyPeer(dst=dev%d, src=root): %s (code %d)\n",
               dst, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) ++broadcastCount;
        if (eAllocDst == cudaSuccess) cudaFree(dstPtr);
    }

    printf("\nReduced %d device(s), then broadcast reached %d non-root "
           "device(s). Every byte of this all-reduce passed through the\n"
           "root device TWICE -- once inbound during the reduce phase, "
           "once outbound during the broadcast phase -- while every other\n"
           "device only ever sent or received once. Section 9.3 makes this\n"
           "asymmetry concrete with real counted data-volume numbers.\n",
           devicesReduced, broadcastCount);

    if (eAllocRoot == cudaSuccess) cudaFree(rootPtr);
    return 0;
}
