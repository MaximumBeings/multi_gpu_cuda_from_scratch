// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 23_naive_allreduce.cu
//
// The naive baseline: reduce every device's buffer to a host
// accumulator (Chapter 8, Section 8.2), write the result back onto
// the root device, then broadcast it out to every other device
// (Chapter 8, Section 8.1). Note the actual data path: the reduce
// phase funnels EVERY device's buffer (the root's included) through
// the HOST, and only the broadcast phase is centered on the root
// device. cudaSetDevice() is called before every per-device
// operation (Chapter 3's rule). Genuinely compiled with a real nvcc
// and genuinely run.
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
    // exactly like Chapter 8, Section 8.2. The hub of this phase is
    // the host: it receives one full buffer from every device.
    long long accumulator[N_INTS] = {0, 0, 0, 0};
    int hostStaging[N_INTS];
    int devicesReduced = 0;
    for (int src = 0; src < deviceCount; ++src) {
        void* devPtr = nullptr;
        cudaSetDevice(src);  // every per-device call below targets src
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
    // SAME reduced value; this two-phase, hub-centered round trip
    // (host hub for the reduce, root hub for the broadcast) is the
    // "naive" all-reduce this chapter improves on.
    int hostResult[N_INTS];
    for (size_t i = 0; i < N_INTS; ++i) hostResult[i] = (int)accumulator[i];

    void* rootPtr = nullptr;
    cudaSetDevice(ROOT);  // the result buffer lives on the root
    cudaError_t eAllocRoot = cudaMalloc(&rootPtr, BYTES);
    cudaError_t eWriteRoot = cudaMemcpy(rootPtr, hostResult, BYTES, cudaMemcpyHostToDevice);
    printf("Broadcast phase, cudaMemcpy(host result -> root device): %s (code %d)\n",
           cudaGetErrorString(eWriteRoot), (int)eWriteRoot);

    int broadcastCount = 0;
    for (int dst = 0; dst < deviceCount; ++dst) {
        if (dst == ROOT) continue;
        void* dstPtr = nullptr;
        cudaSetDevice(dst);  // allocate dst's buffer ON device dst
        cudaError_t eAllocDst = cudaMalloc(&dstPtr, BYTES);
        cudaError_t eCopy = cudaMemcpyPeer(dstPtr, dst, rootPtr, ROOT, BYTES);
        printf("Broadcast phase, cudaMemcpyPeer(dst=dev%d, src=root): %s (code %d)\n",
               dst, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) ++broadcastCount;
        if (eAllocDst == cudaSuccess) cudaFree(dstPtr);
    }

    printf("\nReduced %d device(s) through the host, then broadcast reached\n"
           "%d non-root device(s). Each phase of this all-reduce funnels\n"
           "everything through ONE hub: the HOST receives every device's\n"
           "full buffer during the reduce phase, and the ROOT device sends a\n"
           "full buffer to every other device during the broadcast phase --\n"
           "while every other device only ever sends once and receives once.\n"
           "Section 9.3 puts counted numbers on this star-shaped imbalance.\n",
           devicesReduced, broadcastCount);

    cudaSetDevice(ROOT);  // free the root's buffer with the root current
    if (eAllocRoot == cudaSuccess) cudaFree(rootPtr);
    return 0;
}
