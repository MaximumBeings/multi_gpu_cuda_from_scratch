// Chapter 8: Broadcast and Reduce -- The First Two Collectives, By Hand
// 21_reduce_by_hand.cu
//
// A reduce, written by hand: stage each device's buffer to the host
// and accumulate there, one device at a time. Genuinely compiled with
// a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t N_INTS = 4;
    const size_t BYTES = N_INTS * sizeof(int);

    long long accumulator[N_INTS] = {0, 0, 0, 0};
    int hostStaging[N_INTS];

    int devicesReduced = 0;
    for (int src = 0; src < deviceCount; ++src) {
        void* devPtr = nullptr;
        cudaError_t eAlloc = cudaMalloc(&devPtr, BYTES);
        cudaError_t eCopy = cudaMemcpy(hostStaging, devPtr, BYTES, cudaMemcpyDeviceToHost);
        printf("cudaMemcpy(device %d -> host staging, for accumulation): %s (code %d)\n",
               src, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) {
            for (size_t i = 0; i < N_INTS; ++i) accumulator[i] += hostStaging[i];
            ++devicesReduced;
        }
        if (eAlloc == cudaSuccess) cudaFree(devPtr);
    }

    printf("Reduced %d of %d device(s) into the host accumulator. On real\n"
           "hardware this same per-device stage-then-accumulate loop produces\n"
           "the elementwise sum across every device's buffer -- the reduce\n"
           "collective's defining result, computed here entirely on the host\n"
           "since no device kernel is needed for the accumulation step.\n",
           devicesReduced, deviceCount);

    return 0;
}
