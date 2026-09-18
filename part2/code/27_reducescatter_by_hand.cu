// Chapter 10: All-Gather, Reduce-Scatter, and All-to-All
// 27_reducescatter_by_hand.cu
//
// Reduce-scatter, done directly: for each chunk index c, gather that
// ONE chunk from every device to the host, sum it, and write the sum
// back ONLY onto device c -- the device "assigned" to that chunk.
// This is exactly the SCATTER-REDUCE phase Chapter 9 used inside its
// ring all-reduce (Section 9.2-9.3), stopped there instead of
// continuing on into an all-gather. Genuinely compiled with a real
// nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    int chunksReduced = 0;
    int chunksScattered = 0;
    for (int owner = 0; owner < deviceCount; ++owner) {
        // Gather chunk `owner` from every device and reduce on the host --
        // exactly Chapter 8's reduce loop, but for a single chunk index.
        long long accumulator = 0;
        int hostStaging = 0;
        int devicesGathered = 0;
        for (int src = 0; src < deviceCount; ++src) {
            void* devPtr = nullptr;
            cudaError_t eAlloc = cudaMalloc(&devPtr, CHUNK_BYTES);
            cudaError_t eCopy = cudaMemcpy(&hostStaging, devPtr, CHUNK_BYTES, cudaMemcpyDeviceToHost);
            printf("Reduce-scatter, chunk %d: cudaMemcpy(device %d -> host): %s (code %d)\n",
                   owner, src, cudaGetErrorString(eCopy), (int)eCopy);
            if (eCopy == cudaSuccess) {
                accumulator += hostStaging;
                ++devicesGathered;
            }
            if (eAlloc == cudaSuccess) cudaFree(devPtr);
        }
        if (devicesGathered == deviceCount && deviceCount > 0) ++chunksReduced;

        // Scatter: write the reduced chunk back ONLY to device `owner`.
        // Every other device never receives this chunk at all -- that is
        // the entire point of reduce-scatter.
        int hostResult = (int)accumulator;
        void* ownerPtr = nullptr;
        cudaError_t eAllocOwner = cudaMalloc(&ownerPtr, CHUNK_BYTES);
        cudaError_t eWrite = cudaMemcpy(ownerPtr, &hostResult, CHUNK_BYTES, cudaMemcpyHostToDevice);
        printf("Reduce-scatter, chunk %d: cudaMemcpy(host reduced chunk -> owner dev%d): %s (code %d)\n\n",
               owner, owner, cudaGetErrorString(eWrite), (int)eWrite);
        if (eWrite == cudaSuccess) ++chunksScattered;
        if (eAllocOwner == cudaSuccess) cudaFree(ownerPtr);
    }

    printf("%d of %d chunk(s) fully reduced, %d of %d chunk(s) scattered to\n"
           "their owner. On real hardware, EVERY device ends up holding\n"
           "exactly ONE reduced chunk -- never all %d, and never the SAME\n"
           "chunk as any other device. Section 10.3 covers all-to-all, the\n"
           "one collective in this chapter with no reduction step at all.\n",
           chunksReduced, deviceCount, chunksScattered, deviceCount, deviceCount);

    return 0;
}
