// Chapter 3: The CUDA Multi-GPU Programming Model
// 07_multi_device_loop_pattern.cu
//
// The idiomatic multi-device pattern this book uses from here on: never
// hard-code a device count, loop over whatever cudaGetDeviceCount()
// actually reports, and call cudaSetDevice(i) before every single
// operation meant for device i -- because "current device" is per host
// thread, not per stream or per allocation. A stream or allocation is
// permanently bound to whichever device was current at the moment it
// was created; cudaSetDevice() only changes what happens *next*.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void addOneKernel(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount() reports %d device(s).\n", deviceCount);
    printf("The loop below runs its body exactly %d time(s) as a "
           "direct consequence -- not a special case, just what a "
           "correctly-guarded loop over a real count does when that "
           "count happens to be zero.\n\n", deviceCount);

    std::vector<cudaStream_t> streams(deviceCount);
    std::vector<int*> buffers(deviceCount, nullptr);
    const int N = 8;

    // Phase 1: on EACH device, in turn, set it current, then create
    // that device's own stream and allocate that device's own buffer.
    // The stream created while device i is current belongs to device i
    // permanently -- it is illegal to later enqueue work meant for
    // device j onto it.
    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&buffers[i], N * sizeof(int));
        printf("Device %d: stream and buffer created while device %d "
               "was current.\n", i, i);
    }

    // Phase 2: launch each device's own kernel on its own stream. This
    // is the pattern every collective in Part 2 builds on: one loop,
    // one cudaSetDevice() per iteration, work enqueued asynchronously
    // on each device's own stream so all devices can run concurrently.
    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        addOneKernel<<<1, N, 0, streams[i]>>>(buffers[i], N);
    }

    // Phase 3: synchronize each device's own stream. Note this is a
    // SEPARATE loop from phase 2 -- launching everything first, then
    // waiting, is what lets the devices actually overlap; interleaving
    // launch-then-wait-then-launch-then-wait would serialize them.
    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(buffers[i]);
    }

    printf("Loop body executed %d time(s) total across all three "
           "phases. On real N-GPU hardware, this exact, unmodified "
           "source runs all N devices' kernels concurrently with no "
           "code change -- that's the entire point of writing the loop "
           "this way instead of hard-coding device 0 and device 1.\n",
           deviceCount);

    return 0;
}
