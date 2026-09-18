// Appendix E: Profiling and Benchmarking Multi-GPU Communication
// 133_multi_device_timing.cu
//
// Appendix E.2 -- DSA's own sibling Appendix E.2 established cudaEvent_t
// as the correct way to time ONE kernel on ONE device, because a host
// clock wrapped around an asynchronous launch measures almost nothing.
// A multi-device collective raises the identical problem at one more
// level: cudaEventRecord()/cudaEventElapsedTime() on device 0's OWN
// stream only tells you how long device 0 took -- but a collective is
// not actually finished, anywhere, until every participating device has
// finished its own share of the work. This file builds the real,
// correct pattern: an event pair recorded on EVERY device's own stream,
// synchronized per device, with the collective's true elapsed time being
// the MAXIMUM across devices, not any single device's own number.
//
// Compile: nvcc -arch=sm_80 133_multi_device_timing.cu -o 133_multi_device_timing
// Run:     ./133_multi_device_timing
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void dummy_kernel(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] += 1;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-32s -> %-28s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    printf("=== Section E.2: timing a MULTI-DEVICE collective correctly ===\n\n");

    int deviceCount = 0;
    cudaError_t e0 = cudaGetDeviceCount(&deviceCount);
    report("cudaGetDeviceCount", e0);
    printf("  deviceCount = %d\n\n", deviceCount);

    printf("=== the WRONG way: one event pair, on one device's own stream ===\n\n");
    printf("cudaSetDevice(0); cudaEventRecord(start); ncclAllReduce(...); cudaEventRecord(stop);\n");
    printf("cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);\n\n");
    printf("this measures ONLY how long device 0's own stream took to reach the second\n");
    printf("event -- NCCL enqueues an all-reduce's own work on every participating\n");
    printf("device's stream, and device 0's stream can genuinely reach its own \"done\"\n");
    printf("marker before a slower device (different clock speed, different PCIe link,\n");
    printf("or simply more queued work) has finished its own share of the collective at\n");
    printf("all. A single device's own elapsed time UNDER-reports the real, whole-\n");
    printf("collective completion time whenever devices are not perfectly synchronized.\n\n");

    printf("=== the CORRECT way: one event pair PER device, take the MAXIMUM ===\n\n");
    const int NDEV = 4;   // illustrative -- this environment has 0 real devices
    std::vector<cudaEvent_t> startEvents(NDEV), stopEvents(NDEV);
    std::vector<cudaError_t> setDevErrs(NDEV), startErrs(NDEV), stopErrs(NDEV), syncErrs(NDEV);

    for (int dev = 0; dev < NDEV; ++dev) {
        setDevErrs[dev] = cudaSetDevice(dev);
        startErrs[dev] = cudaEventCreate(&startEvents[dev]);
        stopErrs[dev] = cudaEventCreate(&stopEvents[dev]);
    }

    printf("  %-6s %-20s %-20s %-20s\n", "dev", "cudaSetDevice", "cudaEventCreate(start)", "cudaEventCreate(stop)");
    for (int dev = 0; dev < NDEV; ++dev) {
        printf("  %-6d %-20s %-20s %-20s\n", dev,
               cudaGetErrorName(setDevErrs[dev]), cudaGetErrorName(startErrs[dev]), cudaGetErrorName(stopErrs[dev]));
    }

    bool anyDeviceReal = false;
    for (int dev = 0; dev < NDEV; ++dev) if (startErrs[dev] == cudaSuccess) anyDeviceReal = true;

    std::vector<float> perDeviceMs(NDEV, -1.0f);
    if (anyDeviceReal) {
        for (int dev = 0; dev < NDEV; ++dev) {
            cudaSetDevice(dev);
            cudaEventRecord(startEvents[dev]);
            dummy_kernel<<<1, 32>>>(nullptr, 0);
            cudaEventRecord(stopEvents[dev]);
        }
        for (int dev = 0; dev < NDEV; ++dev) {
            cudaSetDevice(dev);
            syncErrs[dev] = cudaEventSynchronize(stopEvents[dev]);
            cudaEventElapsedTime(&perDeviceMs[dev], startEvents[dev], stopEvents[dev]);
        }
        float maxMs = *std::max_element(perDeviceMs.begin(), perDeviceMs.end());
        printf("\n  per-device elapsed times collected; the collective's own TRUE elapsed\n");
        printf("  time is max(all %d devices' own elapsed times) = %f ms, not device 0's\n", NDEV, maxMs);
        printf("  own number alone.\n");
    } else {
        printf("\n  (every cudaEventCreate() above reports the same honest cudaErrorNoDevice\n");
        printf("   this environment reports for every device-touching call since Chapter 2 --\n");
        printf("   no per-device elapsed time can actually be measured here, but the CORRECT\n");
        printf("   pattern -- one event pair per device, cudaEventSynchronize EACH one, take\n");
        printf("   the MAXIMUM across all of them -- is the real, genuinely different-from-\n");
        printf("   single-device correction this section exists to make, and it is what a\n");
        printf("   reader with real multi-GPU hardware should actually run.)\n");
    }

    bool allSameError = true;
    for (int dev = 1; dev < NDEV; ++dev)
        if (setDevErrs[dev] != setDevErrs[0]) allSameError = false;

    printf("\nself-check: every cudaSetDevice() call across all %d simulated devices reports\n", NDEV);
    printf("the SAME error (%s), confirming the per-device loop was genuinely run to\n",
           cudaGetErrorName(setDevErrs[0]));
    printf("completion for every device rather than short-circuited after the first one:\n");
    printf("%s\n", allSameError ? "confirmed" : "MISMATCH");

    return allSameError ? 0 : 1;
}
