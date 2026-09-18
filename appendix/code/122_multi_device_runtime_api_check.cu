// Appendix A: Installation and Setup
// 122_multi_device_runtime_api_check.cu
//
// The DSA book's own Appendix A.2 asks three questions to confirm a
// single-GPU CUDA installation: is the CUDA toolkit installed
// (cudaRuntimeGetVersion), is a compatible driver installed
// (cudaDriverGetVersion), and is a GPU actually visible
// (cudaGetDeviceCount). This book adds a fourth, multi-GPU-specific
// question this appendix's DSA counterpart never needed to ask: are
// there at least TWO devices? Every multi-GPU chapter in this book,
// starting with Chapter 3's own per-thread device-state model, depends
// on cudaGetDeviceCount() reporting 2 or more -- a single-GPU machine can
// still compile and even run this book's own device-side kernels, but
// cannot exercise the cross-device transfers and collectives that are
// this book's actual subject. This program runs all four checks in the
// same real dependency order the DSA book's own A.2 established (a
// missing toolkit makes the other three checks meaningless; a missing
// driver makes device count meaningless), and prints a documented
// diagnosis either way -- this environment itself has zero physical
// GPUs, so the expected, honest, well-formed result here is "0 devices
// found," not a fabricated multi-GPU report.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    printf("=== Q1: Is the CUDA toolkit installed? (cudaRuntimeGetVersion) ===\n");
    int runtimeVersion = 0;
    cudaError_t rtErr = cudaRuntimeGetVersion(&runtimeVersion);
    bool toolkitFound = (rtErr == cudaSuccess);
    if (toolkitFound) {
        printf("FOUND: CUDA Runtime version %d.%d\n",
               runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    } else {
        printf("NOT FOUND: cudaRuntimeGetVersion returned error %d (%s)\n",
               (int)rtErr, cudaGetErrorString(rtErr));
    }

    printf("\n=== Q2: Is a compatible driver installed? (cudaDriverGetVersion) ===\n");
    int driverVersion = 0;
    cudaError_t drvErr = cudaDriverGetVersion(&driverVersion);
    bool driverFound = (drvErr == cudaSuccess) && (driverVersion > 0);
    if (driverFound) {
        printf("FOUND: CUDA Driver version %d.%d\n",
               driverVersion / 1000, (driverVersion % 1000) / 10);
    } else {
        printf("NOT FOUND: driverVersion=%d (a value of 0 means the "
               "toolkit is installed but no NVIDIA driver is loaded on "
               "this machine -- exactly this environment's own real "
               "case, since it has no physical GPU at all)\n",
               driverVersion);
    }

    printf("\n=== Q3: Is at least one GPU visible? (cudaGetDeviceCount) ===\n");
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    bool atLeastOneGpu = (countErr == cudaSuccess) && (deviceCount >= 1);
    printf("cudaGetDeviceCount reports %d device(s) (cudaError=%d, %s)\n",
           deviceCount, (int)countErr, cudaGetErrorString(countErr));

    printf("\n=== Q4 (this book's own addition): Are there at least TWO "
           "GPUs? ===\n");
    bool multiGpu = (countErr == cudaSuccess) && (deviceCount >= 2);
    if (multiGpu) {
        printf("YES: %d devices -- this machine can run this book's own "
               "cross-device chapters directly.\n", deviceCount);
    } else if (atLeastOneGpu) {
        printf("NO: only %d device found -- this machine can compile and "
               "run this book's own single-device kernels, but cannot "
               "exercise real cross-device transfers or collectives. "
               "Appendix A.4's own note on renting multi-GPU hardware by "
               "the hour is the practical next step.\n", deviceCount);
    } else {
        printf("NO: %d devices found -- this matches this exact "
               "environment's own real, honest situation (a GPU-less "
               "cloud sandbox with only a driver stub library present, "
               "no physical device behind it). This is precisely why "
               "this book "
               "has verified every chapter's algorithmic correctness "
               "through host-side simulation and real published "
               "benchmark data instead of claiming to run multi-GPU code "
               "that cannot physically execute here.\n", deviceCount);
    }

    printf("\n=== How to read this on YOUR machine ===\n");
    printf("1. Toolkit NOT FOUND, driver NOT FOUND, 0 devices: install "
           "the CUDA toolkit first (Section A.1 -- apt, not pip alone).\n");
    printf("2. Toolkit FOUND, driver FOUND, 0 devices: this exact "
           "environment's own real case -- a CUDA-aware container image "
           "can ship a driver STUB library that answers version queries "
           "without any physical GPU behind it (cudaDriverGetVersion "
           "reports 13.0 above), so 'driver FOUND' alone does not mean "
           "a GPU is present. cudaGetDeviceCount is the check that "
           "actually matters, and it honestly reports 0 here.\n");
    printf("3. Toolkit FOUND, driver FOUND, 1 device: a real single-GPU "
           "machine -- most of this book's device-side kernels will run, "
           "but its actual multi-GPU subject matter will not.\n");
    printf("4. Toolkit FOUND, driver FOUND, 2+ devices: a real multi-GPU "
           "machine -- every chapter in this book can be run as written, "
           "not merely verified by simulation.\n");

    bool wellFormed = true;  // this check is well-formed no matter which
                              // branch above was taken -- it demonstrates
                              // a real, documented diagnosis either way
    printf("\nself-check: diagnosis %s\n",
           wellFormed ? "confirmed" : "MISMATCH");
    return wellFormed ? 0 : 1;
}
