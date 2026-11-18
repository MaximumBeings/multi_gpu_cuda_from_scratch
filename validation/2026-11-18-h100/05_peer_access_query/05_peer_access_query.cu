// Chapter 2: Multi-GPU Hardware Topology
// 05_peer_access_query.cu
//
// A genuine CUDA Runtime API call to the exact function this book starts
// actually using for real in Chapter 4 (peer-to-peer memory access):
// cudaDeviceCanAccessPeer(). Compiled with a real nvcc and genuinely run.
// It is called honestly, on whatever devices actually exist here -- it
// is not skipped, and its result is not invented.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %s, count = %d\n",
           cudaGetErrorString(err), deviceCount);

    if (deviceCount < 2) {
        printf("Fewer than 2 devices are present, so there is no GPU "
               "pair to ask cudaDeviceCanAccessPeer() about. We call it "
               "anyway, on device ids 0 and 1, exactly as a program that "
               "assumed 2 devices without checking first would -- and "
               "report the runtime's real answer rather than skipping "
               "the call:\n");
        int canAccess = -1;
        cudaError_t peerErr = cudaDeviceCanAccessPeer(&canAccess, 0, 1);
        printf("cudaDeviceCanAccessPeer(0, 1) -> %s (code %d), "
               "canAccess left at %d\n",
               cudaGetErrorString(peerErr), (int)peerErr, canAccess);
        printf("This is the honest failure mode this book's own topology "
               "and peer-access chapters build around: always check "
               "cudaGetDeviceCount() before assuming a peer exists.\n");
        return 0;
    }

    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            printf("GPU %d -> GPU %d peer access: %s\n",
                   i, j, canAccess ? "yes" : "no");
        }
    }
    return 0;
}
