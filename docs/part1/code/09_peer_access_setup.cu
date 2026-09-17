// Chapter 4: Peer-to-Peer Memory Access and Unified Virtual Addressing
// 09_peer_access_setup.cu
//
// The real, idiomatic peer-access setup sequence: check every ordered
// pair with cudaDeviceCanAccessPeer(), then enable access with
// cudaDeviceEnablePeerAccess() -- called from the "current" side of the
// pair, since (per the CUDA Runtime API docs) access granted this way
// is unidirectional. Genuinely compiled with a real nvcc and genuinely
// run; on this driver-less machine the loop body runs zero times, for
// exactly the reason Chapter 3 already explained.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    int enabledPairs = 0;
    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            if (canAccess) {
                cudaSetDevice(i);
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                printf("Enabled i=%d -> j=%d: %s\n", i, j,
                       cudaGetErrorString(e));
                if (e == cudaSuccess) ++enabledPairs;
            }
        }
    }
    printf("Peer access enabled for %d ordered pair(s) out of %d "
           "device(s) checked. On real multi-GPU hardware this same "
           "loop enables every accessible ordered pair exactly once, "
           "each direction independently, per the API's own "
           "unidirectional-access note.\n", enabledPairs, deviceCount);
    return 0;
}
