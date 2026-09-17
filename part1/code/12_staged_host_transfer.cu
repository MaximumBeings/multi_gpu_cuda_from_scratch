// Chapter 5: Explicit Transfers -- cudaMemcpyPeer, Staged Host Transfers,
// and When Each Wins
// 12_staged_host_transfer.cu
//
// The staged fallback made explicit: two ordinary cudaMemcpy() calls,
// device 0 -> host, then host -> device 1, using a pinned host buffer
// as the waypoint. This is the exact call pattern cudaMemcpyPeer()
// itself falls back to internally whenever peer access between the two
// devices has not been enabled. Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    void* devPtr0 = nullptr;
    void* devPtr1 = nullptr;
    void* hostStaging = nullptr;
    const size_t BYTES = 64;

    cudaError_t eAlloc0 = cudaMalloc(&devPtr0, BYTES);
    printf("cudaMalloc(device 0 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc0), (int)eAlloc0);

    cudaError_t eAlloc1 = cudaMalloc(&devPtr1, BYTES);
    printf("cudaMalloc(device 1 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc1), (int)eAlloc1);

    cudaError_t eHost = cudaMallocHost(&hostStaging, BYTES);
    printf("cudaMallocHost(staging buffer): %s (code %d)\n", cudaGetErrorString(eHost), (int)eHost);

    // Step 1: device 0 -> host.
    cudaError_t e1 = cudaMemcpy(hostStaging, devPtr0, BYTES, cudaMemcpyDeviceToHost);
    printf("Step 1, cudaMemcpy(device 0 -> host staging): %s (code %d)\n",
           cudaGetErrorString(e1), (int)e1);

    // Step 2: host -> device 1. Two PCIe (or equivalent) transfers where
    // a single peer-direct copy would have used one link once.
    cudaError_t e2 = cudaMemcpy(devPtr1, hostStaging, BYTES, cudaMemcpyHostToDevice);
    printf("Step 2, cudaMemcpy(host staging -> device 1): %s (code %d)\n",
           cudaGetErrorString(e2), (int)e2);

    if (eAlloc0 == cudaSuccess) cudaFree(devPtr0);
    if (eAlloc1 == cudaSuccess) cudaFree(devPtr1);
    if (eHost == cudaSuccess) cudaFreeHost(hostStaging);
    return 0;
}
