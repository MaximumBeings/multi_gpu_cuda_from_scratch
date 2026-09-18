// Chapter 5: Explicit Transfers -- cudaMemcpyPeer, Staged Host Transfers,
// and When Each Wins
// 11_memcpy_peer.cu
//
// The real, dedicated cross-device copy calls: cudaMemcpyPeer() and its
// asynchronous, stream-ordered counterpart cudaMemcpyPeerAsync(). Both
// genuinely compiled with a real nvcc and genuinely run; on this
// driver-less machine every call fails at the same first step this
// book has reported since Chapter 1.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    void* devPtr0 = nullptr;
    void* devPtr1 = nullptr;

    cudaError_t eAlloc0 = cudaMalloc(&devPtr0, 64);
    printf("cudaMalloc(device 0 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc0), (int)eAlloc0);

    cudaError_t eAlloc1 = cudaMalloc(&devPtr1, 64);
    printf("cudaMalloc(device 1 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc1), (int)eAlloc1);

    cudaError_t e1 = cudaMemcpyPeer(devPtr1, 1, devPtr0, 0, 64);
    printf("cudaMemcpyPeer(dst=dev1, src=dev0, 64 bytes): %s (code %d)\n",
           cudaGetErrorString(e1), (int)e1);

    cudaStream_t stream;
    cudaError_t eStream = cudaStreamCreate(&stream);
    printf("cudaStreamCreate: %s (code %d)\n", cudaGetErrorString(eStream), (int)eStream);

    cudaError_t e2 = cudaMemcpyPeerAsync(devPtr0, 0, devPtr1, 1, 64, stream);
    printf("cudaMemcpyPeerAsync(dst=dev0, src=dev1, 64 bytes, on stream): %s (code %d)\n",
           cudaGetErrorString(e2), (int)e2);

    if (eAlloc0 == cudaSuccess) cudaFree(devPtr0);
    if (eAlloc1 == cudaSuccess) cudaFree(devPtr1);
    if (eStream == cudaSuccess) cudaStreamDestroy(stream);
    return 0;
}
