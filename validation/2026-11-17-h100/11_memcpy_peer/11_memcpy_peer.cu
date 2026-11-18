// Chapter 5: checked synchronous and asynchronous peer transfers.
// 11_memcpy_peer.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#define CUDA(call) do { cudaError_t status = (call); if (status != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ? 2 : 1; } } while (0)
int main() {
    int count = 0;
    CUDA(cudaGetDeviceCount(&count));
    if (count < 2) { printf("BLOCKED: requires 2 GPUs; found %d.\n", count); return 2; }
    const int n = 4096;
    const size_t bytes = n * sizeof(float);
    std::vector<float> input(n), output(n);
    for (int i = 0; i < n; ++i) input[i] = float((i * 17) % 4093 - 2046) / 8.0f;
    float *source = nullptr, *destination = nullptr;
    CUDA(cudaSetDevice(0));
    CUDA(cudaMalloc(&source, bytes));
    CUDA(cudaMemcpy(source, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaSetDevice(1));
    CUDA(cudaMalloc(&destination, bytes));
    CUDA(cudaMemset(destination, 0, bytes));
    CUDA(cudaMemcpyPeer(destination, 1, source, 0, bytes));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(output.data(), destination, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) if (output[i] != input[i]) {
        fprintf(stderr, "FAIL: synchronous payload[%d]\n", i); return 1;
    }
    // Reverse direction, after poisoning the original buffer, exercises Async.
    CUDA(cudaSetDevice(0));
    CUDA(cudaMemset(source, 0, bytes));
    cudaStream_t stream = nullptr;
    CUDA(cudaStreamCreate(&stream));
    CUDA(cudaMemcpyPeerAsync(source, 0, destination, 1, bytes, stream));
    CUDA(cudaStreamSynchronize(stream));
    CUDA(cudaMemcpy(output.data(), source, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) if (output[i] != input[i]) {
        fprintf(stderr, "FAIL: asynchronous payload[%d]\n", i); return 1;
    }
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(source));
    CUDA(cudaSetDevice(1));
    CUDA(cudaFree(destination));
    printf("PASS: 4096 nonuniform values verified in both peer-copy directions.\n");
    printf("Copy success does not identify the physical transfer route or bandwidth.\n");
    return 0;
}
