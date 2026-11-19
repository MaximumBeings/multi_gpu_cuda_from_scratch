// Chapter 13: Model Parallelism -- Layer ranges and a real activation handoff.
// 37_layer_partition_activation_handoff.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CUDA(call) do { cudaError_t status = (call); if (status != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ? 2 : 1; } } while (0)
void computeLayerRange(int layers, int world, int rank, int* start, int* end) {
    *start = layers * rank / world;
    *end = layers * (rank + 1) / world;
}
static uint16_t pattern(size_t index) { return uint16_t(0x0400 + (index * 17) % 0x7400); }
int main() {
    int count = 0;
    CUDA(cudaGetDeviceCount(&count));
    if (count < 2) { printf("BLOCKED: requires 2 GPUs; found %d.\n", count); return 2; }
    int world = std::min(count, 4);
    for (int rank = 0; rank < world; ++rank) {
        int begin, end;
        computeLayerRange(96, world, rank, &begin, &end);
        printf("rank %d: layer indices [%d, %d)\n", rank, begin, end);
    }
    // GPT-3 dimensions size this handoff, but no GPT-3 layer is executed here.
    // The uint16_t words are varied finite positive binary16 bit patterns.
    const size_t elements = size_t(2048) * 12288;
    const size_t bytes = elements * sizeof(uint16_t);
    std::vector<uint16_t> input(elements), output(elements);
    for (size_t i = 0; i < elements; ++i) input[i] = pattern(i);
    uint16_t *producer = nullptr, *consumer = nullptr;
    CUDA(cudaSetDevice(0));
    CUDA(cudaMalloc(&producer, bytes));
    CUDA(cudaMemcpy(producer, input.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaSetDevice(1));
    CUDA(cudaMalloc(&consumer, bytes));
    CUDA(cudaMemset(consumer, 0, bytes));
    CUDA(cudaMemcpyPeer(consumer, 1, producer, 0, bytes));
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(output.data(), consumer, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < elements; ++i) if (output[i] != pattern(i)) {
        fprintf(stderr, "FAIL: activation word %zu: %u expected %u\n", i, unsigned(output[i]), unsigned(pattern(i)));
        return 1;
    }
    CUDA(cudaFree(consumer));
    CUDA(cudaSetDevice(0));
    CUDA(cudaFree(producer));
    printf("PASS: %zu nonuniform FP16-sized activation words (%zu bytes) handed off exactly.\n", elements, bytes);
    printf("Scope: layer-index partition and transfer payload, not a transformer forward pass.\n");
    return 0;
}
