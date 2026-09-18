// Chapter 3: The CUDA Multi-GPU Programming Model
// 06_single_device_setup.cu
//
// The single-device baseline every multi-device program is built from:
// pick a device, allocate on it, launch on it, wait for it, read the
// result back, free it. Every step's return code is checked -- that
// discipline is what turns "no device present" from a crash into an
// honest, informative report instead.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void addOneKernel(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}

#define CHECK(call, label) do { \
    cudaError_t _e = (call); \
    printf("%-28s %s (code %d)\n", label, cudaGetErrorString(_e), (int)_e); \
    if (_e != cudaSuccess) { \
        printf("Stopping here -- every step after this one depends on " \
               "this one having succeeded.\n"); \
        return 0; \
    } \
} while (0)

int main() {
    const int N = 8;

    CHECK(cudaSetDevice(0), "cudaSetDevice(0)");

    int* d_data = nullptr;
    CHECK(cudaMalloc(&d_data, N * sizeof(int)), "cudaMalloc");

    int h_in[N] = {0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(cudaMemcpy(d_data, h_in, N * sizeof(int), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");

    addOneKernel<<<1, N>>>(d_data, N);
    CHECK(cudaGetLastError(), "kernel launch");
    CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    int h_out[N] = {0};
    CHECK(cudaMemcpy(h_out, d_data, N * sizeof(int), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");

    printf("Result: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_out[i]);
    printf("\n");

    cudaFree(d_data);
    return 0;
}
