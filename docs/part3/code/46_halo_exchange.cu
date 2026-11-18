// Chapter 16: checked nearest-neighbor halo transfers on 2--4 real GPUs.
// 46_halo_exchange.cu
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#define CUDA(call) do { cudaError_t status = (call); if (status != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); \
    return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ? 2 : 1; } } while (0)
constexpr int ROWS = 3, COLS = 11;
constexpr double EDGE = -99.0, POISON = -777.0;
static double value(int iteration, int rank, int row, int column) {
    return double(iteration * 10000 + rank * 1000 + row * COLS + column) / 8.0;
}
int main() {
    int count = 0;
    CUDA(cudaGetDeviceCount(&count));
    if (count < 2) { printf("BLOCKED: requires 2 GPUs; found %d.\n", count); return 2; }
    const int world = std::min(count, 4);
    const size_t row_bytes = COLS * sizeof(double), bytes = (ROWS + 2) * row_bytes;
    std::vector<double*> grids(world, nullptr);
    for (int rank = 0; rank < world; ++rank) {
        CUDA(cudaSetDevice(rank));
        CUDA(cudaMalloc(&grids[rank], bytes));
    }
    for (int iteration = 0; iteration < 3; ++iteration) {
        // Poison every internal halo each generation, and vary all interior data.
        for (int rank = 0; rank < world; ++rank) {
            std::vector<double> input((ROWS + 2) * COLS, POISON);
            for (int row = 1; row <= ROWS; ++row)
                for (int col = 0; col < COLS; ++col) input[row * COLS + col] = value(iteration, rank, row, col);
            if (rank == 0) std::fill(input.begin(), input.begin() + COLS, EDGE);
            if (rank == world - 1) std::fill(input.end() - COLS, input.end(), EDGE);
            CUDA(cudaSetDevice(rank));
            CUDA(cudaMemcpy(grids[rank], input.data(), bytes, cudaMemcpyHostToDevice));
        }
        for (int rank = 0; rank < world; ++rank) {
            CUDA(cudaSetDevice(rank));
            if (rank > 0) CUDA(cudaMemcpyPeer(grids[rank-1] + (ROWS+1)*COLS, rank-1,
                                            grids[rank] + COLS, rank, row_bytes));
            if (rank + 1 < world) CUDA(cudaMemcpyPeer(grids[rank+1], rank+1,
                                                     grids[rank] + ROWS*COLS, rank, row_bytes));
        }
        // One host controls all phase submissions: wait-all before validation.
        for (int rank = 0; rank < world; ++rank) {
            CUDA(cudaSetDevice(rank));
            CUDA(cudaDeviceSynchronize());
        }
        for (int rank = 0; rank < world; ++rank) {
            std::vector<double> output((ROWS + 2) * COLS);
            CUDA(cudaSetDevice(rank));
            CUDA(cudaMemcpy(output.data(), grids[rank], bytes, cudaMemcpyDeviceToHost));
            for (int row = 0; row < ROWS + 2; ++row) for (int col = 0; col < COLS; ++col) {
                double expected = row == 0 ? (rank == 0 ? EDGE : value(iteration, rank-1, ROWS, col))
                    : row == ROWS+1 ? (rank == world-1 ? EDGE : value(iteration, rank+1, 1, col))
                    : value(iteration, rank, row, col);
                if (output[row*COLS+col] != expected) {
                    fprintf(stderr, "FAIL: generation %d rank %d row %d col %d got %.9g expected %.9g\n",
                            iteration, rank, row, col, output[row*COLS+col], expected);
                    return 1;
                }
            }
        }
    }
    for (int rank = 0; rank < world; ++rank) {
        CUDA(cudaSetDevice(rank));
        CUDA(cudaFree(grids[rank]));
    }
    printf("PASS: %d GPUs, 3 halo generations; all interior/halo/boundary values verified.\n", world);
    printf("Scope: halo transport and boundary preservation, not a stencil-kernel benchmark.\n");
    return 0;
}
