// Chapter 14: Tensor Parallelism -- Splitting a Single Matrix Multiply Across Devices
// 39_column_parallel_linear.cu
//
// Megatron-LM's own column-parallel linear layer: split a weight
// matrix's OUTPUT columns across devices, so [Y1, Y2] = [X*A1, X*A2]
// -- each device computes a DIFFERENT slice of the SAME output
// vector, entirely independently. Megatron-LM's own paper states the
// consequence directly: this partitioning "allows the GeLU
// nonlinearity to be independently applied to the output of each
// partitioned GEMM," because an elementwise nonlinearity never needs
// to know what any other element's value is. The result: this
// section is the first in this entire book where there is no real
// device-touching call left to genuinely attempt at all, beyond the
// device count query every chapter has run since Chapter 3 -- and
// that absence IS this section's real content, not an omission.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

const int IN_FEATURES = 2;
const int OUT_FEATURES = 4;

// The full, unsplit weight matrix A (IN_FEATURES x OUT_FEATURES) and
// input X (1 x IN_FEATURES). Small, deterministic, hand-traceable
// values -- nothing about the specific numbers matters except that
// both the split and unsplit paths use the exact same ones.
const double A[IN_FEATURES][OUT_FEATURES] = {
    { 0.1,  0.2, -0.3, 0.4},
    { 0.5, -0.6,  0.7, 0.8}
};
const double X[IN_FEATURES] = {1.0, 2.0};

double relu(double v) { return v > 0.0 ? v : 0.0; }

// One column range's GEMM + ReLU, computed entirely independently of
// any other column range -- this IS the function every device would
// run on its own local column slice of A, with no input from any
// other device.
void columnPartialForward(int colStart, int colEnd, double* out) {
    for (int j = colStart; j < colEnd; ++j) {
        double y = 0.0;
        for (int i = 0; i < IN_FEATURES; ++i) y += X[i] * A[i][j];
        out[j - colStart] = relu(y);
    }
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // The UNSPLIT reference: every output column computed in one pass.
    double reference[OUT_FEATURES];
    columnPartialForward(0, OUT_FEATURES, reference);

    printf("\nUnsplit reference H = ReLU(X * A): [");
    for (int j = 0; j < OUT_FEATURES; ++j) printf("%s%.4f", j ? ", " : "", reference[j]);
    printf("]\n");

    // The SPLIT version: 2 devices, each holding HALF of A's columns.
    // rank 0 computes columns [0,2), rank 1 computes columns [2,4) --
    // genuinely independently; rank 1 needs nothing rank 0 computed,
    // and vice versa.
    const int WORLD_SIZE = 2;
    const int PER_RANK = OUT_FEATURES / WORLD_SIZE;
    double split[OUT_FEATURES];
    for (int rank = 0; rank < WORLD_SIZE; ++rank) {
        int colStart = rank * PER_RANK, colEnd = colStart + PER_RANK;
        double partial[PER_RANK];
        columnPartialForward(colStart, colEnd, partial);
        for (int k = 0; k < PER_RANK; ++k) split[colStart + k] = partial[k];
        printf("rank %d computes columns [%d, %d) independently: [", rank, colStart, colEnd);
        for (int k = 0; k < PER_RANK; ++k) printf("%s%.4f", k ? ", " : "", partial[k]);
        printf("]\n");
    }

    bool exact = true;
    for (int j = 0; j < OUT_FEATURES; ++j) if (split[j] != reference[j]) exact = false;
    printf("\nConcatenating both ranks' independently-computed columns "
           "matches the unsplit reference exactly: %s\n", exact ? "PASS (bit-identical)" : "FAIL");

    printf("\nNotice what this section did NOT need: no cudaMemcpyPeer(),\n"
           "no ncclBroadcast(), no ncclAllReduce() -- nothing beyond the\n"
           "device count query above. Splitting a GEMM's OUTPUT columns,\n"
           "followed by an elementwise nonlinearity, is the one shape of\n"
           "tensor-parallel split this book will show that needs no real\n"
           "collective at all, PROVIDED every device already holds an\n"
           "identical copy of X -- which itself had to get there via a\n"
           "real broadcast (Chapter 8, Chapter 12), before this section's\n"
           "code ever ran. Section 14.2 shows the very next GEMM in the\n"
           "same block, where that free ride ends.\n");

    return exact ? 0 : 1;
}
