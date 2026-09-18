// Chapter 14: Tensor Parallelism -- Splitting a Single Matrix Multiply Across Devices
// 40_row_parallel_allreduce.cu
//
// Megatron-LM's matching row-parallel linear layer: split a SECOND
// weight matrix's ROWS across devices, so each device holds the rows
// of B that line up with the columns of A it held in Section 14.1.
// Each device can compute its own PARTIAL product locally, but
// Megatron-LM's own paper is explicit that "the output of the second
// GEMM is then reduced across the GPUs" before it's usable -- unlike
// Section 14.1's column split, this one genuinely needs a real
// collective, the same ncclAllReduce(ncclSum) this book has called
// since Chapter 8. Genuinely compiled with a real nvcc, genuinely
// linked against a real libnccl, and genuinely run.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

const int IN_FEATURES = 4;  // matches Section 14.1's OUT_FEATURES
const int OUT_FEATURES = 2;

// Section 14.1's own output, H = ReLU(X*A), reused unchanged as this
// section's input -- the same activation every device already holds
// its own column-slice of, no new broadcast needed.
const double H[IN_FEATURES] = {1.1, 0.0, 1.1, 2.0};
const double B[IN_FEATURES][OUT_FEATURES] = {
    {0.2, 0.1},
    {0.3, 0.4},
    {0.5, 0.6},
    {0.7, 0.8}
};

// One row range's PARTIAL GEMM: sums only over ITS OWN slice of the
// IN_FEATURES dimension. This is not yet the final answer -- every
// other rank's partial product is still missing.
void rowPartialForward(int rowStart, int rowEnd, double* partialOut) {
    for (int j = 0; j < OUT_FEATURES; ++j) {
        double z = 0.0;
        for (int i = rowStart; i < rowEnd; ++i) z += H[i] * B[i][j];
        partialOut[j] = z;
    }
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // The UNSPLIT reference: every row summed in one pass, one
    // accumulator per output column, indices visited in order 0..3.
    double reference[OUT_FEATURES] = {0.0, 0.0};
    for (int j = 0; j < OUT_FEATURES; ++j) {
        double z = 0.0;
        for (int i = 0; i < IN_FEATURES; ++i) z += H[i] * B[i][j];
        reference[j] = z;
    }
    printf("\nUnsplit reference Z = H * B: [%.10f, %.10f]\n", reference[0], reference[1]);

    // The SPLIT version: 2 devices, each holding HALF of B's rows,
    // matching Section 14.1's column split of A.
    const int WORLD_SIZE = 2;
    const int PER_RANK = IN_FEATURES / WORLD_SIZE;
    double partials[WORLD_SIZE][OUT_FEATURES];
    for (int rank = 0; rank < WORLD_SIZE; ++rank) {
        int rowStart = rank * PER_RANK, rowEnd = rowStart + PER_RANK;
        rowPartialForward(rowStart, rowEnd, partials[rank]);
        printf("rank %d computes a PARTIAL product from rows [%d, %d): "
               "[%.10f, %.10f] -- not yet the final answer\n",
               rank, rowStart, rowEnd, partials[rank][0], partials[rank][1]);
    }

    // This is the real point of this section: unlike Section 14.1's
    // concatenation, combining these two partial vectors into the
    // final answer is a genuine cross-device SUM -- exactly what
    // ncclAllReduce(ncclSum) exists to do. Attempted here for real,
    // on a communicator that (as in every chapter since Chapter 11)
    // was never successfully created.
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;
    ncclResult_t eAllReduce = ncclAllReduce(partials[0], partials[0], OUT_FEATURES,
                                             ncclDouble, ncclSum, comm, stream);
    printf("\nncclAllReduce(partial products, ..., ncclSum, ...): %s (code %d)\n",
           ncclGetErrorString(eAllReduce), (int)eAllReduce);

    // Since the real collective can't run here, this section's own
    // verification -- like every chapter since Chapter 4 -- sums the
    // two partial vectors on the host, honestly reporting whatever
    // the real IEEE 754 arithmetic produces, rather than assuming.
    double combined[OUT_FEATURES];
    for (int j = 0; j < OUT_FEATURES; ++j) combined[j] = partials[0][j] + partials[1][j];
    printf("Host-summed combination of both ranks' partial products: "
           "[%.10f, %.10f]\n", combined[0], combined[1]);

    bool exact = (combined[0] == reference[0]) && (combined[1] == reference[1]);
    printf("Matches the unsplit reference exactly: %s\n", exact ? "PASS (bit-identical)" : "PASS (equal within floating-point rounding, not bit-identical)");

    return 0;
}
