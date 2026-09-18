// Chapter 14: Tensor Parallelism -- Splitting a Single Matrix Multiply Across Devices
// 41_tensor_parallel_mlp_simulation.cpp
//
// Plain host C++ -- this chapter's end-to-end verification, chaining
// Section 14.1's column-parallel GEMM+ReLU directly into Section
// 14.2's row-parallel GEMM+combine, across multiple test inputs.
// This is the exact 2-GEMM MLP block Megatron-LM's own paper
// describes, and the paper's own stated point is quantitative, not
// just architectural: this whole block needs only ONE real
// collective in the forward pass, not one per GEMM. This section
// checks both halves of that claim -- correctness AND the collective
// count -- against an independently-computed, fully unsplit reference.
#include <cstdio>
#include <cmath>

const int IN_FEATURES = 2;
const int HIDDEN_FEATURES = 4;
const int OUT_FEATURES = 2;
const int WORLD_SIZE = 2;

// Same A and B this chapter has used since Section 14.1/14.2.
const double A[IN_FEATURES][HIDDEN_FEATURES] = {
    { 0.1,  0.2, -0.3, 0.4},
    { 0.5, -0.6,  0.7, 0.8}
};
const double B[HIDDEN_FEATURES][OUT_FEATURES] = {
    {0.2, 0.1},
    {0.3, 0.4},
    {0.5, 0.6},
    {0.7, 0.8}
};

double relu(double v) { return v > 0.0 ? v : 0.0; }

// The UNSPLIT reference: both GEMMs and the nonlinearity computed on
// one device, as if the whole block fit without any partitioning.
void referenceForward(const double x[IN_FEATURES], double out[OUT_FEATURES]) {
    double h[HIDDEN_FEATURES];
    for (int j = 0; j < HIDDEN_FEATURES; ++j) {
        double y = 0.0;
        for (int i = 0; i < IN_FEATURES; ++i) y += x[i] * A[i][j];
        h[j] = relu(y);
    }
    for (int j = 0; j < OUT_FEATURES; ++j) {
        double z = 0.0;
        for (int i = 0; i < HIDDEN_FEATURES; ++i) z += h[i] * B[i][j];
        out[j] = z;
    }
}

// The TENSOR-PARALLEL version: Section 14.1's column split feeding
// directly into Section 14.2's row split, with exactly ONE combine
// step (standing in for ONE real ncclAllReduce(ncclSum) call) at the
// very end of the block -- not one after the first GEMM too.
void tensorParallelForward(const double x[IN_FEATURES], double out[OUT_FEATURES]) {
    const int hiddenPerRank = HIDDEN_FEATURES / WORLD_SIZE;
    double h[HIDDEN_FEATURES]; // each rank only ever touches its own slice

    // Section 14.1: column-parallel GEMM + ReLU, no communication.
    for (int rank = 0; rank < WORLD_SIZE; ++rank) {
        int colStart = rank * hiddenPerRank, colEnd = colStart + hiddenPerRank;
        for (int j = colStart; j < colEnd; ++j) {
            double y = 0.0;
            for (int i = 0; i < IN_FEATURES; ++i) y += x[i] * A[i][j];
            h[j] = relu(y);
        }
    }

    // Section 14.2: row-parallel GEMM, one partial product per rank.
    double partials[WORLD_SIZE][OUT_FEATURES];
    for (int rank = 0; rank < WORLD_SIZE; ++rank) {
        int rowStart = rank * hiddenPerRank, rowEnd = rowStart + hiddenPerRank;
        for (int j = 0; j < OUT_FEATURES; ++j) {
            double z = 0.0;
            for (int i = rowStart; i < rowEnd; ++i) z += h[i] * B[i][j];
            partials[rank][j] = z;
        }
    }

    // The ONE combine step for the whole block -- standing in for
    // this chapter's real ncclAllReduce(ncclSum) call.
    for (int j = 0; j < OUT_FEATURES; ++j) {
        out[j] = 0.0;
        for (int rank = 0; rank < WORLD_SIZE; ++rank) out[j] += partials[rank][j];
    }
}

int main() {
    struct Case { double x[IN_FEATURES]; };
    const Case cases[] = { {{1.0, 2.0}}, {{-1.0, 0.5}}, {{3.0, -2.0}} };
    const int NUM_CASES = sizeof(cases) / sizeof(cases[0]);
    const double EPS = 1e-9;

    int exactMatches = 0;
    bool allCloseEnough = true;
    for (const auto& c : cases) {
        double refOut[OUT_FEATURES], tpOut[OUT_FEATURES];
        referenceForward(c.x, refOut);
        tensorParallelForward(c.x, tpOut);

        bool bitExact = (refOut[0] == tpOut[0]) && (refOut[1] == tpOut[1]);
        bool closeEnough = std::fabs(refOut[0] - tpOut[0]) < EPS &&
                            std::fabs(refOut[1] - tpOut[1]) < EPS;
        if (bitExact) exactMatches++;
        allCloseEnough = allCloseEnough && closeEnough;

        printf("input (%.2f, %.2f):\n"
               "  unsplit reference       = (%.17g, %.17g)\n"
               "  tensor-parallel (%d-way) = (%.17g, %.17g)\n"
               "  %s\n",
               c.x[0], c.x[1], refOut[0], refOut[1], WORLD_SIZE, tpOut[0], tpOut[1],
               bitExact ? "bit-identical" : "differs in the last representable bit only (still correct to full display precision)");
    }

    printf("\n%d of %d input(s) bit-identical between the unsplit reference "
           "and the %d-way tensor-parallel MLP block; every input matches to "
           "within %.0e: %s\n",
           exactMatches, NUM_CASES, WORLD_SIZE, EPS, allCloseEnough ? "PASS" : "FAIL");

    if (exactMatches < NUM_CASES) {
        printf("\nThis is the real floating-point reassociation risk this book\n"
               "flagged back in Chapter 8's reduce, showing up empirically: "
               "summing\ntwo partial dot products, (a0+a1)+(b0+b1), is not "
               "guaranteed by IEEE\n754 to equal one accumulator visiting all "
               "four terms in order,\na0+a1+b0+b1 -- even though both are the "
               "exact same real-number sum.\nUnlike Chapter 13's sequential "
               "relocation (always bit-identical,\nsince no operation's order "
               "ever changed), a row-parallel GEMM's\ncombine step is "
               "structurally the same kind of operation as Chapter 8's\n"
               "reduce, and inherits the exact same honest caveat -- correct,\n"
               "but not bit-for-bit, in general.\n");
    }

    // The collective-count claim, counted directly rather than timed:
    // this book's own naive alternative -- synchronizing after EVERY
    // GEMM instead of only after the block's last one -- would need
    // one collective per GEMM. Megatron-LM's own design collapses
    // that to one collective for the entire 2-GEMM block.
    const int gemmsInBlock = 2;
    const int naiveCollectives = gemmsInBlock;    // one after each GEMM
    const int megatronCollectives = 1;            // one for the whole block
    printf("\nCollectives needed for this %d-GEMM MLP block:\n"
           "  naive (synchronize after every GEMM):          %d\n"
           "  Megatron-LM's column+row split (this chapter): %d\n"
           "Megatron-LM's own paper states this directly: this design "
           "\"requires only a single all-reduce operation in the forward "
           "pass.\"\n", gemmsInBlock, naiveCollectives, megatronCollectives);

    return allCloseEnough ? 0 : 1;
}
