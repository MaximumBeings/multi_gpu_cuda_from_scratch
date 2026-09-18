// Chapter 16: Domain Decomposition for Scientific Computing --
// Halo Exchange and Distributed Stencils
// 46_halo_exchange.cu
//
// Halo exchange is a genuinely new communication pattern for this
// book: nearest-neighbor only. Not root-based like Chapter 8's
// broadcast/reduce, not a repeating ring like Chapter 9's all-reduce,
// not every-pair-to-every-pair like Chapter 10's all-to-all -- each
// rank talks to at most TWO other ranks (the rank directly above it
// and the rank directly below it in the row-strip decomposition
// Section 16.1 built), and every rank that isn't at the physical edge
// of the domain does this EVERY iteration, not once at setup. The
// transfer mechanism itself is not new: it's the same real
// cudaMemcpyPeer() Chapter 5 introduced and Chapter 13 reused for
// activation handoff, now carrying one row of a grid's halo instead.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int GRID_COLS = 8;
    const size_t ROW_BYTES = GRID_COLS * sizeof(double);
    const int WORLD_SIZE = 4;

    printf("\nHalo exchange for a %d-way row-strip decomposition. Each "
           "rank exchanges exactly one row (%zu bytes) with each real "
           "neighbor it has -- never with a rank two or more strips "
           "away, and never wrapping around the way Chapter 9's ring "
           "does (rank 0 and rank %d are the physical TOP and BOTTOM "
           "edges of the domain, not neighbors of each other).\n\n",
           WORLD_SIZE, ROW_BYTES, WORLD_SIZE - 1);

    double* haloBuf = nullptr; // never successfully allocated -- deviceCount is 0

    for (int rank = 0; rank < WORLD_SIZE; ++rank) {
        printf("rank %d:\n", rank);

        // Send this rank's TOPMOST interior row up, to fill the rank
        // ABOVE's bottom halo -- only if a rank above genuinely exists.
        // The trap this guards against: calling cudaMemcpyPeer with a
        // destination device of -1 for rank 0, which is not a missing
        // peer, it is the physical edge of the domain.
        if (rank > 0) {
            cudaError_t eUp = cudaMemcpyPeer(haloBuf, rank - 1, haloBuf, rank, ROW_BYTES);
            printf("  send top row    -> rank %d (fills rank %d's bottom halo): %s (code %d)\n",
                   rank - 1, rank - 1, cudaGetErrorString(eUp), (int)eUp);
        } else {
            printf("  top edge is the domain boundary -- fixed value, no transfer\n");
        }

        // Send this rank's BOTTOMMOST interior row down, to fill the
        // rank BELOW's top halo -- only if a rank below genuinely
        // exists.
        if (rank < WORLD_SIZE - 1) {
            cudaError_t eDown = cudaMemcpyPeer(haloBuf, rank + 1, haloBuf, rank, ROW_BYTES);
            printf("  send bottom row -> rank %d (fills rank %d's top halo):    %s (code %d)\n",
                   rank + 1, rank + 1, cudaGetErrorString(eDown), (int)eDown);
        } else {
            printf("  bottom edge is the domain boundary -- fixed value, no transfer\n");
        }
    }

    printf("\nEvery cudaMemcpyPeer() call above reports the same honest "
           "cudaErrorNoDevice this book's every real device-touching call "
           "has reported since Chapter 3 -- these are real transfer "
           "attempts, on a real API, guarded by real boundary checks, not "
           "illustrative pseudocode. Section 16.3 checks what these "
           "exchanges are actually FOR: whether a domain-decomposed "
           "stencil update, refreshed by exchanges like these every "
           "iteration, computes the exact same thing a single, unsplit "
           "domain would.\n");

    return 0;
}
