// Chapter 16: Domain Decomposition for Scientific Computing --
// Halo Exchange and Distributed Stencils
// 45_domain_decomposition_row_range.cu
//
// This chapter leaves neural-network parallelism behind. What gets
// split across devices here is not a model's layers (Ch13), a single
// matrix multiply (Ch14), or a batch of independent samples (Ch12) --
// it is a literal PHYSICAL GRID, the discretized space a scientific
// simulation runs over. Splitting it is still the same host-arithmetic
// pattern this book has used every time since Chapter 12's
// computeShard(): divide a known total by a known device count,
// compute offsets. What is genuinely new starts in Section 16.2 --
// unlike a batch shard, a grid row-strip is NOT independent of its
// neighbors.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

// Splits GRID_ROWS interior rows of a 2D grid into `worldSize` equal,
// non-overlapping, contiguous row ranges, and returns device `rank`'s
// own [start, end) range -- structurally identical to Chapter 12's
// computeShard() and Chapter 13's computeLayerRange(). Only WHAT is
// being partitioned changes: grid rows instead of samples or layers.
void computeRowRange(int totalRows, int worldSize, int rank, int* start, int* end) {
    int perDevice = totalRows / worldSize; // assumes an even split
    *start = rank * perDevice;
    *end = *start + perDevice;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // A modest 2D grid standing in for a discretized physical domain --
    // e.g. a metal plate's temperature field, one scalar per grid
    // point. 16 interior rows x 8 columns. Real Jacobi solvers used in
    // production run grids orders of magnitude larger; the partitioning
    // arithmetic and the halo-exchange pattern built in this chapter
    // are identical at any size.
    const int GRID_ROWS = 16;
    const int GRID_COLS = 8;

    printf("\nPartitioning a %d x %d grid's rows across this book's own "
           "real deviceCount (%d):\n", GRID_ROWS, GRID_COLS, deviceCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        int start, end;
        computeRowRange(GRID_ROWS, deviceCount, rank, &start, &end);
        printf("  rank %d: rows [%d, %d)\n", rank, start, end);
    }
    printf("  (%d device(s) -- nothing to print above.)\n", deviceCount);

    // Hypothetically, with 4 devices: every rank gets a different,
    // non-overlapping, equal-sized band of consecutive rows -- a
    // row-strip domain decomposition. (Real production codes often
    // decompose in 2D as well, splitting columns too -- NVIDIA's own
    // "Multi-GPU Programming with MPI" materials describe exactly that
    // as "2D domain decomposition with n x k domains." This chapter
    // builds the simpler 1D row-strip case; the halo-exchange pattern
    // Section 16.2 builds generalizes unchanged to 2D, just with four
    // neighbors per rank instead of two.)
    const int HYPOTHETICAL_WORLD_SIZE = 4;
    printf("\nThe same formula, hypothetically, with world size %d:\n", HYPOTHETICAL_WORLD_SIZE);
    for (int rank = 0; rank < HYPOTHETICAL_WORLD_SIZE; ++rank) {
        int start, end;
        computeRowRange(GRID_ROWS, HYPOTHETICAL_WORLD_SIZE, rank, &start, &end);
        // Every rank also needs to know what borders it ABOVE and
        // BELOW its own rows: another rank's rows (a real halo), or
        // the physical edge of the domain itself (a fixed boundary
        // condition, not a transfer at all).
        const char* above = (rank == 0) ? "domain boundary (fixed value)" : "rank above (real halo)";
        const char* below = (rank == HYPOTHETICAL_WORLD_SIZE - 1) ? "domain boundary (fixed value)" : "rank below (real halo)";
        printf("  rank %d: rows [%d, %d)  above: %-30s below: %s\n",
               rank, start, end, above, below);
    }

    // The five-point stencil this chapter's Jacobi solver uses to
    // update one interior grid point (i, j) from its four direct
    // neighbors -- the same formula a well-known HPC teaching exercise
    // states as E_ij = I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1) -
    // 4*I(i,j), rearranged into a Jacobi update:
    //   new(i,j) = 0.25 * ( old(i-1,j) + old(i+1,j)
    //                     + old(i,j-1) + old(i,j+1) )
    // Notice what this formula needs at row boundaries: computing
    // new(i,j) for a row at the very TOP of a rank's own strip needs
    // old(i-1,j) -- a row that belongs to the rank ABOVE. That single
    // dependency is the entire reason Section 16.2 exists.
    printf("\nFive-point stencil (interior update): new(i,j) = 0.25 * "
           "(old(i-1,j) + old(i+1,j) + old(i,j-1) + old(i,j+1))\n"
           "The top row of every rank's own strip needs old(i-1,j) from "
           "the rank ABOVE it; the bottom row needs old(i+1,j) from the "
           "rank BELOW it. Neither value lives on this rank.\n");

    return 0;
}
