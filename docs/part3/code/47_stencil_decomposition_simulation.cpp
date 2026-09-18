// Chapter 16: Domain Decomposition for Scientific Computing --
// Halo Exchange and Distributed Stencils
// 47_stencil_decomposition_simulation.cpp
//
// Plain host C++ -- this chapter's real verification, since no real
// device-to-device transfer could run on this machine. Part A checks
// the one property that actually matters about domain decomposition:
// does a grid split across ranks, with halo exchange refreshing each
// rank's ghost rows every iteration (Section 16.2), compute the exact
// same result a single, unsplit domain running the identical Jacobi
// iteration would? Part B checks the quantitative reason domain
// decomposition uses halo exchange at all instead of just giving every
// rank the whole grid: a closed-form comparison of how much data each
// approach actually moves.
//
// COMPILE WITH -ffp-contract=off on both this book's real toolchains
// (see Chapter 14's own discovery): this file prints exact grid-point
// values and checks exact bit equality between two code paths, so the
// same FMA-contraction discipline Chapter 14 established applies here.
#include <cstdio>
#include <cstring>
#include <vector>

const int GRID_ROWS = 16;   // interior rows of the physical domain
const int GRID_COLS = 8;    // interior columns
const int WORLD_SIZE = 4;   // devices the domain is split across
const int ROWS_PER_RANK = GRID_ROWS / WORLD_SIZE;
const int ITERS = 30;       // Jacobi iterations

// Dirichlet boundary conditions: one hot edge, three cold ones --
// the classic steady-state heat-plate problem these sources' own
// Jacobi/stencil examples use.
const double TOP_BOUNDARY = 100.0;
const double OTHER_BOUNDARY = 0.0;

// The exact five-point stencil formula Section 16.1 introduced,
// applied by BOTH the reference and the decomposed path below through
// this one shared function -- exactly the discipline Chapter 13's
// applyLayer() established: if both paths call the identical function
// on identical operands, the results cannot differ by construction.
inline double stencilUpdate(double up, double down, double left, double right) {
    return 0.25 * (up + down + left + right);
}

// ---------------------------------------------------------------
// Part A: correctness. A single unsplit domain vs. the same domain
// row-strip-decomposed across WORLD_SIZE ranks with halo exchange.
// ---------------------------------------------------------------

// Reference: one (GRID_ROWS+2) x (GRID_COLS+2) grid, boundary rows/
// columns fixed once, interior 0-initialized, updated with real Jacobi
// double-buffering (every cell in an iteration reads only the PREVIOUS
// iteration's values -- never a value already updated this iteration,
// which is what makes it Jacobi rather than Gauss-Seidel).
std::vector<double> referenceRun() {
    const int R = GRID_ROWS + 2, C = GRID_COLS + 2;
    std::vector<double> grid(R * C, 0.0), next(R * C, 0.0);
    auto at = [&](std::vector<double>& g, int i, int j) -> double& { return g[i * C + j]; };

    for (int j = 0; j < C; ++j) { at(grid, 0, j) = TOP_BOUNDARY; at(grid, R - 1, j) = OTHER_BOUNDARY; }
    for (int i = 0; i < R; ++i) { at(grid, i, 0) = OTHER_BOUNDARY; at(grid, i, C - 1) = OTHER_BOUNDARY; }
    next = grid; // boundary values never change across iterations

    for (int it = 0; it < ITERS; ++it) {
        for (int i = 1; i <= GRID_ROWS; ++i) {
            for (int j = 1; j <= GRID_COLS; ++j) {
                at(next, i, j) = stencilUpdate(at(grid, i - 1, j), at(grid, i + 1, j),
                                                at(grid, i, j - 1), at(grid, i, j + 1));
            }
        }
        std::swap(grid, next);
    }
    return grid; // full (GRID_ROWS+2) x (GRID_COLS+2), including boundary rows/cols
}

// Decomposed: WORLD_SIZE local buffers, each (ROWS_PER_RANK+2) x
// (GRID_COLS+2) -- its own interior rows plus one halo row top and
// bottom. Every iteration: (1) exchange halos with real neighbors
// (array copy standing in for Section 16.2's real cudaMemcpyPeer(),
// exactly this book's established host-side simulation technique --
// see getting-started.md), using each other rank's CURRENT committed
// row, never one already updated this iteration; (2) each rank
// computes its own new interior rows from its own old data plus its
// halos; (3) swap old/new per rank, same double-buffering as the
// reference.
std::vector<double> decomposedRun() {
    const int LR = ROWS_PER_RANK + 2, C = GRID_COLS + 2;
    std::vector<std::vector<double>> local(WORLD_SIZE, std::vector<double>(LR * C, 0.0));
    std::vector<std::vector<double>> nextLocal = local;
    auto at = [&](std::vector<double>& g, int i, int j) -> double& { return g[i * C + j]; };

    // Fixed boundary conditions set once: left/right columns on every
    // rank's own rows, and the domain's physical top/bottom edges on
    // rank 0's/rank (WORLD_SIZE-1)'s outer halo row.
    // Order matters at the four corner cells, where a row boundary and
    // a column boundary overlap: the reference below sets row
    // boundaries first, then column boundaries, so column boundary
    // values win at the corners. Matching that exact order here is
    // required for Part A's exact `==` check to pass -- setting them
    // in the opposite order is a real, easy-to-make bug (it was the
    // first version of this file's own bug: corners silently ended up
    // on the wrong boundary value).
    for (int r = 0; r < WORLD_SIZE; ++r) {
        for (int j = 0; j < C; ++j) {
            if (r == 0) at(local[r], 0, j) = TOP_BOUNDARY;                       // domain's top edge
            if (r == WORLD_SIZE - 1) at(local[r], LR - 1, j) = OTHER_BOUNDARY;   // domain's bottom edge
        }
        for (int i = 0; i < LR; ++i) { at(local[r], i, 0) = OTHER_BOUNDARY; at(local[r], i, C - 1) = OTHER_BOUNDARY; }
    }
    nextLocal = local;

    for (int it = 0; it < ITERS; ++it) {
        // Halo exchange -- Section 16.2's pattern, done every
        // iteration: each rank's row 1 (its topmost interior row) is
        // copied into the rank above's bottom halo (row LR-1), and
        // each rank's row LR-2 (its bottommost interior row) is
        // copied into the rank below's top halo (row 0). Ranks at the
        // domain's physical edge skip the side that has no real
        // neighbor -- that side already holds a fixed boundary value.
        for (int r = 0; r < WORLD_SIZE; ++r) {
            if (r > 0) {
                for (int j = 0; j < C; ++j) at(local[r - 1], LR - 1, j) = at(local[r], 1, j);
            }
            if (r < WORLD_SIZE - 1) {
                for (int j = 0; j < C; ++j) at(local[r + 1], 0, j) = at(local[r], LR - 2, j);
            }
        }

        // Each rank updates its own interior rows using ITS OWN old
        // data and its (just-refreshed) halos -- the identical
        // stencilUpdate() call the reference uses, on operands that
        // are bit-for-bit the same values the reference has at the
        // corresponding global grid position.
        for (int r = 0; r < WORLD_SIZE; ++r) {
            for (int i = 1; i <= ROWS_PER_RANK; ++i) {
                for (int j = 1; j <= GRID_COLS; ++j) {
                    at(nextLocal[r], i, j) = stencilUpdate(at(local[r], i - 1, j), at(local[r], i + 1, j),
                                                            at(local[r], i, j - 1), at(local[r], i, j + 1));
                }
            }
        }
        std::swap(local, nextLocal);
    }

    // Reassemble into one (GRID_ROWS+2) x (GRID_COLS+2) grid, same
    // layout the reference returns, for a direct cell-by-cell check.
    const int R = GRID_ROWS + 2;
    std::vector<double> full(R * C, 0.0);
    // Read the boundary rows back from rank 0's and rank (WORLD_SIZE-1)'s
    // own row 0 / row LR-1 -- not hard-coded again here -- so the
    // reassembled grid reflects exactly what each rank actually stored,
    // corner cells included.
    for (int j = 0; j < C; ++j) full[0 * C + j] = at(local[0], 0, j);
    for (int j = 0; j < C; ++j) full[(R - 1) * C + j] = at(local[WORLD_SIZE - 1], LR - 1, j);
    for (int r = 0; r < WORLD_SIZE; ++r) {
        for (int i = 1; i <= ROWS_PER_RANK; ++i) {
            int globalRow = r * ROWS_PER_RANK + i; // 1-indexed into the full grid's interior
            for (int j = 0; j < C; ++j) full[globalRow * C + j] = at(local[r], i, j);
        }
    }
    return full;
}

int main() {
    printf("Part A: domain-decomposed vs. unsplit Jacobi stencil, %d "
           "iterations, %d x %d interior grid, %d-way row-strip "
           "decomposition (%d rows/rank).\n\n",
           ITERS, GRID_ROWS, GRID_COLS, WORLD_SIZE, ROWS_PER_RANK);

    std::vector<double> ref = referenceRun();
    std::vector<double> dec = decomposedRun();

    const int R = GRID_ROWS + 2, C = GRID_COLS + 2;
    int mismatches = 0;
    for (int idx = 0; idx < R * C; ++idx) {
        if (ref[idx] != dec[idx]) ++mismatches;
    }

    printf("Sample interior cell values (row, col are 1-indexed into "
           "the %d x %d interior):\n", GRID_ROWS, GRID_COLS);
    int sampleRows[] = {1, 4, 8, 12, 16};
    for (int i : sampleRows) {
        int j = GRID_COLS / 2;
        double r = ref[i * C + j], d = dec[i * C + j];
        printf("  (row %2d, col %d): unsplit = %.17g   decomposed = %.17g   %s\n",
               i, j, r, d, (r == d) ? "bit-identical" : "DIFFERS");
    }

    printf("\n%d of %d total grid cells (including fixed boundary cells) "
           "mismatched between the unsplit reference and the %d-way "
           "domain-decomposed run: %s\n",
           mismatches, R * C, WORLD_SIZE, (mismatches == 0) ? "PASS (bit-identical)" : "FAIL");

    if (mismatches == 0) {
        printf("\nThis is Chapter 13's own structural argument, showing up "
               "again in a completely different setting: a stencil update "
               "reads a FIXED set of neighbor values and applies the SAME "
               "arithmetic regardless of which device happens to store "
               "each operand. Splitting the grid into row-strips only "
               "relocates WHERE each cell's data lives; halo exchange "
               "delivers the exact same bit pattern a neighboring cell "
               "would have held in an unsplit array. No operation's order "
               "changes, and -- unlike Chapter 14's row-parallel combine, "
               "which really does sum partial results in a different "
               "order -- no reduction happens here at all, so exact `==` "
               "is a valid check, not merely a convenient one.\n");
    }

    // ---------------------------------------------------------------
    // Part B: why halo exchange, not full-domain replication. A
    // closed-form comparison of data moved per iteration, not a timed
    // benchmark -- counting elements, the same technique Chapter 9's
    // 9.3 and Chapter 11's 11.3 used for their own closed-form models.
    // ---------------------------------------------------------------
    printf("\nPart B: data moved per iteration -- halo exchange vs. giving "
           "every rank the whole domain every iteration.\n");

    struct GridCase { const char* label; long rows, cols; int world; };
    GridCase cases[] = {
        {"this chapter's own grid", GRID_ROWS, GRID_COLS, WORLD_SIZE},
        {"a more realistic grid",   1024,       1024,      8},
    };

    for (const auto& c : cases) {
        long haloElemsPerIter = 2L * (c.world - 1) * c.cols;
        long fullReplicationElemsPerIter = (long)c.world * c.rows * c.cols;
        double ratio = (double)fullReplicationElemsPerIter / (double)haloElemsPerIter;
        printf("\n  %s: %ld x %ld interior grid, %d-way decomposition\n"
               "    halo exchange (Section 16.2):        %ld elements/iteration\n"
               "    full-domain replication every iter:  %ld elements/iteration\n"
               "    halo exchange moves %.1fx less data\n",
               c.label, c.rows, c.cols, c.world,
               haloElemsPerIter, fullReplicationElemsPerIter, ratio);
    }

    printf("\nHalo-exchange traffic scales with each rank's BOUNDARY -- "
           "its perimeter, GRID_COLS elements per shared edge -- while "
           "full-domain replication scales with the entire domain's "
           "VOLUME, every rank re-receiving every cell every iteration. "
           "This is exactly the \"surface/volume behavior\" William "
           "Gropp's own halo-exchange lecture names as the reason "
           "domain decomposition communicates only what changed at a "
           "boundary, not the whole state.\n");

    return (mismatches == 0) ? 0 : 1;
}
