# Chapter 16: Domain Decomposition for Scientific Computing: Halo Exchange and Distributed Stencils

**What you will understand by the end of this chapter:**

- Why this chapter splits something fundamentally different from every chapter since Chapter 12: not a batch of independent samples, not a model's layers, not a single matrix multiply, but a literal physical grid -- the discretized space a scientific simulation runs over.
- How to partition a grid's rows across devices using the same host-arithmetic pattern this book has used since Chapter 12's `computeShard()`, and why a grid row-strip, unlike a data-parallel batch shard, is never independent of its neighbors.
- Halo exchange: a genuinely new communication pattern for this book -- nearest-neighbor only, repeated every iteration, built from the same real `cudaMemcpyPeer()` Chapter 5 introduced, but used in a way no earlier chapter's pattern (root-based, ring, or all-to-all) matches.
- Why a domain-decomposed stencil update, refreshed by halo exchange every iteration, produces a bit-identical result to an unsplit one -- and why that still isn't automatic, the way Chapter 13's sequential relocation was: it depends on getting boundary-condition ordering exactly right.
- The real, quantifiable reason halo exchange moves so much less data than giving every device the whole grid: communication scales with a rank's *boundary*, not its *volume*.

**What you need to know first:**

- Chapter 5's real `cudaMemcpyPeer()` call, and Chapter 13's `computeLayerRange()` partitioning pattern this chapter's `computeRowRange()` mirrors exactly.
- Chapter 9's ring all-reduce (this chapter's halo exchange is deliberately contrasted with it -- no wraparound).
- Chapter 13's structural argument for why a split computation can be checked with exact `==` instead of an epsilon tolerance.

---

Every chapter since Chapter 12 has split something that lives only inside a training run: a batch of samples, a model's layers, one matrix multiply, a pipeline of stages. None of those things has an inherent physical shape. This chapter splits something that does. A scientific simulation -- the temperature across a metal plate, the pressure in a fluid, the electric field in a volume -- is usually represented as a grid: a discretized version of real 2D or 3D space, one value per grid point, updated every iteration from its neighbors' values according to the physics being modeled. Splitting that grid across devices is still the same host-arithmetic partitioning this book has used since Chapter 12 -- divide a known total by a known device count, compute offsets. What's new is what happens *after* the split: a grid point at the edge of one device's strip needs its neighbor's value to update correctly, and that neighbor lives on a different device. This chapter builds the pattern that supplies it -- a halo exchange -- and proves it doesn't change the answer.

```text
Chapters 12-15 split something ABSTRACT:        This chapter splits something PHYSICAL:
  data batches, model layers, matrices,            an actual simulation grid -- each
  pipeline stages -- no inherent 2D shape          device owns a strip of real space,
                                                    directly adjacent to its neighbors'

  dev0: [ ALL layers ] <- data[0:4)      dev0: [ rows  0- 3 ]  (grid's TOP edge)
  dev1: [ ALL layers ] <- data[4:8)            [ halo row  4 ]  <- copied from dev1
                                          --------------------------------------
                                          dev1: [ halo row  3 ]  <- copied from dev0
                                                [ rows  4- 7 ]
                                                [ halo row  8 ]  <- copied from dev2
                                          --------------------------------------
                                          dev2: [ halo row  7 ]  <- copied from dev1
                                                [ rows  8-11 ]
                                                ...

  Shards never talk to each other          Every rank needs its neighbors' boundary
  between the periodic all-reduces.        data EVERY single iteration.
```

## 16.1 Domain Decomposition: Splitting the Grid Itself

### Intuition

Imagine a large rectangular metal plate, held at a fixed hot temperature along its top edge and a fixed cold temperature along the other three edges, left to reach a steady-state temperature everywhere in between. Simulating that plate means discretizing it into a grid of points and repeatedly updating each interior point's temperature as the average of its four immediate neighbors -- up, down, left, and right -- until the values stop changing much. That averaging rule is called a five-point stencil, and applying it repeatedly is a Jacobi iteration. Splitting the work across four devices is the obvious move: give each device a horizontal band of rows to own, the same way Chapter 12 gave each device a band of samples and Chapter 13 gave each device a band of layers. But there is an immediate difference this chapter has to deal with that neither of those chapters did. A row at the very bottom of device 0's band needs the row directly below it to compute its own update -- and that row belongs to device 1. Chapter 12's data shards never needed anything from each other between synchronization points; a grid row-strip needs its neighbor's data on *every single iteration*, forever, for as long as the simulation runs.

```text
Interior grid, 16 rows x 8 cols, split 4 ways (row-strip decomposition):

  row  0 ................   <- rank 0 owns rows [ 0, 4)
  row  1 ................
  row  2 ................
  row  3 ................
  row  4 ................   <- rank 1 owns rows [ 4, 8)
  row  5 ................
  row  6 ................
  row  7 ................
  row  8 ................   <- rank 2 owns rows [ 8,12)
  row  9 ................
  row 10 ................
  row 11 ................
  row 12 ................   <- rank 3 owns rows [12,16)
  row 13 ................
  row 14 ................
  row 15 ................

  Each rank owns a CONTIGUOUS band of rows -- the same host arithmetic
  as Chapter 12's computeShard() / Chapter 13's computeLayerRange(),
  now partitioning grid rows instead of samples or layers.
```

### Background

Assigning row ranges is, arithmetically, nothing new: divide `GRID_ROWS` by the device count and compute an offset, exactly the way Chapter 12's `computeShard()` divided a batch and Chapter 13's `computeLayerRange()` divided a stack of layers. What's genuinely new is what each rank has to track *about its neighbors*. Every rank in the middle of the decomposition has a real neighbor both above and below it -- another rank's own rows, which will need to be exchanged before every update. The two ranks at the very top and bottom of the decomposition instead border the physical edge of the domain itself: a fixed boundary condition, not a device, not something that will ever arrive over `cudaMemcpyPeer()`. Getting that distinction right in code -- which side of each rank is a real halo and which is a fixed value -- is what this section builds, before Section 16.2 uses it to guard a real transfer.

```cpp
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
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

Partitioning a 16 x 8 grid's rows across this book's own real deviceCount (0):
  (0 device(s) -- nothing to print above.)

The same formula, hypothetically, with world size 4:
  rank 0: rows [0, 4)  above: domain boundary (fixed value)  below: rank below (real halo)
  rank 1: rows [4, 8)  above: rank above (real halo)         below: rank below (real halo)
  rank 2: rows [8, 12)  above: rank above (real halo)         below: rank below (real halo)
  rank 3: rows [12, 16)  above: rank above (real halo)         below: domain boundary (fixed value)

Five-point stencil (interior update): new(i,j) = 0.25 * (old(i-1,j) + old(i+1,j) + old(i,j-1) + old(i,j+1))
The top row of every rank's own strip needs old(i-1,j) from the rank ABOVE it; the bottom row needs old(i+1,j) from the rank BELOW it. Neither value lives on this rank.
```

Four ranks split the grid's 16 real rows into four genuinely computed, equal, 4-row bands -- the same host arithmetic Chapter 12 and Chapter 13 both used, now partitioning physical space instead of a batch or a stack of layers. What the printed table adds, that neither of those earlier chapters needed, is a classification of *both sides* of every rank's own band: a real neighboring rank, or the domain's own physical edge. Rank 0 and rank 3 each have exactly one side that is a fixed boundary condition, never a transfer; ranks 1 and 2 have two real neighbors, one on each side. The five-point stencil formula makes clear why this distinction has to be tracked precisely: every interior row's update reads a specific neighboring row, and for the rows at the top and bottom of a rank's own band, that neighboring row does not live on this rank at all.

!!! warning "[COMMON TRAP] Assuming a grid row-strip is independent of its neighbors, the way a data-parallel batch shard is"
    Chapter 12's batch shards are deliberately independent: rank 0's samples and rank 1's samples never interact with each other except at a periodic all-reduce boundary, and between those boundaries each rank runs entirely on its own. It's tempting to assume domain decomposition works the same way, since the partitioning arithmetic above looks identical. It doesn't. A grid row-strip's *interior* rows are independent within a single iteration, but its *boundary* rows -- the top and bottom row of every rank's own band -- depend on a neighboring rank's data on every single iteration, not periodically. Skip the exchange for even one iteration and every rank's boundary rows silently compute from stale data forever after; there is no equivalent of Chapter 12's "average the gradients every N steps" flexibility here. Section 16.2 builds the transfer that has to run, without exception, before every stencil update.

## 16.2 Halo Exchange: A Genuinely New Communication Pattern

### Intuition

Every collective this book has built so far has one of three shapes: root-based (Chapter 8's broadcast and reduce -- one device is privileged, every other device talks to it), ring-based (Chapter 9's all-reduce -- every device talks to exactly two neighbors, but the ring wraps around, so those two neighbors eventually reach every device transitively), or all-to-all (Chapter 10 -- every device talks to every other device directly). Halo exchange is none of these. Each rank talks to at most two other ranks -- the one directly above it and the one directly below it in the row-strip decomposition -- and that's the entire communication graph. There's no privileged root, no wraparound, and no need to reach a device three strips away, because a rank's stencil update never needs anything beyond its immediate neighbor's boundary row. The mechanism itself isn't new -- it's the same real `cudaMemcpyPeer()` Chapter 5 introduced and Chapter 13 reused to hand activations across a layer boundary -- but the pattern it's used in is: small, frequent, strictly nearest-neighbor, and repeated every iteration for as long as the simulation runs.

```text
Halo exchange (this chapter):               Ring all-reduce (Chapter 9):

  dev0 <-> dev1 <-> dev2 <-> dev3             dev0 -> dev1 -> dev2 -> dev3
   ^                          ^                ^                        |
   domain boundary            domain            +------------------------+
   (fixed value,               boundary          (wraps back around to dev0)
   no transfer)                (fixed value,
                                no transfer)     Every device has exactly two
                                                  ring-neighbors, and the ring
  Only real, adjacent neighbors exchange.        wraps -- appropriate for
  The two ends of the chain are edges            summing a value that every
  of PHYSICAL SPACE, not neighbors of             device needs; wrong for a
  each other -- wrapping them would               physical domain's own edges,
  connect the top of the domain to                which are not adjacent to
  the bottom, which isn't how the                 each other in real space.
  space this grid represents works.
```

### Background

Building the exchange itself means guarding a real `cudaMemcpyPeer()` call with the same above/below classification Section 16.1 computed: a rank sends its topmost interior row to the rank above it (filling that rank's bottom halo) only if a rank above genuinely exists, and its bottommost interior row to the rank below it (filling that rank's top halo) only if a rank below genuinely exists. Get either guard wrong and the call either targets a device that was never meant to receive anything, or silently skips an exchange a real neighbor was depending on.

```cpp
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
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

Halo exchange for a 4-way row-strip decomposition. Each rank exchanges exactly one row (64 bytes) with each real neighbor it has -- never with a rank two or more strips away, and never wrapping around the way Chapter 9's ring does (rank 0 and rank 3 are the physical TOP and BOTTOM edges of the domain, not neighbors of each other).

rank 0:
  top edge is the domain boundary -- fixed value, no transfer
  send bottom row -> rank 1 (fills rank 1's top halo):    no CUDA-capable device is detected (code 100)
rank 1:
  send top row    -> rank 0 (fills rank 0's bottom halo): no CUDA-capable device is detected (code 100)
  send bottom row -> rank 2 (fills rank 2's top halo):    no CUDA-capable device is detected (code 100)
rank 2:
  send top row    -> rank 1 (fills rank 1's bottom halo): no CUDA-capable device is detected (code 100)
  send bottom row -> rank 3 (fills rank 3's top halo):    no CUDA-capable device is detected (code 100)
rank 3:
  send top row    -> rank 2 (fills rank 2's bottom halo): no CUDA-capable device is detected (code 100)
  bottom edge is the domain boundary -- fixed value, no transfer

Every cudaMemcpyPeer() call above reports the same honest cudaErrorNoDevice this book's every real device-touching call has reported since Chapter 3 -- these are real transfer attempts, on a real API, guarded by real boundary checks, not illustrative pseudocode. Section 16.3 checks what these exchanges are actually FOR: whether a domain-decomposed stencil update, refreshed by exchanges like these every iteration, computes the exact same thing a single, unsplit domain would.
```

Every rank in the middle of the decomposition makes exactly two calls per iteration -- one up, one send down -- and every rank at either edge of the physical domain makes exactly one, skipping the side that borders a fixed boundary condition rather than a device. Six real `cudaMemcpyPeer()` calls happen across the whole 4-rank decomposition per iteration (rank 0 sends down, rank 1 sends up and down, rank 2 sends up and down, rank 3 sends up), each honestly reporting the same `cudaErrorNoDevice` (code 100) this book has reported since Chapter 3. That count -- not `WORLD_SIZE * (WORLD_SIZE - 1)`, which is what an all-to-all pattern like Chapter 10's would need, and not a fixed number independent of `WORLD_SIZE`, which is what a root-based broadcast would need -- is itself evidence that this pattern is genuinely different: it grows with the number of *shared boundaries* (`WORLD_SIZE - 1` of them, each exchanged in both directions), not with the total number of device pairs.

!!! warning "[COMMON TRAP] Treating halo exchange like Chapter 9's ring, wraparound included"
    Chapter 9's ring all-reduce deliberately wraps around: device `N-1` exchanges with device `0` as if they were adjacent, because for summing a value that every device needs, it doesn't matter which device is "first" or "last" -- the ring is a communication topology chosen for the algorithm, not a reflection of physical adjacency. A row-strip domain decomposition's two end ranks are not adjacent in that sense; they are the literal top and bottom edges of a physical domain, and connecting them the way a ring would means telling the simulation that the space wraps around -- which is only correct if the problem being modeled is genuinely periodic (an actual torus-shaped domain, for instance). For the plate-with-a-hot-edge problem this chapter builds, wrapping rank 0's halo to rank 3 would silently feed the plate's hot top edge into its cold bottom edge's neighbor calculation, corrupting the result while producing no error at all -- exactly the kind of silent, plausible-looking wrong answer this book has tried to avoid throughout.

## 16.3 Correctness and Cost: Does the Split Domain Match, and Why Bother Splitting At All

### Intuition

Two questions are worth checking about this chapter's approach, and they're different kinds of questions. The first is correctness: does a grid split into row-strips, refreshed by halo exchange every iteration, converge to the exact same values a single, unsplit grid running the identical Jacobi iteration would? The second is cost: given that halo exchange requires real, repeated communication, why is it worth doing at all instead of just giving every device a full copy of the whole grid and letting each one update it independently? This section answers both, and the first answer depends on a detail easy to get wrong: Jacobi iteration's own definition. A Jacobi update computes every new grid point from the *previous* iteration's values only -- it never uses a value already updated during the current sweep. That's what makes it safe to split across devices in the first place: if a rank used a neighbor's *already-updated* value instead of its previous one, the result would depend on exactly which order ranks happened to finish in, which is precisely the kind of nondeterminism this book has spent sixteen chapters guarding against.

```text
Jacobi (this chapter, order-independent):     Gauss-Seidel (NOT used here):

  read ALL of old[] --------> write new[]       read/write the SAME array in
  (old[] is never touched                       place, in some fixed sweep
   again until the whole                        order -- later cells see
   sweep finishes; then                         EARLIER cells' already-
   old and new swap)                            updated values from this
                                                 same pass, not last pass's

  Every cell's update depends ONLY on           Depends on which order cells
  last iteration's committed values --          are visited in -- correct,
  safe to compute in any order, on any           but NOT safe to split across
  device, split any number of ways.              ranks the same way.
```

### Background

Part A below runs the exact same five-point stencil formula, through one shared function, on two different layouts: a single unsplit `(GRID_ROWS+2) x (GRID_COLS+2)` array with fixed boundary rows and columns, and four per-rank local buffers whose halo rows are refreshed every iteration by an array copy standing in for Section 16.2's real `cudaMemcpyPeer()` -- this book's own established host-side simulation technique, used since Chapter 4, for verifying a multi-device communication pattern without a second real device. Part B then steps back and asks a cost question with a closed-form count, the same technique Chapter 9's 9.3 and Chapter 11's 11.3 used: how much data actually has to move under halo exchange, compared to the alternative of just replicating the entire grid to every device every iteration.

```cpp
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

    // Fixed boundary conditions set once. Order matters at the four
    // corner cells, where a row boundary and a column boundary
    // overlap: the reference above sets row boundaries first, then
    // column boundaries, so column boundary values win at the
    // corners. Matching that exact order here is required for Part
    // A's exact `==` check to pass -- setting them in the opposite
    // order is a real, easy-to-make bug (it was the first version of
    // this file's own bug: corners silently ended up on the wrong
    // boundary value).
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
```

Genuinely compiled with `g++ -ffp-contract=off` (see Chapter 14's toolchain finding) and genuinely run, re-verified identical on both this book's real toolchains. Locked output, deterministic across repeated runs:

```
Part A: domain-decomposed vs. unsplit Jacobi stencil, 30 iterations, 16 x 8 interior grid, 4-way row-strip decomposition (4 rows/rank).

Sample interior cell values (row, col are 1-indexed into the 16 x 8 interior):
  (row  1, col 4): unsplit = 75.112944216675984   decomposed = 75.112944216675984   bit-identical
  (row  4, col 4): unsplit = 23.531589206336115   decomposed = 23.531589206336115   bit-identical
  (row  8, col 4): unsplit = 2.6290237654413451   decomposed = 2.6290237654413451   bit-identical
  (row 12, col 4): unsplit = 0.121397771948485   decomposed = 0.121397771948485   bit-identical
  (row 16, col 4): unsplit = 0.0017304024518540947   decomposed = 0.0017304024518540947   bit-identical

0 of 180 total grid cells (including fixed boundary cells) mismatched between the unsplit reference and the 4-way domain-decomposed run: PASS (bit-identical)

This is Chapter 13's own structural argument, showing up again in a completely different setting: a stencil update reads a FIXED set of neighbor values and applies the SAME arithmetic regardless of which device happens to store each operand. Splitting the grid into row-strips only relocates WHERE each cell's data lives; halo exchange delivers the exact same bit pattern a neighboring cell would have held in an unsplit array. No operation's order changes, and -- unlike Chapter 14's row-parallel combine, which really does sum partial results in a different order -- no reduction happens here at all, so exact `==` is a valid check, not merely a convenient one.

Part B: data moved per iteration -- halo exchange vs. giving every rank the whole domain every iteration.

  this chapter's own grid: 16 x 8 interior grid, 4-way decomposition
    halo exchange (Section 16.2):        48 elements/iteration
    full-domain replication every iter:  512 elements/iteration
    halo exchange moves 10.7x less data

  a more realistic grid: 1024 x 1024 interior grid, 8-way decomposition
    halo exchange (Section 16.2):        14336 elements/iteration
    full-domain replication every iter:  8388608 elements/iteration
    halo exchange moves 585.1x less data

Halo-exchange traffic scales with each rank's BOUNDARY -- its perimeter, GRID_COLS elements per shared edge -- while full-domain replication scales with the entire domain's VOLUME, every rank re-receiving every cell every iteration. This is exactly the "surface/volume behavior" William Gropp's own halo-exchange lecture names as the reason domain decomposition communicates only what changed at a boundary, not the whole state.
```

Every one of the 180 cells in the reassembled grid -- interior points and fixed boundary cells alike -- matches the unsplit reference exactly, to the last bit, after 30 real Jacobi iterations. That result didn't come free: getting it required setting the four corner cells' boundary conditions in the *same order* in both code paths, since a cell where a row boundary and a column boundary overlap depends on which one was written last. That's not a floating-point subtlety like Chapter 14's FMA-contraction finding -- it's a plain logic-ordering bug, and it's exactly the kind of thing domain decomposition code has to get right at every one of a real grid's edges and corners, not just its interior. Part B's count answers the cost question directly: for this chapter's own small grid, halo exchange moves roughly 11 times less data per iteration than replicating the whole domain to every device would; at a more realistic 1024x1024 grid split 8 ways, that gap widens to nearly 600 times less. The reason is structural, not incidental -- a rank's halo is proportional to the *length of its shared boundary* with a neighbor, while full replication is proportional to the *entire domain's area*. Every additional row a grid has costs a halo exchange nothing at all, as long as the number of columns stays fixed; it costs full replication one more full row, on every device, every iteration.

!!! warning "[COMMON TRAP] Updating grid cells in place instead of double-buffering"
    It's tempting, especially when translating this chapter's arithmetic into GPU kernel code, to update a grid's array in place -- write each new value directly over the old one as soon as it's computed, saving the second buffer `nextLocal` needed above. Do that and the algorithm silently stops being Jacobi iteration and becomes something closer to Gauss-Seidel: a cell computed later in the sweep sees its earlier neighbors' *already-updated* values instead of the previous iteration's committed values, and the result now depends on the exact order cells happen to be visited in. That's not simply a different, equally valid answer -- it breaks the entire premise Section 16.1 through this section have relied on, that a rank's update depends only on data that was fixed and available *before* this iteration began. Double-buffering (`grid` and `next`, swapped only after every cell in the sweep has been computed from `grid`) is what keeps every rank's computation independent of every other rank's internal ordering, and it's why this chapter's exact `==` check in Part A is even meaningful to run.

## Chapter Summary

This chapter split something no earlier chapter in Part 3 did: not a batch, a set of layers, a matrix, or a pipeline stage, but a literal physical grid, the discretized space a scientific simulation runs over. Section 16.1 partitioned that grid into row-strips using the same host-arithmetic pattern Chapter 12's `computeShard()` and Chapter 13's `computeLayerRange()` established, then introduced the one thing that makes a grid row-strip different from a data-parallel batch shard: every rank needs real data from its immediate neighbors, every single iteration, not just at periodic synchronization points. Section 16.2 built the transfer that supplies it -- halo exchange -- a genuinely new communication pattern for this book: nearest-neighbor only, no privileged root, no ring wraparound, built from the same real `cudaMemcpyPeer()` Chapter 5 introduced and Chapter 13 reused. Section 16.3 verified two separate claims: that a domain-decomposed Jacobi stencil, refreshed by halo exchange every iteration, produces a result bit-identical to an unsplit domain's -- true here for the same structural reason Chapter 13's split forward pass was exact, provided Jacobi's own double-buffering discipline is followed -- and that halo exchange moves dramatically less data than full-domain replication would, because its traffic scales with each rank's *boundary*, not the domain's entire *volume*. This closes Part 3's parallelization-strategies arc: Chapters 12 through 15 split a neural network's data, weights, matrices, and layers across devices; this chapter split physical space itself, and Part 6's Chapter 29 returns to build a complete distributed Jacobi solver on exactly this foundation.

## Self-Check Questions

1. Section 16.1's `computeRowRange()` is structurally identical to two earlier functions in this book. Name them, and explain what changed between all three.
2. Why does this chapter claim a grid row-strip is *not* independent of its neighbors the way Chapter 12's data shard is? Be specific about when the dependency arises.
3. Halo exchange is deliberately contrasted with Chapter 9's ring all-reduce in Section 16.2. What is the one structural difference between them, and why would using the ring's wraparound behavior for this chapter's plate problem produce a wrong answer with no error at all?
4. Section 16.3's Part A checks its result with exact `==`. Using Chapter 13's own structural argument, explain why that check is valid here, and name the one earlier chapter whose reduce operation could NOT make the same claim.
5. The COMMON TRAP in Section 16.3 distinguishes Jacobi iteration from Gauss-Seidel. In your own words, what would break about this chapter's correctness check if the decomposed code updated its local buffers in place instead of double-buffering?
6. Using Part B's own formulas, compute the halo-exchange and full-replication element counts per iteration for an 8x8 interior grid split 2 ways, and state the ratio.
7. Why does Section 16.3's code set row boundaries before column boundaries in both the reference and the decomposed path? What would happen to a corner cell if that order were reversed in only one of the two paths?
8. Section 16.1's printed table shows rank 1 and rank 2 each with two real halo neighbors, but rank 0 and rank 3 each with only one. Explain why, referring to the row-strip decomposition's shape.
9. This chapter's own code comment admits Part A's decomposed corner-boundary logic had a real bug during this chapter's own development. What was it, and how was it found?

## Where We Go Next

This completes Part 3 -- Chapters 12 through 16 have now covered every major way of splitting work across GPUs this book set out to build: replicated-model/sharded-data (Chapter 12), sharded-model/shared-data (Chapter 13), a single split matrix multiply (Chapter 14), pipelined layers (Chapter 15), and now a split physical domain (Chapter 16). Part 4 turns to a different set of concerns that apply across all of these strategies at once: keeping devices synchronized when their workloads aren't perfectly balanced, and handling the failures and stragglers that a real cluster running any of these patterns will eventually hit. Part 6's Chapter 29, "A Distributed Jacobi Solver: Multi-GPU Scientific Computing With Halo Exchange," returns to this exact chapter's foundation and builds it out into a complete, converging solver.

## Worked Solutions

**1.** `computeRowRange()` is structurally identical to Chapter 12's `computeShard()` and Chapter 13's `computeLayerRange()` -- all three divide a known total by a known device count and compute a contiguous `[start, end)` offset with the exact same two lines of arithmetic. What changed across the three is only *what* is being partitioned: sample indices in Chapter 12, layer indices in Chapter 13, and grid row indices here.

**2.** A grid row-strip's dependency on its neighbors is not periodic the way a data-parallel shard's is -- it arises on *every single iteration*. Chapter 12's shards only need to interact at a chosen synchronization point (Chapter 12 used every step, but the design allows less frequent averaging); a row-strip's top and bottom rows need their neighboring rank's boundary data for every stencil update, with no equivalent flexibility, because skipping even one exchange leaves those rows computing from stale data from that point forward.

**3.** Chapter 9's ring wraps device `N-1` back to device `0`, which is correct for a reduction because a ring is just a chosen communication topology -- it doesn't matter which device is "first." Halo exchange's two end ranks are not topologically arbitrary; they are the literal top and bottom edges of a physical domain. Wrapping them together would silently feed one physical edge's boundary condition into the other edge's neighbor calculation -- for this chapter's hot-top/cold-other-three-edges plate, that means the hot boundary would leak into the bottom-edge rank's stencil update, producing a plausible-looking but physically wrong steady-state temperature field, with no crash or error to flag it.

**4.** Section 16.3's stencil update reads a fixed set of neighbor values and applies the identical `stencilUpdate()` function regardless of which device stores which operand -- splitting the grid only relocates *where* each cell's data lives, it never changes the *order* any arithmetic happens in, exactly Chapter 13's own argument for why its split forward pass could be checked with exact `==`. Chapter 8's reduce could not make that claim, because summing values from different devices in a different grouping order is not guaranteed by IEEE 754 to produce an identical result to summing them in another order.

**5.** Updating in place would mean a cell computed later in a sweep could read a neighboring cell's *already-updated* value from the current iteration instead of the previous iteration's committed value -- turning the update into something order-dependent, like Gauss-Seidel. Since the decomposed path and the reference path would then potentially visit cells in different effective orders (four independent per-rank sweeps vs. one single sweep), their results could differ, and the exact `==` check would no longer be checking two implementations of the *same* algorithm.

**6.** For an 8x8 interior grid (`GRID_ROWS=8`, `GRID_COLS=8`) split 2 ways: halo exchange = `2 * (2-1) * 8 = 16` elements/iteration; full replication = `2 * 8 * 8 = 128` elements/iteration. Ratio: `128 / 16 = 8x` less data moved by halo exchange.

**7.** Setting row boundaries first means the column-boundary loop, which runs second, overwrites whatever a corner cell received from the row-boundary loop -- so every corner cell ends up holding its column's boundary value. If one path (say, the decomposed one) reversed that order while the other (the reference) kept it, the two paths' corner cells would hold different fixed values, producing a mismatch at exactly those cells and nowhere else -- which is precisely the two-cell mismatch this chapter's own code comment describes finding during development.

**8.** Rank 1 and rank 2 sit strictly between two other ranks in the row-strip decomposition -- rank 1 borders rank 0 above and rank 2 below; rank 2 borders rank 1 above and rank 3 below -- so both of their sides are real device neighbors. Rank 0 and rank 3 sit at the two ends of the decomposition, so one of their sides borders another rank while the other side borders the domain's own physical edge, which needs a fixed boundary value instead of a halo.

**9.** The bug was an ordering mismatch between the reference and decomposed boundary-condition setup: the reference set row boundaries first and column boundaries second (so column values won at the four corners), while an earlier version of the decomposed code set them in the opposite order for rank 0 and the last rank, leaving those two ranks' corner cells holding the row's boundary value instead. It was found by running Part A's own mismatch count, which reported exactly 2 of 180 cells differing -- a small, specific number that pointed directly at the two affected corners rather than a broader structural error, and was fixed by making both code paths set row boundaries before column boundaries.

---

**Sources cited in this chapter:**

- Wence, "MPI: domain decomposition and halo exchanges," [PHYS52015 -- Introduction to HPC](https://teaching.wence.uk/phys52015/exercises/mpi-stencil/) -- the exact quotes defining domain decomposition ("divide up the whole domain... approximately equally among all processes"), halo/ghost regions ("extending the extent of the image by the stencil width... creating halo or ghost regions"), the five-point finite-difference stencil formula, and the Jacobi iteration update rule this chapter's `stencilUpdate()` implements.
- [S8314: Multi-GPU Programming with MPI](https://indico.itp.ac.cn/event/25/attachments/42/62/S8314_Multi_GPU_Programming_with_MPI.pdf), Jiri Kraus, NVIDIA (GTC) -- real 2D domain decomposition terminology ("2D domain decomposition with n x k domains"), the `MPI_Sendrecv`-based top/bottom halo exchange pattern this chapter's `cudaMemcpyPeer()` guards mirror, and the manual `cudaMemcpy` host-staging pattern for exchanging boundary data without CUDA-aware MPI.
- William Gropp, ["Lecture 25: Strategies for Parallelism and Halo Exchange"](https://wgropp.cs.illinois.edu/courses/cs598-s15/lectures/lecture25.pdf), UIUC CS598 -- the exact quotes "provide access to remote data through a halo exchange," "decompose mesh into equal sized (work) pieces," and Jacobi iteration's own "surface/volume behavior," cited for Section 16.3's communication-volume argument.
- This book's own Chapter 5 (`cudaMemcpyPeer()`), Chapter 9 (ring all-reduce, contrasted with halo exchange's nearest-neighbor pattern), Chapter 12 (`computeShard()`), Chapter 13 (`computeLayerRange()`, and the exact-`==` correctness argument reused in Section 16.3), and Chapter 14 (the `-ffp-contract=off` toolchain practice reused for this chapter's own `.cpp` file).
