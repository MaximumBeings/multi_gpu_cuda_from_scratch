**What you will understand after reading this chapter:** why solving a linear system with the Jacobi iterative method, distributed across many devices, needs almost exactly the same halo exchange this book built in Chapter 16 for a plain stencil update -- and exactly one genuinely new ingredient on top of it: a global residual-norm reduction to decide when the solver is done. You will see that new ingredient built, verified, and cost-modeled, including a real floating-point subtlety it introduces that the halo exchange itself does not have.

**What you need to know first:** Chapter 16's row-strip domain decomposition and fixed top/bottom halo exchange; Chapter 8's caution about floating-point reduction order; Chapter 9's ring collective round-count model; Chapter 11's ring-vs-tree contrast; Chapter 25's own real one-bit floating-point mismatch inside a collective.

---

Chapter 16 built a distributed 5-point stencil update: a row-strip decomposition across P ranks, each one talking only to its fixed top and bottom neighbors, applying the same local update formula every step, for a fixed number of steps. That chapter's own update never needed to ask "are we done yet?" -- the number of steps was decided in advance.

The Jacobi *iterative method* for solving a linear system Ax = b (Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003, Chapter 4, "Basic Iterative Methods") looks, cell by cell, almost identical to that same stencil update when the system comes from a discretized 2D Poisson equation: each unknown's new value is a weighted average of its neighbors' old values. NVIDIA's own real multi-GPU example library states this directly about its own reference implementation: "This project implements the well known multi GPU Jacobi solver with different multi GPU Programming Models" (NVIDIA, `multi-gpu-programming-models`, `mpi/jacobi.cpp`). But an iterative *solver*, unlike a fixed-step stencil sweep, needs to know when to stop -- and that decision needs a number every single rank agrees on: the residual norm of the *entire* distributed system, not just one rank's own local piece. Computing that number needs a collective every rank participates in, on every rank's own data, added together globally. This chapter builds that solver, shows the update step reusing Chapter 16 unchanged, and then isolates exactly what the new global-agreement requirement costs and where it can go quietly wrong.

```text
Chapter 16's stencil sweep:              This chapter's Jacobi solver:

  rank 0 ---halo--- rank 1                rank 0 ---halo--- rank 1
     |                  |                    |                  |
     +------------------+                    +------------------+
     |                  |                    |                  |
  update             update                update             update
     |                  |                    |                  |
     (repeat FIXED       |                    +------ AND -------+
      number of times)   |                    |    global residual norm
                                               |    (ALL ranks, ALL data)
                                               |
                                        converged? ---no---> loop again
                                               |
                                              yes
                                               |
                                             done
```

## 29.1 The Update Step: Chapter 16's Halo Exchange, Unchanged

### Intuition

Picture a distributed 2D grid again, split into horizontal strips, one strip per rank -- exactly Chapter 16's picture. Each interior cell's new value is the average of its four neighbors' old values plus a small correction term. A cell near the top or bottom edge of a rank's own strip needs a neighbor's row that rank doesn't own; that row arrives through the same fixed top/bottom halo exchange Chapter 16 already built. Nothing about *this* part of the problem cares whether the underlying task is "smooth this stencil" or "solve this linear system" -- the communication shape is identical, because the update formula touching only immediate neighbors is identical.

### Background

NVIDIA's own real MPI Jacobi solver (`multi-gpu-programming-models/mpi/jacobi.cpp`) computes the same fixed neighbor pair Chapter 16 used:

```
const int top = rank > 0 ? rank - 1 : (size - 1);
const int bottom = (rank + 1) % size;
MPI_CALL(MPI_Sendrecv(a_new + iy_start * nx, nx, MPI_REAL_TYPE, top, 0,
                      a_new + (iy_end * nx), nx, MPI_REAL_TYPE, bottom, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE));
MPI_CALL(MPI_Sendrecv(a_new + (iy_end - 1) * nx, nx, MPI_REAL_TYPE, bottom, 0,
                      a_new, nx, MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE));
```

This file's own host-side simulation builds the same shape: P ranks, each owning a contiguous row-strip of a discretized 2D Poisson grid (interior size NY x NX, constant forcing term, zero Dirichlet boundary), exchanging only top/bottom halo rows every iteration, for a fixed 40 iterations. It then checks the assembled distributed grid against a single-process reference **exactly**, cell by cell -- because, exactly as in Chapter 16, each cell's own update touches the same four old neighbor values in the same fixed order regardless of how the grid happens to be partitioned. It goes one step further than Chapter 16 did: it also computes the *global residual norm* two different ways -- once as a single flat pass over the whole grid (the reference), and once as P separate per-rank partial sums that are only combined at the very end (modeling `MPI_Allreduce(..., MPI_SUM, ...)`, the real call in NVIDIA's own `jacobi.cpp`) -- to check whether that reduction is *also* exact, the way the grid update itself is.

```cpp
// Chapter 29: Distributed Jacobi Solver
// 84_distributed_jacobi_correctness_simulation.cpp
//
// Chapter 16's own row-strip domain decomposition and fixed top/bottom-only
// halo exchange (matching NVIDIA's own real multi-GPU Jacobi example,
// multi-gpu-programming-models/mpi/jacobi.cpp: "const int top = rank > 0 ?
// rank - 1 : (size - 1); const int bottom = (rank + 1) % size;") apply
// almost unchanged to the classic Jacobi ITERATIVE METHOD for solving a
// linear system (Saad, "Iterative Methods for Sparse Linear Systems", 2nd
// ed., SIAM 2003, Ch.4 "Basic Iterative Methods") on a discretized 2D
// Poisson equation. This file builds that distributed Jacobi solver as a
// host-side simulation -- P ranks, each owning a contiguous row-strip of a
// (NY x NX) interior grid, exchanging only top/bottom halo rows every
// iteration -- and verifies the assembled grid matches a single-process
// reference EXACTLY (bit-for-bit), across several different values of P,
// because each interior cell's own update touches the same four
// neighbors in the same fixed order regardless of how the grid is
// partitioned. It then also computes the GLOBAL residual norm two ways:
// once as a single flat pass over the whole grid (the reference), and once
// as P separate per-rank partial sums later combined (modeling
// MPI_Allreduce(..., MPI_SUM, ...), the real call in NVIDIA's own
// jacobi.cpp) -- to check whether that reduction is ALSO bit-exact, or
// whether it runs into the same floating-point non-associativity this
// book's own Chapter 8 and Chapter 25 already found for other collectives.
#include <cstdio>
#include <vector>
#include <cmath>

const int NY = 12;   // decomposed dimension (rows)
const int NX = 8;    // non-decomposed dimension (columns)
const double H = 1.0;
const int ITERS = 40;

inline double forcing(int, int) { return 1.0; } // constant source term f(i,j) = 1

inline int idx(int i, int j, int stride) { return i * stride + j; }

// --- Reference: single-process Jacobi sweep over the FULL (NY+2) x (NX+2) grid. ---
std::vector<double> referenceJacobi(int iters) {
    int stride = NX + 2;
    std::vector<double> u((NY + 2) * stride, 0.0);
    std::vector<double> unew(u.size(), 0.0);
    for (int it = 0; it < iters; it++) {
        for (int i = 1; i <= NY; i++) {
            for (int j = 1; j <= NX; j++) {
                double up = u[idx(i - 1, j, stride)];
                double down = u[idx(i + 1, j, stride)];
                double left = u[idx(i, j - 1, stride)];
                double right = u[idx(i, j + 1, stride)];
                unew[idx(i, j, stride)] = 0.25 * (up + down + left + right + H * H * forcing(i, j));
            }
        }
        std::swap(u, unew);
    }
    return u;
}

double referenceResidualNorm(const std::vector<double> &u) {
    int stride = NX + 2;
    double sumsq = 0.0;
    for (int i = 1; i <= NY; i++) {
        for (int j = 1; j <= NX; j++) {
            double up = u[idx(i - 1, j, stride)];
            double down = u[idx(i + 1, j, stride)];
            double left = u[idx(i, j - 1, stride)];
            double right = u[idx(i, j + 1, stride)];
            double r = 4.0 * u[idx(i, j, stride)] - (up + down + left + right) - H * H * forcing(i, j);
            sumsq += r * r;
        }
    }
    return std::sqrt(sumsq);
}

// --- Distributed: P ranks, 1D row-strip decomposition (Chapter 16's own
// scheme), halo exchange with fixed top/bottom neighbors only. ---
struct DistributedResult {
    std::vector<double> assembled;
    double globalResidualNorm;
};

DistributedResult distributedJacobi(int P, int iters) {
    int localRows = NY / P;
    int stride = NX + 2;
    std::vector<std::vector<double>> u(P), unew(P);
    for (int r = 0; r < P; r++) {
        u[r].assign((localRows + 2) * stride, 0.0);
        unew[r].assign((localRows + 2) * stride, 0.0);
    }

    auto exchangeHalos = [&]() {
        for (int r = 0; r < P; r++) {
            for (int j = 0; j < stride; j++) {
                double topHalo = (r > 0) ? u[r - 1][idx(localRows, j, stride)] : 0.0;
                double bottomHalo = (r < P - 1) ? u[r + 1][idx(1, j, stride)] : 0.0;
                u[r][idx(0, j, stride)] = topHalo;
                u[r][idx(localRows + 1, j, stride)] = bottomHalo;
            }
        }
    };

    for (int it = 0; it < iters; it++) {
        exchangeHalos();
        for (int r = 0; r < P; r++) {
            int globalRowBase = r * localRows;
            for (int li = 1; li <= localRows; li++) {
                int gi = globalRowBase + li;
                for (int j = 1; j <= NX; j++) {
                    double up = u[r][idx(li - 1, j, stride)];
                    double down = u[r][idx(li + 1, j, stride)];
                    double left = u[r][idx(li, j - 1, stride)];
                    double right = u[r][idx(li, j + 1, stride)];
                    unew[r][idx(li, j, stride)] = 0.25 * (up + down + left + right + H * H * forcing(gi, j));
                }
            }
        }
        for (int r = 0; r < P; r++) std::swap(u[r], unew[r]);
    }

    // One final halo exchange so every rank's ghost rows hold the truly
    // latest neighbor values before the residual is computed -- otherwise
    // the residual would be computed against one-iteration-stale boundary
    // data, a real correctness detail easy to miss.
    exchangeHalos();

    std::vector<double> assembled((NY + 2) * stride, 0.0);
    for (int r = 0; r < P; r++) {
        int globalRowBase = r * localRows;
        for (int li = 1; li <= localRows; li++) {
            int gi = globalRowBase + li;
            for (int j = 0; j < stride; j++) {
                assembled[idx(gi, j, stride)] = u[r][idx(li, j, stride)];
            }
        }
    }

    // Global residual norm: each rank sums the squared residual over ONLY
    // its own owned rows (a local partial sum), then the P partial sums are
    // combined -- modeling MPI_Allreduce(..., MPI_SUM, ...), the real call
    // in NVIDIA's own jacobi.cpp.
    std::vector<double> partialSumSq(P, 0.0);
    for (int r = 0; r < P; r++) {
        int globalRowBase = r * localRows;
        double s = 0.0;
        for (int li = 1; li <= localRows; li++) {
            int gi = globalRowBase + li;
            for (int j = 1; j <= NX; j++) {
                double up = u[r][idx(li - 1, j, stride)];
                double down = u[r][idx(li + 1, j, stride)];
                double left = u[r][idx(li, j - 1, stride)];
                double right = u[r][idx(li, j + 1, stride)];
                double resid = 4.0 * u[r][idx(li, j, stride)] - (up + down + left + right) - H * H * forcing(gi, j);
                s += resid * resid;
            }
        }
        partialSumSq[r] = s;
    }
    double globalSumSq = 0.0;
    for (int r = 0; r < P; r++) globalSumSq += partialSumSq[r];

    DistributedResult res;
    res.assembled = assembled;
    res.globalResidualNorm = std::sqrt(globalSumSq);
    return res;
}

int main() {
    std::vector<double> ref = referenceJacobi(ITERS);
    double refNorm = referenceResidualNorm(ref);

    printf("Reference (single-process) %dx%d-grid Jacobi solve, %d iterations.\n"
           "Global residual norm (full-grid single-pass sum): %.17g\n\n",
           NY, NX, ITERS, refNorm);

    int Ps[] = {1, 2, 3, 4, 6, 12};
    printf("%-6s %-20s %-28s %-10s\n", "P", "grid EXACT match?",
           "residual norm (per-rank sums)", "norm match?");
    for (int P : Ps) {
        DistributedResult dr = distributedJacobi(P, ITERS);
        bool exact = (dr.assembled == ref);
        bool normExact = (dr.globalResidualNorm == refNorm);
        printf("%-6d %-20s %-28.17g %-10s\n", P, exact ? "YES" : "NO",
               dr.globalResidualNorm, normExact ? "YES" : "NO");
    }

    return 0;
}
```

Compile and run:

```
g++ -O2 -ffp-contract=off 84_distributed_jacobi_correctness_simulation.cpp -o 84_distributed_jacobi_correctness_simulation
./84_distributed_jacobi_correctness_simulation
```

```text
Reference (single-process) 12x8-grid Jacobi solve, 40 iterations.
Global residual norm (full-grid single-pass sum): 1.3874786747088721

P      grid EXACT match?    residual norm (per-rank sums) norm match?
1      YES                  1.3874786747088721           YES       
2      YES                  1.3874786747088719           NO        
3      YES                  1.3874786747088719           NO        
4      YES                  1.3874786747088719           NO        
6      YES                  1.3874786747088719           NO        
12     YES                  1.3874786747088719           NO        
```

The grid itself matches bit-for-bit at every P, exactly as Chapter 16 predicted: the update step's communication pattern and per-cell arithmetic are unchanged by partitioning. But the residual norm does **not** match at P > 1 -- it differs in the last printed digit (`...088721` vs `...088719`). This is not a bug in this file; it is Chapter 8's own non-associativity caution and Chapter 25's own real one-bit mismatch, showing up again here for a completely different reason: summing P separate partial sums and then adding those partial sums together is a different sequence of rounding steps than summing the same values in one flat pass, even though the *set* of values and their *order* along the grid are identical. Section 29.2 explains why this is the real, structural difference between the halo exchange (exact regardless of P) and the residual Allreduce (not exact, in general, once P > 1).

!!! warning "[COMMON TRAP] Assuming a distributed reduction is exact just because the underlying update was"
    Chapter 16's halo exchange and this chapter's update step are exact at every P because each cell's own arithmetic touches a *fixed, small set of operands in a fixed order*, and partitioning never changes that order. A global sum-reduction is different: partitioning changes *how the terms are grouped* before they're added, and floating-point addition is not associative -- `(a+b)+(c+d)` and `((a+b)+c)+d` can differ in their last bit even though both equal the same real number mathematically. Never assume a residual norm, loss value, or other reduced scalar will reproduce bit-for-bit across different rank counts; if exact reproducibility across P is a requirement (e.g., for a regression test), the reduction order itself must be fixed, not just the operand set.

```text
Single flat pass (reference):        Per-rank partial sums, then combined:

  (((0+a)+b)+c)+d)+e)+f)+g)+h           ((0+a)+b)+c)+d)  +  ((0+e)+f)+g)+h)
        one long chain                    two short chains, added last
        of roundings                      -- different rounding sequence,
                                           same real-number answer
```

## 29.2 The New Cost: A Global Residual Every Check

### Intuition

The halo exchange has a fixed cost per iteration that never depends on how many ranks there are -- each rank only ever has two neighbors, and Chapter 16 already established that. The residual-norm Allreduce is the opposite: every single rank must participate, and for a naive algorithm the number of communication rounds needed grows with the number of ranks. Checking convergence every single iteration means paying that scaling cost every single iteration; checking it less often means paying it less often, at the cost of running a few extra iterations past the point where the solver actually converged.

### Background

NVIDIA's own real repository exposes exactly this tradeoff as a command-line flag: "`-nccheck`: How often to check for convergence (default 1)", and even ships a dedicated variant described as "Multi Threaded with OpenMP using GPUDirect P2P mappings for inter GPU communication with delayed norm execution" -- built specifically to decouple the norm computation from every iteration. PETSc's own real user manual states plainly why this matters as rank counts grow: "Standard Krylov methods have one or more global reductions resulting from the computations of inner products or norms in each iteration. These reductions need to block until all MPI processes have received the results. For a large number of MPI processes (this number is machine dependent but can be above 10,000 processes) this synchronization is very time consuming and can significantly slow the computation."

This section's own model quantifies that tradeoff using this book's own established tools: Chapter 9's ring round-count formula and Chapter 11's ring-vs-tree contrast, applied here to a single-scalar reduction rather than a large gradient buffer -- so, per Chapter 26's own finding, it is the **round count**, not the payload size, that matters.

```cpp
// Chapter 29: Distributed Jacobi Solver
// 85_allreduce_check_frequency_cost_model.cpp
//
// Section 29.1's own simulation showed the Jacobi UPDATE itself reuses
// Chapter 16's fixed top/bottom halo exchange exactly, unchanged. This
// section quantifies the genuinely NEW cost this chapter introduces: the
// GLOBAL residual-norm Allreduce needed to check convergence. NVIDIA's own
// real multi-GPU Jacobi example (multi-gpu-programming-models,
// mpi/jacobi.cpp) exposes exactly this as a real, named tunable --
// "-nccheck: How often to check for convergence (default 1)" -- and even
// ships a variant described as using "delayed norm execution" specifically
// to decouple the norm check from every iteration. PETSc's own real user
// manual states plainly why this matters at scale: "Standard Krylov
// methods have one or more global reductions resulting from the
// computations of inner products or norms in each iteration. These
// reductions need to block until all MPI processes have received the
// results. For a large number of MPI processes (this number is machine
// dependent but can be above 10,000 processes) this synchronization is
// very time consuming." This section builds a closed-form ROUND-count
// model (never a fabricated timing, matching this book's own standing
// practice) contrasting the halo exchange's fixed, P-independent cost
// against the residual Allreduce's P-dependent round cost, and shows how
// checking every K iterations (NVIDIA's own real "nccheck") divides the
// Allreduce's total overhead by K.
#include <cstdio>
#include <cmath>

int main() {
    printf("--- Per-ITERATION cost: halo exchange (Section 29.1's update step) ---\n");
    printf("Chapter 16's own real finding still holds here unchanged: each rank "
           "talks to exactly 2 fixed neighbors (top, bottom), exchanging NX "
           "doubles per side, REGARDLESS of P. This chapter's own %d-column grid "
           "(Section 29.1) means a fixed 2*8 = 16 doubles per rank per "
           "iteration, whether P is 2 or 4096 -- it never depends on P.\n\n", 8);

    printf("--- Per-CHECK cost: global residual Allreduce (this chapter's new element) ---\n");
    printf("Unlike the halo exchange, an Allreduce's ROUND count (not its tiny "
           "8-byte payload) is what dominates its cost for a single scalar, "
           "per this book's own Chapter 26 finding (\"round count, not volume\"). "
           "Two real algorithm choices, per this book's own Chapter 9/11 models:\n");
    printf("%-8s %-28s %-28s\n", "P", "naive ring rounds: 2(P-1)", "tree rounds: 2*ceil(log2(P))");
    const int Ps[] = {4, 16, 64, 256, 1024, 4096};
    for (int P : Ps) {
        int ringRounds = 2 * (P - 1);
        int treeRounds = 2 * (int)std::ceil(std::log2((double)P));
        printf("%-8d %-28d %-28d\n", P, ringRounds, treeRounds);
    }
    printf("\nEven for this chapter's single-scalar residual norm, a naive ring "
           "Allreduce's round count still grows LINEARLY with P (same shape as "
           "Chapter 27's mandatory all-gather), while a tree Allreduce's grows "
           "only logarithmically -- confirming Chapter 11's own real reason "
           "production NCCL prefers tree/double-binary-tree algorithms for "
           "small messages at large P.\n\n");

    printf("--- Amortizing the check: NVIDIA's own real '-nccheck K' ---\n");
    printf("Checking convergence every K iterations (K=1 is NVIDIA's own real "
           "default) turns a FIXED per-check round cost into a total overhead "
           "that divides by K across a full solve of ITERS iterations. Using "
           "this chapter's own tree-round model at a representative P:\n\n");

    const int ITERS = 1000;
    const int Ks[] = {1, 10, 100, 1000};
    const int P_example = 1024;
    int treeRoundsPerCheck = 2 * (int)std::ceil(std::log2((double)P_example));
    printf("Fixed for this table: ITERS = %d total Jacobi iterations, P = %d "
           "ranks, tree Allreduce = %d rounds per check.\n\n", ITERS, P_example,
           treeRoundsPerCheck);
    printf("%-8s %-20s %-28s %-16s\n", "K", "# convergence checks",
           "total Allreduce rounds", "overhead reduction");
    long long baselineRounds = 0;
    for (int K : Ks) {
        long long numChecks = ITERS / K;
        long long totalRounds = numChecks * treeRoundsPerCheck;
        if (K == 1) baselineRounds = totalRounds;
        double reduction = (double)baselineRounds / (double)totalRounds;
        printf("%-8d %-20lld %-28lld %-16.1fx\n", K, numChecks, totalRounds, reduction);
    }
    printf("\nThe reduction factor is exactly K, by construction -- the same "
           "shape as Chapter 25's own bucketing tradeoff, but for a collective's "
           "CALL FREQUENCY rather than its per-call MESSAGE SIZE. This is exactly "
           "why NVIDIA's own real repo ships a 'delayed norm execution' variant: "
           "'GPUDirect P2P mappings for inter GPU communication with delayed "
           "norm execution' -- checking less often is a genuine, real, "
           "production optimization, not a hypothetical one.\n");

    return 0;
}
```

Compile and run:

```
g++ -O2 85_allreduce_check_frequency_cost_model.cpp -o 85_allreduce_check_frequency_cost_model
./85_allreduce_check_frequency_cost_model
```

```text
--- Per-ITERATION cost: halo exchange (Section 29.1's update step) ---
Chapter 16's own real finding still holds here unchanged: each rank talks to exactly 2 fixed neighbors (top, bottom), exchanging NX doubles per side, REGARDLESS of P. This chapter's own 8-column grid (Section 29.1) means a fixed 2*8 = 16 doubles per rank per iteration, whether P is 2 or 4096 -- it never depends on P.

--- Per-CHECK cost: global residual Allreduce (this chapter's new element) ---
Unlike the halo exchange, an Allreduce's ROUND count (not its tiny 8-byte payload) is what dominates its cost for a single scalar, per this book's own Chapter 26 finding ("round count, not volume"). Two real algorithm choices, per this book's own Chapter 9/11 models:
P        naive ring rounds: 2(P-1)    tree rounds: 2*ceil(log2(P))
4        6                            4                           
16       30                           8                           
64       126                          12                          
256      510                          16                          
1024     2046                         20                          
4096     8190                         24                          

Even for this chapter's single-scalar residual norm, a naive ring Allreduce's round count still grows LINEARLY with P (same shape as Chapter 27's mandatory all-gather), while a tree Allreduce's grows only logarithmically -- confirming Chapter 11's own real reason production NCCL prefers tree/double-binary-tree algorithms for small messages at large P.

--- Amortizing the check: NVIDIA's own real '-nccheck K' ---
Checking convergence every K iterations (K=1 is NVIDIA's own real default) turns a FIXED per-check round cost into a total overhead that divides by K across a full solve of ITERS iterations. Using this chapter's own tree-round model at a representative P:

Fixed for this table: ITERS = 1000 total Jacobi iterations, P = 1024 ranks, tree Allreduce = 20 rounds per check.

K        # convergence checks total Allreduce rounds       overhead reduction
1        1000                 20000                        1.0             x
10       100                  2000                         10.0            x
100      10                   200                          100.0           x
1000     1                    20                           1000.0          x

The reduction factor is exactly K, by construction -- the same shape as Chapter 25's own bucketing tradeoff, but for a collective's CALL FREQUENCY rather than its per-call MESSAGE SIZE. This is exactly why NVIDIA's own real repo ships a 'delayed norm execution' variant: 'GPUDirect P2P mappings for inter GPU communication with delayed norm execution' -- checking less often is a genuine, real, production optimization, not a hypothetical one.
```

```text
Halo exchange cost (per iteration):        Allreduce cost (per CHECK):

  fixed, 2 neighbors                         grows with P (ring: linear,
  NEVER depends on P                         tree: log P) -- and this
  --------------------------->               cost is only PAID when a
        (flat line as P grows)               convergence check happens

  Checking every K iterations divides the Allreduce's TOTAL overhead by K:

  K=1:    [chk][chk][chk][chk][chk][chk]...  (expensive: every iteration)
  K=10:   [chk]---------[chk]---------...    (10x fewer collective calls)
```

!!! warning "[COMMON TRAP] Treating a tiny payload as a tiny cost"
    An 8-byte residual scalar looks negligible next to Chapter 25's multi-megabyte gradient buffers, which tempts the conclusion that checking convergence "costs nothing." Chapter 26 already established the real reason this is wrong: for a small message, the collective's cost is dominated by its **round count** (how many communication steps are needed), not by how many bytes travel in each step. A naive ring Allreduce's round count still grows linearly with P regardless of payload size, so checking convergence every iteration at large P can genuinely dominate total solve time even though the number being reduced is a single double.

## 29.3 Why Jacobi Itself Is Rarely the Production Choice

### Intuition

This chapter picked the Jacobi method because its update step is the cleanest possible way to show a halo-exchange-plus-global-reduction pattern with nothing else in the way. That simplicity has a real, well-known cost: Jacobi converges very slowly compared to other methods that solve the exact same problem. The communication *pattern* this chapter built -- neighbor exchange for the update, global reduction for the convergence check -- carries over directly to those faster methods; only the arithmetic inside each step changes.

### Background

James Demmel's own real UC Berkeley CS267 lecture notes give the classic convergence-rate result for Jacobi applied to an n x n grid discretization of the Poisson equation: the number of iterations needed just to halve the error once is "m ~ ((n+1)/pi)^2" -- growing with the *square* of the grid dimension. The same notes give a real serial/parallel complexity table (N = n^2 unknowns) contrasting Jacobi's O(N^2) serial cost against Conjugate Gradient's and SOR's O(N^1.5), and Multigrid's O(N); and state that Gauss-Seidel "converge[s] twice as fast (rho_GaussSeidel(n) = rho_Jacobi(n)^2)" than Jacobi, at the cost of a data dependency within each sweep that breaks the clean once-per-iteration halo exchange this chapter relied on.

```cpp
// Chapter 29: Distributed Jacobi Solver
// 86_jacobi_convergence_rate_model.cpp
//
// Sections 29.1-29.2 built and cost-modeled a real distributed Jacobi
// solver. This section closes with a real, cited reason Jacobi itself is
// almost never the production numerical method of choice, even though its
// COMMUNICATION PATTERN (this chapter's own real subject) carries over
// directly to the smarter methods that replace it. James Demmel's own real
// UC Berkeley CS267 lecture notes give the classic result for an n x n
// grid discretized Poisson problem: Jacobi needs "m ~ ((n+1)/pi)^2"
// iterations just to HALVE the error once, and state a real serial
// complexity table (N = n^2 unknowns) contrasting Jacobi's O(N^2) serial
// cost against Conjugate Gradient's O(N^1.5), SOR's O(N^1.5), and
// Multigrid's O(N) -- and separately that Gauss-Seidel "converge[s] twice
// as fast (rho_GaussSeidel(n) = rho_Jacobi(n)^2)" than Jacobi. This
// section computes Demmel's own real formula at several real grid sizes,
// and closes with NVIDIA's own real AmgX paper (Naumov et al., SIAM J.
// Sci. Comput. 37(5), 2015) -- a real production GPU multigrid library
// that reuses this exact chapter's own halo-exchange-plus-Allreduce
// communication pattern, just applied to a far-faster-converging method.
#include <cstdio>
#include <cmath>

int main() {
    printf("--- Demmel's own real formula: Jacobi iterations to HALVE the error, "
           "n x n grid ---\n");
    printf("m ~ ((n+1)/pi)^2   (CS267 Lecture 24, Demmel)\n\n");
    printf("%-10s %-24s %-28s\n", "n (grid)", "m: iters to halve error",
           "iters for error <= 1e-6 (~20 halvings)");
    const int ns[] = {10, 100, 1000, 10000};
    const double PI = 3.14159265358979323846;
    // log2(1/1e-6) = log2(1e6) ~= 19.93 halvings needed to shrink error by 1e-6
    double halvingsNeeded = std::log2(1.0e6);
    for (int n : ns) {
        double m = ((double)(n + 1) / PI) * ((double)(n + 1) / PI);
        double totalIters = m * halvingsNeeded;
        printf("%-10d %-24.1f %-28.1f\n", n, m, totalIters);
    }
    printf("\nFor n = 10,000 (a 10,000 x 10,000 grid, N = 10^8 unknowns -- a "
           "realistic production PDE mesh size), Jacobi alone would need on "
           "the order of %.0f iterations just to reach a modest 1e-6 relative "
           "error. This is exactly why this chapter's own solver is a "
           "DIDACTIC vehicle for the communication pattern, not a claim that "
           "Jacobi itself is production-grade.\n\n",
           ((double)(10001) / PI) * ((double)(10001) / PI) * halvingsNeeded);

    printf("--- Demmel's own real serial/PRAM complexity table (N = n^2 unknowns) ---\n");
    printf("%-14s %-14s %-18s\n", "Method", "Serial", "PRAM (parallel depth)");
    printf("%-14s %-14s %-18s\n", "Jacobi", "N^2", "N");
    printf("%-14s %-14s %-18s\n", "CG", "N^1.5", "N^0.5 * log N");
    printf("%-14s %-14s %-18s\n", "SOR", "N^1.5", "N^0.5");
    printf("%-14s %-14s %-18s\n", "Multigrid", "N", "(log N)^2");
    printf("\nGauss-Seidel's own real relation to Jacobi (Demmel, CS267): "
           "\"converge[s] twice as fast (rho_GaussSeidel(n) = "
           "rho_Jacobi(n)^2)\" -- a real, easy, one-line improvement this "
           "chapter's own solver deliberately does NOT take, precisely "
           "because Gauss-Seidel's own update makes each cell depend on "
           "NEIGHBOR VALUES ALREADY UPDATED THIS SWEEP, which breaks the "
           "clean halo-exchange-once-per-iteration pattern Section 29.1 "
           "relied on for its bit-exact correctness check.\n\n");

    printf("--- Real production successor: NVIDIA AmgX ---\n");
    printf("Naumov, Arsaev, Castonguay, Cohen, Demouth, Eaton, Layton, "
           "Markovskiy, Reguly, Sakharnykh, Sellappan & Strzodka, \"AmgX: A "
           "Library for GPU Accelerated Algebraic Multigrid and "
           "Preconditioned Iterative Methods,\" SIAM J. Sci. Comput. "
           "37(5):S602-S626, 2015. Real cited result: \"The algebraic "
           "multigrid algorithm implemented in the AmgX library achieves "
           "2-5x speedup on a single GPU against a competitive "
           "implementation on the CPU.\" AmgX's own multi-GPU solves reuse "
           "the exact same two ingredients this chapter built by hand: "
           "neighbor halo exchange for the local sparse matrix-vector "
           "product, and a global Allreduce for the residual norm/inner "
           "products the method's convergence check needs -- just wrapped "
           "around Multigrid instead of Jacobi.\n");

    return 0;
}
```

Compile and run:

```
g++ -O2 86_jacobi_convergence_rate_model.cpp -o 86_jacobi_convergence_rate_model
./86_jacobi_convergence_rate_model
```

```text
--- Demmel's own real formula: Jacobi iterations to HALVE the error, n x n grid ---
m ~ ((n+1)/pi)^2   (CS267 Lecture 24, Demmel)

n (grid)   m: iters to halve error  iters for error <= 1e-6 (~20 halvings)
10         12.3                     244.4                       
100        1033.6                   20600.8                     
1000       101523.9                 2023531.1                   
10000      10134144.9               201989403.8                 

For n = 10,000 (a 10,000 x 10,000 grid, N = 10^8 unknowns -- a realistic production PDE mesh size), Jacobi alone would need on the order of 201989404 iterations just to reach a modest 1e-6 relative error. This is exactly why this chapter's own solver is a DIDACTIC vehicle for the communication pattern, not a claim that Jacobi itself is production-grade.

--- Demmel's own real serial/PRAM complexity table (N = n^2 unknowns) ---
Method         Serial         PRAM (parallel depth)
Jacobi         N^2            N                 
CG             N^1.5          N^0.5 * log N     
SOR            N^1.5          N^0.5             
Multigrid      N              (log N)^2         

Gauss-Seidel's own real relation to Jacobi (Demmel, CS267): "converge[s] twice as fast (rho_GaussSeidel(n) = rho_Jacobi(n)^2)" -- a real, easy, one-line improvement this chapter's own solver deliberately does NOT take, precisely because Gauss-Seidel's own update makes each cell depend on NEIGHBOR VALUES ALREADY UPDATED THIS SWEEP, which breaks the clean halo-exchange-once-per-iteration pattern Section 29.1 relied on for its bit-exact correctness check.

--- Real production successor: NVIDIA AmgX ---
Naumov, Arsaev, Castonguay, Cohen, Demouth, Eaton, Layton, Markovskiy, Reguly, Sakharnykh, Sellappan & Strzodka, "AmgX: A Library for GPU Accelerated Algebraic Multigrid and Preconditioned Iterative Methods," SIAM J. Sci. Comput. 37(5):S602-S626, 2015. Real cited result: "The algebraic multigrid algorithm implemented in the AmgX library achieves 2-5x speedup on a single GPU against a competitive implementation on the CPU." AmgX's own multi-GPU solves reuse the exact same two ingredients this chapter built by hand: neighbor halo exchange for the local sparse matrix-vector product, and a global Allreduce for the residual norm/inner products the method's convergence check needs -- just wrapped around Multigrid instead of Jacobi.
```

```text
Jacobi's own real slowness (Demmel):        What production solvers actually do:

  n=10,000 grid                                same halo exchange
  needs ~2*10^8 iterations                     +
  to reach 1e-6 error                           same Allreduce residual check
  --------------------------->                  +
  O(N^2) serial cost                            a FASTER-CONVERGING method
                                                 (Multigrid: O(N))
                                                 = AmgX and similar libraries
```

!!! warning "[COMMON TRAP] Concluding this chapter's solver is production-ready because its communication pattern is real"
    Every collective call and cost tradeoff in this chapter is real and reused by production libraries. The Jacobi *method* itself is not what those libraries actually run at scale -- Demmel's own real complexity table shows Multigrid needing only O(N) serial work against Jacobi's O(N^2), and NVIDIA's own real AmgX library is built around Multigrid, not Jacobi, for exactly that reason. Learning this chapter's communication pattern transfers directly to reading or building a real production solver; assuming the *numerical method* transfers too does not.

## Chapter Summary

Distributed Jacobi combines two ingredients this book already built separately: Chapter 16's fixed-neighbor halo exchange for the local update step, verified in Section 29.1 to still be bit-exact across every partition tested, and a global residual-norm Allreduce -- the first genuinely new requirement an *iterative solver* adds on top of a fixed-step stencil sweep. Section 29.1 showed that the update step's exactness does not automatically extend to the reduction: summing the residual as P separate partial sums, then combining them, produced a real one-bit mismatch against a single flat pass, for the same non-associativity reason Chapter 8 and Chapter 25 already documented. Section 29.2 quantified the reduction's real cost using Chapter 9's round-count model and Chapter 11's ring-vs-tree contrast, and showed -- using NVIDIA's own real `-nccheck` tunable and PETSc's own real statement about large-P reduction cost -- that checking convergence every K iterations divides the reduction's total overhead by K. Section 29.3 closed with Demmel's own real convergence-rate formula, showing Jacobi's O(N^2) serial complexity makes it impractical at production grid sizes, and pointed to NVIDIA's own real AmgX library as a production successor that reuses this exact communication pattern around a faster-converging method.

## Self-Check Questions

1. Why does Section 29.1's distributed Jacobi *update* step match a single-process reference bit-for-bit at every tested value of P, while the *residual norm* does not?
2. In the final halo exchange added after the last Jacobi update iteration in `84_distributed_jacobi_correctness_simulation.cpp`, what would go wrong with the residual computation if that final exchange were skipped?
3. Why is an Allreduce's round count, rather than its payload size, the dominant cost for a single-scalar residual norm, per Chapter 26's own finding?
4. What does NVIDIA's own real `-nccheck` flag actually control, and what tradeoff does increasing it from 1 to some K > 1 create?
5. Why does checking convergence via a naive ring Allreduce scale worse with P than checking it via a tree Allreduce, even though both reduce the exact same single double?
6. Why does Gauss-Seidel's real convergence-rate advantage over Jacobi (twice as fast, per Demmel) come at the cost of breaking the clean per-iteration halo exchange this chapter relied on?
7. According to Demmel's own real complexity table, how does Jacobi's serial cost compare to Multigrid's as the number of unknowns N grows, and why does this make Jacobi a poor production choice at large grid sizes?
8. What two ingredients does NVIDIA's own real AmgX library reuse from this chapter's own solver, according to Section 29.3?

## Where We Go Next

Chapter 30, "Multi-GPU Ray Tracing and Rendering," leaves the iterative-solver world behind for a workload with a completely different communication profile: instead of neighbor-only halo exchange or a periodic global reduction, distributing a rendering workload usually means partitioning *independent* units of work (rays, tiles, or frames) that need little or no communication during computation, only a final assembly step -- a genuinely different shape from every case study since Chapter 24.

## Worked Solutions

1. Each interior cell's own update formula touches a fixed, small set of neighbor values in a fixed arithmetic order regardless of how the grid is partitioned, so partitioning never changes the sequence of floating-point roundings for the update. A sum-reduction is different: partitioning the summands into P groups and combining the group sums afterward changes the *grouping* of the additions, and floating-point addition is not associative, so the reduction's result can differ in its last bit even though the same real-number answer is being approximated.
2. Without that final exchange, each rank's ghost rows would still hold the neighbor's boundary values from *before* the last update (one iteration stale), so the residual computed near every rank boundary would be measuring a mismatched, out-of-date pair of grid states rather than the true, fully up-to-date residual -- silently corrupting the convergence check exactly where two ranks meet.
3. A single double is only 8 bytes, so the bandwidth term of any reasonable collective algorithm is negligible; what actually costs time is the number of sequential communication steps ("rounds") the algorithm needs before every rank has the combined result, and that round count depends on the algorithm and P, not on the payload size.
4. `-nccheck` sets how many Jacobi iterations pass between convergence checks; increasing it from 1 reduces the total number of Allreduce calls (and therefore total reduction overhead) by that same factor, at the cost of potentially running a few extra iterations past the true convergence point before the next check notices.
5. A naive ring Allreduce needs a number of rounds that grows linearly with P (2(P-1) in this chapter's own model) because data has to pass around the entire ring of ranks; a tree Allreduce needs only a number of rounds that grows with log2(P), because the reduction can fan in and the broadcast can fan out along a tree rather than a single ring, and this difference in round count applies regardless of how small the actual payload is.
6. Gauss-Seidel's update uses each cell's *already-updated* neighbor values from earlier in the same sweep, rather than strictly the previous iteration's values everywhere; that in-sweep ordering dependency means a rank cannot safely compute its own updates purely from halo values exchanged once at the start of the iteration, the way this chapter's Jacobi solver does, breaking the simple communicate-once-then-update-everything pattern this chapter's correctness check depended on.
7. Demmel's own real table gives Jacobi an O(N^2) serial cost against Multigrid's O(N) for the same N unknowns, so as the grid grows, Jacobi's cost grows quadratically while Multigrid's grows only linearly -- meaning Jacobi becomes disproportionately, and eventually prohibitively, slower than Multigrid as production-scale grids get larger.
8. Section 29.3 states that AmgX's own multi-GPU solves reuse the same two ingredients this chapter built by hand: neighbor halo exchange for the local sparse matrix-vector product, and a global Allreduce for the residual norm and inner products the method's convergence check needs -- just applied to a faster-converging numerical method (Multigrid) instead of Jacobi.

---

**Sources cited in this chapter:**

- Yousef Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003, Chapter 1 (Theorem 1.10) and Chapter 4 ("Basic Iterative Methods").
- NVIDIA, `multi-gpu-programming-models` (GitHub repository), `mpi/jacobi.cpp` and top-level `README.md`.
- PETSc Users Manual, "Pipelined Krylov Methods" section, https://petsc.org/release/manual/ksp/.
- James Demmel, UC Berkeley CS267, "Notes for Lectures 15 and 16" / Lecture 24 course notes, https://people.eecs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html.
- M. Naumov, M. Arsaev, P. Castonguay, J. Cohen, J. Demouth, J. Eaton, S. Layton, N. Markovskiy, I. Reguly, N. Sakharnykh, V. Sellappan, R. Strzodka, "AmgX: A Library for GPU Accelerated Algebraic Multigrid and Preconditioned Iterative Methods," *SIAM Journal on Scientific Computing*, 37(5): S602-S626, 2015.
