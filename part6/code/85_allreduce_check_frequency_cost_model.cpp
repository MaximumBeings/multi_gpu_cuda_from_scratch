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
