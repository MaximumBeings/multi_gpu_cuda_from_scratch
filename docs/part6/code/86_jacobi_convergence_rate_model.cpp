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
