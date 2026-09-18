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
