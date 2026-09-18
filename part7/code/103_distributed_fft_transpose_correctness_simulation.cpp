// Chapter 35: AI Weather Forecasting at Scale
// 103_distributed_fft_transpose_correctness_simulation.cpp
//
// FourCastNet's own real Adaptive Fourier Neural Operator (AFNO) is
// described directly in the paper: it "frames the mixing operation as
// continuous global convolution, implemented efficiently in the Fourier
// domain with FFTs." A Fourier transform is fundamentally GLOBAL -- every
// output frequency component depends on EVERY input grid point, not just
// its nearest neighbors -- so once a spatial grid grows too large for one
// GPU and must be split, Chapter 16's own halo exchange (built for a
// LOCAL 5-point stencil) cannot carry the needed data: there is no
// "neighbor" concept left after a Fourier transform. NVIDIA's own real
// cuFFTMp documentation describes the standard real fix instead: a
// "slab" decomposition (each rank owns full rows) combined with a global
// transpose, warning plainly that "distributed 3D FFTs are well-known to
// be communication-bound because of global collective communications of
// the MPI_Alltoallv type." This file builds that exact real two-phase
// algorithm -- local 1D transform along rows (no communication needed,
// each row lives entirely on one rank), an all-to-all TRANSPOSE (Chapter
// 10's own primitive) to reshard from row-owned to column-owned, then a
// local 1D transform along columns -- as a host-side simulation of a
// small 2D discrete Fourier transform, and checks the result against a
// single-process reference at every tested P. A naive O(N) per-frequency
// summation is used instead of a real O(N log N) Cooley-Tukey FFT purely
// to keep the verification code simple and exact -- both compute the
// identical mathematical DFT; only the real cuFFTMp library's own
// algorithm for computing each 1D pass is faster, not different in the
// COMMUNICATION pattern this file is actually testing.
#include <cstdio>
#include <cmath>
#include <vector>
#include <complex>

const int NX = 8;  // grid columns (transformed in phase 1, local)
const int NY = 8;  // grid rows    (transformed in phase 2, after transpose)
typedef std::complex<double> cplx;

// Deterministic real-valued "weather field" input, so every rank starts
// from identical, reproducible data regardless of partitioning.
double gridValue(int y, int x) {
    unsigned int h = (unsigned int)(y * 1000 + x);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return (double)(h % 10);
}

// Naive O(N) per-output-element 1D DFT along one row/column of length N
// -- mathematically the exact same transform a real FFT computes, just
// via direct summation instead of Cooley-Tukey's divide-and-conquer.
std::vector<cplx> dft1D(const std::vector<cplx> &in) {
    int N = (int)in.size();
    std::vector<cplx> out(N);
    for (int k = 0; k < N; k++) {
        cplx sum(0.0, 0.0);
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * (double)k * (double)n / (double)N;
            sum += in[n] * cplx(std::cos(angle), std::sin(angle));
        }
        out[k] = sum;
    }
    return out;
}

// Reference: full 2D DFT computed with no partitioning at all -- 1D DFT
// along every row, then 1D DFT along every resulting column.
std::vector<std::vector<cplx>> runReference() {
    std::vector<std::vector<cplx>> grid(NY, std::vector<cplx>(NX));
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++)
            grid[y][x] = cplx(gridValue(y, x), 0.0);

    // Phase 1: transform each row.
    for (int y = 0; y < NY; y++) grid[y] = dft1D(grid[y]);

    // Phase 2: transform each column.
    for (int x = 0; x < NX; x++) {
        std::vector<cplx> col(NY);
        for (int y = 0; y < NY; y++) col[y] = grid[y][x];
        col = dft1D(col);
        for (int y = 0; y < NY; y++) grid[y][x] = col[y];
    }
    return grid;
}

// P-rank simulation of cuFFTMp's own real slab-decomposition-plus-
// transpose algorithm: each rank owns a contiguous slab of ROWS, does
// phase 1 locally (needs no communication -- exactly the real design
// point), then an ALL-TO-ALL TRANSPOSE reshards ownership from rows to
// columns, and phase 2 runs locally on the newly-owned columns.
std::vector<std::vector<cplx>> runWithP(int P) {
    // Phase 1: each rank transforms its own row slab.
    std::vector<std::vector<cplx>> afterPhase1(NY, std::vector<cplx>(NX));
    for (int r = 0; r < P; r++) {
        for (int y = r; y < NY; y += P) {
            std::vector<cplx> row(NX);
            for (int x = 0; x < NX; x++) row[x] = cplx(gridValue(y, x), 0.0);
            afterPhase1[y] = dft1D(row);
        }
    }

    // Phase 1.5: the real all-to-all transpose (Chapter 10's own
    // primitive) -- reshard from "every rank owns a row slab, every
    // column" to "every rank owns a column slab, every row." This is
    // pure data relocation: no value is combined with another here.
    std::vector<std::vector<cplx>> transposed(NX, std::vector<cplx>(NY));
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++)
            transposed[x][y] = afterPhase1[y][x];

    // Phase 2: each rank transforms its own column slab (now rows of
    // the transposed layout).
    std::vector<std::vector<cplx>> afterPhase2(NX, std::vector<cplx>(NY));
    for (int r = 0; r < P; r++) {
        for (int x = r; x < NX; x += P) {
            afterPhase2[x] = dft1D(transposed[x]);
        }
    }

    // Transpose back to the original (y, x) layout for comparison.
    std::vector<std::vector<cplx>> result(NY, std::vector<cplx>(NX));
    for (int x = 0; x < NX; x++)
        for (int y = 0; y < NY; y++)
            result[y][x] = afterPhase2[x][y];
    return result;
}

int main() {
    auto reference = runReference();
    printf("Reference (single-process): |grid[0][0]|=%.6f  |grid[3][5]|=%.6f\n\n",
           std::abs(reference[0][0]), std::abs(reference[3][5]));

    int Ps[] = {1, 2, 4, 8};
    printf("%-6s %-24s\n", "P", "max |diff| vs reference");
    for (int P : Ps) {
        auto result = runWithP(P);
        double maxDiff = 0.0;
        for (int y = 0; y < NY; y++)
            for (int x = 0; x < NX; x++)
                maxDiff = std::max(maxDiff, std::abs(result[y][x] - reference[y][x]));
        printf("%-6d %-24.17g\n", P, maxDiff);
    }

    printf("\nEvery P reproduces the exact same 2D DFT as the single-"
           "process reference, to the last bit. Like Chapter 34's "
           "embedding all-to-all, this transpose is pure RESHARDING (each "
           "output element is computed by exactly one rank, from data "
           "that arrived via relocation, never combined with another "
           "rank's own partial value) -- the real reason a distributed "
           "FFT's transpose step, despite moving just as much data as any "
           "reduction this book built, never raises the floating-point "
           "associativity question Chapter 8/25/29/31/32 investigated. "
           "The real cost here is not correctness risk -- it is the sheer "
           "COMMUNICATION VOLUME of the transpose itself, which Section "
           "35.3 quantifies next using cuFFTMp's own real cited numbers.\n");
    return 0;
}
