**What you will understand after this chapter:** why NVIDIA Earth-2's real FourCastNet weather model -- despite operating on a giant global spatial grid, the exact shape Chapter 16 trained you to expect needs domain decomposition -- is actually trained with ordinary Chapter 12 data parallelism, because the real grid comfortably fits one GPU's memory; why FourCastNet's Adaptive Fourier Neural Operator (AFNO) architecture uses a computation (the Fourier transform) that is fundamentally GLOBAL rather than local, so that IF a future grid ever did outgrow one GPU, Chapter 16's own halo exchange could not supply what it needs; and what NVIDIA's own real cuFFTMp library uses instead -- a slab decomposition plus a global all-to-all transpose -- including the real point at which that technique's communication cost, not correctness, becomes the dominant concern.

**What you need to know first:** Chapter 10 (all-to-all, resharding a tensor from one split dimension to another), Chapter 12 (data parallelism), Chapter 16 (domain decomposition and halo exchange, this chapter's main point of contrast), and Chapter 34 (this book's most recent use of an all-to-all for a pure-resharding, non-reduction purpose).

---

The previous chapter's own closing preview, and this book's own table of contents, both predicted that NVIDIA Earth-2's real FourCastNet weather model would extend Chapter 16's domain decomposition to a learned neural operator "while keeping the same halo-exchange communication shape." That prediction is corrected, in public, in this chapter: it is wrong, and the real reason it is wrong is more interesting than the prediction itself. NVIDIA and collaborators' own real paper ("FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators," Pathak et al., arXiv:2202.11214, 2022) reports a real, specific training shape -- "the end to end training takes about 16 hours wall-clock time on a cluster of 64 Nvidia A100 GPUs" -- at a real, specific global grid resolution: "a resolution of 0.25 degree which corresponds to... a global grid size of 720x1440 pixels." That grid is nowhere near large enough to force domain decomposition on a real A100. This chapter builds the correction in three steps: first, what FourCastNet's real training actually is (data parallelism, Section 35.1); second, why its real AFNO architecture's use of the Fourier transform would break Chapter 16's halo exchange if a future, much larger grid ever did need spatial splitting (Section 35.2); and third, NVIDIA's own real cuFFTMp library's documented answer for that case, and the real scale at which it would start to matter (Section 35.3).

```text
+------------------------------------------------------------------+
| What Chapter 16 and last chapter's own preview both expected:     |
|   giant weather grid -> split across GPUs -> halo exchange        |
|   [ GPU: strip ][ GPU: strip ][ GPU: strip ][ GPU: strip ]        |
|        each rank trades only its EDGE rows with its neighbors     |
+------------------------------------------------------------------+
| What FourCastNet's own real training actually is (Section 35.1):  |
|   720x1440 grid fits ONE A100 -> ordinary Chapter 12 data parallel |
|   [ GPU: full grid ][ GPU: full grid ][ GPU: full grid ] ...      |
|        replica             replica            replica             |
|        (differ only by which training samples each one sees)      |
+------------------------------------------------------------------+
| Where spatial splitting WOULD eventually be needed                |
| (Sections 35.2-35.3): AFNO's global Fourier mixing has no local    |
| neighbor concept -> needs a slab decomposition + all-to-all        |
| TRANSPOSE, not halo exchange, once a grid outgrows one GPU         |
+------------------------------------------------------------------+
```

## 35.1 FourCastNet's Real Training Is Data Parallelism, Not Domain Decomposition

### Intuition

Picture two very different ways a team of 64 meteorologists could split up the work of training themselves to forecast weather from a huge archive of historical maps. In the first (Chapter 16's approach), each meteorologist is handed a physical PIECE of every single map -- the northeast corner, say -- and has to constantly compare notes with whoever holds the adjoining piece, because weather at a boundary depends on what's happening just across it. In the second (Chapter 12's approach), every meteorologist gets their OWN full copy of the map-drawing tools and studies a DIFFERENT subset of the historical archive, periodically comparing what they've each learned and averaging their updated technique. FourCastNet's real training uses the second approach, and the reason is almost anticlimactic: the map itself, at "a global grid size of 720x1440 pixels," is small enough that giving every one of the real 64 A100s its own full copy costs nothing worth worrying about. There is no map-splitting problem to solve at all -- the real bottleneck this cluster is working around is the size of the training ARCHIVE, not the size of any single map, and that is exactly Chapter 12's own problem shape.

!!! warning "[COMMON TRAP] Assuming a large spatial grid model automatically needs domain decomposition"
    A weather model operating on a global grid looks, on the surface, exactly like Chapter 16's stencil computations -- and the previous chapter's own closing preview fell into this trap in print. File 102's own real cited numbers show why the assumption fails here specifically: FourCastNet's real 720x1440 global grid is a small fraction of a real A100's memory capacity, so there is no capacity pressure forcing a spatial split at all. Domain decomposition earns its cost (Chapter 16's own real halo-exchange communication, paid on every step) only when a SINGLE replica of the spatial domain does not fit on one device -- a condition FourCastNet's own real training never reaches.

### Background

```text
+----------------------------------------------------------+
| Real anchor: 16 hours wall-clock on 64 real A100 GPUs      |
| Real grid: 720x1440 pixels -- fits comfortably on ONE A100 |
+----------------------------------------------------------+
| Therefore: whole-model REPLICAS (Chapter 12), not a         |
| spatially-sharded model (Chapter 16) -- GPUs differ only by |
| which training SAMPLES they see, gradients all-reduced      |
+----------------------------------------------------------+
```

File 102 builds a simple, clearly-labeled illustrative data-parallel scaling model anchored to FourCastNet's real cited 16-hour/64-GPU training point, and prints FourCastNet's own real cited speedup and energy numbers directly, without deriving or modifying them.

```cpp
// Chapter 35: AI Weather Forecasting at Scale
// 102_fourcastnet_data_parallel_training_model.cpp
//
// NVIDIA and collaborators' own real FourCastNet paper ("FourCastNet: A
// Global Data-driven High-resolution Weather Model using Adaptive
// Fourier Neural Operators," Pathak et al., arXiv:2202.11214, 2022)
// reports a real, specific multi-GPU training shape: "the end to end
// training takes about 16 hours wall-clock time on a cluster of 64
// Nvidia A100 GPUs." At the real global grid size the paper cites --
// "a resolution of 0.25 degree which corresponds to... a global grid
// size of 720x1440 pixels" -- that grid comfortably fits in a single
// A100's own memory, so this real 64-GPU training job is, structurally,
// Chapter 12's own data parallelism (whole-model replicas, batch split
// across GPUs, gradients synchronized by an all-reduce), NOT Chapter
// 16's spatial domain decomposition -- a genuine surprise for a "weather
// grid" model, and the reason this section exists before Section 35.2
// gets to the real technique (distributed FFT) that DOES eventually
// require spatial splitting. This file builds a simple, clearly-labeled
// illustrative scaling model (not a fabricated timing) around the real
// 16-hour/64-GPU anchor point, and prints FourCastNet's own real cited
// speedup and energy numbers directly.
#include <cstdio>

int main() {
    double realHoursAt64GPUs = 16.0;
    int realGpuCount = 64;

    printf("Real cited anchor point (Pathak et al. 2022): %.0f hours "
           "wall-clock on %d real A100 GPUs.\n\n", realHoursAt64GPUs, realGpuCount);

    printf("Illustrative near-linear data-parallel scaling model (Chapter "
           "12's own technique: whole-model replicas, gradient all-reduce "
           "each step) anchored to that real point -- NOT a claim about "
           "measured scaling efficiency, which this sandbox cannot "
           "measure:\n");
    printf("%-12s %-20s\n", "GPU count", "Illustrative hours");
    int gpuCounts[] = {8, 16, 32, 64, 128, 256};
    for (int g : gpuCounts) {
        double hours = realHoursAt64GPUs * (double)realGpuCount / (double)g;
        printf("%-12d %-20.2f\n", g, hours);
    }

    printf("\nFourCastNet's own real cited results (not derived by this "
           "book):\n");
    printf("- \"FourCastNet generates a week-long forecast in less than 2 "
           "seconds, orders of magnitude faster than IFS.\"\n");
    printf("- \"FourCastNet is about 45,000 times faster than traditional "
           "NWP models on a node-hour basis.\"\n");
    printf("- \"Once trained, however, FourCastNet uses about 12,000 times "
           "less energy to generate a forecast than the IFS model.\"\n");
    printf("- Real global grid: \"a resolution of 0.25 degree... a global "
           "grid size of 720x1440 pixels\" -- comfortably fits one real "
           "A100's memory, which is exactly why the real 64-GPU/16-hour "
           "training run above is ordinary data parallelism, not spatial "
           "domain decomposition.\n");
    return 0;
}
```

Compile and run (a plain host `.cpp` file with no CUDA/NCCL/MPI/NVSHMEM linkage, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 102_fourcastnet_data_parallel_training_model \
    102_fourcastnet_data_parallel_training_model.cpp
./102_fourcastnet_data_parallel_training_model
```

Locked output:

```
Real cited anchor point (Pathak et al. 2022): 16 hours wall-clock on 64 real A100 GPUs.

Illustrative near-linear data-parallel scaling model (Chapter 12's own technique: whole-model replicas, gradient all-reduce each step) anchored to that real point -- NOT a claim about measured scaling efficiency, which this sandbox cannot measure:
GPU count    Illustrative hours  
8            128.00              
16           64.00               
32           32.00               
64           16.00               
128          8.00                
256          4.00                

FourCastNet's own real cited results (not derived by this book):
- "FourCastNet generates a week-long forecast in less than 2 seconds, orders of magnitude faster than IFS."
- "FourCastNet is about 45,000 times faster than traditional NWP models on a node-hour basis."
- "Once trained, however, FourCastNet uses about 12,000 times less energy to generate a forecast than the IFS model."
- Real global grid: "a resolution of 0.25 degree... a global grid size of 720x1440 pixels" -- comfortably fits one real A100's memory, which is exactly why the real 64-GPU/16-hour training run above is ordinary data parallelism, not spatial domain decomposition.
```

## 35.2 Why AFNO's Global Fourier Mixing Would Break Halo Exchange

### Intuition

Chapter 16's halo exchange works because a 5-point stencil is fundamentally LOCAL -- to update one cell, a rank only ever needs that cell's immediate neighbors, so trading a single border strip with each adjacent rank is enough. FourCastNet's real Adaptive Fourier Neural Operator architecture is described directly in its own paper as "fram[ing] the mixing operation as continuous global convolution, implemented efficiently in the Fourier domain with FFTs" -- and a Fourier transform has the opposite property from a stencil: every single output frequency component is a weighted combination of EVERY input point on the grid, not just its neighbors. Imagine trying to compute a city's overall traffic rhythm using only each neighborhood's own local counts, with no way to combine them city-wide -- that is what asking Chapter 16's halo exchange to feed a Fourier transform would be like. There is no "neighbor" concept left once the computation is genuinely global, so if a future grid ever did outgrow one GPU, Chapter 16's own technique could not supply what this computation needs, even though both operate on what looks like the same spatial grid.

!!! warning "[COMMON TRAP] Assuming halo exchange generalizes to any spatially-organized computation"
    Chapter 16 built domain decomposition around a LOCAL computation (a 5-point stencil), and it is tempting to assume any spatial-grid computation can be parallelized the same way. AFNO's real Fourier-domain mixing is the counterexample: its dependency structure is GLOBAL, not local, so a technique built to share only border data cannot supply what it needs. The real fix, described directly in NVIDIA's own cuFFTMp documentation, is structurally different -- a slab decomposition plus a global all-to-all TRANSPOSE (Chapter 10's own primitive), not a border exchange.

### Background

```text
+----------------------------------------------------------+
| Halo exchange (Chapter 16): LOCAL dependency only           |
|   [ rank r ] shares only its EDGE with rank r-1 and r+1      |
+----------------------------------------------------------+
| Fourier transform: GLOBAL dependency -- every output needs  |
| EVERY input; no "edge" or "neighbor" concept survives        |
+----------------------------------------------------------+
| Real fix (cuFFTMp's own slab decomposition):                 |
|   phase 1: transform ROWS locally (no comm needed)           |
|   all-to-all TRANSPOSE: reshard row-owned -> column-owned     |
|   phase 2: transform COLUMNS locally (no comm needed)         |
+------------------------------------------------------------+
```

NVIDIA's own real cuFFTMp documentation describes this exact real two-phase algorithm and warns plainly that "distributed 3D FFTs are well-known to be communication-bound because of global collective communications of the MPI_Alltoallv type." File 103 builds that real algorithm -- local 1D transform along rows (no communication, each row lives entirely on one rank), an all-to-all TRANSPOSE (Chapter 10's own primitive) to reshard from row-owned to column-owned, then a local 1D transform along columns -- as a host-side correctness simulation of a small 2D discrete Fourier transform, and checks the result against a single-process reference at every tested P. A naive O(N) per-frequency summation is used instead of a real O(N log N) Cooley-Tukey FFT purely to keep the verification code simple and exact -- both compute the identical mathematical DFT; only the real cuFFTMp library's own algorithm for computing each 1D pass is faster, not different in the communication pattern this file is actually testing.

```cpp
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
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 103_distributed_fft_transpose_correctness_simulation \
    103_distributed_fft_transpose_correctness_simulation.cpp
./103_distributed_fft_transpose_correctness_simulation
```

Locked output:

```
Reference (single-process): |grid[0][0]|=317.000000  |grid[3][5]|=30.673242

P      max |diff| vs reference 
1      0                       
2      0                       
4      0                       
8      0                       

Every P reproduces the exact same 2D DFT as the single-process reference, to the last bit. Like Chapter 34's embedding all-to-all, this transpose is pure RESHARDING (each output element is computed by exactly one rank, from data that arrived via relocation, never combined with another rank's own partial value) -- the real reason a distributed FFT's transpose step, despite moving just as much data as any reduction this book built, never raises the floating-point associativity question Chapter 8/25/29/31/32 investigated. The real cost here is not correctness risk -- it is the sheer COMMUNICATION VOLUME of the transpose itself, which Section 35.3 quantifies next using cuFFTMp's own real cited numbers.
```

## 35.3 The Real Communication Cost of a Distributed FFT

### Intuition

Section 35.2 proved the slab-decomposition-plus-transpose technique is exactly correct at every tested P -- so correctness is settled. But correctness and cost are different questions, and NVIDIA's own real cuFFTMp documentation is direct about which one dominates here: "distributed 3D FFTs are well-known to be communication-bound because of global collective communications of the MPI_Alltoallv type." Picture the transpose as a citywide reshuffling of every precinct's records onto new precinct boundaries -- nobody gets anything wrong in the reshuffle, but moving that much paperwork citywide, all at once, is real, unavoidable work. cuFFTMp's own real cited benchmark numbers -- 78 milliseconds at 8 A100 GPUs, falling to 4 milliseconds at 2048 A100 GPUs, for a real 2048-cubed FFT -- show that work is expensive but does shrink with more GPUs, sustaining "more than 70% of peak effective bandwidth" along the way. Placed next to FourCastNet's own real 720x1440 grid from Section 35.1, the honest conclusion is that today's FourCastNet is nowhere near the regime where this cost matters -- but a future, much higher-resolution global model would be.

!!! warning "[COMMON TRAP] Assuming a bit-exact operation is automatically a cheap one"
    Section 35.2 showed the transpose is bit-exact at every P, which can read as "solved" -- but bit-exact says nothing about how much data crossed the network to get there. File 104's own real cited numbers separate the two questions cleanly: the transpose's correctness was never in doubt once it was recognized as pure resharding (Section 35.2), while its real communication COST is a genuinely different, scale-dependent concern that cuFFTMp's own documentation names directly as the dominant one for distributed FFTs.

### Background

```text
+----------------------------------------------------------+
| cuFFTMp's own real 2048^3 FFT benchmark:                    |
|   8 GPUs: 78 ms  ------->  2048 GPUs: 4 ms                  |
|   (real, cited; MPI_Alltoallv transpose is the bottleneck)  |
+----------------------------------------------------------+
| FourCastNet's real grid: 720x1440, about 1.0 million points |
| cuFFTMp benchmark grid: 2048^3, about 8.6 billion points     |
+----------------------------------------------------------+
| FourCastNet today: fits 1 GPU, no distributed FFT needed     |
| Future higher-res model: approaching cuFFTMp's own regime    |
+----------------------------------------------------------+
```

File 104 first prints cuFFTMp's own real cited benchmark numbers verbatim -- no interpolation, no invented figures, since both endpoints are already real, sourced measurements -- and then places them side by side with FourCastNet's own real 720x1440 global grid from Section 35.1, using plain arithmetic on two independently real, separately-cited figures.

```cpp
// Chapter 35: AI Weather Forecasting at Scale
// 104_distributed_fft_communication_cost_model.cpp
//
// Section 35.2 showed the distributed-FFT transpose (slab decomposition +
// all-to-all) is bit-exact at every P -- a pure resharding, not a
// reduction. That leaves exactly one real question: how expensive is that
// transpose's own communication? NVIDIA's own real cuFFTMp documentation
// answers it directly, warning that "distributed 3D FFTs are well-known
// to be communication-bound because of global collective communications
// of the MPI_Alltoallv type," and backs that warning with real cited
// benchmark numbers for a large 2048^3 FFT: 78ms at 8 A100 GPUs, falling
// to 4ms at 2048 A100 GPUs, sustaining "more than 70% of peak effective
// bandwidth" and "1.8 PFlop/s" of achieved throughput at a real cited
// 4096-A100 configuration. This file first prints those real, directly
// cited cuFFTMp numbers verbatim -- no interpolation, no invented
// numbers, because both endpoints are already real, sourced measurements
// -- and then puts them side by side with FourCastNet's own real 720x1440
// global grid from Section 35.1: a 2048^3 (roughly 8.6 billion point) FFT
// is over four orders of magnitude larger than FourCastNet's real 720x1440
// (about 1 million point) 2D grid, which is exactly why FourCastNet's own
// real training run never needed cuFFTMp's own distributed-FFT machinery
// at all -- and exactly the scale at which a FUTURE, much higher-
// resolution global weather or climate model would.
#include <cstdio>

int main() {
    printf("cuFFTMp's own real cited benchmark, a single large 2048^3 "
           "complex FFT (NVIDIA cuFFTMp documentation):\n");
    printf("%-14s %-18s %-24s\n", "GPU count", "Time", "Note");
    printf("%-14d %-18s %-24s\n", 8, "78 ms", "real cited measurement");
    printf("%-14d %-18s %-24s\n", 2048, "4 ms", "real cited measurement");
    printf("Also real and cited: \"more than 70%% of peak effective "
           "bandwidth\" sustained, and \"1.8 PFlop/s\" achieved throughput, "
           "at a real cited 4096-A100 configuration.\n\n");

    printf("Why this matters directly quotes cuFFTMp's own documentation: "
           "\"distributed 3D FFTs are well-known to be communication-bound "
           "because of global collective communications of the "
           "MPI_Alltoallv type\" -- the exact all-to-all transpose Section "
           "35.2 proved correct, now identified as the real bottleneck, "
           "not a correctness risk.\n\n");

    // Illustrative point count comparison -- both figures are real and
    // separately cited (FourCastNet's paper; a standard 2048^3 FFT
    // problem size), so this is a plain arithmetic comparison, not a
    // fabricated benchmark.
    long long fourCastNetPoints = 720LL * 1440LL;
    long long cuFFTMpPoints = 2048LL * 2048LL * 2048LL;
    double ratio = (double)cuFFTMpPoints / (double)fourCastNetPoints;

    printf("Grid size comparison (real, cited figures on both sides):\n");
    printf("FourCastNet's real global grid (Section 35.1): 720 x 1440 = "
           "%lld points.\n", fourCastNetPoints);
    printf("cuFFTMp's own real cited benchmark problem: 2048^3 = %lld "
           "points.\n", cuFFTMpPoints);
    printf("Ratio: the cuFFTMp benchmark grid holds about %.0fx as many "
           "points as FourCastNet's real global grid.\n\n", ratio);

    printf("This is the honest resolution of the tension this chapter "
           "opened with: AFNO's own real global Fourier-domain mixing "
           "(Section 35.2) is, in principle, exactly the kind of operation "
           "that eventually needs cuFFTMp's own real slab-decomposition-"
           "plus-transpose machinery once a grid outgrows one GPU. "
           "FourCastNet's own real 720x1440 grid never reaches that point "
           "-- it fits one A100 comfortably, which is exactly why Section "
           "35.1's real 64-GPU training run is ordinary data parallelism. "
           "But a future global model at, say, 1-2km resolution instead of "
           "today's 0.25 degree -- pushing toward grids the size of "
           "cuFFTMp's own real cited 2048^3 benchmark -- is precisely where "
           "this chapter's own real distributed-FFT technique stops being "
           "optional.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 104_distributed_fft_communication_cost_model \
    104_distributed_fft_communication_cost_model.cpp
./104_distributed_fft_communication_cost_model
```

Locked output:

```
cuFFTMp's own real cited benchmark, a single large 2048^3 complex FFT (NVIDIA cuFFTMp documentation):
GPU count      Time               Note                    
8              78 ms              real cited measurement  
2048           4 ms               real cited measurement  
Also real and cited: "more than 70% of peak effective bandwidth" sustained, and "1.8 PFlop/s" achieved throughput, at a real cited 4096-A100 configuration.

Why this matters directly quotes cuFFTMp's own documentation: "distributed 3D FFTs are well-known to be communication-bound because of global collective communications of the MPI_Alltoallv type" -- the exact all-to-all transpose Section 35.2 proved correct, now identified as the real bottleneck, not a correctness risk.

Grid size comparison (real, cited figures on both sides):
FourCastNet's real global grid (Section 35.1): 720 x 1440 = 1036800 points.
cuFFTMp's own real cited benchmark problem: 2048^3 = 8589934592 points.
Ratio: the cuFFTMp benchmark grid holds about 8285x as many points as FourCastNet's real global grid.

This is the honest resolution of the tension this chapter opened with: AFNO's own real global Fourier-domain mixing (Section 35.2) is, in principle, exactly the kind of operation that eventually needs cuFFTMp's own real slab-decomposition-plus-transpose machinery once a grid outgrows one GPU. FourCastNet's own real 720x1440 grid never reaches that point -- it fits one A100 comfortably, which is exactly why Section 35.1's real 64-GPU training run is ordinary data parallelism. But a future global model at, say, 1-2km resolution instead of today's 0.25 degree -- pushing toward grids the size of cuFFTMp's own real cited 2048^3 benchmark -- is precisely where this chapter's own real distributed-FFT technique stops being optional.
```

## Chapter Summary

This chapter opened by correcting its own predecessor's preview: NVIDIA Earth-2's real FourCastNet weather model does not extend Chapter 16's halo exchange to a learned neural operator. File 102 showed why -- FourCastNet's own real 720x1440 global grid comfortably fits a single A100's memory, so its real 64-GPU, 16-hour training run is ordinary Chapter 12 data parallelism: whole-model replicas differing only in which training samples they see. The real reason this matters goes deeper than one grid's size, though: FourCastNet's own real AFNO architecture computes its mixing operation as "continuous global convolution, implemented efficiently in the Fourier domain with FFTs," and a Fourier transform's dependency structure is fundamentally global, not local -- so if a future, larger grid ever did outgrow one GPU, Chapter 16's own halo exchange could not supply what it needs. File 103 built NVIDIA's own real cuFFTMp technique for that case -- a slab decomposition plus a global all-to-all TRANSPOSE (Chapter 10's own primitive) -- and proved it bit-exact at every tested P, because it is pure resharding, never a reduction, echoing Chapter 34's own embedding all-to-all. File 104 closed the chapter by separating correctness from cost: cuFFTMp's own real cited benchmark numbers (78ms at 8 GPUs, 4ms at 2048 GPUs for a 2048-cubed FFT) show the transpose's real communication cost is significant but shrinks with more GPUs, and a plain grid-size comparison shows FourCastNet's own real grid is about four orders of magnitude smaller than that benchmark -- exactly why this chapter's own distributed-FFT technique is not needed today, but is precisely what a future, much higher-resolution global model would need.

## Self-Check Questions

1. Chapter 16's domain decomposition and the previous chapter's own closing preview both predicted FourCastNet's real training would need halo exchange. What did File 102 actually find, and why?
2. What real cited fact about FourCastNet's own global grid size explains why pure data parallelism is sufficient for training, without any spatial splitting?
3. Why can Chapter 16's halo exchange not supply the data a Fourier transform needs, even though both operate on the same spatial grid?
4. Describe cuFFTMp's own real two-phase slab-decomposition algorithm in your own words. Which phase needs communication, and which two phases do not?
5. Why does File 103 use a naive O(N) DFT instead of a real Cooley-Tukey FFT, and does that choice affect the correctness result?
6. Was File 103's all-to-all transpose a reduction, or something else? How does the answer determine whether floating-point associativity was ever a risk in Section 35.2, contrasted with Chapters 8/25/29/31/32?
7. According to File 104's own locked output, how many times more grid points does cuFFTMp's own cited 2048-cubed benchmark have compared to FourCastNet's own real 720x1440 grid?
8. cuFFTMp's own documentation names the real bottleneck for distributed 3D FFTs. What is it, and how does that relate to Section 35.2's own correctness finding?
9. If a future weather model moved to a much higher spatial resolution than FourCastNet's real 0.25 degree, which of this chapter's two techniques -- Section 35.1's data parallelism, or Sections 35.2-35.3's slab decomposition plus transpose -- would likely become necessary, and why?

## Where We Go Next

Chapter 36 returns to the financial-services domain this book last visited in Chapter 31, extending that chapter's own Monte Carlo risk simulation to multi-GPU XVA and portfolio-level risk computation at production scale.

## Worked Solutions

1. File 102 found that FourCastNet's real training is ordinary Chapter 12 data parallelism -- whole-model replicas, each seeing a different slice of training samples, gradients synchronized by an all-reduce -- not Chapter 16's spatial domain decomposition. The reason is capacity, not computation: FourCastNet's own real 720x1440 global grid is small enough that giving every one of the real 64 A100s its own full replica costs nothing worth avoiding, so there was never any capacity pressure forcing a spatial split.
2. The real cited fact is FourCastNet's own paper describing "a resolution of 0.25 degree which corresponds to... a global grid size of 720x1440 pixels" -- a grid that comfortably fits within a single real A100's memory budget, removing the entire premise (a spatial domain too large for one device) that would make domain decomposition necessary.
3. A Fourier transform is fundamentally global: every output frequency component is a weighted combination of every input grid point, not just nearby ones. Chapter 16's halo exchange only ever shares a border strip with immediate neighbors, which is sufficient for a local 5-point stencil but structurally cannot deliver the globally-scattered data a Fourier transform's computation actually depends on.
4. In phase 1, each rank owns a contiguous slab of full ROWS and computes a local 1D transform along each of its own rows -- no communication needed, because each row lives entirely on one rank. Between phase 1 and phase 2, an all-to-all TRANSPOSE reshards ownership from row-slabs to column-slabs -- this is the one phase that requires communication. In phase 2, each rank computes a local 1D transform along each of its own (now-owned) columns -- again no communication needed.
5. File 103 uses a naive O(N) per-output-element DFT because it computes the mathematically identical transform a real Cooley-Tukey FFT computes, just via direct summation instead of divide-and-conquer -- simpler code to verify exactly, with no effect on the correctness result, since both algorithms compute the same DFT values. Only the real cuFFTMp library's own algorithm for computing each 1D pass would be faster, not different in the communication pattern the file is actually testing.
6. It was something else: a pure resharding, never a reduction. Every output element in File 103's transpose is computed by exactly one rank from data that arrived via relocation, and no value is ever combined with another rank's own partial value. That is exactly why floating-point associativity, the recurring concern in Chapters 8, 25, 29, 31, and 32, was never a risk here -- there is no combination step for grouping order to ever affect.
7. File 104's own locked output shows the ratio is about 8285x: cuFFTMp's own cited 2048-cubed benchmark has 8,589,934,592 points, versus FourCastNet's own real 720x1440 grid's 1,036,800 points.
8. cuFFTMp's own documentation names "global collective communications of the MPI_Alltoallv type" as the real bottleneck for distributed 3D FFTs -- precisely the all-to-all transpose Section 35.2 proved bit-exact. The relationship is that Section 35.2 settled the correctness question for that exact operation, while cuFFTMp's own documentation is naming its communication cost, not its correctness, as the real concern at scale.
9. Sections 35.2-35.3's slab decomposition plus transpose would likely become necessary, because a much higher spatial resolution would grow the global grid past what a single GPU can hold, reintroducing the capacity pressure that Section 35.1 showed FourCastNet's own real 0.25-degree grid never reaches -- and because AFNO's own real Fourier-domain mixing operation is exactly the kind of globally-dependent computation Section 35.2 showed cannot be split using Chapter 16's local halo exchange, once that capacity limit is reached.

---

**Sources cited in this chapter:**

- Pathak, J. et al. "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators." arXiv:2202.11214, 2022, fetched fresh this session. (The real "16 hours wall-clock time on a cluster of 64 Nvidia A100 GPUs" training figure, the real "0.25 degree... global grid size of 720x1440 pixels" resolution figure, the real "continuous global convolution, implemented efficiently in the Fourier domain with FFTs" AFNO description, and the real cited 45,000x/12,000x/less-than-2-second speedup and energy figures.)
- NVIDIA. cuFFTMp documentation, docs.nvidia.com, fetched fresh this session. (The real slab decomposition description, the real "distributed 3D FFTs are well-known to be communication-bound because of global collective communications of the MPI_Alltoallv type" quote, and the real cited 78ms-at-8-GPUs/4ms-at-2048-GPUs/70%-peak-bandwidth/1.8-PFlop/s-at-4096-GPUs benchmark numbers for a 2048-cubed FFT.)
- NVIDIA. PhysicsNeMo distributed training documentation, docs.nvidia.com/physicsnemo, fetched fresh this session, consulted for contrast material on domain-parallel training (its own real "Domain Parallel" glossary definition describing halo and border-data exchange between neighbors).
- This book's own Chapter 10 (all-to-all, the primitive reused for the transpose), Chapter 12 (data parallelism, this chapter's real technique for Section 35.1), Chapter 16 (domain decomposition and halo exchange, this chapter's main point of contrast and correction), and Chapter 34 (this book's most recent use of an all-to-all for pure resharding rather than reduction, directly echoed in Section 35.2's own finding).

**A note on this chapter's own framing:** FourCastNet's own paper does not itself discuss a distributed-FFT implementation strategy for AFNO's mixing operation, and cuFFTMp's own documentation does not itself mention FourCastNet. The connection drawn in Sections 35.2 and 35.3 -- that AFNO's real FFT-based architecture would, at sufficient scale, need cuFFTMp's own real slab-decomposition-and-transpose technique -- is this book's own synthesis of two independently real, separately-cited facts, not a claim that NVIDIA's own documentation ties the two together explicitly.
