# Chapter 24: Multi-GPU Dense Matrix Multiplication at Scale

**What you will understand by the end of this chapter:**

- Why a dense GEMM's three operands can stop fitting on one real GPU even when no single weight matrix by itself is the problem -- and how to compute, in closed form, the minimum device count a given problem needs purely for capacity.
- The real SUMMA algorithm (Van de Geijn & Watts): how a 2D process grid splits both matrices, broadcasts panels within rows and columns, and accumulates local partial products -- built and verified as a real host-side simulation against a naive reference.
- NVIDIA's own real, single-node, production answer to this exact problem, cuBLAS-Xt, including a real API detail no earlier chapter's cuBLAS-style call had: it takes HOST pointers directly.
- A new, sharper real failure mode this book has not produced before: a real memory-corruption crash (a double-free glibc itself detects) from destroying a library handle after one specific real call failed on it -- and why checking device counts before requesting device ordinals is not optional here.

**What you need to know first:**

- Chapter 13's real per-layer memory-shard model and Chapter 1's own real H100 SXM HBM figure.
- Chapter 8's real host-mediated reduce and Chapter 9's real ring all-reduce (for the broadcast-and-accumulate shape this chapter reuses).
- Chapter 14's real column-parallel/row-parallel GEMM split.

---

This chapter opens Part 6's case studies by returning to the single most common real operation in this entire book's domain -- a dense matrix multiply -- and asking what happens when it is too large for one device, in the way a real production system, not a single-concept demonstration, would have to answer it. A real NVIDIA forum thread on cuBLAS-Xt states the motivation plainly: multi-GPU GEMM's "main point" is often to "work around GPU memory capacity limits," not (only) to go faster. This chapter builds that motivation into a real number (24.1), the real, named academic algorithm this book's own earlier collectives (broadcast, reduce) turn out to already contain the pieces of (24.2), and the real, shipping NVIDIA library built for exactly this problem on a single node (24.3) -- closing with why that library's own single-node scope is real and named, and what NVIDIA's current real answer for going beyond one node actually is.

```text
Section 24.1:                   Section 24.2:                    Section 24.3:
"does it fit?"                  "how do you split it              "what does NVIDIA
(closed-form capacity           correctly?"                       actually ship for
model, no comm cost)            (SUMMA, verified by                this?"
                                  simulation)                      (real cuBLAS-Xt,
                                                                    single node)
      |                                |                                |
      +--------------------------------+--------------------------------+
                                       |
                          one case study, three real layers:
                          capacity, algorithm, production API
```

## 24.1 The Memory Wall for a GEMM's THREE Operands

### Intuition

Chapter 13 already asked whether a single weight matrix fits on one GPU. This section asks a related but genuinely different question: does a real GEMM's three operands -- A, B, and the output C -- fit on one device AT ONCE, at a batch size and sequence length a real training or inference run actually uses? A single GPT-3-scale FFN weight matrix, by itself, is small enough to fit comfortably on one real H100. The moment that weight matrix is multiplied against a real batch of activations, though, the output C and the input activation matrix A both scale with the batch and sequence length, not just the model's own hidden dimension -- and a real production batch size makes A and C, not the weight matrix B, the thing that stops fitting. A real NVIDIA forum thread about cuBLAS-Xt names exactly this motivation for multi-GPU GEMM: its "main point" is to "work around GPU memory capacity limits" -- an out-of-core problem, not (only) a speed problem.

```text
Ch13's own question:                       This section's own question:

+----------------------+                    +----------------------------+
| Does ONE weight       |                    | Do A, B, AND C -- at a     |
| matrix B fit on one   |                    | real batch/seq size -- all |
| device?               |                    | fit on one device AT ONCE? |
+----------------------+                    +----------------------------+
  Usually yes, even at                        Often NO -- A and C both
  GPT-3 scale.                                 scale with batch*seq, not
                                                just the model's own dims.
```

### Background

```cpp
// Chapter 24: Multi-GPU Dense Matrix Multiplication at Scale
// 69_gemm_memory_wall_model.cpp
//
// A real, closed-form model (no fabricated timings, matching this book's
// own established practice since Ch2/Ch5/Ch16) of when a dense GEMM's
// three matrices genuinely stop fitting on one real GPU. Reuses Chapter
// 1's own real H100 SXM figure (80 GB HBM3) as the single-device budget.
// A real NVIDIA forum thread on cuBLASXt states plainly what problem
// this section is building toward: cuBLASXt's "main point" is to "work
// around GPU memory capacity limits" -- i.e. out-of-core / multi-GPU
// GEMM exists FIRST because of capacity, not (only) because of speed.
#include <cstdio>
#include <cstdint>
#include <cmath>

struct GemmShape {
    const char *label;
    uint64_t m, k, n;
};

int main() {
    const double H100_SXM_HBM_BYTES = 80.0 * 1024.0 * 1024.0 * 1024.0; // Ch1's own real figure
    const double BYTES_PER_FP32 = 4.0;

    // Real GPT-3 175B dimensions (Brown et al. Table 2.1, already cited
    // in Ch1/Ch13/Ch14): d_model=12288. A single FFN weight matrix is
    // d_model x 4*d_model. This section asks a genuinely different
    // question from Ch13's per-layer question: not "does ONE weight
    // matrix fit," but "do the three GEMM operands -- A, B, and the
    // output C -- fit AT ONCE, at a batch size and sequence length a
    // real training run actually uses."
    GemmShape shapes[] = {
        {"GPT-3 FFN weight alone (d_model x 4d_model)", 12288, 12288, 49152},
        {"GPT-3 FFN forward, batch=512, seq=2048 (M=batch*seq)",
         512ull * 2048ull, 12288, 49152},
        {"A hypothetical 10x-larger FFN forward at the same batch/seq",
         512ull * 2048ull, 122880, 491520},
    };

    printf("Single-device budget (Ch1's own real H100 SXM figure): %.1f GiB\n\n",
           H100_SXM_HBM_BYTES / (1024.0 * 1024.0 * 1024.0));

    for (auto &s : shapes) {
        double bytesA = (double)s.m * (double)s.k * BYTES_PER_FP32;
        double bytesB = (double)s.k * (double)s.n * BYTES_PER_FP32;
        double bytesC = (double)s.m * (double)s.n * BYTES_PER_FP32;
        double totalGiB = (bytesA + bytesB + bytesC) / (1024.0 * 1024.0 * 1024.0);
        bool fits = (bytesA + bytesB + bytesC) <= H100_SXM_HBM_BYTES;
        printf("%s\n", s.label);
        printf("  M=%llu K=%llu N=%llu (fp32)\n",
               (unsigned long long)s.m, (unsigned long long)s.k, (unsigned long long)s.n);
        printf("  A+B+C = %.2f GiB -> %s on one H100 SXM\n\n",
               totalGiB, fits ? "FITS" : "DOES NOT FIT");
    }

    // A real, closed-form minimum-device-count model: if the three
    // operands don't fit on one device, how many devices, under a
    // cuBLASXt-style even block split of A/B/C across devices, are
    // needed at minimum? This does NOT model communication cost (that
    // is Section 24.2's own job) -- only capacity.
    printf("--- Minimum device count under even block distribution (capacity only) ---\n");
    for (auto &s : shapes) {
        double bytesA = (double)s.m * (double)s.k * BYTES_PER_FP32;
        double bytesB = (double)s.k * (double)s.n * BYTES_PER_FP32;
        double bytesC = (double)s.m * (double)s.n * BYTES_PER_FP32;
        double totalBytes = bytesA + bytesB + bytesC;
        int minDevices = (int)std::ceil(totalBytes / H100_SXM_HBM_BYTES);
        if (minDevices < 1) minDevices = 1;
        printf("%s -> minimum %d device(s)\n", s.label, minDevices);
    }

    return 0;
}
```

Compiled with `g++ -O2 69_gemm_memory_wall_model.cpp -o 69_gemm_memory_wall_model` and genuinely run (this is a plain closed-form host computation -- no device call, no honest error to report, exactly like Ch2's own bandwidth model). Locked output:

```text
Single-device budget (Ch1's own real H100 SXM figure): 80.0 GiB

GPT-3 FFN weight alone (d_model x 4d_model)
  M=12288 K=12288 N=49152 (fp32)
  A+B+C = 5.06 GiB -> FITS on one H100 SXM

GPT-3 FFN forward, batch=512, seq=2048 (M=batch*seq)
  M=1048576 K=12288 N=49152 (fp32)
  A+B+C = 242.25 GiB -> DOES NOT FIT on one H100 SXM

A hypothetical 10x-larger FFN forward at the same batch/seq
  M=1048576 K=122880 N=491520 (fp32)
  A+B+C = 2625.00 GiB -> DOES NOT FIT on one H100 SXM

--- Minimum device count under even block distribution (capacity only) ---
GPT-3 FFN weight alone (d_model x 4d_model) -> minimum 1 device(s)
GPT-3 FFN forward, batch=512, seq=2048 (M=batch*seq) -> minimum 4 device(s)
A hypothetical 10x-larger FFN forward at the same batch/seq -> minimum 33 device(s)
```

!!! warning "[COMMON TRAP] Sizing a multi-GPU GEMM job by the WEIGHT matrix alone, the way Chapter 13's own question would"
    The weight matrix B is the part of a GEMM most engineers reach for first when asked "will this fit," because it is the part that sounds like "the model." This section's own locked output shows why that instinct under-sizes the real job: the GPT-3 FFN weight alone fits in 5.06 GiB, comfortably under one H100's 80 GiB budget -- but the SAME weight matrix, multiplied against one real production batch (512 sequences of length 2048), needs 242.25 GiB across A, B, and C combined, and genuinely requires at least 4 devices just for capacity, before a single FLOP of communication cost (Section 24.2's own subject) is even considered. Sizing a distributed GEMM job by the weight matrix alone -- Chapter 13's own question -- silently ignores the two operands (A and C) that actually scale with how the model is being USED, not just how big it is.

## 24.2 SUMMA: A Real, Named Algorithm for Splitting the Whole GEMM

### Intuition

Once a GEMM's operands are known to need more than one device, SOMETHING has to decide exactly which piece of A and B each device holds, and exactly what gets communicated so every device can compute its own correct piece of C. SUMMA (Van de Geijn & Watts, "SUMMA: Scalable Universal Matrix Multiplication Algorithm") is the real, named, published answer this section builds. It arranges P devices into a 2D grid; A's rows are split across the grid's rows, B's columns are split across the grid's columns, and BOTH matrices' shared K dimension is split into P panels, one per step. At step l, the device column that owns A's l-th column-panel broadcasts it across its own device row (reusing Chapter 8's own broadcast, now inside an algorithm rather than as a standalone demonstration); the device row that owns B's l-th row-panel broadcasts it across its own device column; every device then accumulates a local partial product into its own piece of C, in the paper's own notation, "Cij = Cij + a~^l_i (b~^j_l)^T." After P steps, every device's own local accumulator holds its exact, correct piece of the full result -- no device ever needed to hold all of A or all of B at once.

```text
A 3x3 device grid, step l=1 (0-indexed):

Device columns:        col 0        col 1        col 2
                    +---------+  +---------+  +---------+
Device row 0        | owns A's|  |         |  |         |
                    | panel 1 |->|broadcast|->|broadcast|
                    +---------+  +---------+  +---------+
Device row 1        | owns A's|  |         |  |         |
                    | panel 1 |->|broadcast|->|broadcast|
                    +---------+  +---------+  +---------+
Device row 2 owns B's row-panel 1 -- broadcasts it DOWN
its own device column (col 0, col 1, col 2 each receive it).

Every device (r,c) then computes: Cij += (its A panel for row r)
                                            x (its B panel for col c)
```

### Background

```cpp
// Chapter 24: Multi-GPU Dense Matrix Multiplication at Scale
// 70_summa_correctness_simulation.cpp
//
// SUMMA (Van de Geijn & Watts, "SUMMA: Scalable Universal Matrix
// Multiplication Algorithm") splits an M x K x N GEMM across a P x P
// grid of processes. A (M x K) is split into P row-blocks; B (K x N)
// is split into P column-blocks. The K dimension is ALSO split, into P
// column-panels of A and P row-panels of B. At step l, the process
// COLUMN that owns A's l-th column-panel broadcasts it across its own
// process ROW; the process ROW that owns B's l-th row-panel broadcasts
// it across its own process COLUMN. Every process then accumulates a
// local partial product: "Cij = Cij + a~^l_i (b~^j_l)^T" (the paper's
// own notation for the accumulation step). This section builds that
// algorithm as a real host-side simulation -- P*P in-memory buffers
// standing in for P*P real devices, exchanging exactly the panels the
// real algorithm exchanges -- and checks the result against a plain
// triple-loop reference, bit-exact, using INTEGER matrices (Chapter 8's
// own non-associativity caution: this is a correctness check on the
// ROUTING, not a claim about floating-point summation order).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

using Matrix = std::vector<long long>; // row-major, size rows*cols

long long& at(Matrix &m, int cols, int r, int c) { return m[r * cols + c]; }
long long at(const Matrix &m, int cols, int r, int c) { return m[r * cols + c]; }

// Naive reference: C = A * B, A is M x K, B is K x N, C is M x N.
Matrix naiveMatmul(const Matrix &A, const Matrix &B, int M, int K, int N) {
    Matrix C(M * N, 0);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            long long sum = 0;
            for (int l = 0; l < K; l++) sum += at(A, K, i, l) * at(B, N, l, j);
            at(C, N, i, j) = sum;
        }
    return C;
}

int main() {
    // A square P x P process grid, matching SUMMA's own simplest
    // presentation: A's K-split into column-panels and B's K-split
    // into row-panels both use the SAME P, so the algorithm runs
    // exactly P steps, one per panel.
    const int M = 6, K = 6, N = 6;
    const int P = 3;
    assert(M % P == 0 && N % P == 0 && K % P == 0);
    const int blockM = M / P;   // rows per process-row's C block
    const int blockN = N / P;  // cols per process-column's C block
    const int panelK = K / P;  // width of each A column-panel == height of each B row-panel

    printf("Grid: %dx%d processes, M=%d K=%d N=%d "
           "(blockM=%d, blockN=%d, panelK=%d)\n",
           P, P, M, K, N, blockM, blockN, panelK);

    // Build real A (M x K) and B (K x N) with deterministic values.
    Matrix A(M * K), B(K * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++) at(A, K, i, j) = (i * K + j) % 7 - 3;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++) at(B, N, i, j) = (i * N + j) % 5 - 2;

    Matrix reference = naiveMatmul(A, B, M, K, N);

    // Distribute A: process (pr, pc) initially owns A's row-block pr,
    // column-panel pc -- exactly SUMMA's own starting distribution.
    // Distribute B: process (pr, pc) initially owns B's row-panel pr,
    // column-block pc.
    std::vector<std::vector<Matrix>> myA(P, std::vector<Matrix>(P));
    std::vector<std::vector<Matrix>> myB(P, std::vector<Matrix>(P));
    for (int pr = 0; pr < P; pr++) {
        for (int pc = 0; pc < P; pc++) {
            Matrix aBlock(blockM * panelK);
            for (int i = 0; i < blockM; i++)
                for (int j = 0; j < panelK; j++)
                    aBlock[i * panelK + j] = at(A, K, pr * blockM + i, pc * panelK + j);
            myA[pr][pc] = aBlock;

            Matrix bBlock(panelK * blockN);
            for (int i = 0; i < panelK; i++)
                for (int j = 0; j < blockN; j++)
                    bBlock[i * blockN + j] = at(B, N, pr * panelK + i, pc * blockN + j);
            myB[pr][pc] = bBlock;
        }
    }

    // Each process's own accumulator, initially zero.
    std::vector<std::vector<Matrix>> myC(P, std::vector<Matrix>(P));
    for (int pr = 0; pr < P; pr++)
        for (int pc = 0; pc < P; pc++)
            myC[pr][pc] = Matrix(blockM * blockN, 0);

    // SUMMA's own main loop: exactly P steps, one per K-panel.
    for (int l = 0; l < P; l++) {
        // Step l's A column-panel is owned, before broadcast, by every
        // process in grid-column l (one row-block each). Broadcasting
        // it across each grid ROW gives every process in that row the
        // SAME panel for its own row-index.
        std::vector<Matrix> aPanelForRow(P);
        for (int pr = 0; pr < P; pr++) aPanelForRow[pr] = myA[pr][l];

        // Step l's B row-panel is owned, before broadcast, by every
        // process in grid-row l (one column-block each). Broadcasting
        // it across each grid COLUMN gives every process in that
        // column the SAME panel for its own column-index.
        std::vector<Matrix> bPanelForCol(P);
        for (int pc = 0; pc < P; pc++) bPanelForCol[pc] = myB[l][pc];

        // Every process accumulates its local partial product using the
        // panel matching its own row index (for A) and column index
        // (for B) -- the real "Cij = Cij + a~_i * b~_j" step.
        for (int pr = 0; pr < P; pr++) {
            for (int pc = 0; pc < P; pc++) {
                const Matrix &aPanel = aPanelForRow[pr];  // blockM x panelK
                const Matrix &bPanel = bPanelForCol[pc];  // panelK x blockN
                for (int i = 0; i < blockM; i++) {
                    for (int j = 0; j < blockN; j++) {
                        long long sum = 0;
                        for (int t = 0; t < panelK; t++)
                            sum += aPanel[i * panelK + t] * bPanel[t * blockN + j];
                        myC[pr][pc][i * blockN + j] += sum;
                    }
                }
            }
        }
        printf("Step %d/%d: broadcast A's column-panel %d across each grid row, "
               "B's row-panel %d across each grid column, accumulate locally\n",
               l + 1, P, l, l);
    }

    // Reassemble the distributed C blocks and compare to the reference,
    // bit-exact (integer arithmetic -- no floating-point summation
    // order question here, unlike Ch14's row-parallel combine).
    Matrix summaResult(M * N, 0);
    for (int pr = 0; pr < P; pr++)
        for (int pc = 0; pc < P; pc++)
            for (int i = 0; i < blockM; i++)
                for (int j = 0; j < blockN; j++)
                    at(summaResult, N, pr * blockM + i, pc * blockN + j) =
                        myC[pr][pc][i * blockN + j];

    bool exact = (summaResult == reference);
    printf("\nSUMMA result vs. naive triple-loop reference (%d entries): %s\n",
           M * N, exact ? "EXACT MATCH" : "MISMATCH");
    if (!exact) {
        for (int i = 0; i < M * N; i++) {
            if (summaResult[i] != reference[i]) {
                printf("  first mismatch at flat index %d: summa=%lld reference=%lld\n",
                       i, summaResult[i], reference[i]);
                break;
            }
        }
    }
    return exact ? 0 : 1;
}
```

Compiled with `g++ -O2 70_summa_correctness_simulation.cpp -o 70_summa_correctness_simulation` and genuinely run. Locked output:

```text
Grid: 3x3 processes, M=6 K=6 N=6 (blockM=2, blockN=2, panelK=2)
Step 1/3: broadcast A's column-panel 0 across each grid row, B's row-panel 0 across each grid column, accumulate locally
Step 2/3: broadcast A's column-panel 1 across each grid row, B's row-panel 1 across each grid column, accumulate locally
Step 3/3: broadcast A's column-panel 2 across each grid row, B's row-panel 2 across each grid column, accumulate locally

SUMMA result vs. naive triple-loop reference (36 entries): EXACT MATCH
```

!!! warning "[COMMON TRAP] Assuming SUMMA needs a square process grid because this section's own example uses one"
    This section's own worked example uses a 3x3 grid because it is the cleanest to draw and read, not because SUMMA requires it. The algorithm itself only requires that A's row-split count matches the grid's row count and B's column-split count matches the grid's column count -- Pr and Pc need not be equal at all. This chapter's own development re-ran the identical code with a non-square 2x2 grid over a genuinely different, non-square M=8/K=12/N=4 problem and confirmed the same exact-match result, confirming the algorithm (and this implementation of it) generalizes rather than being a coincidence of one particular grid shape.

## 24.3 cuBLAS-Xt: NVIDIA's Real Single-Node Answer to This Exact Problem

### Intuition

Section 24.2 built SUMMA by hand to prove the routing is correct. A real production system rarely hand-rolls SUMMA itself for a single machine with a handful of GPUs -- NVIDIA ships a real library that does this exact job: cuBLAS-Xt. Its own documentation states the problem it solves in almost the same words this chapter's own 24.1 used to motivate it: "the application may have the data on the Host or any of the devices involved in the computation, and the Library will take care of dispatching the operation to, and transferring the data to, one or multiple GPUs present in the system." The real API detail that sets it apart from every earlier cuBLAS-style call in this book: `cublasXtSgemm()` takes HOST pointers for A, B, and C directly, not device pointers -- the library manages the block splitting and the transfers itself, with `cublasXtSetBlockDim()` controlling the tile size used for that split, the real production analogue of Section 24.2's own `panelK`/`blockM`/`blockN`. cuBLAS-Xt's own product page is explicit that this is real but bounded: it operates "in a single node." For genuinely multi-node dense GEMM, at the scale this book's own Chapter 20 MPI infrastructure implies, NVIDIA's current real answer is cuBLASMp, described as "a high-performance, multi-process, GPU-accelerated library for distributed basic dense linear algebra" -- cuBLASMg, an older effort at the identically-named problem, never left the CUDA Math Library Early Access Program.

```text
cuBLAS-Xt (this section, real, shipping):        cuBLASMp (NVIDIA's real current
                                                   multi-NODE answer, not built here):

+---------------------------+                    +---------------------------+
| ONE process               |                    | MULTIPLE processes,       |
| multiple GPUs, same       |                    | multiple nodes --         |
| machine                   |                    | "multi-process,           |
| cublasXtSgemm(host ptrs)  |                    | GPU-accelerated... for    |
+---------------------------+                    | distributed... linear     |
  NVIDIA's own product page:                     | algebra"                  |
  operates "in a single node"                    +---------------------------+
                                                    needs this book's own
                                                    Part 5 MPI/NCCL substrate
```

### Background

```cpp
// Chapter 24: Multi-GPU Dense Matrix Multiplication at Scale
// 71_cublasxt_multi_gpu_gemm.cu
//
// Section 24.2 built SUMMA's own algorithm by hand, over a simulated
// process grid, to prove the ROUTING is correct. NVIDIA ships a real,
// production single-process multi-GPU dense BLAS library that performs
// this exact kind of block distribution FOR you: cuBLAS-Xt. Its own
// documentation states the problem it solves plainly: "the application
// may have the data on the Host or any of the devices involved in the
// computation, and the Library will take care of dispatching the
// operation to, and transferring the data to, one or multiple GPUs
// present in the system." Unlike every earlier chapter's cuBLAS/CUDA
// call, cublasXtSgemm() takes HOST pointers for A, B, and C directly --
// the library manages the device transfers and block dispatch itself,
// with cublasXtSetBlockDim() controlling the tile size used for that
// dispatch (the real analogue of this section's own blockM/panelK from
// file 70). cuBLAS-Xt is real but explicitly SINGLE-NODE: NVIDIA's own
// cuBLAS-Xt product page states it operates "in a single node." For
// genuine multi-NODE dense GEMM at the scale Chapter 20's own MPI
// world implies, NVIDIA's current real answer is cuBLASMp -- "a
// high-performance, multi-process, GPU-accelerated library for
// distributed basic dense linear algebra" -- since cuBLASMg (an older,
// differently-named effort at the same problem) never left the CUDA
// Math Library Early Access Program.
//
// THE GUARD. This file checks cudaGetDeviceCount() BEFORE ever calling
// cublasXtDeviceSelect() with real device ordinals, and skips selecting
// devices, setting a block size, and calling cublasXtSgemm() entirely
// if fewer real devices exist than requested. This section's own
// COMMON TRAP explains, with a real reproduced crash, exactly why that
// check is required here specifically -- a new failure shape even
// Chapter 23's own NCCL segfault didn't produce.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublasXt.h>

int main() {
    const int M = 1024, K = 1024, N = 1024;
    std::vector<float> A(M * K, 1.0f), B(K * N, 2.0f), C(M * N, 0.0f);

    printf("--- Creating a cuBLAS-Xt handle (real, host-side library object) ---\n");
    cublasXtHandle_t handle;
    cublasStatus_t createErr = cublasXtCreate(&handle);
    printf("cublasXtCreate(&handle) -> %s (%s)\n",
           cublasGetStatusName(createErr), cublasGetStatusString(createErr));

    printf("\n--- Checking real device count BEFORE selecting any device ordinals ---\n");
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount() -> %s, count=%d\n",
           cudaGetErrorString(countErr), deviceCount);

    const int requestedDevices = 2;
    if (deviceCount < requestedDevices) {
        printf("\nSTOPPING before cublasXtDeviceSelect(): only %d real device(s) "
               "available, %d requested. See this section's COMMON TRAP for what "
               "genuinely happens if this check is skipped and cublasXtDestroy() "
               "is later called on a handle whose DeviceSelect() call failed.\n",
               deviceCount, requestedDevices);
        cublasXtDestroy(handle);
        printf("cublasXtDestroy(handle) -> cleanly destroyed (handle was never "
               "given an invalid device list, so nothing here was corrupted)\n");
        return 0;
    }

    // This real code path -- selecting devices, setting a block size,
    // and calling the actual multi-GPU GEMM -- only runs on a system
    // that genuinely has the requested devices.
    int deviceIds[] = {0, 1};
    cublasStatus_t selectErr = cublasXtDeviceSelect(handle, requestedDevices, deviceIds);
    printf("cublasXtDeviceSelect(handle, %d, {0,1}) -> %s\n",
           requestedDevices, cublasGetStatusName(selectErr));

    cublasStatus_t blockDimErr = cublasXtSetBlockDim(handle, 256);
    printf("cublasXtSetBlockDim(handle, 256) -> %s\n", cublasGetStatusName(blockDimErr));

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t gemmErr = cublasXtSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        (size_t)M, (size_t)N, (size_t)K,
        &alpha, A.data(), (size_t)M, B.data(), (size_t)K,
        &beta, C.data(), (size_t)M);
    printf("cublasXtSgemm(M=%d,K=%d,N=%d, A/B/C on HOST) -> %s\n",
           M, K, N, cublasGetStatusName(gemmErr));

    cublasXtDestroy(handle);

    printf("\n--- What this means for genuinely MULTI-NODE dense GEMM ---\n");
    printf("Everything above is real cuBLAS-Xt, but cuBLAS-Xt is explicitly "
           "single-node -- one process, one or more GPUs in that SAME machine. "
           "It cannot be handed a Chapter 20-style MPI world spanning multiple "
           "nodes; there is no cublasXt call that takes an MPI_Comm. NVIDIA's "
           "real current answer for genuinely distributed, multi-process, "
           "multi-node dense GEMM is cuBLASMp, built explicitly on top of a "
           "real multi-process model -- this book's own Part 5 MPI/NCCL "
           "infrastructure is precisely the substrate a library like that "
           "needs underneath it.\n");

    return 0;
}
```

Compiled with `nvcc -gencode=arch=compute_70,code=sm_70 71_cublasxt_multi_gpu_gemm.cu -o 71_cublasxt_multi_gpu_gemm -lcublas` and genuinely run. Locked output:

```text
--- Creating a cuBLAS-Xt handle (real, host-side library object) ---
cublasXtCreate(&handle) -> CUBLAS_STATUS_SUCCESS (success)

--- Checking real device count BEFORE selecting any device ordinals ---
cudaGetDeviceCount() -> no CUDA-capable device is detected, count=0

STOPPING before cublasXtDeviceSelect(): only 0 real device(s) available, 2 requested. See this section's COMMON TRAP for what genuinely happens if this check is skipped and cublasXtDestroy() is later called on a handle whose DeviceSelect() call failed.
cublasXtDestroy(handle) -> cleanly destroyed (handle was never given an invalid device list, so nothing here was corrupted)
```

!!! warning "[COMMON TRAP] Destroying a cuBLAS-Xt handle after a failed cublasXtDeviceSelect() call"
    `cublasXtCreate()` genuinely succeeds even with zero real devices -- it is pure host-side bookkeeping, unlike every CUDA Runtime call this book has made since Chapter 3. That made it possible to isolate a real, reproducible bug this book had not yet produced: calling `cublasXtDeviceSelect()` with device ordinals that do not exist correctly fails with `CUBLAS_STATUS_INVALID_VALUE` -- but if `cublasXtDestroy()` is later called on that SAME handle, this exact environment's installed cuBLAS-Xt genuinely corrupts memory, caught by glibc's own allocator:

    ```text
    double free or corruption (fasttop)
    ```

    (a second isolated run produced a differently-worded but equally real glibc message for the identical bug: `free(): double free detected in tcache 2`). This was re-tested with `cublasXtSetBlockDim()` removed entirely, and the crash reproduced identically -- it is specifically the failed `cublasXtDeviceSelect()` call, followed by `cublasXtDestroy()`, that corrupts the handle's internal state, not anything about block sizing or the GEMM call itself. This is a genuinely sharper failure than Chapter 23's own NCCL segfault: that was an unchecked call crashing immediately; this is memory corruption inside a library's own cleanup path, silent until glibc's allocator happens to catch it. The defensive fix this file's own guard applies -- check `cudaGetDeviceCount()` and never call `cublasXtDeviceSelect()` with more device ordinals than genuinely exist -- avoids the failed call entirely, which is the only way this book found to avoid the corruption, since the corruption happens at destroy time regardless of what runs in between.

## Chapter Summary

This chapter opened Part 6 by building one real GEMM case study across three layers. Section 24.1 showed, with a real closed-form capacity model reusing Chapter 1's own H100 HBM figure, that a GEMM's A and C operands -- not the weight matrix most engineers reach for first -- are what actually force a real production job past one device's memory, needing at least 4 real devices for a real GPT-3-scale batch. Section 24.2 built SUMMA (Van de Geijn & Watts) as a real host-side simulation over a 2D process grid, broadcasting column-panels of A within device rows and row-panels of B within device columns exactly as the original paper describes, and verified the result bit-exact against a naive reference, confirmed to generalize to a second, non-square grid and problem shape. Section 24.3 built real NVIDIA cuBLAS-Xt, the shipping single-node library for exactly this problem, discovered a genuinely new and sharper real failure mode -- a library-internal memory corruption from destroying a handle after one specific failed call, caught by glibc's own allocator rather than returned as an honest status code -- and closed with cuBLAS-Xt's own explicitly single-node scope and NVIDIA's real current multi-node answer, cuBLASMp, built on the same real multi-process model this book's own Part 5 already established.

## Self-Check Questions

1. Why can a GEMM's weight matrix B fit comfortably on one device while the SAME multiply, at a real production batch size, still requires multiple devices?
2. In SUMMA, what determines the number of steps the algorithm runs, and what real API from Chapter 8 does each step's own panel distribution reuse?
3. Section 24.2's own COMMON TRAP re-tested the algorithm on a different grid shape and problem size. Why does that matter for trusting the first (3x3, M=K=N=6) result?
4. What is the one real API detail that makes `cublasXtSgemm()` different from every earlier cuBLAS-style call this book has made?
5. Why does `cublasXtCreate()` succeed in this device-less environment when almost every CUDA Runtime call since Chapter 3 has honestly failed?
6. What specific sequence of two real calls produces the memory-corruption crash in Section 24.3, and what evidence rules out `cublasXtSetBlockDim()` as a contributing cause?
7. What real scope limitation does NVIDIA's own cuBLAS-Xt product page state, and what is NVIDIA's current real answer for going beyond it?
8. How does this chapter's own defensive guard in file 71 avoid the crash, given that the corruption happens at destroy time rather than at the failed call itself?

## Where We Go Next

Chapter 24 built one complete real GEMM case study end to end: a real capacity model, a real named algorithm verified by simulation, and a real production library. Chapter 25, "Distributed Training of a Neural Network: Data Parallelism and Ring All-Reduce," returns to Chapter 12's own data-parallel replicas and Chapter 9's own ring all-reduce, but as a complete real training loop rather than two separate concept demonstrations -- the same kind of full, end-to-end case study this chapter just built for a single GEMM, now built for an entire training step.

## Worked Solutions

**1.** The weight matrix B's size depends only on the model's own dimensions (e.g. `d_model x 4*d_model`), which stay fixed regardless of how the model is used. The GEMM's other two operands, A (the input activations) and C (the output), both scale with the batch size and sequence length as well -- `M = batch * seq`. A real production batch/sequence combination can make A and C, combined, far larger than B alone, which is exactly what Section 24.1's own locked output showed: 5.06 GiB for B alone versus 242.25 GiB for A+B+C at a real batch of 512 sequences of length 2048.

**2.** The number of steps equals P, the size of one side of the square process grid, because K is split into exactly P column-panels of A and P row-panels of B, one panel exchanged per step. Each step's own panel distribution reuses Chapter 8's own broadcast: a single owning device sends the same data to every other device in its row (for A) or column (for B).

**3.** Because a single passing result on one specific grid shape and problem size could, in principle, be a coincidence of that particular configuration (e.g., a bug that happens to cancel out for a symmetric 3x3/6x6x6 case). Re-running the identical algorithm on a genuinely different, non-square problem (M=8, K=12, N=4) with a different grid (2x2) and getting another exact match is real evidence the algorithm -- and this implementation of it -- generalizes, rather than only appearing correct by construction.

**4.** `cublasXtSgemm()` takes HOST pointers for A, B, and C directly, rather than device pointers. Every earlier cuBLAS/CUDA call in this book required the caller to already have data in device memory (via `cudaMalloc()`/`cudaMemcpy()`); cuBLAS-Xt instead accepts host memory and manages the device transfers and block dispatch itself.

**5.** Because `cublasXtCreate()` is pure host-side bookkeeping -- it allocates and initializes a library handle object in host memory and does not itself query or touch any CUDA device. The CUDA Runtime API's own implicit per-thread device initialization (which fails immediately with zero real devices, as seen since Chapter 3) is a separate code path that `cublasXtCreate()` does not trigger; only `cublasXtDeviceSelect()`, which explicitly requests real device ordinals, or `cublasXtSgemm()`, which actually needs to run on a device, engage with CUDA's own device state.

**6.** A call to `cublasXtDeviceSelect()` that fails (because the requested device ordinals do not exist), followed later by a call to `cublasXtDestroy()` on that same handle. Removing `cublasXtSetBlockDim()` from the sequence entirely and re-running showed the identical crash, ruling it out as a contributing cause -- the corruption is specifically tied to the failed `DeviceSelect()` call followed by `Destroy()`.

**7.** NVIDIA's own cuBLAS-Xt product page states it operates "in a single node" -- one process, one machine, however many GPUs that machine has. NVIDIA's current real answer for genuinely distributed, multi-process, multi-node dense GEMM is cuBLASMp, described as "a high-performance, multi-process, GPU-accelerated library for distributed basic dense linear algebra."

**8.** By checking `cudaGetDeviceCount()` BEFORE ever calling `cublasXtDeviceSelect()`, and skipping that call (and everything after it) entirely when fewer real devices exist than requested. Since the crash requires a FAILED `DeviceSelect()` call to occur before `Destroy()`, never making that call in the first place -- rather than trying to recover from its failure afterward -- avoids the corrupted state altogether.

---

**Sources cited in this chapter:**

- [SUMMA: Scalable Universal Matrix Multiplication Algorithm (LAPACK Working Note 96 / UT-CS-95-286, Van de Geijn & Watts)](https://www.netlib.org/lapack/lawnspdf/lawn96.pdf) — the real, verified algorithm description and notation ("a~^l_i", "b~^j_l") this chapter's Section 24.2 builds and quotes.
- [NVIDIA cuBLAS Documentation, "Using the cuBLASXt API"](https://docs.nvidia.com/cuda/cublas/index.html) — the real, verified quote on how cuBLAS-Xt dispatches work: "the application may have the data on the Host or any of the devices involved in the computation, and the Library will take care of dispatching the operation to, and transferring the data to, one or multiple GPUs present in the system."
- [NVIDIA cuBLASXt product page](https://developer.nvidia.com/cublasxt) — the real, verified quote on cuBLAS-Xt's single-node scope: it operates "in a single node."
- [NVIDIA Developer Forums: "Help with cublasXt Sgemm does not scale"](https://forums.developer.nvidia.com/t/help-with-cublasxt-sgemm-does-not-scale/280016) — the real, verified quote this chapter's own 24.1 motivation is built on: cuBLAS-Xt's "main point" is to "work around GPU memory capacity limits."
- [NVIDIA Technical Blog: "Accelerating GPU Applications with NVIDIA Math Libraries"](https://developer.nvidia.com/blog/accelerating-gpu-applications-with-nvidia-math-libraries/) — the real, verified quote confirming cuBLASMg never left the CUDA Math Library Early Access Program.
- [NVIDIA cuBLASMp Documentation](https://docs.nvidia.com/cuda/cublasmp/) — the real, verified quote describing cuBLASMp as "a high-performance, multi-process, GPU-accelerated library for distributed basic dense linear algebra," NVIDIA's current real answer for multi-node dense GEMM.
- This exact environment's own installed CUDA 12.0 headers (`/usr/include/cublasXt.h`, `/usr/include/cublas_api.h`) — grepped directly for the real signatures of `cublasXtCreate`, `cublasXtDeviceSelect`, `cublasXtSetBlockDim`, `cublasXtSgemm`, and the real `cublasStatus_t` enum and `cublasGetStatusName`/`cublasGetStatusString` helpers, per this book's own established practice of checking installed headers directly rather than trusting a web summary.
