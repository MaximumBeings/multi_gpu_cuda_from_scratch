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
