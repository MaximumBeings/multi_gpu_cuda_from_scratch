// Chapter 24: Multi-GPU Dense Matrix Multiplication at Scale
// 70_summa_correctness_simulation.cpp
//
// SUMMA (Van de Geijn & Watts, "SUMMA: Scalable Universal Matrix
// Multiplication Algorithm") splits an M x K x N GEMM across a Pr x Pc
// grid of processes. A (M x K) is split into Pr row-blocks and Pc
// column-blocks; B (K x N) is split into Pr row-blocks and Pc
// column-blocks; process (pr, pc) owns block (pr, pc) of each. The
// algorithm then walks the shared K dimension in panels of width kb
// (any width dividing both K/Pc and K/Pr). At step l, the process
// COLUMN that owns A's l-th column-panel broadcasts it across its own
// process ROW; the process ROW that owns B's l-th row-panel broadcasts
// it across its own process COLUMN. Every process then accumulates a
// local partial product: "Cij = Cij + a~^l_i (b~^j_l)^T" (the paper's
// own notation for the accumulation step). On a square P x P grid with
// kb = K/P this is exactly P steps, panel l owned by grid column l and
// grid row l; on a non-square grid, A's and B's panel owners advance at
// DIFFERENT rates, which is the case worth testing separately. This
// section builds that algorithm as a real host-side simulation -- Pr*Pc
// in-memory buffers standing in for Pr*Pc real devices, exchanging
// exactly the panels the real algorithm exchanges -- and checks the
// result against a plain triple-loop reference, bit-exact, using
// INTEGER matrices (Chapter 8's own non-associativity caution: this is
// a correctness check on the ROUTING, not a claim about floating-point
// summation order).
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

using Matrix = std::vector<long long>; // row-major, size rows*cols

long long& at(Matrix &m, int cols, int r, int c) { return m[r * cols + c]; }
long long at(const Matrix &m, int cols, int r, int c) { return m[r * cols + c]; }

int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }

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

// Runs SUMMA on a Pr x Pc grid and checks it against the reference.
bool runSumma(int M, int K, int N, int Pr, int Pc) {
    assert(M % Pr == 0 && N % Pc == 0 && K % Pc == 0 && K % Pr == 0);
    const int blockM = M / Pr;      // rows of A / C per grid row
    const int blockN = N / Pc;      // cols of B / C per grid column
    const int aColsK = K / Pc;      // K-width of each process's A block
    const int bRowsK = K / Pr;      // K-height of each process's B block
    const int kb = gcd(aColsK, bRowsK); // panel width: divides both
    const int steps = K / kb;

    printf("Grid: %dx%d processes, M=%d K=%d N=%d "
           "(blockM=%d, blockN=%d, A block K-width=%d, B block K-height=%d, "
           "panel kb=%d, %d steps)\n",
           Pr, Pc, M, K, N, blockM, blockN, aColsK, bRowsK, kb, steps);

    // Build real A (M x K) and B (K x N) with deterministic values.
    Matrix A(M * K), B(K * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++) at(A, K, i, j) = (i * K + j) % 7 - 3;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++) at(B, N, i, j) = (i * N + j) % 5 - 2;

    Matrix reference = naiveMatmul(A, B, M, K, N);

    // Distribute A: process (pr, pc) owns A's row-block pr, column-block
    // pc (blockM x aColsK). Distribute B: process (pr, pc) owns B's
    // row-block pr, column-block pc (bRowsK x blockN).
    std::vector<std::vector<Matrix>> myA(Pr, std::vector<Matrix>(Pc));
    std::vector<std::vector<Matrix>> myB(Pr, std::vector<Matrix>(Pc));
    for (int pr = 0; pr < Pr; pr++) {
        for (int pc = 0; pc < Pc; pc++) {
            Matrix aBlock(blockM * aColsK);
            for (int i = 0; i < blockM; i++)
                for (int j = 0; j < aColsK; j++)
                    aBlock[i * aColsK + j] = at(A, K, pr * blockM + i, pc * aColsK + j);
            myA[pr][pc] = aBlock;

            Matrix bBlock(bRowsK * blockN);
            for (int i = 0; i < bRowsK; i++)
                for (int j = 0; j < blockN; j++)
                    bBlock[i * blockN + j] = at(B, N, pr * bRowsK + i, pc * blockN + j);
            myB[pr][pc] = bBlock;
        }
    }

    // Each process's own accumulator, initially zero.
    std::vector<std::vector<Matrix>> myC(Pr, std::vector<Matrix>(Pc));
    for (int pr = 0; pr < Pr; pr++)
        for (int pc = 0; pc < Pc; pc++)
            myC[pr][pc] = Matrix(blockM * blockN, 0);

    // SUMMA's own main loop: one step per K-panel of width kb.
    for (int l = 0; l < steps; l++) {
        const int k0 = l * kb;              // global K offset of this panel
        const int ownerCol = k0 / aColsK;   // grid column holding A's panel
        const int aOff = k0 % aColsK;       // offset inside that A block
        const int ownerRow = k0 / bRowsK;   // grid row holding B's panel
        const int bOff = k0 % bRowsK;       // offset inside that B block

        // Grid column ownerCol broadcasts A's panel across each grid
        // ROW: every process in row pr receives the same blockM x kb
        // slice for its own row-block.
        std::vector<Matrix> aPanelForRow(Pr, Matrix(blockM * kb));
        for (int pr = 0; pr < Pr; pr++)
            for (int i = 0; i < blockM; i++)
                for (int t = 0; t < kb; t++)
                    aPanelForRow[pr][i * kb + t] = myA[pr][ownerCol][i * aColsK + aOff + t];

        // Grid row ownerRow broadcasts B's panel down each grid COLUMN.
        std::vector<Matrix> bPanelForCol(Pc, Matrix(kb * blockN));
        for (int pc = 0; pc < Pc; pc++)
            for (int t = 0; t < kb; t++)
                for (int j = 0; j < blockN; j++)
                    bPanelForCol[pc][t * blockN + j] = myB[ownerRow][pc][(bOff + t) * blockN + j];

        // Every process accumulates its local partial product using the
        // panel matching its own row index (for A) and column index
        // (for B) -- the real "Cij = Cij + a~_i * b~_j" step.
        for (int pr = 0; pr < Pr; pr++) {
            for (int pc = 0; pc < Pc; pc++) {
                const Matrix &aPanel = aPanelForRow[pr];  // blockM x kb
                const Matrix &bPanel = bPanelForCol[pc];  // kb x blockN
                for (int i = 0; i < blockM; i++) {
                    for (int j = 0; j < blockN; j++) {
                        long long sum = 0;
                        for (int t = 0; t < kb; t++)
                            sum += aPanel[i * kb + t] * bPanel[t * blockN + j];
                        myC[pr][pc][i * blockN + j] += sum;
                    }
                }
            }
        }
        printf("Step %d/%d: K-panel %d -- A's panel from grid column %d "
               "broadcast across each grid row, B's panel from grid row %d "
               "broadcast down each grid column, accumulate locally\n",
               l + 1, steps, l, ownerCol, ownerRow);
    }

    // Reassemble the distributed C blocks and compare to the reference,
    // bit-exact (integer arithmetic -- no floating-point summation
    // order question here, unlike Ch14's row-parallel combine).
    Matrix summaResult(M * N, 0);
    for (int pr = 0; pr < Pr; pr++)
        for (int pc = 0; pc < Pc; pc++)
            for (int i = 0; i < blockM; i++)
                for (int j = 0; j < blockN; j++)
                    at(summaResult, N, pr * blockM + i, pc * blockN + j) =
                        myC[pr][pc][i * blockN + j];

    bool exact = (summaResult == reference);
    printf("SUMMA result vs. naive triple-loop reference (%d entries): %s\n\n",
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
    return exact;
}

int main() {
    // 1. A square 3x3 grid, SUMMA's own simplest presentation.
    bool ok = runSumma(6, 6, 6, 3, 3);
    // 2. A genuinely NON-square 2x3 grid (Pr != Pc) on a non-square
    //    problem: A's and B's panel owners advance at different rates.
    ok = runSumma(8, 12, 6, 2, 3) && ok;
    // 3. The transposed shape, 3x2, for good measure.
    ok = runSumma(6, 12, 8, 3, 2) && ok;
    return ok ? 0 : 1;
}
