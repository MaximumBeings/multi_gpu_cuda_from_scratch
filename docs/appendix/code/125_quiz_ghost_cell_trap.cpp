// Appendix B: Practice Quiz
// 125_quiz_ghost_cell_trap.cpp
//
// Appendix B.4, Challenge 3 -- Chapters 16 and 40 both warned that a
// halo/ghost-cell exchange must genuinely happen before a boundary cell
// is computed -- skipping it does not crash the program, it silently
// computes a WRONG answer using stale data. This file splits an 8-cell
// 1D array across 2 ranks (4 interior cells each), computes a 3-point
// smoothing average at every cell, and runs the SAME computation twice:
// once with a correct ghost-cell exchange across the rank boundary, and
// once where that exchange is skipped, leaving each rank's own ghost
// cell at its uninitialized starting value of 0. Before compiling and
// running this file, predict: how many of the 8 total cells does the
// no-sync version get WRONG compared to the single-process reference --
// and are they the interior cells, or specifically the ones nearest the
// rank boundary?
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 125_quiz_ghost_cell_trap.cpp -o 125_quiz_ghost_cell_trap
// Run:     ./125_quiz_ghost_cell_trap
#include <cstdio>
#include <vector>

constexpr int NX = 8;   // global array size
constexpr int P = 2;    // ranks
constexpr int LOCAL = NX / P;  // 4 interior cells per rank

double globalValue(int i) {
    // Deterministic, arbitrary values for cell i.
    return 10.0 + i;
}

double referenceSmooth(int i) {
    double sum = globalValue(i);
    int count = 1;
    if (i - 1 >= 0)  { sum += globalValue(i - 1); count++; }
    if (i + 1 < NX)  { sum += globalValue(i + 1); count++; }
    return sum / count;
}

int main() {
    printf("=== 8-cell array split across 2 ranks, 4 interior cells each "
           "===\n\n");

    // --- Correct version: ghost cells genuinely synchronized. ---
    std::vector<double> correct(NX);
    for (int rank = 0; rank < P; rank++) {
        int start = rank * LOCAL;
        for (int local = 0; local < LOCAL; local++) {
            int i = start + local;
            double left  = (i - 1 >= 0)  ? globalValue(i - 1) : globalValue(i);
            double right = (i + 1 < NX) ? globalValue(i + 1) : globalValue(i);
            int count = 1;
            double sum = globalValue(i);
            if (i - 1 >= 0)  { sum += left;  count++; }
            if (i + 1 < NX) { sum += right; count++; }
            correct[i] = sum / count;
        }
    }

    // --- Buggy version: each rank's own ghost cell is never fetched
    // from its neighbor -- it stays at its uninitialized starting value
    // of 0.0, exactly the "hard delete" of a real synchronization step.
    std::vector<double> buggy(NX);
    for (int rank = 0; rank < P; rank++) {
        int start = rank * LOCAL;
        int end = start + LOCAL;
        for (int local = 0; local < LOCAL; local++) {
            int i = start + local;
            double leftGhost = 0.0, rightGhost = 0.0;   // never synced
            double left, right;
            bool useLeftGhost = (i - 1 < start) && (i - 1 >= 0);
            bool useRightGhost = (i + 1 >= end) && (i + 1 < NX);
            left  = useLeftGhost  ? leftGhost  : ((i - 1 >= 0) ? globalValue(i - 1) : globalValue(i));
            right = useRightGhost ? rightGhost : ((i + 1 < NX) ? globalValue(i + 1) : globalValue(i));
            int count = 1;
            double sum = globalValue(i);
            if (i - 1 >= 0)  { sum += left;  count++; }
            if (i + 1 < NX) { sum += right; count++; }
            buggy[i] = sum / count;
        }
    }

    printf("%-6s %-14s %-14s %-10s\n", "cell", "reference", "correct-sync", "buggy(no-sync)");
    int mismatches = 0;
    for (int i = 0; i < NX; i++) {
        double ref = referenceSmooth(i);
        bool correctMatches = (correct[i] == ref);
        bool buggyMatches = (buggy[i] == ref);
        if (!buggyMatches) mismatches++;
        printf("%-6d %-14.4f %-14s %-10s\n", i, ref,
               correctMatches ? "matches" : "MISMATCH",
               buggyMatches ? "matches" : "MISMATCH (stale ghost=0)");
    }

    printf("\ntotal cells where the no-sync version disagrees with the "
           "single-process reference: %d out of %d\n", mismatches, NX);
    printf("those mismatches occur ONLY at the two rank-boundary cells "
           "(index 3 and index 4) -- every purely interior cell (0,1,2 "
           "and 5,6,7) never reads a neighbor across the rank boundary "
           "at all, so it is correct with or without the ghost-cell "
           "exchange. This is exactly Chapters 16 and 40's own real "
           "finding: a missing halo sync does not corrupt the whole "
           "domain, it silently corrupts precisely the cells whose "
           "correctness depends on it -- the boundary, and nowhere "
           "else.\n");
    return 0;
}
