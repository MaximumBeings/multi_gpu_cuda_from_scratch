// Appendix E: Profiling and Benchmarking Multi-GPU Communication
// 132_round_count_vs_timing.cpp
//
// Appendix E.1 -- this book has never reported a wall-clock number as
// evidence that one collective strategy beats another (see
// getting-started.md's own honesty discipline). What it has used
// instead, since Chapter 9, is a deterministic, hardware-independent
// closed-form formula: Chapter 9's own real round-count result, a naive
// direct-send all-reduce needs 2(N-1) message ROUNDS while a real ring
// all-reduce needs only 2(N-1)/N times the data volume moved per GPU.
// This file makes that comparison directly, with ZERO timing calls
// anywhere -- computing, for a range of real GPU counts, exactly how
// many multiples of a GPU's own local buffer size each strategy moves.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 132_round_count_vs_timing.cpp -o 132_round_count_vs_timing
// Run:     ./132_round_count_vs_timing
#include <cstdio>
#include <vector>

int main() {
    printf("=== Section E.1: comparing two all-reduce strategies by DATA VOLUME MOVED, "
           "not wall-clock time ===\n\n");

    printf("Chapter 9's own real formulas, reused here directly with no clock involved:\n");
    printf("  naive (direct-send) all-reduce: each of N GPUs sends its own buffer to\n");
    printf("    every other GPU -- (N-1) sends per GPU, each of the FULL buffer size K\n");
    printf("  ring all-reduce:                two phases (reduce-scatter + all-gather),\n");
    printf("    2(N-1) total steps, but each step moves only K/N of the buffer\n\n");

    printf("%-6s %-28s %-28s %-10s\n", "N", "naive: volume/GPU (x K)", "ring: volume/GPU (x K)", "ring/naive");
    std::vector<int> ns = {2, 4, 8, 16, 32, 64};
    for (int n : ns) {
        double naive_volume = (double)(n - 1);              // (N-1) full-size sends
        double ring_volume = 2.0 * (n - 1) / n;              // Chapter 9's own real formula
        double ratio = ring_volume / naive_volume;
        printf("%-6d %-28.4f %-28.4f %-10.4f\n", n, naive_volume, ring_volume, ratio);
    }

    printf("\nnone of the numbers above came from running anything -- they are all read\n");
    printf("directly off Chapter 9's own closed-form formulas, computed identically and\n");
    printf("reproducibly on ANY machine, ANY run, ANY compiler, unlike a wall-clock\n");
    printf("measurement of an actual NCCL call would be.\n");

    // Self-check: ring's own volume NEVER exceeds naive's volume for any N, and is
    // STRICTLY less for every N > 2 -- at N=2 the two strategies are identical (one
    // GPU sending its whole buffer to the other, however you name the algorithm),
    // so N=2 is an expected, honest equality, not a bug. Confirms Chapter 9's own
    // real "ring moves no more data per GPU than naive, and strictly less once
    // there are more than two GPUs" finding.
    bool ok = true;
    for (int n : ns) {
        double naive_volume = (double)(n - 1);
        double ring_volume = 2.0 * (n - 1) / n;
        if (ring_volume > naive_volume) ok = false;
        if (n > 2 && !(ring_volume < naive_volume)) ok = false;
    }
    double ring_at_64 = 2.0 * 63 / 64;
    printf("\nself-check: ring's own volume-per-GPU never exceeds naive's at any tested N,\n");
    printf("and is strictly less for every N > 2 (approaching but never reaching 2x K as N\n");
    printf("grows large -- %.4f at N=64, vs naive's 63x K at the same N); N=2 is the one\n",
           ring_at_64);
    printf("honest exception, where both strategies reduce to the same single transfer: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
