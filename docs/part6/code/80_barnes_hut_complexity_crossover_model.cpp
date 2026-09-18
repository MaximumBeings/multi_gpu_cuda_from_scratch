// Chapter 27: N-Body Simulation at Scale
// 80_barnes_hut_complexity_crossover_model.cpp
//
// Sections 27.1 and 27.2 both showed all-pairs N-body force calculation
// getting worse at scale: O(N^2) compute, and per-GPU communication
// that stays close to the FULL dataset size no matter how many GPUs
// are added. Barnes & Hut's own real, named 1986 Nature paper offers
// the real algorithmic answer: a "tree-structured hierarchical
// subdivision of space into cubic cells, each of which is recursively
// divided into eight subcells whenever more than one particle is found
// to occupy the same cell," which lets distant groups of bodies be
// approximated as single aggregate sources instead of visited
// individually -- turning O(N^2) into O(N log N). This section builds
// this chapter's own closed-form operation-count model of that
// crossover (not a runtime measurement), and cites REAL measured
// numbers for a REAL GPU Barnes-Hut implementation (Burtscher &
// Pingali) to show the algorithmic idea also holds up on real
// hardware -- kept clearly separate from this section's own model, the
// same discipline Chapter 25 and Chapter 26 both already applied to
// their own closed-form communication-cost sections.
#include <cstdio>
#include <cmath>
#include <cstdint>

int main() {
    printf("--- This chapter's own worked model: O(N^2) vs O(N log N) "
           "operation counts (NOT a runtime measurement) ---\n");
    printf("%-12s %-16s %-16s %-16s\n", "N", "N^2 ops", "N*log2(N) ops",
           "ratio N/log2(N)");
    const uint64_t Ns[] = {1000ULL, 10000ULL, 100000ULL, 1000000ULL,
                            5000000ULL, 50000000ULL};
    for (uint64_t N : Ns) {
        double nSquared = (double)N * (double)N;
        double log2N = std::log2((double)N);
        double nLogN = (double)N * log2N;
        double ratio = (double)N / log2N;
        printf("%-12llu %-16.3e %-16.3e %-16.1fx\n",
               (unsigned long long)N, nSquared, nLogN, ratio);
    }
    printf("\nAs N grows, the ratio N/log2(N) grows WITHOUT BOUND -- "
           "all-pairs' own O(N^2) cost doesn't just stay worse than "
           "Barnes-Hut's O(N log N), it gets combinatorially WORSE, "
           "relative to Barnes-Hut, the larger the simulation gets. This "
           "compounds Section 27.2's own finding that all-pairs' "
           "per-GPU COMMUNICATION also does not improve by adding more "
           "GPUs -- both the compute and the communication argument "
           "point the same real direction at scale.\n\n");

    printf("--- Real, cited measured numbers for an ACTUAL GPU Barnes-Hut "
           "implementation (Burtscher & Pingali) -- NOT computed by this "
           "program ---\n");
    printf("\"Our CUDA code takes 5.2 seconds to simulate one time step "
           "with 5,000,000 bodies on a 1.3 GHz Quadro FX 5800 GPU with "
           "240 cores, which is 74 times faster than an optimized serial "
           "implementation running on a 2.53 GHz Xeon E5540 CPU.\"\n");
    printf("\"With increasing input size, the parallel CUDA Barnes Hut "
           "code is 5, 35, 66, 74, and 53 times faster than the serial C "
           "code.\"\n");
    printf("\"the Barnes Hut implementation reaches a respectable 75 "
           "Gflop/s.\"\n");
    printf("\"The most important kernel, kernel 5, is over 90 times "
           "faster.\"\n\n");

    printf("--- What this chapter does NOT claim ---\n");
    printf("Burtscher & Pingali's own real speedup figures are for ONE "
           "GPU against a serial CPU baseline, not a multi-GPU "
           "distributed Barnes-Hut communication measurement. This "
           "chapter's own reasoning -- that a distributed Barnes-Hut "
           "implementation could exchange far LESS than Section 27.2's "
           "own full O(N) all-gather, by approximating a distant "
           "region's bodies as a single aggregate value rather than "
           "requiring every individual body's raw data -- is this "
           "chapter's own logical extension of the tree algorithm's own "
           "real approximation idea, not a number measured or published "
           "by any source this chapter cites.\n");

    return 0;
}
