// Chapter 31: Multi-GPU Monte Carlo Risk Simulation
// 90_independent_path_correctness_simulation.cpp
//
// Chapter 30's rendering case needed zero communication during
// computation AND had no final reduction at all. Monte Carlo risk
// simulation is close but not identical: Boyle's own real classic paper
// ("Options: A Monte Carlo Approach," Journal of Financial Economics,
// 1977) describes simulating independent price paths and using their
// average payoff to value a derivative -- each PATH is independent
// (like Chapter 30's pixels), but the whole point is to COMBINE every
// path's result into one final statistic, which Chapter 30 never needed.
// NVIDIA's own real "Monte Carlo Option Pricing" whitepaper (Podlozhnyuk
// & Harris, 2008) describes exactly this split: "The set of input
// options is divided into contiguous subsets (the number of subsets
// equals the number of CUDA-capable GPUs installed in the system)," each
// GPU thread "computes and sums the payoff for multiple integration
// paths and stores the sum and the sum of squares... into a device
// memory array," and the FINAL step "uses a parallel reduction to
// compute the sums" across every thread's own partial result. This file
// builds that shape: a fixed set of paths, split across P ranks (each
// rank touching ONLY its own paths, using a counter-based deterministic
// per-path value -- modeling NVIDIA's own real cuRAND Philox generator,
// whose documentation states "each thread of computation should be
// assigned a unique id number" so that independent streams need NO
// coordination between them), with each rank's own partial (count, mean,
// M2) combined into a global statistic using Chan, Golub & LeVeque's own
// real pairwise combination formula ("Updating Formulae and a Pairwise
// Algorithm for Computing Sample Variances," Stanford CS Tech Report
// STAN-CS-79-773, 1979), and checked against a single-process reference.
#include <cstdio>
#include <vector>
#include <cmath>

const long long N_PATHS = 24;
const long long S0 = 100;    // starting price (illustrative integer units)
const long long K = 100;     // strike price
const long long RANGE = 40;  // terminal price spread around S0

// A deterministic, counter-based per-path "random" value -- a simplified
// stand-in for a real counter-based generator (cuRAND's own real Philox
// family): given ONLY the path index, this returns the SAME value no
// matter which rank calls it and no matter what order paths are
// processed in -- exactly the real property that lets cuRAND's own real
// subsequence/offset scheme hand out independent streams with ZERO
// coordination between threads or GPUs.
unsigned int counterBasedHash(long long pathIndex) {
    unsigned int x = (unsigned int)pathIndex;
    x = ((x >> 16) ^ x) * 0x45d9f3bU;
    x = ((x >> 16) ^ x) * 0x45d9f3bU;
    x = (x >> 16) ^ x;
    return x;
}

// Simplified terminal price and call-option payoff for one path -- pure
// integer arithmetic, exact regardless of partitioning.
long long payoffForPath(long long pathIndex) {
    unsigned int h = counterBasedHash(pathIndex);
    long long terminalPrice = S0 + (long long)(h % (2 * RANGE + 1)) - RANGE;
    long long payoff = terminalPrice - K;
    return payoff > 0 ? payoff : 0;
}

struct Stats { long long count; double mean; double M2; };

// Two-pass (numerically stable) mean/M2 over an explicit list of path
// indices -- used both for the reference (all paths) and for each
// rank's own local statistics (its own paths only).
Stats computeStats(const std::vector<long long> &pathIndices) {
    long long n = (long long)pathIndices.size();
    long long sum = 0;
    for (long long idx : pathIndices) sum += payoffForPath(idx);
    double mean = (double)sum / (double)n;
    double m2 = 0.0;
    for (long long idx : pathIndices) {
        double d = (double)payoffForPath(idx) - mean;
        m2 += d * d;
    }
    return {n, mean, m2};
}

// Chan, Golub & LeVeque's own real pairwise combination formula (1979):
// combine two groups' (count, mean, M2) into one, WITHOUT re-touching
// any individual data point.
Stats combinePairwise(const Stats &a, const Stats &b) {
    long long n = a.count + b.count;
    double delta = b.mean - a.mean;
    double mean = (a.count * a.mean + b.count * b.mean) / (double)n;
    double m2 = a.M2 + b.M2 + delta * delta * (double)a.count * (double)b.count / (double)n;
    return {n, mean, m2};
}

int main() {
    std::vector<long long> allPaths;
    for (long long i = 0; i < N_PATHS; i++) allPaths.push_back(i);

    Stats reference = computeStats(allPaths);
    printf("Reference (single-process): count=%lld  mean=%.17g  M2=%.17g\n\n",
           reference.count, reference.mean, reference.M2);

    int Ps[] = {1, 2, 3, 4, 6, 8, 12, 24};
    printf("%-6s %-10s %-14s %-24s %-12s %-12s\n", "P", "count",
           "combined mean", "combined M2", "mean match", "M2 match");
    for (int P : Ps) {
        long long pathsPerRank = N_PATHS / P;
        Stats combined = {0, 0.0, 0.0};
        bool first = true;
        for (int r = 0; r < P; r++) {
            std::vector<long long> rankPaths;
            for (long long i = r * pathsPerRank; i < (r + 1) * pathsPerRank; i++)
                rankPaths.push_back(i);
            Stats local = computeStats(rankPaths);
            combined = first ? local : combinePairwise(combined, local);
            first = false;
        }
        bool meanMatch = (combined.mean == reference.mean);
        bool m2Match = (combined.M2 == reference.M2);
        printf("%-6d %-10lld %-14.6f %-24.17g %-12s %-12s\n",
               P, combined.count, combined.mean, combined.M2,
               meanMatch ? "YES" : "NO", m2Match ? "YES" : "NO");
    }
    printf("\nEvery combined count matches exactly (integer payoff sums are "
           "exactly associative, unlike the floating-point mean and M2). The "
           "combined MEAN matches the reference exactly at every P tested here "
           "-- Chan et al.'s own pairwise mean formula happens to reproduce "
           "this dataset's reference mean bit-for-bit. The combined M2 does "
           "NOT match at P > 1: a real, tiny (last-few-digits) floating-point "
           "difference, the same non-associativity finding Chapter 8, "
           "Chapter 25, and Chapter 29 already documented -- Chan et al.'s "
           "own real algorithm is famous for being NUMERICALLY STABLE, not "
           "for being BIT-IDENTICAL to a differently-grouped two-pass "
           "reference; those are two different guarantees.\n");

    return 0;
}
