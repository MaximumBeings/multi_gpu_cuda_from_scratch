// Chapter 31: Multi-GPU Monte Carlo Risk Simulation
// 92_var_percentile_and_allgather_cost_model.cpp
//
// Sections 31.1-31.2 showed that a MEAN and VARIANCE can be combined
// from per-rank partial statistics using Chan et al.'s own real pairwise
// formula, sending only 3 numbers (count, mean, M2) per rank regardless
// of how many paths that rank simulated. This section shows a real,
// genuinely different closing complication: real bank risk reporting
// does not stop at a mean and a variance. The Basel Committee on
// Banking Supervision's own real 1996 market-risk amendment states that
// "'value-at-risk' be computed daily, using a 99th percentile, one-
// tailed confidence interval" -- a PERCENTILE of the loss distribution,
// not a mean or a variance, and Philippe Jorion's own real definition
// states plainly that VaR "summarizes the worst loss over a target
// horizon with a given level of confidence." Unlike a mean or a
// variance, an exact percentile of a distributed dataset genuinely
// CANNOT be computed from a few summary numbers per rank -- it requires
// knowing the relative rank/order of every single simulated loss, which
// means gathering the actual data, reusing this book's own real
// Chapter 10/27 all-gather pattern, not Chan et al.'s O(1)-per-rank
// message.
#include <cstdio>
#include <vector>
#include <algorithm>

const long long N_PATHS = 100;
const long long S0 = 100, RANGE = 40;

unsigned int counterBasedHash(long long pathIndex) {
    unsigned int x = (unsigned int)pathIndex;
    x = ((x >> 16) ^ x) * 0x45d9f3bU;
    x = ((x >> 16) ^ x) * 0x45d9f3bU;
    x = (x >> 16) ^ x;
    return x;
}
// Portfolio loss for one path: positive when the simulated terminal
// price falls below the starting price S0.
long long lossForPath(long long pathIndex) {
    unsigned int h = counterBasedHash(pathIndex);
    long long terminalPrice = S0 + (long long)(h % (2 * RANGE + 1)) - RANGE;
    return S0 - terminalPrice;
}

int main() {
    std::vector<long long> allLosses;
    for (long long i = 0; i < N_PATHS; i++) allLosses.push_back(lossForPath(i));

    std::vector<long long> sortedRef = allLosses;
    std::sort(sortedRef.begin(), sortedRef.end());
    long long var99Ref = sortedRef[98]; // 99th of 100 sorted losses (nearest-rank method)

    printf("Reference (single-process), %lld simulated paths.\n", N_PATHS);
    printf("Basel's own real quantitative standard: \"'value-at-risk' be "
           "computed daily, using a 99th percentile, one-tailed confidence "
           "interval.\"\n");
    printf("Reference VaR_99 (99th percentile of the loss distribution): %lld\n\n",
           var99Ref);

    printf("--- What Section 31.1's Chan-et-al. combination could NOT tell us ---\n");
    printf("Chan et al.'s own real pairwise formula combines (count, mean, M2) "
           "into a global mean and variance using exactly 3 numbers per rank, "
           "regardless of how many paths that rank ran. A MEAN and a VARIANCE "
           "describe the SHAPE of a distribution in summary; a PERCENTILE "
           "requires knowing exactly how many individual outcomes fall below "
           "a given threshold -- information that 3 summary numbers per rank "
           "simply do not contain.\n\n");

    int Ps[] = {1, 2, 4, 5, 10, 20};
    printf("%-8s %-28s %-28s %-16s\n", "P", "Chan et al.: values/rank",
           "exact VaR: values/rank (all-gather)", "VaR_99 EXACT match?");
    for (int P : Ps) {
        long long pathsPerRank = N_PATHS / P;
        // Distributed: each rank computes its own local losses; an exact
        // global percentile needs ALL of them gathered somewhere, exactly
        // Chapter 10/27's own real all-gather shape -- NX values per rank,
        // not O(1).
        std::vector<long long> gathered;
        for (int r = 0; r < P; r++) {
            for (long long i = r * pathsPerRank; i < (r + 1) * pathsPerRank; i++)
                gathered.push_back(lossForPath(i));
        }
        std::sort(gathered.begin(), gathered.end());
        long long var99Dist = gathered[98];
        bool exact = (var99Dist == var99Ref);
        printf("%-8d %-28s %-28lld %-16s\n", P, "3 (count, mean, M2)",
               pathsPerRank, exact ? "YES" : "NO");
    }
    printf("\nEvery rank's own message size for the MEAN/VARIANCE path stays at "
           "3 values, always -- but an EXACT VaR needs each rank to contribute "
           "its full local share of the dataset (N/P values per rank) to a "
           "gather step, because only the full sorted set determines which "
           "value sits at the 99th percentile. Real production systems facing "
           "this exact tradeoff at genuinely large scale use approximate "
           "streaming-percentile techniques (e.g., t-digest-style sketches) "
           "specifically to avoid gathering every raw value -- an extension "
           "this chapter's own exact all-gather model does not build, but "
           "names honestly as the real next step production systems take.\n");

    return 0;
}
