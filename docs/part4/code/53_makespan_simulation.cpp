// Chapter 18: Load Balancing Across Heterogeneous GPUs
// 53_makespan_simulation.cpp
//
// Plain host C++ -- this chapter's real verification. Chapter 17's
// own barrier guarantees every rank waits for the SLOWEST one to
// arrive before any rank proceeds -- which means the real wall-clock
// cost of one iteration is exactly the LONGEST of any rank's own
// finish times (its "makespan"), no matter how quickly the other
// ranks finished. Part A computes that makespan under Section 18.2's
// equal split and weighted split, for the continuous (infinitely
// divisible) ideal case, where the algebra says a proportional split
// makes every rank's finish time exactly equal. Part B re-runs the
// same comparison using REAL integer-quantized work units (the same
// computeWeightedShard() Section 18.2 built, reproduced here), to
// check honestly whether that ideal survives rounding to whole units.
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>

// Reproduced verbatim from Section 18.2's own 52_weighted_shard_partition.cpp
// -- see that section for the full explanation.
std::vector<long> computeWeightedShard(long totalWork, const std::vector<double>& speeds) {
    double speedSum = 0.0;
    for (double s : speeds) speedSum += s;
    std::vector<long> shares(speeds.size());
    long assigned = 0;
    for (size_t r = 0; r < speeds.size(); ++r) {
        if (r + 1 < speeds.size()) {
            shares[r] = (long)(totalWork * (speeds[r] / speedSum));
            assigned += shares[r];
        } else {
            shares[r] = totalWork - assigned;
        }
    }
    return shares;
}
std::vector<long> computeEqualShard(long totalWork, int worldSize) {
    std::vector<long> shares(worldSize);
    long perRank = totalWork / worldSize;
    long assigned = 0;
    for (int r = 0; r + 1 < worldSize; ++r) { shares[r] = perRank; assigned += perRank; }
    shares[worldSize - 1] = totalWork - assigned;
    return shares;
}

int main() {
    const std::vector<double> speeds = {1.0, 1.0, 1.5, 0.5};
    const long TOTAL_WORK = 900;
    const int WORLD_SIZE = (int)speeds.size();
    double speedSum = 0.0;
    for (double s : speeds) speedSum += s;

    // ---------------------------------------------------------------
    // Part A: the continuous ideal. If work could be divided into
    // infinitely fine units, a proportional split makes every rank's
    // finish time exactly the same: work_i/speed_i = (totalWork *
    // speed_i/speedSum) / speed_i = totalWork/speedSum, for every i,
    // regardless of speed_i -- the speed_i term cancels algebraically.
    // ---------------------------------------------------------------
    printf("Part A: continuous (infinitely divisible) ideal, %d ranks, "
           "speeds %.1fx/%.1fx/%.1fx/%.1fx, %ld total work units.\n\n",
           WORLD_SIZE, speeds[0], speeds[1], speeds[2], speeds[3], TOTAL_WORK);

    double idealEqualShare = (double)TOTAL_WORK / WORLD_SIZE;
    double idealEqualMakespan = 0.0;
    for (double s : speeds) idealEqualMakespan = std::max(idealEqualMakespan, idealEqualShare / s);
    double idealWeightedMakespan = (double)TOTAL_WORK / speedSum; // same for every rank, algebraically

    printf("  equal split:    every rank gets %.4f units; slowest rank "
           "(speed %.1fx) finishes at t=%.4f -- every OTHER rank sits "
           "idle at Chapter 17's barrier until then.\n",
           idealEqualShare, *std::min_element(speeds.begin(), speeds.end()), idealEqualMakespan);
    printf("  weighted split: EVERY rank finishes at t=%.4f -- the "
           "algebra says this is exact, for any speeds, in the "
           "continuous case.\n", idealWeightedMakespan);
    printf("  ideal speedup: %.2fx\n", idealEqualMakespan / idealWeightedMakespan);

    // ---------------------------------------------------------------
    // Part B: reality. Work comes in whole units (samples, batches,
    // grid rows), and Section 18.2's own computeWeightedShard() has
    // to round to an integer number of them. Does the continuous
    // ideal above survive that?
    // ---------------------------------------------------------------
    printf("\nPart B: the same comparison using REAL integer-quantized "
           "shares from Section 18.2's own computeWeightedShard().\n\n");

    std::vector<long> equalShares = computeEqualShard(TOTAL_WORK, WORLD_SIZE);
    std::vector<long> weightedShares = computeWeightedShard(TOTAL_WORK, speeds);

    double equalMakespan = 0.0, weightedMakespan = 0.0;
    double weightedMin = 1e18, weightedMax = 0.0;
    printf("%-6s %-8s %-14s %-14s %-14s %-14s\n",
           "rank", "speed", "equal share", "equal finish", "weighted share", "weighted finish");
    for (int r = 0; r < WORLD_SIZE; ++r) {
        double equalFinish = equalShares[r] / speeds[r];
        double weightedFinish = weightedShares[r] / speeds[r];
        equalMakespan = std::max(equalMakespan, equalFinish);
        weightedMakespan = std::max(weightedMakespan, weightedFinish);
        weightedMin = std::min(weightedMin, weightedFinish);
        weightedMax = std::max(weightedMax, weightedFinish);
        printf("%-6d %-8.1f %-14ld %-14.3f %-14ld %-14.3f\n",
               r, speeds[r], equalShares[r], equalFinish, weightedShares[r], weightedFinish);
    }

    double weightedSpread = weightedMax - weightedMin;
    printf("\n  equal split makespan (real bottleneck rank's finish time): %.3f\n"
           "  weighted split makespan:                                    %.3f\n"
           "  real speedup from weighted partitioning:                    %.2fx\n"
           "  weighted split's own finish-time spread (max - min):        %.3f "
           "(the continuous ideal predicts exactly 0.0)\n",
           equalMakespan, weightedMakespan, equalMakespan / weightedMakespan, weightedSpread);

    const double SPREAD_TOLERANCE = 2.0; // whole work units' worth of rounding, not zero
    bool spreadIsSmall = weightedSpread < SPREAD_TOLERANCE;
    printf("\n  weighted split's spread (%.3f) is within %.1f work-units'-"
           "worth of the continuous ideal's exact 0.0: %s\n",
           weightedSpread, SPREAD_TOLERANCE, spreadIsSmall ? "PASS" : "FAIL");

    if (spreadIsSmall) {
        printf("\nThe continuous ideal's EXACT equality doesn't survive "
               "rounding to whole work units -- the fastest- and "
               "slowest-finishing ranks under the weighted split finish "
               "%.3f apart, not identically -- but the gap is tiny "
               "relative to either rank's own share, because "
               "computeWeightedShard() can only be off by less than one "
               "whole unit of work per rank. This is the same honest "
               "distinction this book drew in Chapter 13 vs. Chapter 14: "
               "an exact algebraic guarantee (Chapter 13's relocation, "
               "this section's continuous ideal) is a different, "
               "stronger claim than an empirically-checked, epsilon-"
               "bounded one (Chapter 14's reduce, this section's real "
               "integer-quantized shares) -- and this section's own "
               "result belongs to the second kind, checked, not assumed.\n",
               weightedSpread);
    }

    return spreadIsSmall ? 0 : 1;
}
