// Chapter 18: Load Balancing Across Heterogeneous GPUs
// 52_weighted_shard_partition.cpp
//
// Plain host C++ -- computeWeightedShard() takes one more real input
// than Chapter 12's computeShard(), Chapter 13's computeLayerRange(),
// and Chapter 16's computeRowRange() ever did: each rank's OWN
// relative speed, not just its rank and the world size. The real
// technique it implements is the one Um et al.'s "Cephalo" paper
// (2024) describes plainly: "the batch of inputs is distributed
// unevenly across GPUs according to their relative computational
// speeds," and, in describing Cephalo's own design, "partitions the
// global batch of training inputs unevenly across GPUs to control
// the computational workload assigned to each GPU."
#include <cstdio>
#include <vector>

// Splits `totalWork` units of work across `worldSize` ranks
// PROPORTIONALLY to each rank's own relative speed, instead of
// evenly. A rank twice as fast as another gets (approximately) twice
// as much work -- the goal, made precise in Section 18.3, is that
// every rank's own work/speed (its finish time) comes out equal.
std::vector<long> computeWeightedShard(long totalWork, const std::vector<double>& speeds) {
    double speedSum = 0.0;
    for (double s : speeds) speedSum += s;

    std::vector<long> shares(speeds.size());
    long assigned = 0;
    for (size_t r = 0; r < speeds.size(); ++r) {
        // Every rank but the last gets its proportional share, rounded
        // down; the last rank absorbs whatever integer remainder is
        // left, the same "assumes an even split" caveat Chapter 12's
        // computeShard() carried, now generalized to an uneven one.
        if (r + 1 < speeds.size()) {
            shares[r] = (long)(totalWork * (speeds[r] / speedSum));
            assigned += shares[r];
        } else {
            shares[r] = totalWork - assigned;
        }
    }
    return shares;
}

// The equal-split baseline every earlier chapter actually used --
// Chapter 12's computeShard(), specialized to a single dimension,
// reproduced here for a direct side-by-side comparison.
std::vector<long> computeEqualShard(long totalWork, int worldSize) {
    std::vector<long> shares(worldSize);
    long perRank = totalWork / worldSize;
    long assigned = 0;
    for (int r = 0; r + 1 < worldSize; ++r) { shares[r] = perRank; assigned += perRank; }
    shares[worldSize - 1] = totalWork - assigned;
    return shares;
}

int main() {
    // The same hypothetical 4-device cluster Section 18.1 introduced:
    // two ordinary A100s, one faster H100 SXM, and one thermally
    // throttled A100. Relative speed here is expressed directly as a
    // measured throughput factor (the real, already-established
    // technique this book has used since Chapter 2's bandwidth
    // model -- cite a real, already-measured number, don't re-derive
    // one from first principles) rather than a raw SM count, since SM
    // count alone can't see rank 3's real thermal throttling either.
    std::vector<double> speeds = {1.0, 1.0, 1.5, 0.5};
    const long TOTAL_WORK = 900; // e.g. 900 training samples in a global batch

    printf("4-device cluster, relative measured speeds: rank0=%.1fx, "
           "rank1=%.1fx, rank2=%.1fx (faster H100), rank3=%.1fx "
           "(throttled A100). Total work to split: %ld units.\n\n",
           speeds[0], speeds[1], speeds[2], speeds[3], TOTAL_WORK);

    std::vector<long> equal = computeEqualShard(TOTAL_WORK, (int)speeds.size());
    std::vector<long> weighted = computeWeightedShard(TOTAL_WORK, speeds);

    long equalSum = 0, weightedSum = 0;
    printf("%-6s %-10s %-10s %-14s %-14s\n", "rank", "speed", "equal", "weighted", "weighted/speed");
    for (size_t r = 0; r < speeds.size(); ++r) {
        equalSum += equal[r];
        weightedSum += weighted[r];
        printf("%-6zu %-10.1f %-10ld %-14ld %-14.2f\n",
               r, speeds[r], equal[r], weighted[r], weighted[r] / speeds[r]);
    }
    printf("total: equal split sums to %ld, weighted split sums to %ld "
           "(both must equal %ld): %s\n",
           equalSum, weightedSum, TOTAL_WORK,
           (equalSum == TOTAL_WORK && weightedSum == TOTAL_WORK) ? "PASS" : "FAIL");

    printf("\nEvery rank's equal share is identical (%ld units each) "
           "regardless of speed -- exactly what every computeShard()-"
           "style function in this book has done since Chapter 12. The "
           "weighted share instead scales with each rank's own speed: "
           "rank 2 (the faster H100) gets the largest share, rank 3 "
           "(throttled) gets the smallest. Section 18.3 checks what "
           "this buys, concretely: whether it actually equalizes how "
           "long each rank takes to finish its own share.\n", equal[0]);

    return 0;
}
