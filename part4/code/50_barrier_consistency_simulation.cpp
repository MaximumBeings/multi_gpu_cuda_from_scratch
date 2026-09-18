// Chapter 17: Barriers and Global Synchronization Across Devices
// 50_barrier_consistency_simulation.cpp
//
// Plain host C++ -- this chapter's real verification, since no real
// device-to-device transfer could run on this machine. Part A checks
// the one property a barrier actually buys you: with heterogeneous
// per-rank progress (some ranks reach a checkpoint faster than
// others, exactly the situation Chapter 18 will call load imbalance),
// does every rank see the SAME, COMPLETE shared state after the
// checkpoint, or does a rank that arrives early race ahead and read
// an incomplete view? Part B is a closed-form cost model: a barrier
// built from a throwaway AllReduce (Section 17.2) costs exactly the
// same number of communication rounds as a real AllReduce of any
// other size, reusing Chapter 9's ring formula and Chapter 11's tree
// formula rather than fabricating new timing numbers.
#include <cstdio>
#include <vector>
#include <algorithm>

const int WORLD_SIZE = 4;

// Each rank's simulated "arrival time" at the checkpoint -- standing
// in for real, heterogeneous per-device progress (different clock
// speeds, different local work, exactly what Chapter 18 addresses
// directly). Rank 2 arrives first; rank 1 arrives last.
const int arrivalTime[WORLD_SIZE] = {3, 7, 2, 5};
// Each rank's own contribution to a shared checkpoint value -- e.g.
// standing in for one term of a real distributed sum.
const int contribution[WORLD_SIZE] = {10, 20, 30, 40};
const int REFERENCE_TOTAL = 10 + 20 + 30 + 40; // = 100, computed independently

// ---------------------------------------------------------------
// Part A: consistency. WITHOUT a barrier, a rank reads the shared
// total as soon as IT PERSONALLY arrives -- exactly what happens if
// nothing stops a rank from proceeding past a checkpoint before every
// other rank has written its own contribution. WITH a barrier, every
// rank's write is guaranteed complete before any rank's read begins.
// ---------------------------------------------------------------

std::vector<int> withoutBarrier() {
    // Ranks are processed in arrival order -- the real order events
    // would happen in, since nothing enforces anything else. Each
    // rank writes its contribution, then immediately reads whatever
    // the shared total is AT THAT MOMENT (which may or may not
    // include ranks that haven't arrived yet).
    std::vector<int> order(WORLD_SIZE);
    for (int r = 0; r < WORLD_SIZE; ++r) order[r] = r;
    std::sort(order.begin(), order.end(), [](int a, int b) { return arrivalTime[a] < arrivalTime[b]; });

    int sharedTotal = 0;
    std::vector<int> observed(WORLD_SIZE, 0);
    for (int r : order) {
        sharedTotal += contribution[r];       // this rank's own write
        observed[r] = sharedTotal;             // this rank's own read, right after its own write
    }
    return observed;
}

std::vector<int> withBarrier() {
    // Every rank's write happens first, in any order -- it doesn't
    // matter which, because no rank is allowed to READ until a real
    // barrier (Section 17.2's throwaway AllReduce, followed by a real
    // stream/device sync) confirms every rank's write is done.
    int sharedTotal = 0;
    for (int r = 0; r < WORLD_SIZE; ++r) sharedTotal += contribution[r];
    // -- barrier here: every rank has now written, before any reads --
    std::vector<int> observed(WORLD_SIZE, sharedTotal); // every rank reads the SAME, COMPLETE total
    return observed;
}

int main() {
    printf("Part A: %d ranks, heterogeneous arrival times %d/%d/%d/%d "
           "(rank 2 arrives first, rank 1 arrives last), each "
           "contributing to a shared checkpoint total. Independently "
           "computed reference total: %d.\n\n",
           WORLD_SIZE, arrivalTime[0], arrivalTime[1], arrivalTime[2], arrivalTime[3],
           REFERENCE_TOTAL);

    std::vector<int> withoutB = withoutBarrier();
    std::vector<int> withB = withBarrier();

    printf("WITHOUT a barrier -- each rank reads immediately after its own write:\n");
    int correctWithout = 0;
    for (int r = 0; r < WORLD_SIZE; ++r) {
        bool matches = (withoutB[r] == REFERENCE_TOTAL);
        if (matches) ++correctWithout;
        printf("  rank %d (arrived at t=%d): observed total = %-4d  %s\n",
               r, arrivalTime[r], withoutB[r], matches ? "matches reference" : "STALE -- missing later contributions");
    }
    printf("  %d of %d ranks saw the correct, complete total.\n\n", correctWithout, WORLD_SIZE);

    printf("WITH a barrier -- every write finishes before any read:\n");
    int correctWith = 0;
    for (int r = 0; r < WORLD_SIZE; ++r) {
        bool matches = (withB[r] == REFERENCE_TOTAL);
        if (matches) ++correctWith;
        printf("  rank %d: observed total = %-4d  %s\n",
               r, withB[r], matches ? "matches reference" : "STALE");
    }
    printf("  %d of %d ranks saw the correct, complete total: %s\n",
           correctWith, WORLD_SIZE, (correctWith == WORLD_SIZE) ? "PASS" : "FAIL");

    if (correctWithout < WORLD_SIZE && correctWith == WORLD_SIZE) {
        printf("\nOnly the LAST rank to arrive (rank 1, at t=%d) happened to "
               "see the correct total without a barrier -- purely because "
               "every other rank had already written by the time it read. "
               "That's not a guarantee, it's an accident of this particular "
               "arrival order; swap any two arrival times and a DIFFERENT "
               "subset of ranks would see a stale value, silently, with no "
               "error raised anywhere. This is exactly the failure mode "
               "Chapter 16's own halo exchange had to guard against every "
               "iteration, generalized to any shared checkpoint, not just a "
               "grid's neighboring rows.\n", arrivalTime[1]);
    }

    // ---------------------------------------------------------------
    // Part B: cost. A barrier built from a throwaway AllReduce costs
    // the same number of communication rounds as a real AllReduce of
    // any other payload size -- reusing Chapter 9's ring formula and
    // Chapter 11's tree formula directly, not fabricating a new one.
    // ---------------------------------------------------------------
    printf("\nPart B: a barrier's real communication cost, in rounds -- "
           "reusing Chapter 9's ring formula 2*(N-1) and Chapter 11's "
           "tree formula 2*ceil(log2(N)).\n\n");

    int worldSizes[] = {4, 8, 35}; // 35 reuses Ch13's own GPT-3 shard count
    printf("%-10s %-18s %-18s\n", "N", "ring rounds", "tree rounds");
    for (int n : worldSizes) {
        int ringRounds = 2 * (n - 1);
        int log2n = 0;
        while ((1 << log2n) < n) ++log2n; // ceil(log2(n))
        int treeRounds = 2 * log2n;
        printf("%-10d %-18d %-18d\n", n, ringRounds, treeRounds);
    }

    printf("\nA barrier's payload is the smallest possible -- one element, "
           "not a real gradient tensor or activation -- but the ROUND "
           "COUNT above doesn't depend on payload size at all; it depends "
           "only on N, the number of ranks. A barrier is exactly as "
           "latency-expensive, per call, as any other AllReduce with the "
           "same N -- it is cheap in bytes moved, never in synchronization "
           "rounds paid. Calling one every iteration, for every one of "
           "Part 3's parallelization strategies, is a real, recurring cost, "
           "not a free correctness guarantee.\n");

    return (correctWith == WORLD_SIZE) ? 0 : 1;
}
