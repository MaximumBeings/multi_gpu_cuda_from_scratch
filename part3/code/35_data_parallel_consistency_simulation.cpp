// Chapter 12: Data Parallelism -- Replicated Model, Sharded Data
// 35_data_parallel_consistency_simulation.cpp
//
// Plain host C++ -- verifying the one property PyTorch's own
// DistributedDataParallel paper states as DDP's actual correctness
// guarantee: "all model replicas start from the exact same model
// state, and see the same parameter gradients after every backward
// pass" (cited in Sources). N real in-memory scalars stand in for N
// real devices' weight replicas; a tiny synthetic linear-regression
// gradient (grad = w - target) stands in for a real backward pass,
// small enough to check by hand.
#include <cstdio>

int main() {
    const int N = 4;
    const double targets[N] = {12.0, 8.0, 20.0, 16.0}; // each device's own data-shard "target"
    const double LR = 0.1;
    const int STEPS = 5;

    double meanTarget = 0.0;
    for (int i = 0; i < N; ++i) meanTarget += targets[i];
    meanTarget /= N;

    // ---------------------------------------------------------------
    // (a) WITH synchronization: every step, average all N replicas'
    // local gradients (this book's real ncclAllReduce(..., ncclAvg,
    // ...) from Section 12.2), then apply that SAME averaged update
    // to every replica -- exactly the mechanism this chapter's cited
    // sources describe.
    // ---------------------------------------------------------------
    double synced[N];
    for (int i = 0; i < N; ++i) synced[i] = 0.0; // Section 12.1's broadcast initial weight

    // Independent reference: a single, non-replicated model trained
    // directly on the mean target -- mathematically what training on
    // the true GLOBAL batch (the union of every shard) would do,
    // since the shards are equal-sized. Computed completely
    // separately from the loop below.
    double reference = 0.0;

    printf("WITH synchronization (average gradients via ncclAvg every step):\n");
    printf("step 0: ");
    for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, synced[i]);
    printf(" | reference=%.4f\n", reference);

    for (int step = 1; step <= STEPS; ++step) {
        // Every replica computes its OWN local gradient from its OWN
        // data shard -- and since every replica's weight is still
        // identical going into this step, this is genuinely N
        // separate (but, this step, numerically equal-to-each-other's
        // starting point) local computations, not one computation
        // copied N times.
        double localGrad[N];
        for (int i = 0; i < N; ++i) localGrad[i] = synced[i] - targets[i];

        // ncclAllReduce(..., ncclAvg, ...): average across all N.
        double avgGrad = 0.0;
        for (int i = 0; i < N; ++i) avgGrad += localGrad[i];
        avgGrad /= N;

        // Every replica applies the SAME averaged gradient.
        for (int i = 0; i < N; ++i) synced[i] = synced[i] - LR * avgGrad;

        reference = reference - LR * (reference - meanTarget);

        printf("step %d: ", step);
        for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, synced[i]);
        printf(" | reference=%.4f\n", reference);
    }

    bool allSyncedEqual = true;
    bool allMatchReference = true;
    for (int i = 0; i < N; ++i) {
        if (synced[i] != synced[0]) allSyncedEqual = false;
        if (synced[i] != reference) allMatchReference = false;
    }
    printf("After %d step(s): every replica identical to every other: %s; "
           "every replica matches the independent single-model reference: %s\n\n",
           STEPS, allSyncedEqual ? "PASS" : "FAIL", allMatchReference ? "PASS" : "FAIL");

    // ---------------------------------------------------------------
    // (b) WITHOUT synchronization: every replica applies its OWN
    // local gradient, with no all-reduce at all -- the mistake this
    // book's entire Part 2 exists to prevent.
    // ---------------------------------------------------------------
    double unsynced[N];
    for (int i = 0; i < N; ++i) unsynced[i] = 0.0;

    printf("WITHOUT synchronization (each replica updates from its own local gradient only):\n");
    printf("step 0: ");
    for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, unsynced[i]);
    printf("\n");

    for (int step = 1; step <= STEPS; ++step) {
        double localGrad[N];
        for (int i = 0; i < N; ++i) localGrad[i] = unsynced[i] - targets[i];
        for (int i = 0; i < N; ++i) unsynced[i] = unsynced[i] - LR * localGrad[i]; // no averaging

        printf("step %d: ", step);
        for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, unsynced[i]);
        printf("\n");
    }

    bool anyDivergence = false;
    for (int i = 1; i < N; ++i) {
        if (unsynced[i] != unsynced[0]) anyDivergence = true;
    }
    printf("After %d step(s) with no synchronization: replicas have "
           "diverged from each other: %s\n", STEPS,
           anyDivergence ? "PASS (divergence correctly demonstrated)" : "FAIL (unexpectedly still identical)");

    return (allSyncedEqual && allMatchReference && anyDivergence) ? 0 : 1;
}
