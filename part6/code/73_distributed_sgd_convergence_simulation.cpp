// Chapter 25: Distributed Training of a Neural Network: Data Parallelism
// and Ring All-Reduce
// 73_distributed_sgd_convergence_simulation.cpp
//
// Chapter 12's own consistency proof used ONE all-reduce call and a
// specially-chosen mean-target trick to get an EXACT match. This section
// builds the real thing that trick was standing in for: a genuine
// multi-step SGD training loop, training a real (if tiny) model --
// linear regression, y = w*x + b -- with the training data SHARDED
// across N replicas (Chapter 12's own computeShard() pattern) and each
// step's gradient combined across replicas via a real ring all-reduce
// (Chapter 9's own algorithm, reused unchanged: pass a running sum
// around the ring N-1 times, so every replica ends the step holding the
// exact same combined gradient). The comparison this section actually
// cares about: does K steps of this genuinely distributed loop end at
// the SAME learned (w, b) as a single-process reference trained with
// full-batch gradient descent on the SAME data, every step? Chapter 8's
// own non-associativity caution applies here for real, not just as a
// caveat -- summing the SAME numbers in a DIFFERENT grouping (per-shard
// partial sums combined across replicas, vs. one flat pass over every
// sample) is a real floating-point question this section actually
// tests, rather than assumes.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

struct Sample { float x, y; };

// Chapter 12's own computeShard() pattern, applied to training samples
// instead of a batch dimension.
void computeShard(int totalSamples, int worldSize, int rank, int &start, int &count) {
    int base = totalSamples / worldSize;
    start = rank * base;
    count = base; // totalSamples is chosen to divide evenly below
}

// Chapter 9's own real ring all-reduce algorithm, applied to a tiny
// fixed-size gradient vector (gradW, gradB) instead of a large array.
// Every replica's own local value CIRCULATES around the ring exactly
// once: at each of N-1 steps, replica r receives whatever its neighbor
// is currently forwarding, adds it to its own running accumulator, and
// then forwards that SAME received value onward next step (not its own
// original value again) -- this is what guarantees every one of the N
// local values passes every OTHER replica exactly once, so after N-1
// steps every replica's accumulator holds the exact same full sum.
void ringAllReduceSum(std::vector<float> &accW, std::vector<float> &accB,
                       const std::vector<float> &localGradW,
                       const std::vector<float> &localGradB, int N) {
    std::vector<float> messageW = localGradW, messageB = localGradB;
    for (int r = 0; r < N; r++) { accW[r] = localGradW[r]; accB[r] = localGradB[r]; }

    for (int step = 0; step < N - 1; step++) {
        std::vector<float> receivedW(N), receivedB(N);
        for (int r = 0; r < N; r++) {
            int src = (r - 1 + N) % N;
            receivedW[r] = messageW[src];
            receivedB[r] = messageB[src];
        }
        for (int r = 0; r < N; r++) {
            accW[r] += receivedW[r];
            accB[r] += receivedB[r];
        }
        // Forward what was just RECEIVED, not the original local value --
        // this is the step that makes each value circulate the whole ring.
        messageW = receivedW;
        messageB = receivedB;
    }
}

int main() {
    // A small, hand-checkable dataset: y = 3x + 2, plus a touch of real
    // per-sample variation so gradient descent has genuine work to do.
    const int M = 8; // total samples
    std::vector<Sample> data(M);
    for (int i = 0; i < M; i++) {
        float x = (float)(i + 1);
        data[i].x = x;
        data[i].y = 3.0f * x + 2.0f + ((i % 2 == 0) ? 0.5f : -0.5f);
    }

    const int N = 4;      // replicas
    const int K = 20;     // training steps
    const float lr = 0.01f;

    // --- Reference: single process, full-batch gradient descent, one
    // flat pass over all M samples in original order, every step. ---
    float wRef = 0.0f, bRef = 0.0f;
    for (int step = 0; step < K; step++) {
        float gW = 0.0f, gB = 0.0f;
        for (int i = 0; i < M; i++) {
            float pred = wRef * data[i].x + bRef;
            float err = pred - data[i].y;
            gW += err * data[i].x;
            gB += err;
        }
        gW = (2.0f / M) * gW;
        gB = (2.0f / M) * gB;
        wRef -= lr * gW;
        bRef -= lr * gB;
    }

    // --- Distributed: N replicas, each with its own shard, each step
    // computes a LOCAL partial sum, combines via a real ring all-reduce,
    // then every replica applies the SAME update independently
    // (Chapter 12's own "every replica stays identical" invariant). ---
    std::vector<float> wDist(N, 0.0f), bDist(N, 0.0f);
    for (int step = 0; step < K; step++) {
        std::vector<float> localGW(N), localGB(N);
        for (int r = 0; r < N; r++) {
            int start, count;
            computeShard(M, N, r, start, count);
            float gW = 0.0f, gB = 0.0f;
            for (int i = start; i < start + count; i++) {
                float pred = wDist[r] * data[i].x + bDist[r];
                float err = pred - data[i].y;
                gW += err * data[i].x;
                gB += err;
            }
            localGW[r] = gW;
            localGB[r] = gB;
        }
        std::vector<float> globalGW(N), globalGB(N);
        ringAllReduceSum(globalGW, globalGB, localGW, localGB, N);
        for (int r = 0; r < N; r++) {
            float gW = (2.0f / M) * globalGW[r];
            float gB = (2.0f / M) * globalGB[r];
            wDist[r] -= lr * gW;
            bDist[r] -= lr * gB;
        }
    }

    printf("After %d steps (M=%d samples, N=%d replicas, lr=%.3f):\n", K, M, N, lr);
    printf("  Reference (full-batch, single process): w=%.9f b=%.9f\n", wRef, bRef);
    for (int r = 0; r < N; r++)
        printf("  Replica %d (sharded + ring all-reduce):  w=%.9f b=%.9f\n",
               r, wDist[r], bDist[r]);

    bool allReplicasIdentical = true;
    for (int r = 1; r < N; r++)
        if (wDist[r] != wDist[0] || bDist[r] != bDist[0]) allReplicasIdentical = false;
    printf("\nEvery replica identical to each other: %s\n",
           allReplicasIdentical ? "YES" : "NO");

    bool exactMatchToReference = (wDist[0] == wRef && bDist[0] == bRef);
    printf("Distributed result EXACTLY matches full-batch reference: %s\n",
           exactMatchToReference ? "YES" : "NO");
    if (!exactMatchToReference) {
        printf("  w difference: %.3e   b difference: %.3e\n",
               (double)std::fabs(wDist[0] - wRef), (double)std::fabs(bDist[0] - bRef));
    }

    return 0;
}
