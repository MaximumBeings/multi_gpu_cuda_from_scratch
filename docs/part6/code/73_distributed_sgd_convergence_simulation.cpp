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
// step's gradient combined across replicas via Chapter 9's own two-phase
// ring all-reduce: a SCATTER-REDUCE phase (N-1 steps) that leaves each
// chunk fully summed on exactly one owner replica, then an ALL-GATHER
// phase (N-1 steps) that COPIES each owner's finished chunk to everyone.
// Two questions are tested, not assumed:
//   1. Do all N replicas end every step with bit-identical gradients
//      (and therefore bit-identical weights after K steps)?
//   2. Does the distributed result match a single-process full-batch
//      reference bit for bit, even though the distributed version sums
//      per-shard partial sums (a different grouping than one flat pass)?
// For contrast, it also runs a deliberately WRONG "all-reduce" in which
// every replica adds all N values itself, starting from its own value --
// the shape of algorithm that genuinely CAN disagree across replicas.
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

// Chapter 9's own ring all-reduce (scatter-reduce, then all-gather), on
// N per-replica buffers of length L, where L is a multiple of N and the
// buffer is split into N equal chunks of C = L/N elements.
//
// Scatter-reduce, step s = 0..N-2: replica r sends chunk (r - s) mod N
// to replica (r+1) mod N, which ADDS it into its own copy of that chunk.
// After N-1 steps, replica r holds the fully-reduced chunk (r+1) mod N.
//
// All-gather, step s = 0..N-2: replica r sends chunk (r + 1 - s) mod N
// to replica (r+1) mod N, which OVERWRITES its copy (no arithmetic).
//
// Every chunk's final value is therefore computed by exactly ONE chain
// of additions, on ONE replica, and then copied verbatim to the others:
// all replicas end bit-identical by construction, regardless of the
// floating-point summation order used inside that one chain.
// Returns the number of elements each replica sent (same for all).
long long ringAllReduceSum(std::vector<std::vector<float>> &buf, int N) {
    const int L = (int)buf[0].size();
    const int C = L / N;
    long long sentPerReplica = 0;

    // Phase 1: scatter-reduce.
    for (int s = 0; s < N - 1; s++) {
        // Snapshot every outgoing message first: all N sends in one step
        // happen "simultaneously", exactly as on real hardware.
        std::vector<std::vector<float>> msg(N, std::vector<float>(C));
        for (int r = 0; r < N; r++) {
            int c = ((r - s) % N + N) % N;
            for (int k = 0; k < C; k++) msg[r][k] = buf[r][c * C + k];
        }
        for (int r = 0; r < N; r++) {
            int dst = (r + 1) % N;
            int c = ((r - s) % N + N) % N;
            for (int k = 0; k < C; k++) buf[dst][c * C + k] += msg[r][k];
        }
        sentPerReplica += C;
    }

    // Phase 2: all-gather (pure copies of already-final chunks).
    for (int s = 0; s < N - 1; s++) {
        std::vector<std::vector<float>> msg(N, std::vector<float>(C));
        for (int r = 0; r < N; r++) {
            int c = ((r + 1 - s) % N + N) % N;
            for (int k = 0; k < C; k++) msg[r][k] = buf[r][c * C + k];
        }
        for (int r = 0; r < N; r++) {
            int dst = (r + 1) % N;
            int c = ((r + 1 - s) % N + N) % N;
            for (int k = 0; k < C; k++) buf[dst][c * C + k] = msg[r][k];
        }
        sentPerReplica += C;
    }
    return sentPerReplica;
}

// A deliberately WRONG "all-reduce", kept only as a contrast: every
// replica receives every other replica's FULL local value and adds all N
// of them itself, starting from its OWN value. Each replica therefore
// performs its own chain of additions in its own order -- and fp32
// addition is not associative, so the replicas can disagree.
void everyReplicaSumsItself(std::vector<std::vector<float>> &buf, int N) {
    const int L = (int)buf[0].size();
    std::vector<std::vector<float>> orig = buf;
    for (int r = 0; r < N; r++)
        for (int k = 0; k < L; k++) {
            float acc = orig[r][k];
            for (int j = 1; j < N; j++) acc += orig[(r - j + N) % N][k];
            buf[r][k] = acc;
        }
}

// Runs K steps of sharded SGD with the given all-reduce; returns final
// per-replica (w, b).
template <typename AllReduce>
void trainDistributed(const std::vector<Sample> &data, int N, int K, float lr,
                      std::vector<float> &w, std::vector<float> &b,
                      AllReduce allReduce) {
    const int M = (int)data.size();
    w.assign(N, 0.0f);
    b.assign(N, 0.0f);
    for (int step = 0; step < K; step++) {
        // Gradient buffer = [gradW, gradB], zero-padded to N chunks of
        // one element each so the ring can split it N ways.
        std::vector<std::vector<float>> grad(N, std::vector<float>(N, 0.0f));
        for (int r = 0; r < N; r++) {
            int start, count;
            computeShard(M, N, r, start, count);
            float gW = 0.0f, gB = 0.0f;
            for (int i = start; i < start + count; i++) {
                float pred = w[r] * data[i].x + b[r];
                float err = pred - data[i].y;
                gW += err * data[i].x;
                gB += err;
            }
            grad[r][0] = gW;
            grad[r][1] = gB;
        }
        allReduce(grad, N);
        for (int r = 0; r < N; r++) {
            float gW = (2.0f / M) * grad[r][0];
            float gB = (2.0f / M) * grad[r][1];
            w[r] -= lr * gW;
            b[r] -= lr * gB;
        }
    }
}

bool allSame(const std::vector<float> &w, const std::vector<float> &b) {
    for (size_t r = 1; r < w.size(); r++)
        if (w[r] != w[0] || b[r] != b[0]) return false;
    return true;
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

    // --- Distributed, Chapter 9's real ring all-reduce. ---
    std::vector<float> wDist, bDist;
    long long sent = 0;
    trainDistributed(data, N, K, lr, wDist, bDist,
                     [&](std::vector<std::vector<float>> &g, int n) {
                         sent = ringAllReduceSum(g, n);
                     });

    printf("After %d steps (M=%d samples, N=%d replicas, lr=%.3f):\n", K, M, N, lr);
    printf("  Reference (full-batch, single process): w=%.9f b=%.9f\n", wRef, bRef);
    for (int r = 0; r < N; r++)
        printf("  Replica %d (sharded + ring all-reduce):  w=%.9f b=%.9f\n",
               r, wDist[r], bDist[r]);

    printf("\nEvery replica identical to each other: %s\n",
           allSame(wDist, bDist) ? "YES" : "NO");
    bool exactMatchToReference = (wDist[0] == wRef && bDist[0] == bRef);
    printf("Distributed result EXACTLY matches full-batch reference: %s\n",
           exactMatchToReference ? "YES" : "NO");
    if (!exactMatchToReference) {
        printf("  w difference: %.3e   b difference: %.3e\n",
               (double)std::fabs(wDist[0] - wRef), (double)std::fabs(bDist[0] - bRef));
    }
    printf("Elements sent per replica per all-reduce: %lld "
           "(= 2(N-1)/N * L with L=%d, Ch9's formula)\n", sent, N);

    // --- Contrast: the WRONG all-reduce where each replica sums all N
    // values itself, in its own rotated order. ---
    std::vector<float> wBad, bBad;
    trainDistributed(data, N, K, lr, wBad, bBad,
                     [](std::vector<std::vector<float>> &g, int n) {
                         everyReplicaSumsItself(g, n);
                     });
    printf("\nContrast -- every replica sums all %d values itself, own order:\n", N);
    for (int r = 0; r < N; r++)
        printf("  Replica %d:  w=%.9f b=%.9f\n", r, wBad[r], bBad[r]);
    printf("Every replica identical to each other: %s\n",
           allSame(wBad, bBad) ? "YES" : "NO");

    return 0;
}
