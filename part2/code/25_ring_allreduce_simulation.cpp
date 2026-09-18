// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 25_ring_allreduce_simulation.cpp
//
// Plain host C++ -- verifying the real ring all-reduce algorithm's
// correctness, and computing its real, counted data-volume advantage
// over Section 9.1's naive baseline. N real in-memory buffers stand in
// for N real device memories; every step below moves exactly one
// chunk per device, exactly like Section 9.2's single real
// cudaMemcpyPeer step, repeated 2*(N-1) times.
#include <cstdio>
#include <vector>

using Chunks = std::vector<int>; // one device's chunks, indexed by chunk id

int main() {
    const int N = 4;

    std::vector<Chunks> devices(N, Chunks(N));
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < N; ++c)
            devices[i][c] = (i + 1) * 10 + c;

    Chunks reference(N, 0);
    for (int c = 0; c < N; ++c)
        for (int i = 0; i < N; ++i)
            reference[c] += devices[i][c];

    printf("Independent reference (sum of all %d devices' chunks): ", N);
    for (int c = 0; c < N; ++c) printf("%d ", reference[c]);
    printf("\n\n");

    const int totalSteps = 2 * (N - 1);
    for (int step = 0; step < totalSteps; ++step) {
        std::vector<Chunks> snapshot = devices;
        bool scatterReducePhase = (step < N - 1);
        for (int j = 0; j < N; ++j) {
            int sendChunk = ((j - step) % N + N) % N;
            int next = (j + 1) % N;
            if (scatterReducePhase) {
                devices[next][sendChunk] = snapshot[next][sendChunk] + snapshot[j][sendChunk];
            } else {
                devices[next][sendChunk] = snapshot[j][sendChunk];
            }
        }
    }

    bool allMatch = true;
    for (int i = 0; i < N; ++i) {
        if (devices[i] != reference) allMatch = false;
    }
    printf("After %d ring steps (scatter-reduce + all-gather), all %d "
           "devices hold the reference sum: %s\n",
           totalSteps, N, allMatch ? "PASS" : "FAIL");

    // Real, counted (not fabricated) data-volume comparison against
    // Section 9.1's naive reduce-then-broadcast baseline, for this
    // exact same problem size. "Traffic" below counts BOTH directions
    // (sent + received) for every device, so the two algorithms are
    // compared on the same basis.
    const int bufferSize = N; // elements per device (N chunks of 1 element each)

    // Naive: root receives (N-1)*bufferSize during reduce, then sends
    // (N-1)*bufferSize during broadcast -- both directions through
    // the SAME device.
    long long naiveRootTraffic = 2LL * (N - 1) * bufferSize;
    // Every other device just sends once and receives once, each of
    // one full buffer.
    long long naiveOtherTraffic = 2LL * bufferSize;

    // Ring: every device sends exactly 1 chunk AND receives exactly 1
    // chunk at every one of the 2*(N-1) steps.
    long long ringPerDeviceTraffic = 2LL * totalSteps * 1;

    printf("\nData-volume comparison for this exact problem (N=%d devices, "
           "%d elements/device):\n", N, bufferSize);
    printf("  Naive root device's traffic:      %lld elements (bottleneck)\n", naiveRootTraffic);
    printf("  Naive non-root device's traffic:  %lld elements\n", naiveOtherTraffic);
    printf("  Ring: EVERY device's traffic:      %lld elements\n", ringPerDeviceTraffic);
    printf("  Ratio (naive root / ring device):  %.2fx -- equals N/2 exactly.\n",
           (double)naiveRootTraffic / (double)ringPerDeviceTraffic);

    return allMatch ? 0 : 1;
}
