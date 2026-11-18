// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 25_ring_allreduce_simulation.cpp
//
// Plain host C++ -- verifying the real ring all-reduce algorithm's
// correctness, and comparing its data volume against Section 9.1's
// naive baseline. The ring's traffic is COUNTED from the simulation
// loop itself; the naive baseline's is a closed-form formula. N real in-memory buffers stand in
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
    std::vector<long long> sent(N, 0), received(N, 0); // elements, counted
    for (int step = 0; step < totalSteps; ++step) {
        std::vector<Chunks> snapshot = devices;
        bool scatterReducePhase = (step < N - 1);
        for (int j = 0; j < N; ++j) {
            int sendChunk = ((j - step) % N + N) % N;
            int next = (j + 1) % N;
            ++sent[j];        // one 1-element chunk leaves device j...
            ++received[next]; // ...and arrives at device next
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

    // Data-volume comparison against Section 9.1's naive
    // reduce-then-broadcast baseline, for this exact same problem size.
    // "Traffic" below counts BOTH directions (sent + received) for
    // every device, so the two algorithms are compared on the same
    // basis. Gibiansky's 2(N-1)K/N counts ONE direction (what each GPU
    // sends, which equals what it receives), so it is exactly half of
    // the two-direction figure -- printed separately below.
    const int bufferSize = N; // elements per device (N chunks of 1 element each)

    // Naive (closed form, idealized root-centered star): the hub
    // receives (N-1)*bufferSize during reduce, then sends
    // (N-1)*bufferSize during broadcast. (File 23 splits this hub
    // between the host and the root; the star shape is the same.)
    long long naiveRootTraffic = 2LL * (N - 1) * bufferSize;
    // Every other device just sends once and receives once, each of
    // one full buffer.
    long long naiveOtherTraffic = 2LL * bufferSize;

    // Ring: counted from the loop above. Every device must have the
    // SAME count -- that equality is the "no device singled out" claim.
    long long ringPerDeviceTraffic = sent[0] + received[0];
    bool ringBalanced = true;
    for (int i = 0; i < N; ++i)
        if (sent[i] + received[i] != ringPerDeviceTraffic || sent[i] != received[i])
            ringBalanced = false;

    printf("\nData-volume comparison for this exact problem (N=%d devices, "
           "%d elements/device):\n", N, bufferSize);
    printf("  Naive hub's traffic (closed form): %lld elements (bottleneck)\n", naiveRootTraffic);
    printf("  Naive non-hub device's traffic:   %lld elements\n", naiveOtherTraffic);
    printf("  Ring: EVERY device's traffic:      %lld elements (counted; all devices equal: %s)\n",
           ringPerDeviceTraffic, ringBalanced ? "PASS" : "FAIL");
    printf("  Ring: elements SENT per device:    %lld = Gibiansky's 2(N-1)K/N = %d\n",
           sent[0], 2 * (N - 1) * bufferSize / N);
    printf("  Ratio (naive hub / ring device):   %.2fx -- equals N/2 exactly.\n",
           (double)naiveRootTraffic / (double)ringPerDeviceTraffic);

    return (allMatch && ringBalanced) ? 0 : 1;
}
