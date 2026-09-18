// Chapter 8: Broadcast and Reduce -- The First Two Collectives, By Hand
// 22_collectives_simulation.cpp
//
// Plain host C++ -- this book's own honesty-discipline technique,
// used here for the first time to verify a genuinely multi-device
// ALGORITHM (not just a single API call's honest failure): N real
// in-memory buffers stand in for N real device memories, and we run
// the exact real message pattern -- one root, N-1 recipients for
// broadcast; N sources, one accumulator for reduce -- checked against
// an independently-computed reference.
#include <cstdio>
#include <vector>

using Buffer = std::vector<int>;

// Broadcast: every non-root device ends up holding an exact copy of
// the root's data. This is the simulated equivalent of Section 8.1's
// per-device cudaMemcpyPeer loop.
void broadcast(std::vector<Buffer>& devices, int root) {
    for (size_t i = 0; i < devices.size(); ++i) {
        if ((int)i == root) continue;
        devices[i] = devices[root];
    }
}

// Reduce: sum every device's buffer, elementwise, into a single
// result. This is the simulated equivalent of Section 8.2's
// per-device stage-then-accumulate loop -- summed here in device
// order 0, 1, 2, ... N-1.
Buffer reduceSumInOrder(const std::vector<Buffer>& devices) {
    Buffer result(devices[0].size(), 0);
    for (const Buffer& d : devices) {
        for (size_t i = 0; i < result.size(); ++i) result[i] += d[i];
    }
    return result;
}

// The same reduction, summed in the REVERSE device order. For
// integers, exactly like this section's own accumulator, addition is
// associative and commutative, so this must produce an identical
// result to reduceSumInOrder -- a guarantee that would NOT hold if
// these were floating-point values (see this chapter's Common Trap).
Buffer reduceSumReverseOrder(const std::vector<Buffer>& devices) {
    Buffer result(devices[0].size(), 0);
    for (size_t i = 0; i < result.size(); ++i) {
        long long sum = 0;
        for (int d = (int)devices.size() - 1; d >= 0; --d) sum += devices[d][i];
        result[i] = (int)sum;
    }
    return result;
}

int main() {
    const int NUM_DEVICES = 4;
    const int N = 4;

    // Broadcast check.
    std::vector<Buffer> broadcastDevices(NUM_DEVICES, Buffer(N, 0));
    Buffer rootData = {7, 14, 21, 28};
    broadcastDevices[0] = rootData;
    broadcast(broadcastDevices, 0);

    bool broadcastPass = true;
    for (int i = 0; i < NUM_DEVICES; ++i) {
        if (broadcastDevices[i] != rootData) broadcastPass = false;
    }
    printf("Broadcast: all %d devices hold an identical copy of the root's "
           "data: %s\n", NUM_DEVICES, broadcastPass ? "PASS" : "FAIL");

    // Reduce check.
    std::vector<Buffer> reduceDevices = {
        {1, 2, 3, 4},
        {10, 20, 30, 40},
        {100, 200, 300, 400},
        {1000, 2000, 3000, 4000},
    };
    Buffer reference = {1111, 2222, 3333, 4444}; // computed independently, by hand

    Buffer sumInOrder = reduceSumInOrder(reduceDevices);
    Buffer sumReverse = reduceSumReverseOrder(reduceDevices);

    bool reduceInOrderPass = (sumInOrder == reference);
    bool reduceReversePass = (sumReverse == reference);
    bool orderIndependent = (sumInOrder == sumReverse);

    printf("Reduce (device order 0..3): result matches independent "
           "reference: %s\n", reduceInOrderPass ? "PASS" : "FAIL");
    printf("Reduce (device order 3..0): result matches independent "
           "reference: %s\n", reduceReversePass ? "PASS" : "FAIL");
    printf("Both accumulation orders produce the IDENTICAL result "
           "(true for integers; NOT guaranteed for floating point): %s\n",
           orderIndependent ? "PASS" : "FAIL");

    return (broadcastPass && reduceInOrderPass && reduceReversePass && orderIndependent) ? 0 : 1;
}
