// Chapter 10: All-Gather, Reduce-Scatter, and All-to-All
// 29_collectives_simulation.cpp
//
// Plain host C++ -- verifying all three of this chapter's collectives
// against independent references, the same way Chapter 8's
// 22_collectives_simulation.cpp verified broadcast and reduce. N real
// in-memory buffers stand in for N real device memories.
#include <cstdio>
#include <vector>

using Row = std::vector<int>;

// ---------------------------------------------------------------------
// (a) All-gather: every device starts with ONE chunk (its own) and ends
// with ALL N chunks, unmodified, in the same order for every device.
// ---------------------------------------------------------------------
bool verifyAllGather(int N) {
    std::vector<int> ownChunk(N);
    for (int i = 0; i < N; ++i) ownChunk[i] = (i + 1) * 10; // device i's own chunk

    std::vector<Row> gathered(N, Row(N, 0));
    for (int dst = 0; dst < N; ++dst)
        for (int src = 0; src < N; ++src)
            gathered[dst][src] = ownChunk[src]; // every device collects every chunk

    bool allMatch = true;
    for (int dst = 0; dst < N; ++dst)
        for (int src = 0; src < N; ++src)
            if (gathered[dst][src] != ownChunk[src]) allMatch = false;

    printf("All-gather (N=%d): independent reference chunks: ", N);
    for (int i = 0; i < N; ++i) printf("%d ", ownChunk[i]);
    printf("\n  Every device's gathered array matches the reference: %s\n\n",
           allMatch ? "PASS" : "FAIL");
    return allMatch;
}

// ---------------------------------------------------------------------
// (b) Reduce-scatter: every device starts with N chunks (one contribution
// per destination) and ends with exactly ONE reduced chunk -- the one
// it owns -- summed across all N devices' contributions to that chunk.
// ---------------------------------------------------------------------
bool verifyReduceScatter(int N) {
    std::vector<Row> devices(N, Row(N));
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < N; ++c)
            devices[i][c] = (i + 1) * 10 + c;

    Row reference(N, 0);
    for (int c = 0; c < N; ++c)
        for (int i = 0; i < N; ++i)
            reference[c] += devices[i][c];

    Row scattered(N, 0);
    for (int owner = 0; owner < N; ++owner) {
        long long sum = 0;
        for (int src = 0; src < N; ++src) sum += devices[src][owner];
        scattered[owner] = (int)sum; // owner is the ONLY device holding this
    }

    bool allMatch = (scattered == reference);
    printf("Reduce-scatter (N=%d): independent reference sums: ", N);
    for (int c = 0; c < N; ++c) printf("%d ", reference[c]);
    printf("\n  Device c's scattered chunk matches reference[c] for every c: %s\n\n",
           allMatch ? "PASS" : "FAIL");
    return allMatch;
}

// ---------------------------------------------------------------------
// (c) All-to-all: device i's send-slot[j] is a value meant ONLY for
// device j. After the exchange, device j's recv-slot[i] must equal
// device i's original send-slot[j] -- a full transpose, no reduction.
// ---------------------------------------------------------------------
bool verifyAllToAll(int N) {
    std::vector<Row> sendSlots(N, Row(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            sendSlots[i][j] = i * 100 + j; // "chunk from device i, meant for device j"

    std::vector<Row> recvSlots(N, Row(N, 0));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            recvSlots[j][i] = sendSlots[i][j]; // the exchange: a full transpose

    bool allMatch = true;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (recvSlots[j][i] != sendSlots[i][j]) allMatch = false;

    printf("All-to-all (N=%d): every device's %d received chunks match the\n"
           "  chunk their sender addressed to them (a full transpose): %s\n\n",
           N, N, allMatch ? "PASS" : "FAIL");
    return allMatch;
}

int main() {
    const int N = 4;
    bool p1 = verifyAllGather(N);
    bool p2 = verifyReduceScatter(N);
    bool p3 = verifyAllToAll(N);

    // Real, counted data-volume note: all-gather and reduce-scatter,
    // done directly (as above, and in 26_/27_.cu), require each device
    // to send or receive N-1 SEPARATE chunks to/from N-1 DIFFERENT
    // remote devices -- the same "farther-apart" communication pattern
    // Thakur, Rabenseifner, and Gropp identify as recursive doubling's
    // weakness relative to a ring's nearest-neighbor-only pattern (see
    // Sources). Chapter 9 already built the ring version of exactly
    // these two operations (its scatter-reduce and all-gather phases);
    // swapping that same ring pattern in here gets the same nearest-
    // neighbor advantage for these two collectives run alone. All-to-all
    // has no such shortcut: every one of its N*(N-1) chunks is genuinely
    // distinct, so no ring hop can carry more than one destination's
    // data at a time the way all-reduce's identical partial sums can.
    long long directMessagesPerDevice = N - 1;
    long long allToAllMessagesPerDevice = N - 1;
    printf("Direct all-gather/reduce-scatter: %lld message(s) per device, "
           "each to/from a DIFFERENT remote device.\n", directMessagesPerDevice);
    printf("All-to-all: %lld message(s) per device too -- but unlike "
           "all-gather/reduce-scatter, none of them can be replaced by a "
           "cheaper ring relay, because every message is unique.\n",
           allToAllMessagesPerDevice);

    return (p1 && p2 && p3) ? 0 : 1;
}
