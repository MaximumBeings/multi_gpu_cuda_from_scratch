// Chapter 4: Peer-to-Peer Memory Access and Unified Virtual Addressing
// 10_p2p_simulation.cpp
//
// Plain host C++ -- this book's own honesty-discipline technique for
// multi-device logic (see getting-started.md): N real in-memory buffers
// stand in for N real device memories, and we exchange data between
// them using the exact same message pattern real code would use --
// either a direct write (only valid where peer access exists, mirroring
// a UVA pointer dereference) or a copy staged through a separate host
// buffer (always valid, mirroring the universal fallback) -- then check
// the result against an independently-kept reference.
#include <cstdio>
#include <vector>
#include <cstring>

using Buffer = std::vector<int>;

// Which ordered pairs have peer access, in this simulated 4-device
// system. Modeled after Chapter 2's "partial mesh" topology: device 0
// has a direct link to device 1 (peer access possible), but NOT to
// device 2 (no direct link -- must fall back to a host-staged copy).
bool peerAccessible(int src, int dst) {
    static const bool table[4][4] = {
        /*        d0     d1     d2     d3 */
        /*d0*/ { false,  true, false,  true },
        /*d1*/ {  true, false,  true, false },
        /*d2*/ { false,  true, false,  true },
        /*d3*/ {  true, false,  true, false },
    };
    return table[src][dst];
}

// Direct P2P write: only valid if peer access exists between src and
// dst. This is the simulated equivalent of a kernel on the source
// device dereferencing a UVA pointer that resolves directly into the
// destination device's memory -- no host involved at all.
bool transferDirectP2P(std::vector<Buffer>& devices, int src, int dst) {
    if (!peerAccessible(src, dst)) return false;
    devices[dst] = devices[src]; // the "direct write", simulated
    return true;
}

// Host-staged transfer: always valid, exactly the universal fallback
// real cudaMemcpy-through-host-buffer code uses when peer access isn't
// available for a given pair.
void transferStagedViaHost(std::vector<Buffer>& devices, Buffer& hostStaging,
                            int src, int dst) {
    hostStaging = devices[src];  // device -> host
    devices[dst] = hostStaging;  // host -> device
}

int main() {
    const int NUM_DEVICES = 4;
    const int N = 6;

    std::vector<Buffer> devices(NUM_DEVICES, Buffer(N, 0));
    Buffer hostStaging(N, 0);

    // Independent reference, kept completely separate from the
    // simulated device buffers, so the check below cannot pass just
    // because both sides share the same underlying bug.
    Buffer reference = {10, 20, 30, 40, 50, 60};
    devices[0] = reference;

    printf("Device 0 initialized. Attempting device 0 -> device 1 "
           "(peer-accessible per the simulated topology) and "
           "device 0 -> device 2 (NOT peer-accessible).\n\n");

    bool usedDirectFor1 = transferDirectP2P(devices, 0, 1);
    printf("Transfer 0->1: %s\n",
           usedDirectFor1 ? "direct P2P write (peer access exists)"
                          : "fell back (should not happen for this pair)");

    bool usedDirectFor2 = transferDirectP2P(devices, 0, 2);
    if (!usedDirectFor2) {
        printf("Transfer 0->2: direct P2P refused (no peer access, "
               "correctly) -- falling back to host-staged copy.\n");
        transferStagedViaHost(devices, hostStaging, 0, 2);
    }

    bool pass1 = (devices[1] == reference);
    bool pass2 = (devices[2] == reference);

    printf("\nCorrectness check, device 1 contents vs. independent "
           "reference: %s\n", pass1 ? "PASS" : "FAIL");
    printf("Correctness check, device 2 contents vs. independent "
           "reference: %s\n", pass2 ? "PASS" : "FAIL");

    printf("\nBoth destinations hold identical, correct data despite "
           "using two different transfer mechanisms -- exactly the "
           "property real P2P code depends on: the *method* changes "
           "with topology, the *result* must not.\n");

    return (pass1 && pass2) ? 0 : 1;
}
