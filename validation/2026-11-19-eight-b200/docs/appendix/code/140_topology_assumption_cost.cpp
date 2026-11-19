// Appendix G: Common Failure Modes -- Deadlocks, Silent Corruption From
// Missed Synchronization, and Topology Mismatches
// 140_topology_assumption_cost.cpp
//
// Appendix G.3 -- Chapter 4's own File 9 established the real,
// idiomatic sequence: query cudaDeviceCanAccessPeer() for a pair before
// assuming anything about it, then enable access only for pairs that
// genuinely support it. Chapter 5's own File 11/13 showed why the query
// matters for CORRECTNESS: cudaMemcpyPeer() and cudaMemcpyPeerAsync()
// still route correctly (falling back to a staged host copy) even when
// peer access was never enabled for a pair -- so skipping the query
// never produces a wrong ANSWER. What it can silently produce is a
// wrong PERFORMANCE ASSUMPTION: code that hardcodes "this pair is
// NVLink-wired" (a real, common shortcut -- baking in a number from one
// deployment's topology rather than re-querying it) pays no error and
// no crash when that assumption is false on a DIFFERENT deployment's
// topology, only a silent multiple of Chapter 5's own already-cited
// bandwidth figures. This file extends Chapter 5's own File 13 cost
// model with exactly that comparison: the real cost of each of the
// three topology classes Chapter 2 already established, against what
// code that assumed the WRONG one actually pays.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 140_topology_assumption_cost.cpp -o 140_topology_assumption_cost
// Run:     ./140_topology_assumption_cost
#include <cstdio>
#include <cmath>

int main() {
    printf("=== Section G.3: assuming a topology class instead of querying it ===\n\n");

    printf("Chapter 4's own real idiom (File 9): cudaDeviceCanAccessPeer(&canAccess,\n");
    printf("i, j), checked for every ordered pair, BEFORE calling\n");
    printf("cudaDeviceEnablePeerAccess(). Chapter 5's own File 11/13 showed why\n");
    printf("skipping it never breaks CORRECTNESS -- cudaMemcpyPeer() falls back to a\n");
    printf("staged host copy transparently for any pair without enabled peer access.\n");
    printf("What it CAN silently break is a performance assumption baked into calling\n");
    printf("code that never re-queries the real topology for the deployment it is\n");
    printf("actually running on.\n\n");

    // Chapter 2's own already-cited bandwidth figures, reused by
    // Chapter 5's own File 13 (not re-derived here).
    const double PCIE_GBPS = 31.5;
    const double NVLINK_GBPS = 50.0;
    const double transferSizeGB = 1.0;

    double nvlinkTime = transferSizeGB / NVLINK_GBPS;              // 1-hop, NVLink-wired pair
    double pcieDirect = transferSizeGB / PCIE_GBPS;                // 1-hop, PCIe-only direct pair
    double stagedTime = 2.0 * (transferSizeGB / PCIE_GBPS);        // 2-hop, no direct link at all

    printf("Chapter 2/5's own three real topology classes for a device pair, and the\n");
    printf("real transfer time for %.1f GB under each:\n\n", transferSizeGB);
    printf("%-32s %14s\n", "actual topology class", "real time (s)");
    printf("%-32s %14.6f\n", "NVLink-wired pair (1 hop)", nvlinkTime);
    printf("%-32s %14.6f\n", "PCIe-only direct pair (1 hop)", pcieDirect);
    printf("%-32s %14.6f\n", "no direct link (2-hop staged)", stagedTime);
    printf("\n");

    printf("=== the silent cost of code that ASSUMED NVLink and never re-queried ===\n\n");
    printf("code that hardcodes the NVLink figure (a real, common shortcut when a\n");
    printf("number is copied from one cluster's own topology report into a capacity\n");
    printf("plan or a scheduling deadline) pays NO error and NO crash when deployed\n");
    printf("on a DIFFERENT pair's real topology -- cudaMemcpyPeer() still returns\n");
    printf("cudaSuccess either way -- only a silent multiple of the assumed number:\n\n");

    printf("%-32s %14s %12s\n", "actual topology at runtime", "real time (s)", "x assumed");
    printf("%-32s %14.6f %12.4f\n", "NVLink (assumption correct)", nvlinkTime, nvlinkTime / nvlinkTime);
    printf("%-32s %14.6f %12.4f\n", "PCIe-only direct", pcieDirect, pcieDirect / nvlinkTime);
    printf("%-32s %14.6f %12.4f\n", "no direct link (staged)", stagedTime, stagedTime / nvlinkTime);
    printf("\n");

    double pcieSlowdown = pcieDirect / nvlinkTime;
    double stagedSlowdown = stagedTime / nvlinkTime;

    printf("a schedule or capacity plan built around the NVLink number silently runs\n");
    printf("%.4fx slower than planned on a PCIe-only-direct pair, and %.4fx slower on\n",
           pcieSlowdown, stagedSlowdown);
    printf("a pair with no direct link at all -- Chapter 2's own partial-mesh topology\n");
    printf("model already established that NOT every pair on a real multi-GPU machine\n");
    printf("shares the same link class, so this is not a hypothetical: two GPUs on the\n");
    printf("SAME machine can genuinely fall into different rows of this table.\n\n");

    printf("=== the real fix: query, don't assume ===\n\n");
    printf("Chapter 4's own File 9 sequence -- cudaDeviceCanAccessPeer() for the\n");
    printf("SPECIFIC pair actually in use, re-run on the actual deployment rather than\n");
    printf("hardcoded from a different one -- is the only way to know which row of\n");
    printf("this table applies BEFORE building a schedule or capacity plan around it.\n");
    printf("Real device attribute queries (Chapter 2's own cudaDeviceProp fields, and\n");
    printf("cudaDeviceGetP2PAttribute() for the specific link's own reported\n");
    printf("performance rank) are what genuinely observe the topology; nothing about\n");
    printf("cudaMemcpyPeer()'s own return code ever will.\n\n");

    bool orderingCorrect = (nvlinkTime < pcieDirect) && (pcieDirect < stagedTime);
    bool slowdownsPositive = (pcieSlowdown > 1.0) && (stagedSlowdown > pcieSlowdown);
    printf("self-check: the three real topology classes are strictly ordered fastest-\n");
    printf("to-slowest (NVLink < PCIe-direct < staged, %s), and an NVLink-based\n",
           orderingCorrect ? "confirmed" : "MISMATCH");
    printf("assumption's own real slowdown strictly increases as the actual topology\n");
    printf("gets worse (%s): %s\n", slowdownsPositive ? "confirmed" : "MISMATCH",
           (orderingCorrect && slowdownsPositive) ? "confirmed" : "MISMATCH");

    return (orderingCorrect && slowdownsPositive) ? 0 : 1;
}
