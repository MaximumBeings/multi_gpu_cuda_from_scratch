// Chapter 2: Multi-GPU Hardware Topology
// 04_topology_bandwidth_model.cpp
//
// Plain host C++. Models three real interconnect topologies as bandwidth
// matrices built from cited, real per-link figures, then computes a
// deterministic "ring bottleneck bandwidth" for each -- previewing why
// Chapter 9's ring all-reduce cares about topology at all. Nothing here
// is a hardware measurement; it's exact arithmetic over cited constants,
// with every modeling simplification called out in a comment.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <limits>

constexpr int N = 8; // 8 GPUs, matching a typical single-node DGX-class box

using Matrix = std::vector<std::vector<double>>;

// Topology A: PCIe-only, dual-socket. GPUs 0-3 sit under CPU0's PCIe
// switch, GPUs 4-7 under CPU1's. Same-switch pairs get a full PCIe 4.0
// x16 link (31.5 GB/s per direction -- real, cited figure). Cross-socket
// pairs must cross the inter-CPU link; we model that, explicitly as a
// simplification (not a vendor spec), as half of the same-switch
// bandwidth, representing the extra hop and shared contention.
Matrix pcieTreeTopology() {
    const double SAME_SWITCH = 31.5;   // PCIe 4.0 x16, per direction
    const double CROSS_SOCKET = SAME_SWITCH / 2.0; // modeling simplification
    Matrix bw(N, std::vector<double>(N, 0.0));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) continue;
            bool sameSocket = (i / 4) == (j / 4);
            bw[i][j] = sameSocket ? SAME_SWITCH : CROSS_SOCKET;
        }
    }
    return bw;
}

// Topology B: a partial NVLink mesh, in the style of pre-NVSwitch
// systems such as DGX-1 -- illustrative of the PATTERN (not a claim to
// reproduce any specific real system's exact wiring diagram). Each GPU
// is directly NVLink-wired to 4 of the other 7 (real NVLink 2.0 figure:
// 25 GB/s per link, 2 links per wired pair = 50 GB/s, V100-class). The
// remaining, non-wired pairs are NOT reachable by relaying through a
// third GPU on these systems -- they fall back to plain PCIe, the same
// real PCIe 4.0 x16 figure (31.5 GB/s) used in the tree topology above.
// This is the actual, well-documented DGX-1-era problem this section's
// prose describes: which pair you pick determines whether you get
// NVLink or PCIe speed, and there is no in-between.
Matrix partialNvlinkMeshTopology() {
    const double DIRECT = 50.0;      // NVLink 2.0, 2 links, 25 GB/s each
    const double PCIE_FALLBACK = 31.5; // PCIe 4.0 x16, same figure as above
    Matrix bw(N, std::vector<double>(N, PCIE_FALLBACK));
    for (int i = 0; i < N; ++i) bw[i][i] = 0.0;
    // Direct NVLink wiring: each GPU i <-> (i+1)%N and i <-> (i+2)%N (a
    // simple illustrative ring-plus-skip graph giving degree 4 per node,
    // matching DGX-1's own degree-4-per-GPU hybrid cube-mesh pattern).
    for (int i = 0; i < N; ++i) {
        for (int step : {1, 2, N - 1, N - 2}) {
            int j = (i + step) % N;
            bw[i][j] = DIRECT;
        }
    }
    return bw;
}

// Topology C: NVSwitch, DGX H100-style. Every GPU reaches every other
// GPU at the full per-GPU NVLink 4.0 aggregate bandwidth (900 GB/s,
// real, cited H100 SXM figure) -- that's the entire point of a
// non-blocking crossbar switch.
Matrix nvswitchTopology() {
    const double UNIFORM = 900.0;
    Matrix bw(N, std::vector<double>(N, UNIFORM));
    for (int i = 0; i < N; ++i) bw[i][i] = 0.0;
    return bw;
}

// The bandwidth achievable by a ring collective is set by its weakest
// link -- every GPU in the ring sends and receives every round, so the
// ring's throughput cannot exceed its slowest edge.
double ringBottleneck(const Matrix& bw, const std::vector<int>& order) {
    double bottleneck = std::numeric_limits<double>::max();
    for (size_t i = 0; i < order.size(); ++i) {
        int a = order[i];
        int b = order[(i + 1) % order.size()];
        bottleneck = std::min(bottleneck, bw[a][b]);
    }
    return bottleneck;
}

void printMatrix(const char* label, const Matrix& bw) {
    printf("%s (GB/s):\n", label);
    for (int i = 0; i < N; ++i) {
        printf("  ");
        for (int j = 0; j < N; ++j) printf("%6.1f", bw[i][j]);
        printf("\n");
    }
}

int main() {
    Matrix pcie = pcieTreeTopology();
    Matrix mesh = partialNvlinkMeshTopology();
    Matrix nvswitch = nvswitchTopology();

    printMatrix("PCIe tree (dual-socket)", pcie);
    printMatrix("Partial NVLink mesh", mesh);
    printMatrix("NVSwitch (full crossbar)", nvswitch);

    // Natural order: GPU 0,1,2,...,7 -- the order you'd get by just
    // indexing devices, with no topology awareness at all.
    std::vector<int> naturalOrder = {0, 1, 2, 3, 4, 5, 6, 7};

    printf("\nRing bottleneck bandwidth, GPUs visited in natural index "
           "order 0->1->2->...->7->0:\n");
    printf("  PCIe tree:      %6.2f GB/s\n", ringBottleneck(pcie, naturalOrder));
    printf("  Partial mesh:   %6.2f GB/s\n", ringBottleneck(mesh, naturalOrder));
    printf("  NVSwitch:       %6.2f GB/s\n", ringBottleneck(nvswitch, naturalOrder));

    printf("\nFor the partial mesh specifically, the natural order above "
           "happens to only use directly-wired (50 GB/s) links -- every "
           "step is a step-1 hop, and step-1 is one of this graph's wired "
           "offsets. Reordering to include one step-3 pair (not wired, "
           "so it falls back to plain PCIe) shows what happens when the "
           "visiting order doesn't match the physical wiring:\n");
    std::vector<int> hitsUnwiredPair = {0, 3, 1, 2, 4, 5, 6, 7}; // 0->3 is step-3, unwired
    printf("  Order that includes one unwired (step-3) pair: %6.2f GB/s\n",
           ringBottleneck(mesh, hitsUnwiredPair));

    printf("\nNVSwitch's whole point: bottleneck bandwidth is %.1f GB/s "
           "regardless of visiting order, because there is no such thing "
           "as an indirect hop -- every pair is one switch away.\n", 900.0);

    return 0;
}
