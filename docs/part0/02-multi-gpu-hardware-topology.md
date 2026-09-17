# Chapter 2: Multi-GPU Hardware Topology: PCIe, NVLink, and NVSwitch

**What you will understand by the end of this chapter:**

- Why "GPU-to-GPU bandwidth" is a matrix, not a single number, and what determines which entry in that matrix two specific GPUs get.
- The real bandwidth figures for PCIe (generations 3 through 5), NVLink (generations 1.0 through 4.0), and NVSwitch, and how each one fills in that matrix differently.
- Why a communication pattern's achievable bandwidth is set by its single weakest link, not its average link — and why that makes topology, not just raw link speed, a first-class design concern.
- What the CUDA Runtime API can honestly report about device-pair connectivity even without real multi-GPU hardware present.

**What you need to know first:**

- Chapter 1's memory-capacity and compute-throughput walls, and why splitting work across devices is the fix for both.
- No prior knowledge of PCIe, NVLink, or NVSwitch is assumed — this chapter builds all three from their real published specifications.

---

Chapter 1 established that splitting work across GPUs solves both the capacity wall and the throughput wall, but introduces a new one: the devices now have to talk to each other. This chapter is about the physical layer that conversation happens over.

## 2.1 Bandwidth Is a Graph, Not a Number

### Intuition

A single GPU has exactly one bandwidth figure worth knowing: how fast it talks to its own memory. The moment a second GPU enters the picture, "bandwidth" stops being a single fact about the machine and becomes a fact about a *pair* — like asking "how far is it" without saying between which two cities. Some pairs of GPUs sit right next to each other on the fastest link the system has; others are separated by several intermediate hops, and the answer for that pair can be an order of magnitude worse, on the exact same machine.

### Background

Three real technologies fill in that pairwise bandwidth matrix in practice, in increasing order of how good an answer they give: PCIe (every GPU has it, and it's the universal fallback), NVLink (a direct, high-bandwidth point-to-point link some GPU pairs have), and NVSwitch (a crossbar that makes *every* pair behave like a direct NVLink link). The rest of this chapter goes through each with real published bandwidth figures, then measures what a difference in topology actually costs a real communication pattern.

```text
PCIe-only pair (through a shared switch):        NVLink pair (direct):
GPU0 --- PCIe switch --- GPU1                    GPU0 ======= NVLink ======= GPU1
(moderate, shared bandwidth)                     (direct, much higher bandwidth)
```

## 2.2 PCIe: The Baseline Every GPU Has

### Intuition

Every GPU, however it's also connected, sits on a PCI Express link to its host system — think of it as the one road every property is guaranteed to have a driveway onto, even if a private highway also happens to connect two particular houses directly. PCIe bandwidth has roughly doubled with each generation, but the number that matters for a specific *pair* of GPUs depends on where in the PCIe road network they each sit, not just which generation the roads are paved to.

### Background

| Generation | x16 link bandwidth (per direction) |
|---|---|
| PCIe 3.0 | 15.8 GB/s |
| PCIe 4.0 | 31.5 GB/s |
| PCIe 5.0 | 63.0 GB/s |

Two GPUs sharing the same PCIe switch get the full per-generation bandwidth between them. Two GPUs on different CPU sockets have to cross both PCIe switches *and* whatever link joins the two sockets (a NUMA interconnect such as Intel's UPI or AMD's Infinity Fabric) — a shared, finite resource every other cross-socket transfer on the machine is also competing for:

```text
CPU0 -- PCIe switch -- GPU0, GPU1, GPU2, GPU3   (same-switch pairs: full x16 bandwidth)
CPU1 -- PCIe switch -- GPU4, GPU5, GPU6, GPU7   (same-switch pairs: full x16 bandwidth)
CPU0 <---- inter-socket link, shared/finite ----> CPU1
(a pair split across sockets, e.g. GPU0<->GPU4, crosses this shared link too)
```

The topology model in Section 2.5 makes the size of this gap concrete rather than just asserted: a same-switch PCIe 4.0 pair gets 31.5 GB/s, while a cross-socket pair (modeled here as half that, to represent the extra hop and shared contention) gets half.

!!! warning "[COMMON TRAP] Assuming "PCIe 4.0 everywhere" means uniform bandwidth"
    Knowing every GPU in a machine is on PCIe 4.0 tells you the *ceiling* for any pair, not the bandwidth any specific pair actually gets. A same-switch pair reaches that ceiling; a cross-socket pair does not, regardless of generation, because the bottleneck moves to a completely different, shared link the generation number says nothing about.

## 2.3 NVLink and NVSwitch: When PCIe Isn't Enough

### Intuition

NVLink is a private highway built directly between two specific houses — dramatically faster than the shared public road (PCIe), but only for the houses it was actually built to connect. NVSwitch is what happens when, instead of building private highways between some pairs of houses and not others, someone builds one central interchange that every house connects to, so any two houses get highway-speed travel between them, with no favorites.

### Background

Each NVLink generation has shipped with a new GPU architecture and a real, published bandwidth jump:

| NVLink generation | Introduced with | Per-link bandwidth | Links per GPU | Total per-GPU bandwidth |
|---|---|---|---|---|
| 1.0 | P100 (Pascal) | 20 GB/s | 4 | 160 GB/s |
| 2.0 | V100 (Volta) | 25 GB/s | 6 | 300 GB/s |
| 3.0 | A100 (Ampere) | 25 GB/s | 12 | 600 GB/s |
| 4.0 | H100 (Hopper) | 25 GB/s | 18 | 900 GB/s |

An H100's 900 GB/s of total NVLink bandwidth is roughly **29 times** a PCIe 4.0 x16 link's 31.5 GB/s. But those 18 links don't have to all go to the same neighbor — with only point-to-point NVLink and a fixed number of links per GPU, a system with enough GPUs eventually runs out of direct links to go around. Early multi-GPU NVLink systems (the DGX-1 generation) had exactly this shape: each GPU directly wired to some, but not all, of the others, with the rest falling back to plain PCIe.

NVSwitch removes that tradeoff by putting a crossbar switch chip between the GPUs instead of wiring them directly to each other:

```text
Partial NVLink mesh (some pairs direct,           NVSwitch (every pair, same bandwidth):
some fall back to PCIe):
                                                          [ NVSwitch crossbar ]
  GPU0 == GPU1                                          /   |   |   |   |   \
   |  \  /  |          <- diagonals are PCIe          GPU0 GPU1 GPU2 GPU3 ...
  GPU3 == GPU2             fallback, not NVLink       (every spoke: full NVLink
                                                        bandwidth, no exceptions)
```

The practical consequence: on an NVSwitch-equipped system (DGX A100, DGX H100, and later), *every* GPU pair gets the GPU's full NVLink bandwidth — 900 GB/s for H100 — with no "did I get lucky with placement" question at all.

!!! warning "[COMMON TRAP] Assuming a GPU's total NVLink bandwidth means every pair gets that much"
    An H100's "900 GB/s of NVLink" describes the total across all 18 of its links, not a guarantee that any two specific H100s in a system share that much. On a partial-mesh system, a directly-wired pair can approach that figure; a non-wired pair gets ordinary PCIe speed instead, a difference Section 2.5 measures directly. Only a full NVSwitch fabric makes the total-bandwidth figure apply uniformly to every pair.

## 2.4 What Topology Costs a Real Communication Pattern

### Intuition

Imagine a relay race where the runners are connected pairwise by roads of different speeds. The team's overall time is decided entirely by its *slowest* leg, no matter how fast the other legs are — a blazing final stretch cannot undo a bottleneck earlier in the chain. A ring of communicating GPUs works the same way: the ring's throughput is exactly its slowest link's bandwidth, not an average of all the links.

### Background

Ordinary host C++, no CUDA, models each topology as a bandwidth matrix built from the real per-link figures above, then computes the bandwidth a *ring* of GPUs can sustain — set by its single weakest link, which is exactly what Chapter 9's ring all-reduce depends on.

```cpp
// 04_topology_bandwidth_model.cpp
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
// real, cited H100 SXM figure).
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 04_topology_bandwidth_model.cpp -o topology_model
./topology_model
```

Output (genuinely compiled and run twice; identical both times):

```text
Ring bottleneck bandwidth, GPUs visited in natural index order 0->1->2->...->7->0:
  PCIe tree:      15.75 GB/s
  Partial mesh:   50.00 GB/s
  NVSwitch:       900.00 GB/s

For the partial mesh specifically, the natural order above happens to only use directly-wired (50 GB/s) links -- every step is a step-1 hop, and step-1 is one of this graph's wired offsets. Reordering to include one step-3 pair (not wired, so it falls back to plain PCIe) shows what happens when the visiting order doesn't match the physical wiring:
  Order that includes one unwired (step-3) pair:  31.50 GB/s
```

Three things fall out of this. First, the PCIe tree's ring bottleneck (15.75 GB/s) is the *cross-socket* figure, not the faster same-switch figure — a ring visiting all 8 GPUs in index order necessarily crosses the socket boundary, and that crossing sets the whole ring's speed. Second, the partial mesh's ring bandwidth (50 GB/s) depends entirely on the visiting order matching the physical wiring — swapping in a single unwired pair drops the same 8 GPUs' ring bandwidth to 31.5 GB/s, a 37% cut, from a reordering that looks harmless on paper. Third, NVSwitch's number (900 GB/s) doesn't move no matter what order you pick, which removes an entire class of silent performance bug the other two topologies are exposed to.

## 2.5 What the CUDA Runtime Can Honestly Tell You About Topology, Right Now

### Intuition

Even without a working truck fleet in the lot, you can still call the dispatcher and ask "if I had truck A and truck B, could they share a loading dock?" — the answer is honest and useful even when there are currently zero trucks to actually test it with.

### Background

The exact API this book uses for real starting in Chapter 4 — `cudaDeviceCanAccessPeer()` — can be called for real today, and its honest behavior is worth seeing before it matters.

```cpp
// 05_peer_access_query.cu
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %s, count = %d\n",
           cudaGetErrorString(err), deviceCount);

    if (deviceCount < 2) {
        int canAccess = -1;
        cudaError_t peerErr = cudaDeviceCanAccessPeer(&canAccess, 0, 1);
        printf("cudaDeviceCanAccessPeer(0, 1) -> %s (code %d), "
               "canAccess left at %d\n",
               cudaGetErrorString(peerErr), (int)peerErr, canAccess);
        return 0;
    }

    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            printf("GPU %d -> GPU %d peer access: %s\n",
                   i, j, canAccess ? "yes" : "no");
        }
    }
    return 0;
}
```

```bash
nvcc -arch=sm_80 05_peer_access_query.cu -o peer_access_query
./peer_access_query
```

Output (genuinely compiled with a real `nvcc` and genuinely run; identical across three consecutive runs):

```text
cudaGetDeviceCount(): no CUDA-capable device is detected, count = 0
cudaDeviceCanAccessPeer(0, 1) -> no CUDA-capable device is detected (code 100), canAccess left at -1
```

On real hardware, the standard tool for seeing this chapter's whole topology story at a glance is `nvidia-smi topo -m`, which prints exactly the kind of matrix Section 2.4 built by hand — but for the real system it's run on, straight from the driver. This environment has no driver to run it against, so that table isn't reproduced here as if it were captured; a reader with real multi-GPU hardware is the one who can genuinely fill it in.

!!! warning "[COMMON TRAP] Calling a peer-access function without checking device count first"
    `cudaDeviceCanAccessPeer(&canAccess, 0, 1)` is a well-formed call even when device 1 (or device 0) doesn't exist — it doesn't crash, it returns an honest error code, exactly as shown above. Code that skips checking `cudaGetDeviceCount()` first and just assumes at least two devices exist will still "work" in the sense of not crashing, but every peer-access decision it makes from that point on is built on a check that silently failed.

## Chapter Summary

GPU-to-GPU bandwidth is a matrix indexed by pair, not a single fact about a machine. PCIe is the universal fallback, shaped by the tree it sits in — same-switch pairs are fast, cross-socket pairs are not, regardless of generation (Section 2.2). NVLink replaces specific pairs' PCIe hop with a much faster direct one, but only for the pairs it's actually wired to; NVSwitch removes that distinction entirely by making every pair look like a direct NVLink pair (Section 2.3). A ring communication pattern's throughput is set by its single weakest link, which Section 2.4 measured directly: 15.75 GB/s for a PCIe tree, 50 GB/s for a partial mesh visited in the right order (31.5 GB/s in the wrong one), and a constant 900 GB/s for NVSwitch regardless of order. Section 2.5's `cudaDeviceCanAccessPeer()` call previews the exact real API Chapter 4 builds on, honestly reporting no devices here.

## Self-Check Questions

1. Two GPUs are on the same PCIe switch; two others are on different sockets. Using Section 2.2's figures, what real-world factor explains the second pair's lower bandwidth, given both pairs are nominally "PCIe 4.0 x16"?
2. An engineer claims "our H100s have 900 GB/s of NVLink between every pair" based only on knowing the GPU model. What additional fact about the system would you need to confirm before trusting that claim?
3. In Section 2.4's partial-mesh model, why does reordering the ring to include a single unwired pair drop the bottleneck bandwidth to exactly the PCIe fallback figure (31.5 GB/s), rather than to some intermediate value?
4. Section 2.4 computes a ring's bottleneck as the *minimum* bandwidth across its edges rather than the *average*. Explain, using the relay-race intuition from Section 2.4, why the minimum is the physically correct choice.
5. Why does NVSwitch's ring bottleneck (900 GB/s) not change at all under any reordering, while the partial mesh's does?
6. `05_peer_access_query.cu` calls `cudaDeviceCanAccessPeer(&canAccess, 0, 1)` even when `cudaGetDeviceCount()` already reported 0 devices. What real value does making that call anyway provide, versus simply skipping it once the count is known to be 0?
7. Suppose a real 8-GPU DGX-A100-class (NVSwitch) system and an 8-GPU partial-mesh system both need to run the identical ring collective. Which system's performance is more sensitive to how the software assigns GPU indices to ring positions, and why?

## Where We Go Next

Chapter 3 builds the CUDA-side vocabulary this book needs to actually address more than one device from code — devices, contexts, and streams — which sits on top of whichever of this chapter's three topologies a given machine actually has. Chapter 4 then starts actually moving data across the links this chapter described.

## Worked Solutions

**1.** The same-switch pair's transfer stays entirely within one PCIe switch, reaching the full generation bandwidth (31.5 GB/s for Gen4 x16). The cross-socket pair's transfer must additionally cross the inter-CPU link (e.g. Intel UPI or AMD Infinity Fabric) joining the two sockets — a separate, shared, finite resource that Section 2.2 models as halving the effective bandwidth, independent of what PCIe generation either GPU uses.

**2.** Whether the system has NVSwitch. The 900 GB/s figure is the total bandwidth across all of one GPU's NVLink connections, which only applies uniformly to *every* pair when a crossbar switch connects all GPUs equally (Section 2.3). On a partial-mesh (pre-NVSwitch) system, only directly-wired pairs approach that figure; others fall back to plain PCIe.

**3.** Because the model in Section 2.4 gives every non-wired pair the exact same constant fallback bandwidth (31.5 GB/s, plain PCIe 4.0 x16) rather than a range of degraded values — there's no partial credit for "almost" having a direct NVLink connection. A pair either has the direct wire (50 GB/s) or it doesn't, in which case it gets exactly the PCIe figure, with nothing in between in this model.

**4.** In a ring, every participating GPU both sends and receives on every round, so the round cannot complete faster than its slowest edge allows — a fast edge finishing early still has to wait for the slowest edge before the next round can begin, exactly like a relay team's total time being decided by its slowest leg no matter how fast the others run. Averaging would imply the fast edges could somehow compensate for the slow one, which they cannot in a synchronous ring.

**5.** NVSwitch gives every ordered pair of GPUs the identical bandwidth (900 GB/s, Section 2.3), so no matter which pair a given ring edge connects, that edge's bandwidth is always 900 GB/s — there is no "wrong" edge to accidentally include. The partial mesh instead has two different bandwidth values (50 GB/s wired, 31.5 GB/s fallback) depending on which specific pair an edge connects, so the ring's bottleneck depends on which pairs the chosen order happens to use.

**6.** Making the call anyway, and reporting its real return code, is what lets this exact source run correctly and diagnostically on any machine — including one with fewer than 2 devices — without a separate code path. It also demonstrates, honestly, that the API doesn't crash or need special-casing for an invalid device id; it degrades to a clear error, which is itself useful information for a caller that didn't check the count first (Section 2.5's own Common Trap).

**7.** The partial-mesh system is far more sensitive. On the NVSwitch system, every possible assignment of GPU indices to ring positions gives the identical 900 GB/s bottleneck, so index assignment doesn't matter at all. On the partial-mesh system, Section 2.4 showed a single wrong adjacent pair drops the ring's bottleneck by 37% (50 to 31.5 GB/s) — so the same collective's real performance on that system depends directly on whether the software's chosen ring order happens to match the physical NVLink wiring.

---

**Sources cited in this chapter:**

- [PCIe Bandwidth Chart — GB/s for Gen 1–7 at x1–x16](https://pciesimulator.com/pcie-bandwidth-table/) -- PCIe 3.0/4.0/5.0 x16 per-direction bandwidth figures.
- [NVLink — Wikipedia](https://en.wikipedia.org/wiki/NVLink) -- NVLink generation-by-generation per-link bandwidth and links-per-GPU figures (1.0 through 4.0).
- [NVIDIA H100 GPU product page](https://www.nvidia.com/en-us/data-center/h100/) -- H100 SXM total NVLink bandwidth (900 GB/s).
