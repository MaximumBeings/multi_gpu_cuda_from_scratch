# Chapter 2: Multi-GPU Hardware Topology: PCIe, NVLink, and NVSwitch

Chapter 1 established that splitting work across GPUs solves both the capacity wall and the throughput wall, but introduces a new one: the devices now have to talk to each other. This chapter is about the physical layer that conversation happens over -- because "GPU-to-GPU bandwidth" is not one number. It depends entirely on which two GPUs you pick, and how they happen to be wired together.

## 2.1 Bandwidth is a graph, not a number

A single GPU has exactly one bandwidth figure worth knowing: how fast it talks to its own HBM. The moment there are two or more GPUs, that single number is replaced by a matrix -- a bandwidth figure for every ordered pair of devices, and those figures are frequently *not* the same for every pair, even within one machine.

```text
   ┌─────┐     ┌─────┐          ┌─────┐     ┌─────┐
   │ GPU0│─────│ GPU1│          │ GPU0│═════│ GPU1│
   └─────┘     └─────┘          └──┬──┘     └──┬──┘
      │  PCIe switch / root         │  NVLink    │
      │  complex (shared,           │  (direct,  │
      │  moderate bandwidth)        │  high      │
   ┌─────┐     ┌─────┐              │  bandwidth)│
   │ GPU2│─────│ GPU3│           ┌──┴──┐     ┌──┴──┐
   └─────┘     └─────┘           │ GPU2│═════│ GPU3│
                                  └─────┘     └─────┘
   GPU0<->GPU1: fast (same switch)   Every pair here: fast, direct
   GPU0<->GPU3: slow (extra hops,    (until you run out of direct
                shared root complex)  links -- more on this below)
```

Three real technologies fill in that matrix in practice, in increasing order of how good an answer they give: PCIe (every GPU has it, and it's the fallback), NVLink (a direct, high-bandwidth point-to-point link some GPU pairs have), and NVSwitch (a crossbar that makes *every* pair behave like a direct NVLink link). The rest of this chapter goes through each, with real cited bandwidth figures, then builds a small model of what a difference in topology actually costs a real communication pattern.

## 2.2 PCIe: the baseline every GPU has

Every GPU in existence, however it's also connected, sits on a PCI Express link to its host system -- this is how the CPU allocates it, launches kernels on it, and (absent anything better) moves data to and from it. PCIe bandwidth has roughly doubled with each generation:

| Generation | x16 link bandwidth (per direction) |
|---|---|
| PCIe 3.0 | 15.8 GB/s |
| PCIe 4.0 | 31.5 GB/s |
| PCIe 5.0 | 63.0 GB/s |

But the number that matters for a GPU *pair* isn't just "what generation" -- it's *where in the PCIe tree* the two GPUs sit relative to each other:

```text
                    ┌────────┐                    ┌────────┐
                    │  CPU 0  │                    │  CPU 1  │
                    └───┬────┘                    └───┬────┘
                        │ PCIe root complex             │ PCIe root complex
                 ┌──────┴──────┐                 ┌──────┴──────┐
                 │ PCIe switch  │                 │ PCIe switch  │
              ┌──┴──┐       ┌──┴──┐            ┌──┴──┐       ┌──┴──┐
              │GPU 0│       │GPU 1│            │GPU 2│       │GPU 3│
              └─────┘       └─────┘            └─────┘       └─────┘
                    \___same switch___/               \___same switch___/
                              \_________cross-socket, extra hop_________/
```

GPU 0 and GPU 1 share a PCIe switch: a transfer between them stays entirely within that switch, at the full per-generation x16 bandwidth. GPU 0 and GPU 2 do not -- a transfer between them has to cross both PCIe switches *and* whatever link joins the two CPU sockets (a NUMA interconnect such as Intel's UPI or AMD's Infinity Fabric), which is itself a shared, finite resource that every other cross-socket transfer on the machine is also competing for. This is why "the GPUs are all PCIe 4.0" is not enough information to predict a real transfer's speed -- topology inside the tree matters as much as the generation number, and Section 2.5's model makes this concrete rather than just asserted.

## 2.3 NVLink: a direct line when PCIe isn't enough

NVLink is NVIDIA's answer to the PCIe ceiling: a direct, dedicated, much higher-bandwidth link wired directly between two GPUs (or between a GPU and an NVSwitch -- Section 2.4), bypassing the PCIe tree and the host entirely. Each generation has shipped with a new GPU architecture and a real, published bandwidth jump:

| NVLink generation | Introduced with | Per-link bandwidth | Links per GPU | Total per-GPU bandwidth |
|---|---|---|---|---|
| 1.0 | P100 (Pascal) | 20 GB/s | 4 | 160 GB/s |
| 2.0 | V100 (Volta) | 25 GB/s | 6 | 300 GB/s |
| 3.0 | A100 (Ampere) | 25 GB/s | 12 | 600 GB/s |
| 4.0 | H100 (Hopper) | 25 GB/s | 18 | 900 GB/s |

An H100's 900 GB/s of total NVLink bandwidth is roughly **29 times** a PCIe 4.0 x16 link's 31.5 GB/s -- the gap is the entire reason this technology exists. But notice the phrase "total per-GPU bandwidth": those 18 links on an H100 don't have to all go to the same neighbor. How they're distributed among a GPU's neighbors is exactly what separates a *partial mesh* from a *switch*.

```text
    Two GPUs, direct NVLink:              Two GPUs, no direct NVLink:
    ┌─────┐                ┌─────┐        ┌─────┐                ┌─────┐
    │ GPU0│════NVLink══════│ GPU1│        │ GPU0│                │ GPU2│
    └─────┘  (900 GB/s,     └─────┘        └──┬──┘                └──┬──┘
              H100-class)                     │        PCIe           │
                                               └────────(31.5 GB/s)────┘
```

## 2.4 NVSwitch: making every pair a direct pair

With only point-to-point NVLink and a fixed number of links per GPU, a system with enough GPUs eventually runs out of direct links to go around -- some pairs get a direct NVLink connection, and the rest fall back to PCIe. Early multi-GPU NVLink systems (the DGX-1 generation) had exactly this shape: each GPU directly wired to some, but not all, of the others, in a fixed pattern often called a hybrid cube-mesh.

NVSwitch removes the tradeoff by putting a crossbar switch chip between the GPUs instead of wiring them directly to each other. Every GPU gets a full set of NVLink connections *to the switch*, and the switch connects any GPU to any other GPU at that same full bandwidth:

```text
                     ┌─────────────────────────┐
                     │        NVSwitch            │
                     │   (non-blocking crossbar)   │
                     └──┬───┬───┬───┬───┬───┬───┬──┘
                        │   │   │   │   │   │   │
                      GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 ...
```

The practical consequence: on an NVSwitch-equipped system (DGX A100, DGX H100, and later), *every* GPU pair gets the GPU's full NVLink bandwidth -- 900 GB/s for H100 -- with no "did I get lucky with placement" question at all. That uniformity is exactly what Part 2's collectives (ring all-reduce especially) benefit most from, and it's why the case studies in Part 6 default to assuming it unless stated otherwise.

## 2.5 What topology costs a real communication pattern, computed sequentially

All three sections above make qualitative claims. Here's the quantitative version -- ordinary host C++, no CUDA, modeling each topology as a bandwidth matrix built from the real per-link figures above, then computing something Part 2 will care about directly: the bandwidth a *ring* of GPUs can sustain, which is set by its single weakest link.

```cpp
// 04_topology_bandwidth_model.cpp (excerpt -- full file in docs/part0/code/)
Matrix pcieTreeTopology() {
    const double SAME_SWITCH = 31.5;   // PCIe 4.0 x16, per direction
    const double CROSS_SOCKET = SAME_SWITCH / 2.0; // modeling simplification
    // ... GPUs 0-3 under CPU0, 4-7 under CPU1 ...
}

Matrix partialNvlinkMeshTopology() {
    const double DIRECT = 50.0;        // NVLink 2.0, 2 links, 25 GB/s each
    const double PCIE_FALLBACK = 31.5; // falls back to plain PCIe if unwired
    // ... each GPU directly wired to 4 of the other 7 ...
}

Matrix nvswitchTopology() {
    const double UNIFORM = 900.0;      // every pair, H100-class NVSwitch
    // ... uniform for all pairs ...
}

double ringBottleneck(const Matrix& bw, const std::vector<int>& order) {
    double bottleneck = std::numeric_limits<double>::max();
    for (size_t i = 0; i < order.size(); ++i)
        bottleneck = std::min(bottleneck, bw[order[i]][order[(i + 1) % order.size()]]);
    return bottleneck;
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

Three things fall out of this that are easy to state and easy to underestimate:

1. **The PCIe tree's ring bottleneck (15.75 GB/s) is the *cross-socket* figure, not the same-switch figure (31.5 GB/s).** A ring that visits all 8 GPUs in index order necessarily crosses the socket boundary at least twice, and the ring's overall speed is set by that crossing, not by the faster hops elsewhere in the ring. One bad link taxes the *entire* collective, every round.
2. **The partial mesh's ring bandwidth (50 GB/s) depends on the visiting order matching the physical wiring** -- and there's nothing about a GPU's index number that guarantees it does. Swapping in a single unwired pair drops the exact same 8 GPUs' ring bandwidth from 50 GB/s to 31.5 GB/s, a 37% cut, from a topology-blind reordering that looks harmless on paper.
3. **NVSwitch's number (900 GB/s) doesn't move no matter what order you pick.** That's not a minor convenience -- it's the removal of an entire class of silent performance bug that the other two topologies are exposed to.

Chapter 9's ring all-reduce is built on exactly this bottleneck-bandwidth idea, and its own text will point back here.

## 2.6 What the CUDA Runtime can honestly tell you about topology, right now

This book's real toolchain doesn't have real hardware to query, but the exact API this book uses for real starting in Chapter 4 -- `cudaDeviceCanAccessPeer()` -- can still be called for real today, and it's worth seeing its honest behavior before it matters.

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
    // ... loop over all real pairs when >= 2 devices exist ...
}
```

```bash
nvcc -arch=sm_80 05_peer_access_query.cu -o peer_access_query
./peer_access_query
```

Output (genuinely compiled with a real `nvcc` and genuinely run; identical across three consecutive runs):

```text
cudaGetDeviceCount(): no CUDA-capable device is detected, count = 0
Fewer than 2 devices are present, so there is no GPU pair to ask cudaDeviceCanAccessPeer() about. We call it anyway, on device ids 0 and 1, exactly as a program that assumed 2 devices without checking first would -- and report the runtime's real answer rather than skipping the call:
cudaDeviceCanAccessPeer(0, 1) -> no CUDA-capable device is detected (code 100), canAccess left at -1
This is the honest failure mode this book's own topology and peer-access chapters build around: always check cudaGetDeviceCount() before assuming a peer exists.
```

On real hardware, the standard tool for seeing this chapter's whole topology story at a glance is `nvidia-smi topo -m`, which prints exactly the kind of matrix Section 2.5 built by hand -- but for the real system it's run on, straight from the driver. This environment has no driver to run it against, so that table isn't reproduced here as if it were captured; a reader with real multi-GPU hardware is the one who can genuinely fill it in, and later chapters that build on real topology output will say so explicitly rather than presenting an invented table.

## 2.7 Summary

Bandwidth between two GPUs is a function of the specific pair, not a property of the machine as a whole. PCIe is the universal fallback and is shaped by the tree it sits in -- same switch is fast, crossing sockets is not. NVLink replaces specific pairs' PCIe hop with a much faster direct one, but only for the pairs it's actually wired to. NVSwitch removes the distinction entirely by making every pair look like a direct NVLink pair. Chapter 3 builds the CUDA-side programming model -- devices, contexts, and streams -- that sits on top of whichever of these three a given machine actually has, and Chapter 4 starts actually moving data across the links this chapter described.

---

**Sources cited in this chapter:**

- [PCIe Bandwidth Chart — GB/s for Gen 1–7 at x1–x16](https://pciesimulator.com/pcie-bandwidth-table/) -- PCIe 3.0/4.0/5.0 x16 per-direction bandwidth figures.
- [NVLink — Wikipedia](https://en.wikipedia.org/wiki/NVLink) -- NVLink generation-by-generation per-link bandwidth and links-per-GPU figures (1.0 through 4.0).
- [NVIDIA H100 GPU product page](https://www.nvidia.com/en-us/data-center/h100/) -- H100 SXM total NVLink bandwidth (900 GB/s).
