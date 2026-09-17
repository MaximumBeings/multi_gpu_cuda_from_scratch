# Chapter 5: Explicit Transfers -- cudaMemcpyPeer, Staged Host Transfers, and When Each Wins

**What you will understand by the end of this chapter:**

- Why cross-device copies have their own dedicated Runtime API call, `cudaMemcpyPeer()`, rather than relying on the ordinary `cudaMemcpy()` with `cudaMemcpyDefault` to infer the route automatically.
- What `cudaMemcpyPeer()` actually falls back to, silently, whenever peer access between two devices has not been enabled -- and why that fallback is worth writing out explicitly, by hand, at least once.
- How to reason quantitatively about which route wins, using only this book's own already-cited hardware bandwidth figures rather than any new fabricated timing.

**What you need to know first:**

- Chapter 4's Unified Virtual Addressing and peer-access setup sequence (`cudaDeviceCanAccessPeer()` / `cudaDeviceEnablePeerAccess()`), and its host-side simulation of correct P2P routing logic.
- Chapter 2's PCIe and NVLink bandwidth figures and partial-mesh topology model.
- Ordinary C++ and the CUDA Runtime API calls from Chapters 1-4. No new CUDA concepts beyond this chapter's own two functions.

---

Chapter 4 established *whether* two devices are allowed to touch each other's memory directly. This chapter is about the calls that actually move the bytes once that permission exists -- and, just as importantly, about what happens when it doesn't. CUDA gives cross-device copies their own dedicated function rather than folding them into the ordinary `cudaMemcpy()` interface, and that function's behavior quietly changes shape depending on Chapter 4's peer-access state: fast and direct when peer access is enabled, staged through the host and slower when it isn't, with the same function signature either way. This chapter makes both paths concrete, and then asks the question a real multi-GPU program actually has to answer: given a specific pair of devices and a specific topology, which route wins, and by how much?

## 5.1 cudaMemcpyPeer: The Dedicated Cross-Device Copy Call

### Intuition

`cudaMemcpy()` with `cudaMemcpyDefault` (Chapter 4) can infer a copy's direction because UVA makes every pointer's owner identifiable. But identifying *what* a pointer is isn't the same as deciding *how* to move it when the source and destination belong to two different devices -- that decision depends on Chapter 4's peer-access state, which is per-pair, queryable, and can change at runtime. Rather than hide that decision inside the general-purpose `cudaMemcpy()`, CUDA gives cross-device copies their own explicit call, `cudaMemcpyPeer()`, that names both devices directly. One function, two very different execution paths underneath, chosen automatically based on whether Chapter 4's setup sequence has already run for that specific ordered pair.

### Background

`cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count)` copies `count` bytes from `src` on `srcDevice` to `dst` on `dstDevice`. Per the CUDA Runtime API documentation, this call "is asynchronous with respect to the host, but serialized with respect [to] all pending and future asynchronous work in the current device, `srcDevice`, and `dstDevice`" -- meaning it returns to the host immediately, but the copy itself still waits its turn against other work already queued on any of the three devices involved. `cudaMemcpyPeerAsync()` adds an explicit stream argument for finer-grained ordering against other stream-issued work, the same stream-ordering model Chapter 4 introduced. Crucially, the CUDA Programming Guide's multi-GPU chapter states plainly what determines the actual route taken: "If peer-to-peer access is enabled between two devices, ... peer-to-peer memory copies between these two devices no longer need to be staged through the host and are therefore faster." The function signature and its performance are two separate questions -- the call always works the same way from the caller's side; only its internal routing depends on Chapter 4's state.

```cpp
// Chapter 5: Explicit Transfers -- cudaMemcpyPeer, Staged Host Transfers,
// and When Each Wins
// 11_memcpy_peer.cu
//
// The real, dedicated cross-device copy calls: cudaMemcpyPeer() and its
// asynchronous, stream-ordered counterpart cudaMemcpyPeerAsync(). Both
// genuinely compiled with a real nvcc and genuinely run; on this
// driver-less machine every call fails at the same first step this
// book has reported since Chapter 1.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    void* devPtr0 = nullptr;
    void* devPtr1 = nullptr;

    cudaError_t eAlloc0 = cudaMalloc(&devPtr0, 64);
    printf("cudaMalloc(device 0 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc0), (int)eAlloc0);

    cudaError_t eAlloc1 = cudaMalloc(&devPtr1, 64);
    printf("cudaMalloc(device 1 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc1), (int)eAlloc1);

    cudaError_t e1 = cudaMemcpyPeer(devPtr1, 1, devPtr0, 0, 64);
    printf("cudaMemcpyPeer(dst=dev1, src=dev0, 64 bytes): %s (code %d)\n",
           cudaGetErrorString(e1), (int)e1);

    cudaStream_t stream;
    cudaError_t eStream = cudaStreamCreate(&stream);
    printf("cudaStreamCreate: %s (code %d)\n", cudaGetErrorString(eStream), (int)eStream);

    cudaError_t e2 = cudaMemcpyPeerAsync(devPtr0, 0, devPtr1, 1, 64, stream);
    printf("cudaMemcpyPeerAsync(dst=dev0, src=dev1, 64 bytes, on stream): %s (code %d)\n",
           cudaGetErrorString(e2), (int)e2);

    if (eAlloc0 == cudaSuccess) cudaFree(devPtr0);
    if (eAlloc1 == cudaSuccess) cudaFree(devPtr1);
    if (eStream == cudaSuccess) cudaStreamDestroy(stream);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaMalloc(device 0 buffer): no CUDA-capable device is detected (code 100)
cudaMalloc(device 1 buffer): no CUDA-capable device is detected (code 100)
cudaMemcpyPeer(dst=dev1, src=dev0, 64 bytes): no CUDA-capable device is detected (code 100)
cudaStreamCreate: no CUDA-capable device is detected (code 100)
cudaMemcpyPeerAsync(dst=dev0, src=dev1, 64 bytes, on stream): no CUDA-capable device is detected (code 100)
```

Every call fails with the same `cudaErrorNoDevice` this book has reported consistently since Chapter 1 -- there is no context to allocate the buffers into, let alone copy between them. What this section can verify honestly is narrower but still real: the exact calls, arguments, and compiled binary that a real multi-GPU program would use for this operation, genuinely built by a genuine `nvcc`.

!!! warning "[COMMON TRAP] Assuming cudaMemcpyPeer() itself tells you whether the fast path was used"
    `cudaMemcpyPeer()` returns `cudaSuccess` whether it took the direct peer route or silently fell back to staging through the host -- the function's return value reports *correctness*, not *which mechanism ran*. There is no flag in its signature that reports back which path was taken. The only way to know is to have already checked `cudaDeviceCanAccessPeer()` (Chapter 4, Section 4.2) for that exact ordered pair *before* calling it -- the copy's success tells you the data arrived; it says nothing about how.

## 5.2 Staged Host Transfers: The Fallback, Made Explicit

### Intuition

Section 5.1 quoted the documentation's own description of what happens when peer access isn't enabled: the copy is "staged through the host." That phrase describes a real, ordinary two-step process -- copy the source device's data to a host buffer, then copy that host buffer to the destination device -- using nothing more exotic than two calls this book already introduced in Chapter 3 and 4. Writing that fallback out explicitly, by hand, rather than only invoking it implicitly inside `cudaMemcpyPeer()`, makes concrete exactly what "staged" costs: two separate transfers over the host's PCIe links, each moving the *entire* payload, rather than one transfer over whatever direct link (if any) connects the two devices.

### Background

The staged pattern is two ordinary `cudaMemcpy()` calls with a host buffer as the waypoint: `cudaMemcpy(hostBuffer, srcDevicePtr, bytes, cudaMemcpyDeviceToHost)` followed by `cudaMemcpy(dstDevicePtr, hostBuffer, bytes, cudaMemcpyHostToDevice)`. This is not a simplification of what `cudaMemcpyPeer()` does internally when peer access is absent -- it is, functionally, the same two-hop path, made visible instead of hidden inside one function call. Using a pinned (`cudaMallocHost()`-allocated) host buffer as the waypoint matters here for the same reason it mattered in Chapter 4's UVA discussion: a pinned allocation is a fixed, DMA-eligible physical address the driver can transfer to and from directly, rather than a page the OS might need to fault in or relocate mid-copy.

```cpp
// Chapter 5: Explicit Transfers -- cudaMemcpyPeer, Staged Host Transfers,
// and When Each Wins
// 12_staged_host_transfer.cu
//
// The staged fallback made explicit: two ordinary cudaMemcpy() calls,
// device 0 -> host, then host -> device 1, using a pinned host buffer
// as the waypoint. This is the exact call pattern cudaMemcpyPeer()
// itself falls back to internally whenever peer access between the two
// devices has not been enabled. Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    void* devPtr0 = nullptr;
    void* devPtr1 = nullptr;
    void* hostStaging = nullptr;
    const size_t BYTES = 64;

    cudaError_t eAlloc0 = cudaMalloc(&devPtr0, BYTES);
    printf("cudaMalloc(device 0 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc0), (int)eAlloc0);

    cudaError_t eAlloc1 = cudaMalloc(&devPtr1, BYTES);
    printf("cudaMalloc(device 1 buffer): %s (code %d)\n", cudaGetErrorString(eAlloc1), (int)eAlloc1);

    cudaError_t eHost = cudaMallocHost(&hostStaging, BYTES);
    printf("cudaMallocHost(staging buffer): %s (code %d)\n", cudaGetErrorString(eHost), (int)eHost);

    // Step 1: device 0 -> host.
    cudaError_t e1 = cudaMemcpy(hostStaging, devPtr0, BYTES, cudaMemcpyDeviceToHost);
    printf("Step 1, cudaMemcpy(device 0 -> host staging): %s (code %d)\n",
           cudaGetErrorString(e1), (int)e1);

    // Step 2: host -> device 1. Two PCIe (or equivalent) transfers where
    // a single peer-direct copy would have used one link once.
    cudaError_t e2 = cudaMemcpy(devPtr1, hostStaging, BYTES, cudaMemcpyHostToDevice);
    printf("Step 2, cudaMemcpy(host staging -> device 1): %s (code %d)\n",
           cudaGetErrorString(e2), (int)e2);

    if (eAlloc0 == cudaSuccess) cudaFree(devPtr0);
    if (eAlloc1 == cudaSuccess) cudaFree(devPtr1);
    if (eHost == cudaSuccess) cudaFreeHost(hostStaging);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaMalloc(device 0 buffer): no CUDA-capable device is detected (code 100)
cudaMalloc(device 1 buffer): no CUDA-capable device is detected (code 100)
cudaMallocHost(staging buffer): no CUDA-capable device is detected (code 100)
Step 1, cudaMemcpy(device 0 -> host staging): no CUDA-capable device is detected (code 100)
Step 2, cudaMemcpy(host staging -> device 1): no CUDA-capable device is detected (code 100)
```

The same honest `cudaErrorNoDevice` at every step, for the same reason as Section 5.1. What this section fixes in place is the *shape* of the fallback: two calls, two full-payload transfers, one intermediate buffer -- the concrete cost Section 5.3 now puts a number on.

!!! warning "[COMMON TRAP] Believing the staged fallback moves half the data each hop"
    Both steps in this section's code copy the *entire* `BYTES`-sized payload -- step 1 moves all of it device-to-host, and step 2 moves all of it again host-to-device. There is no sense in which the two hops split the work; the staged path pays the *full* transfer cost twice, once per hop, which is exactly why Section 5.3's cost model below counts it as `2 * (size / bandwidth)` rather than a single transfer split across two links.

## 5.3 When Each Wins: Peer Access, Topology, and the Real Cost of Staging

### Intuition

Section 5.1 and 5.2 established the two mechanisms; now the actual question a real program has to answer: for a given pair of devices, is the direct peer route worth checking for, or is staging good enough? The honest answer only needs numbers this book has already sourced and cited -- Chapter 2's real PCIe and NVLink bandwidth figures -- arranged into the two routes' costs. No new hardware measurement is needed, and none is fabricated: this is a closed-form arithmetic comparison, exactly like Chapter 2's own ring-bottleneck model.

### Background

The staged route always pays the PCIe rate twice, for the reason Section 5.2's Common Trap just established: `2 * (size / PCIE_GBPS)`. The peer-direct route pays whichever direct link actually connects the two devices exactly once: the NVLink figure if the pair is wired (Chapter 2's partial-mesh model), or the PCIe figure if their only direct connection is PCIe itself. Reusing this book's own already-cited numbers -- PCIe 4.0 x16 at 31.5 GB/s, and a direct wired NVLink pair at 50 GB/s (both from Chapter 2's topology chapter) -- turns "which route wins" into one division per route.

```cpp
// Chapter 5: Explicit Transfers -- cudaMemcpyPeer, Staged Host Transfers,
// and When Each Wins
// 13_when_each_wins_model.cpp
//
// Plain host C++ cost model, reusing this book's own already-cited
// bandwidth figures (Chapter 2, docs/part0/02-multi-gpu-hardware-topology.md):
// PCIe 4.0 x16 per-direction bandwidth (31.5 GB/s) and a direct, wired
// NVLink connection in a partial-mesh topology (50 GB/s). This is a
// closed-form arithmetic model of transfer TIME, not a measurement of
// real hardware -- it answers "which route wins, and by how much" using
// numbers this book has already sourced and cited, not new fabricated
// timings.
#include <cstdio>

int main() {
    const double PCIE_GBPS = 31.5;   // direct PCIe 4.0 x16, per direction
    const double NVLINK_GBPS = 50.0; // direct wired NVLink pair, partial mesh

    const double transferSizeGB = 1.0;

    // Peer-direct route: one hop, over whichever direct link exists.
    // If the pair is NVLink-wired, that's the NVLink figure; if the
    // pair's only direct link is PCIe, that's the PCIe figure.
    double peerTimeNVLink = transferSizeGB / NVLINK_GBPS;
    double peerTimePCIe   = transferSizeGB / PCIE_GBPS;

    // Staged route: two hops through the host, device->host and
    // host->device, each paying the PCIe rate -- this is exactly what
    // Section 5.2's explicit two-step transfer measured the call
    // pattern for, and what cudaMemcpyPeer itself falls back to
    // whenever peer access has not been enabled for that pair.
    double stagedTime = 2.0 * (transferSizeGB / PCIE_GBPS);

    printf("Transfer size: %.1f GB\n\n", transferSizeGB);
    printf("Peer-direct route, NVLink-wired pair:   %.6f s (%.1f GB/s, one hop)\n",
           peerTimeNVLink, NVLINK_GBPS);
    printf("Peer-direct route, PCIe-only direct pair: %.6f s (%.1f GB/s, one hop)\n",
           peerTimePCIe, PCIE_GBPS);
    printf("Staged host route (any pair):            %.6f s (%.1f GB/s, two hops)\n\n",
           stagedTime, PCIE_GBPS);

    double speedupNVLinkVsStaged = stagedTime / peerTimeNVLink;
    double speedupPCIeDirectVsStaged = stagedTime / peerTimePCIe;

    printf("Peer-direct over NVLink is %.4fx faster than staging, for a wired pair.\n",
           speedupNVLinkVsStaged);
    printf("Peer-direct over a direct PCIe link is %.4fx faster than staging,\n"
           "for a pair with a direct PCIe route but no NVLink wire.\n",
           speedupPCIeDirectVsStaged);
    printf("\nWhen the pair has NO direct link at all (per Chapter 2's partial-mesh\n"
           "topology), peer access cannot be enabled between them in the first\n"
           "place (Chapter 4, Section 4.2) -- staging is then not merely faster\n"
           "or slower, it is the ONLY route that exists.\n");

    return 0;
}
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
Transfer size: 1.0 GB

Peer-direct route, NVLink-wired pair:   0.020000 s (50.0 GB/s, one hop)
Peer-direct route, PCIe-only direct pair: 0.031746 s (31.5 GB/s, one hop)
Staged host route (any pair):            0.063492 s (31.5 GB/s, two hops)

Peer-direct over NVLink is 3.1746x faster than staging, for a wired pair.
Peer-direct over a direct PCIe link is 2.0000x faster than staging,
for a pair with a direct PCIe route but no NVLink wire.

When the pair has NO direct link at all (per Chapter 2's partial-mesh
topology), peer access cannot be enabled between them in the first
place (Chapter 4, Section 4.2) -- staging is then not merely faster
or slower, it is the ONLY route that exists.
```

Peer-direct wins whenever it's *available* -- by a factor of exactly 2x when the direct link is itself PCIe (staging simply pays that same rate twice), and by a larger 3.17x when the direct link is a faster NVLink wire. But "when each wins" has a sharper answer than a speed comparison alone: peer-direct is only ever a *choice* to make when Chapter 4's `cudaDeviceCanAccessPeer()` already says the pair has a direct link at all. When it doesn't, there is no decision to make -- staging isn't the slower option, it's the only one.

!!! warning "[COMMON TRAP] Treating this cost model's numbers as real, measured transfer times"
    Every figure in this section's output is a closed-form arithmetic result from bandwidth numbers Chapter 2 already sourced and cited to real vendor documentation -- not a timing measurement taken on real hardware. Real transfer time also includes fixed per-call latency, driver overhead, and whatever else is competing for the same link, none of which this simple `size / bandwidth` model captures. This model answers "which route is structurally faster, and by roughly what multiple" honestly and exactly -- it does not, and cannot, stand in for benchmarking a real transfer on real GPUs.

## Chapter Summary

`cudaMemcpyPeer()` is CUDA's dedicated cross-device copy call, and per the CUDA Programming Guide's own words, its routing -- direct or staged through the host -- depends entirely on whether Chapter 4's peer-access permission has already been granted for that specific ordered pair, with the function's return value reporting only correctness, never which path ran (Section 5.1). The staged fallback, written out explicitly in Section 5.2 as two ordinary `cudaMemcpy()` calls through a pinned host buffer, makes concrete that staging always pays the full transfer cost twice -- once per hop -- never split across the two links. Putting real, already-cited numbers into that shape (Section 5.3) shows peer-direct winning by 2x to 3.17x whenever it's available, but also reframes the real question: peer-direct is a choice only when a direct link exists between the two specific devices at all; when it doesn't, staging is not the slower option but the only one. Chapter 6 builds directly on this by bringing streams and events into the picture -- overlapping a transfer like this chapter's with concurrent kernel execution, and synchronizing correctly across devices when that overlap is in play.

## Self-Check Questions

1. `cudaMemcpyPeer()` returns `cudaSuccess` in both of the following cases: peer access was enabled and the copy went direct, and peer access was not enabled and the copy was staged through the host. What, if anything, in the function's return value tells the caller which case occurred?
2. Section 5.2's staged transfer uses a pinned (`cudaMallocHost()`-allocated) host buffer rather than an ordinary `new`-allocated one. Explain, in your own words, why the physical-address stability of a pinned allocation matters for a DMA-based transfer.
3. Suppose a payload is 2 GB instead of Section 5.3's 1 GB. Using the same bandwidth figures, compute the new staged-route time and the new peer-direct-over-NVLink time. Does the speedup ratio between them change? Why or why not?
4. Section 5.3's model treats the peer-direct route's bandwidth as either the NVLink figure or the PCIe figure, never something in between. Referencing Chapter 2's partial-mesh topology, explain why there is no meaningful "in between" bandwidth for a specific ordered pair of devices.
5. A programmer concludes: "peer-direct access is always the better choice, so I'll always call `cudaDeviceEnablePeerAccess()` for every pair before transferring." Using this chapter's own conclusion, explain what's incomplete about that plan.
6. Section 5.2's code performs two separate `cudaMemcpy()` calls. If a third device were introduced and the data needed to go from device 0 to device 2 by way of being staged through device 1's memory instead of the host, would that still be a "staged" transfer in the sense this chapter uses the term? Explain the distinction.
7. Why does this chapter's cost model (Section 5.3) reuse Chapter 2's bandwidth figures rather than deriving new ones, and why is that a better choice than fabricating a plausible-sounding timing number?
8. Using the field descriptions from the CUDA Runtime API reference (not the cost model), name one reason a real device's *achieved* bandwidth for a staged transfer could come in below the theoretical PCIe rate this chapter's model assumes.

## Where We Go Next

Chapter 6 introduces streams and events as the mechanism for overlapping a transfer -- either of this chapter's two routes -- with concurrent kernel execution on the same or a different device, and for synchronizing correctly across devices once operations are running concurrently rather than one at a time.

## Worked Solutions

**1.** Nothing. `cudaMemcpyPeer()`'s return value (a `cudaError_t`) reports whether the copy completed successfully, not which internal route it took. The only way to know in advance which path will run is to have already called `cudaDeviceCanAccessPeer()` for that exact ordered pair before making the copy call at all.

**2.** A DMA (direct memory access) transfer engine reads or writes host memory using physical addresses, programmed once at the start of the transfer. If the underlying page could be moved or swapped out mid-transfer -- which an ordinary, non-pinned allocation permits, since the OS is free to relocate it -- the DMA engine's already-programmed physical address would become invalid partway through. A pinned (page-locked) allocation is guaranteed by the OS not to move or be paged out for as long as it's pinned, so its physical address stays valid for the DMA engine's entire operation.

**3.** Staged time doubles to `2 * (2.0/31.5) = 0.126984` s, and peer-direct-over-NVLink time doubles to `2.0/50.0 = 0.04` s. The speedup ratio (`stagedTime / peerTimeNVLink`) stays exactly the same, `3.1746`, because both times scaled by the identical factor (2x the data), and that common factor cancels out of the ratio -- the *relative* advantage of peer-direct over staging depends only on the bandwidth figures, not on the transfer size.

**4.** In Chapter 2's partial-mesh topology, a specific ordered pair of devices either has a direct wired NVLink connection between them or it doesn't -- there is no partial, intermediate wiring state for one specific pair. The pair's peer-direct bandwidth is therefore always exactly one of the two discrete figures (NVLink if wired, PCIe if the only direct route is PCIe), never an average or blend of the two, because a "half-wired" link isn't a real hardware configuration.

**5.** Chapter 4's Section 4.2 established that `cudaDeviceEnablePeerAccess()` only succeeds (or is even meaningful to call) between pairs where `cudaDeviceCanAccessPeer()` already returns true -- meaning a direct physical link exists. Calling it "for every pair" doesn't create a link that doesn't exist; between pairs with no direct wiring, the enable call itself does nothing useful, and the program still has to use the staged route for those pairs regardless of intent. Peer-direct access isn't a setting to always turn on -- it's a fact about specific pairs' physical topology that either holds or doesn't.

**6.** No -- this chapter's term "staged" specifically means routed *through host memory*, which is what `cudaMemcpy(..., cudaMemcpyDeviceToHost)` followed by `cudaMemcpy(..., cudaMemcpyHostToDevice)` does. A transfer from device 0 to device 2 that goes by way of device 1's memory (rather than the host's) would be a different thing entirely -- a multi-hop device-to-device relay, which Chapter 2's ring-bottleneck model already discussed in a different context (Section 2.4) and which this chapter's two functions don't perform on their own.

**7.** Chapter 2's figures were already sourced from real vendor documentation (PCIe and NVLink specification tables) and cited at the end of that chapter -- reusing them keeps this chapter's numbers traceable to the same real sources rather than introducing a second, unverified data point. Fabricating a plausible-sounding timing number instead would break this book's own stated honesty discipline: every number in this book is either a genuine, verified program output or a real, cited fact, and a made-up transfer time would be neither.

**8.** The CUDA Runtime API reference notes that a device's actual achieved bandwidth depends on real factors this simple `size / bandwidth` model doesn't include -- for example, fixed per-transfer latency overhead that matters more for small transfers, and contention from other traffic sharing the same PCIe root complex or switch at the same time (Chapter 2's own same-switch-vs-cross-socket distinction already showed shared links being a real, modeled cost). A real transfer's *achieved* rate is a ceiling this model's theoretical rate can approach but a real system rarely reaches under contention.

---

**Sources cited in this chapter:**

- [CUDA Programming Guide — Multi-GPU Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html) -- peer-to-peer copies no longer needing to be staged through the host once peer access is enabled; the four dedicated peer-copy functions (`cudaMemcpyPeer`, `cudaMemcpyPeerAsync`, `cudaMemcpy3DPeer`, `cudaMemcpy3DPeerAsync`).
- [CUDA Runtime API Reference — Memory Management (`cudaMemcpyPeer`)](https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g046702971bc5a66d9bc6000682a6d844.html) -- `cudaMemcpyPeer()` signature and its "asynchronous with respect to the host, but serialized" behavior.
- [CUDA Programming Guide — Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) -- pinned/page-locked host memory as the requirement for genuinely asynchronous transfers.
- Chapter 2 of this book ("Multi-GPU Hardware Topology: PCIe, NVLink, and NVSwitch") -- the PCIe 4.0 x16 (31.5 GB/s) and direct wired NVLink (50 GB/s) figures this chapter's cost model reuses, themselves cited to real vendor sources in that chapter.
