# Chapter 4: Peer-to-Peer Memory Access and Unified Virtual Addressing

**What you will understand by the end of this chapter:**

- Why every pointer in a modern multi-GPU CUDA program — host or device, no matter which device allocated it — lives in one single, flat address space, and what `cudaPointerGetAttributes()` can honestly tell you about any pointer in that space.
- Why "peer access" is a real, queryable, *unidirectional* permission between two specific devices, not an automatic consequence of Unified Virtual Addressing existing — and exactly which two Runtime API calls establish it.
- How this book verifies peer-to-peer transfer *logic* — which buffer goes where, under which topology — without a second real device, using the host-side simulation technique this book commits to using honestly whenever real multi-device hardware isn't available.

**What you need to know first:**

- Chapter 3's per-thread current-device state, and the idiomatic multi-device loop pattern (create → launch → synchronize, kept as three separate loops).
- Chapter 2's topology models: which device pairs have a direct link, and which fall back to a shared, slower path.
- Ordinary C++ containers (`std::vector`) and pointers. No prior CUDA experience beyond Chapters 1-3.

---

Part 0 established *where* the devices are and *whose problem* it is to pick one (a host thread's current-device state) and *when* work runs concurrently (streams bound to whichever device was current at creation time). None of that, so far, has let one GPU touch another GPU's memory. This chapter is where that changes. Unified Virtual Addressing is the feature that makes "one GPU's pointer" and "another GPU's pointer" comparable at all — they live in the same address space — and peer-to-peer access is the specific, opt-in permission that lets a kernel or a copy actually dereference across that boundary. Both are real, queryable pieces of CUDA Runtime API state, and both are things this environment's `cudaErrorNoDevice` machine can genuinely ask about and honestly report on, even with zero real devices attached.

## 4.1 Unified Virtual Addressing: One Address Space, Many Memories

### Intuition

Before CUDA 4.0, a raw pointer value was ambiguous: the same numeric address could be a valid host address on one system and simultaneously mean nothing (or something else entirely) as a device address, so the runtime had to be told explicitly, for every copy, which direction the data was moving (`cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, and so on). Unified Virtual Addressing removes that ambiguity by construction: the host and every visible CUDA device share one flat virtual address range, so a given pointer *value* uniquely identifies where it points — the host's memory, or a specific device's memory, and if the latter, which device. That single fact is what lets `cudaMemcpyDefault` figure out a copy's direction on its own, and — more importantly for this chapter — it's the precondition for peer-to-peer access even being a coherent idea: two devices can only dereference each other's pointers directly because those pointers already live in address ranges both devices agree on the meaning of.

### Background

The CUDA Runtime API's `cudaPointerGetAttributes()` function is the honest, queryable answer to "whose memory does this pointer actually point into?" Given any pointer, it reports (among other fields) a `type` — is this host memory, device memory, or something else — and, when relevant, which `device` ordinal owns it. On a UVA-enabled system (every system this book's target hardware runs on; UVA has applied to all CUDA-capable 64-bit systems for well over a decade), this call works uniformly on *any* pointer, host or device, without the caller needing to already know which kind it is. That is the entire point: the caller can ask the question generically. On this machine, with zero real devices, both possible inputs to that question are worth trying honestly: an ordinary host stack array, and a pointer returned by `cudaMalloc`. Since there is no real driver here at all, `cudaMalloc` itself cannot succeed — so the code below tries both, and honestly reports whichever real Runtime API error each one hits.

```cpp
// Chapter 4: Peer-to-Peer Memory Access and Unified Virtual Addressing
// 08_pointer_attributes.cu
//
// A genuine cudaPointerGetAttributes() call, compiled with a real nvcc
// and genuinely run. Under Unified Virtual Addressing, a single flat
// address space covers the host and every device, so this call is what
// answers "whose memory does this pointer actually point into" -- the
// exact mechanism cudaMemcpyDefault relies on to auto-detect a copy's
// direction without being told explicitly.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int hostArr[4] = {1, 2, 3, 4};
    cudaPointerAttributes attr;
    cudaError_t e = cudaPointerGetAttributes(&attr, hostArr);
    printf("cudaPointerGetAttributes(ordinary stack array): %s (code %d)\n",
           cudaGetErrorString(e), (int)e);
    if (e == cudaSuccess) {
        printf("  type=%d device=%d\n", (int)attr.type, attr.device);
    }

    void* devPtr = nullptr;
    cudaError_t e2 = cudaMalloc(&devPtr, 16);
    printf("cudaMalloc: %s (code %d)\n", cudaGetErrorString(e2), (int)e2);

    if (e2 == cudaSuccess) {
        cudaPointerAttributes attr2;
        cudaError_t e3 = cudaPointerGetAttributes(&attr2, devPtr);
        printf("cudaPointerGetAttributes(device ptr): %s\n", cudaGetErrorString(e3));
        cudaFree(devPtr);
    }
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs on this machine:

```
cudaPointerGetAttributes(ordinary stack array): no CUDA-capable device is detected (code 100)
cudaMalloc: no CUDA-capable device is detected (code 100)
```

Both calls fail with the same `cudaErrorNoDevice` this book has reported since Chapter 1 — for the same reason: every CUDA Runtime API call that needs a real device context, including one asking about an *ordinary host pointer*, must first initialize that context, and there is no physical GPU here to initialize one against. This is itself an honest, useful data point: even a query about host memory is not a "host-only" operation once UVA is involved, because answering it correctly requires consulting the same device state that a device-to-device query would.

!!! warning "[COMMON TRAP] Assuming a pointer's *type* tells you whether an operation on it is legal"
    `cudaPointerGetAttributes()` tells you what a pointer *is* — host memory, or device memory belonging to a specific device ordinal. It does not tell you what operations are currently *permitted* on it from somewhere else. Knowing that a pointer belongs to device 1 does not mean device 0 can dereference it; that is a completely separate, unidirectional permission — the subject of the next section — that has to be established on its own, regardless of what UVA has already told you about the pointer's identity.

## 4.2 Checking and Enabling Peer Access

### Intuition

UVA means every device's pointers are meaningful, comparable values in one address space. It does not mean every device is *allowed* to dereference every other device's pointers directly. That permission is a separate, explicit, per-ordered-pair setting, because allowing it has real hardware and driver consequences (a PCIe or NVLink route has to actually exist and be usable for the access to work at all — recall Chapter 2's topology models, where not every pair of devices has a direct link). The CUDA Runtime API therefore splits this into two calls: one that only *asks* whether a given ordered pair could work, and a second, separate call that actually *grants* it — and the grant is deliberately one-directional, so a fully bidirectional relationship between two devices takes two calls, one from each side.

### Background

`cudaDeviceCanAccessPeer(&canAccess, device, peerDevice)` answers, for one ordered pair, whether `device` could access `peerDevice`'s memory directly if asked to. It changes nothing by itself. `cudaDeviceEnablePeerAccess(peerDevice, flags)` is what actually grants that access — and it does so from the perspective of *whichever device is currently the caller's current device* (Chapter 3's per-thread state again), granting that current device the ability to access `peerDevice`. The CUDA Runtime API documentation is explicit that the access this grants is unidirectional: enabling device 0 to access device 1 says nothing about whether device 1 can access device 0 — that direction, if wanted, needs its own separate call with the current device set to 1. The idiomatic setup sequence, then, is a loop over every *ordered* pair — not every unordered pair — checking and enabling each direction independently. The CUDA Programming Guide's multi-GPU chapter also notes a real, finite ceiling on this: on non-NVSwitch-enabled systems, each device supports a system-wide maximum of eight peer connections at once, which is exactly the kind of topology-shaped constraint Chapter 2's partial-mesh models already anticipated.

```cpp
// Chapter 4: Peer-to-Peer Memory Access and Unified Virtual Addressing
// 09_peer_access_setup.cu
//
// The real, idiomatic peer-access setup sequence: check every ordered
// pair with cudaDeviceCanAccessPeer(), then enable access with
// cudaDeviceEnablePeerAccess() -- called from the "current" side of the
// pair, since (per the CUDA Runtime API docs) access granted this way
// is unidirectional. Genuinely compiled with a real nvcc and genuinely
// run; on this driver-less machine the loop body runs zero times, for
// exactly the reason Chapter 3 already explained.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    int enabledPairs = 0;
    for (int i = 0; i < deviceCount; ++i) {
        for (int j = 0; j < deviceCount; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            if (canAccess) {
                cudaSetDevice(i);
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                printf("Enabled i=%d -> j=%d: %s\n", i, j,
                       cudaGetErrorString(e));
                if (e == cudaSuccess) ++enabledPairs;
            }
        }
    }
    printf("Peer access enabled for %d ordered pair(s) out of %d "
           "device(s) checked. On real multi-GPU hardware this same "
           "loop enables every accessible ordered pair exactly once, "
           "each direction independently, per the API's own "
           "unidirectional-access note.\n", enabledPairs, deviceCount);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).
Peer access enabled for 0 ordered pair(s) out of 0 device(s) checked. On real multi-GPU hardware this same loop enables every accessible ordered pair exactly once, each direction independently, per the API's own unidirectional-access note.
```

With zero devices reported, both loops have zero iterations, so `cudaDeviceCanAccessPeer()` and `cudaDeviceEnablePeerAccess()` are never even called — there is no ordered pair to check. This is the same honest degradation Chapter 2's peer-access query (`05_peer_access_query.cu`) already demonstrated for a single hardcoded pair; this section's version is the general loop form that real multi-device code actually uses, shown here running correctly down to zero real iterations rather than crashing or needing a special case for "fewer than two devices."

!!! warning "[COMMON TRAP] Enabling peer access once and assuming it covers both directions"
    A single call to `cudaDeviceEnablePeerAccess(1, 0)` made while device 0 is current grants device 0 the ability to access device 1 — and only that direction. If code on device 1 later tries to access device 0's memory without a separate call (`cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);`), that access is not permitted, even though it looks symmetric to the first call. The loop in this section's code checks and enables *both* orderings of every pair for exactly this reason — skipping half the loop silently leaves half the possible accesses disabled.

## 4.3 Verifying the Transfer Logic Without a Second Device

### Intuition

Sections 4.1 and 4.2 established what this machine can honestly say about UVA and peer access with zero real devices: both real Runtime API calls, both consistently and correctly reporting `cudaErrorNoDevice`. But this book's job isn't only to report that error correctly — it's to teach the *logic* of how a multi-device program should route its transfers once real devices exist: use a direct peer write where peer access exists, fall back to a host-staged copy where it doesn't, and — the property that actually matters — guarantee both routes land the same, correct data at the destination. That logic can be verified completely honestly right now, using this book's own simulation technique: real in-memory buffers standing in for real device memories, exchanging data through the exact same message pattern real code would use, checked against an independently-kept reference.

### Background

The simulation below models a 4-device system with the same "partial mesh" shape Chapter 2 used for its topology bandwidth model: device 0 has a direct link to device 1 (so peer access is possible and the direct-write path is used), but no direct link to device 2 (so the code must fall back to staging the copy through a separate host buffer). Both transfer functions are written to mirror what real code actually does at each step, not to simplify the problem: `transferDirectP2P()` performs a plain assignment only when a hardcoded peer-access table says the ordered pair is accessible — the simulated equivalent of a kernel dereferencing a UVA pointer that resolves directly into another device's memory, no host involved at all. `transferStagedViaHost()` always works, by construction, because it never depends on peer access at all — it is the same universal fallback real `cudaMemcpy` calls use when peer access isn't available for a given pair. The correctness check compares both destinations against a `reference` buffer that is never touched by either transfer function, so the check cannot pass merely because both sides share a common bug.

```cpp
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
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
Device 0 initialized. Attempting device 0 -> device 1 (peer-accessible per the simulated topology) and device 0 -> device 2 (NOT peer-accessible).

Transfer 0->1: direct P2P write (peer access exists)
Transfer 0->2: direct P2P refused (no peer access, correctly) -- falling back to host-staged copy.

Correctness check, device 1 contents vs. independent reference: PASS
Correctness check, device 2 contents vs. independent reference: PASS

Both destinations hold identical, correct data despite using two different transfer mechanisms -- exactly the property real P2P code depends on: the *method* changes with topology, the *result* must not.
```

Both destinations end up holding the identical, correct reference data — device 1 via the direct-write path, device 2 via the host-staged fallback — which is exactly the property real multi-device code depends on: which mechanism a transfer uses is a topology decision, but the *result* it produces must never depend on which mechanism was chosen. This is what this simulation verifies honestly, without needing a second real GPU: not the real transfer's bandwidth or latency (Chapter 2 already handled those questions, honestly, as separate cost models), but the correctness of the routing logic itself.

!!! warning "[COMMON TRAP] Treating this simulation as a stand-in for measuring real P2P performance"
    This section's simulation verifies *logic* — does the right buffer end up with the right data, regardless of which transfer path was used — not *performance*. A real direct P2P write over NVLink and a real host-staged copy over PCIe have very different latency and bandwidth characteristics (Chapter 2 modeled exactly that difference in raw GB/s terms), and nothing about this host-only simulation can honestly speak to those numbers. Confusing "the routing logic is correct" with "the routing choice is fast" would undo the entire honesty discipline this book commits to: this simulation earns you confidence in *correctness*, and real hardware is the only thing that can honestly earn you confidence in *speed*.

## Chapter Summary

Unified Virtual Addressing puts the host and every device's memory into one flat address space, which is what makes `cudaPointerGetAttributes()` able to identify any pointer's owner generically — measured directly in Section 4.1, where even a query about an ordinary host stack array requires the same device-context machinery this book has reported as `cudaErrorNoDevice` since Chapter 1. UVA alone does not grant any device permission to dereference another device's pointers; that permission is a separate, queryable, and deliberately unidirectional setting established by `cudaDeviceCanAccessPeer()` and `cudaDeviceEnablePeerAccess()` — measured directly in Section 4.2's ordered-pair loop, which correctly runs zero iterations against zero reported devices rather than crashing or needing a special case. Because this environment cannot exercise a real transfer across that permission, Section 4.3 verified the *routing logic* multi-device code depends on — direct write where peer access exists, host-staged fallback where it doesn't — using this book's host-side simulation technique, confirming both paths converge on identical, correct data against an independent reference. Chapter 5 builds directly on this: the actual transfer *mechanism* — `cudaMemcpy` variants and asynchronous, stream-ordered copies — that Section 4.3's simulation stood in for.

## Self-Check Questions

1. Section 4.1's code calls `cudaPointerGetAttributes()` on an ordinary host stack array, not a device pointer, and it still fails with `cudaErrorNoDevice`. Explain why a query about host memory needs the same device-context machinery a device-memory query would.
2. What specific field does `cudaPointerGetAttributes()` return that lets a caller distinguish "this pointer belongs to device 2" from "this pointer belongs to device 3," and why is that distinction only meaningful because of UVA?
3. Section 4.2 states that `cudaDeviceEnablePeerAccess()` grants access from the perspective of whichever device is currently the caller's *current device*. If a program wants device 0 and device 1 to access each other's memory, how many calls to `cudaDeviceEnablePeerAccess()` are required, and what must happen between them?
4. Section 4.2's loop iterates over every *ordered* pair `(i, j)` with `i != j`, rather than every unordered pair. Explain concretely why an unordered-pair loop would be insufficient, using the unidirectional-access fact from Section 4.2.
5. In Section 4.3's simulation, `peerAccessible(0, 2)` returns `false`. Trace through exactly what `transferDirectP2P(devices, 0, 2)` does in that case, and explain why the calling code in `main()` correctly detects this and falls back to `transferStagedViaHost()`.
6. Why does Section 4.3's `main()` keep a `reference` buffer that neither `transferDirectP2P()` nor `transferStagedViaHost()` ever writes to, instead of just comparing `devices[1]` and `devices[2]` to each other directly?
7. The CUDA Programming Guide notes that non-NVSwitch systems support a maximum of eight peer connections per device. Using Chapter 2's partial-mesh topology model, explain why this ceiling is a real design constraint rather than an arbitrary software limit.
8. This chapter's Common Trap in Section 4.3 warns against treating the simulation as a performance measurement. Referencing Chapter 2's actual bandwidth figures, give one concrete number the simulation cannot honestly produce, and explain why not.
9. Suppose a fifth simulated device, device 4, is added to Section 4.3's model with peer access to device 1 but not to device 0. Would `transferDirectP2P(devices, 0, 4)` succeed? Would `transferStagedViaHost(devices, hostStaging, 0, 4)` still work? Explain both answers.

## Where We Go Next

Chapter 5 picks up exactly where Section 4.3's simulation left off — the actual `cudaMemcpy` variants (including the peer-aware ones) and asynchronous, stream-ordered copies that a real multi-device program uses to perform the transfers this chapter's simulation only modeled the logic of, plus what the CUDA Runtime API can honestly report about overlapping a copy with concurrent kernel execution.

## Worked Solutions

**1.** UVA means the runtime cannot answer "what kind of memory is this pointer" by inspecting the pointer's bits alone — the same numeric range could, in principle, need to be resolved against live device-context state to know for certain. Since no device context can be initialized at all on this machine (there is no real driver or GPU), any call that would need to consult that state — including one about a plain host array — fails at the same first step every other Runtime API call in this book has failed at: `cudaErrorNoDevice`.

**2.** The `device` field of the `cudaPointerAttributes` struct — an integer device ordinal. This distinction is only meaningful because of UVA: without a single shared address space, there would be no way to even ask "which device does this pointer belong to" using the pointer's value alone, since the same address could independently exist in multiple devices' separate address spaces with different meanings in each.

**3.** Two calls are required: `cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);` grants device 0 access to device 1, and separately `cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);` grants device 1 access to device 0. Between them, the current device must actually change — from 0 to 1 — because each call's effect applies to whichever device is current *at the time of the call*, not to some fixed pair.

**4.** An unordered-pair loop (checking only `(0,1)` and never separately `(1,0)`) would enable at most one direction of access per pair, since `cudaDeviceEnablePeerAccess()` only affects the current device's ability to access the named peer — the reverse direction, needed for device 1 to access device 0's memory, would never be requested at all, silently leaving half of all possible accesses disabled even though the loop appears to have "handled" that pair.

**5.** `peerAccessible(0, 2)` returns `false`, so `transferDirectP2P()`'s very first line, `if (!peerAccessible(src, dst)) return false;`, returns `false` immediately without touching `devices[2]` at all. Back in `main()`, `usedDirectFor2` is set to that returned `false`, and the `if (!usedDirectFor2)` branch correctly detects the refusal and calls `transferStagedViaHost()` — the function only takes the fallback path when the direct function has already, honestly, reported that it did nothing.

**6.** Comparing `devices[1]` to `devices[2]` directly would only prove the two destinations agree with *each other* — which could pass even if both had been corrupted identically by a shared bug in, say, how `Buffer` is initialized. Comparing each destination independently to a `reference` buffer that neither transfer function ever writes to rules that out: the only way both checks pass is if each destination genuinely received the correct original data, through its own transfer path, with no shared blind spot between the check and the thing being checked.

**7.** On a partial-mesh system (Chapter 2's model, without NVSwitch), every peer connection corresponds to an actual physical NVLink wire between two specific GPUs, and each GPU has a genuinely finite number of NVLink connectors and PCIe/driver resources available to manage active peer mappings. Eight is a real hardware and driver ceiling on how many such physical, resource-backed connections one device can maintain simultaneously — not a software policy that could simply be raised, which is exactly the kind of physical constraint Chapter 2's bandwidth models were already built around.

**8.** The simulation cannot honestly produce the 50 GB/s wired / 31.5 GB/s fallback distinction Chapter 2's topology bandwidth model (`04_topology_bandwidth_model.cpp`) computed for exactly this same partial-mesh shape, because the host-side simulation moves `std::vector` contents with ordinary CPU memory operations — it has no relationship whatsoever to real PCIe or NVLink transfer rates, and reporting a number from it as if it were real GPU-to-GPU bandwidth would be exactly the kind of fabricated measurement this book's honesty discipline forbids.

**9.** `transferDirectP2P(devices, 0, 4)` would fail (return `false`), because `peerAccessible(0, 4)` is `false` by the stated setup — device 4 only has peer access to device 1, not to device 0. `transferStagedViaHost(devices, hostStaging, 0, 4)` would still work without any changes at all, because that function never consults `peerAccessible()` in the first place — its entire purpose, and the reason it exists as the fallback, is that it is valid for *every* pair regardless of topology.

---

**Sources cited in this chapter:**

- [CUDA Programming Guide — Multi-GPU Systems](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html) -- Unified Virtual Addressing across host and devices; unidirectional nature of `cudaDeviceEnablePeerAccess()`; eight-peer-connection ceiling on non-NVSwitch systems.
- [CUDA Runtime API — Peer Device Memory Access](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PEER.html) -- `cudaDeviceCanAccessPeer()` and `cudaDeviceEnablePeerAccess()` signatures and documented semantics.
