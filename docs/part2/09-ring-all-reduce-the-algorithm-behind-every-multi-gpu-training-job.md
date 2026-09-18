# Chapter 9: Ring All-Reduce: The Algorithm Behind Every Multi-GPU Training Job

**What you will understand by the end of this chapter:**

- Why all-reduce -- every device ending up with the *same* combined result, not just one root -- is a genuinely different goal from Chapter 8's reduce, and why the simplest way to get there (chain Chapter 8's two collectives together) concentrates all the real work on a single device.
- How ring all-reduce spreads that same work evenly across every device by having each one talk only to its immediate ring-neighbor, in two phases: scatter-reduce, then all-gather.
- The real, closed-form reason this algorithm is the one essentially every real multi-GPU training job actually uses for gradient synchronization: it moves a fixed, bounded amount of data per device, no matter how large the group of devices grows.

**What you need to know first:**

- Chapter 8's broadcast and reduce, and its host-side simulation technique, now Part 2's primary verification tool.
- Chapter 2's `ringBottleneck()` model and the general idea that a ring's throughput is set by its slowest edge.
- No new CUDA concepts beyond composing calls this book has already introduced.

---

Chapter 8 built reduce -- every device's data combined into one correct result, on one root device. All-reduce asks for something related but distinct: every device ending up holding that *same* combined result, not just the root. Getting there is easy in principle -- reduce to a root, then broadcast the root's result back out, using nothing but Chapter 8's own two collectives, one after the other. It's also exactly the wrong way to do it at scale, because every single byte of that approach passes through one specific device, twice. Ring all-reduce fixes this by rethinking who talks to whom: instead of a star with one root at the center, every device sits in a ring and only ever exchanges data with its two immediate neighbors. This is the algorithm underneath essentially every real multi-GPU training job's gradient synchronization step, and this chapter builds it, by hand, directly on top of what Chapters 2 and 8 already gave this book.

```text
NAIVE (Chapter 8's two collectives, chained):        RING (this chapter):

        dev1     dev2     dev3                         dev0 -----> dev1
          \        |        /                            ^           |
           v       v       v                              |           v
        [        root (dev0)        ]                    dev3 <----- dev2
           |       |        |
           v       v        v
        dev1     dev2     dev3

  Every byte passes through the root            Every device only ever talks to
  TWICE (in during reduce, out during            its two ring neighbors -- no
  broadcast). The root's load grows with N;      device is ever singled out, and
  everyone else is idle most of the time.        the per-device load stays bounded.
```

## 9.1 The Naive Baseline: Reduce-to-Root, Then Broadcast

### Intuition

The most direct way to make every device hold the same combined result is to not invent anything new at all: run Chapter 8's reduce to get the correct sum onto one root device, then run Chapter 8's broadcast to send that root's result to everyone else. This is completely correct -- every device really does end up with the right answer -- and it's the natural first thing to reach for. Its cost is concentrated in exactly one place: the root device receives everyone else's data during the reduce phase, and then turns around and sends its own data to everyone else during the broadcast phase. Every other device only ever does half that work.

### Background

The code below is literally Section 8.2's reduce loop followed by Section 8.1's broadcast loop, with one connecting step in between: the host-computed sum has to be written back onto the root device before it can be broadcast from there, since the accumulation itself happened on the host, not on any device. Nothing about either loop changes from Chapter 8 -- what's new here is only that they're run back-to-back, on the same buffer, which is exactly what turns "a reduce" and "a broadcast" into "an all-reduce."

```cpp
// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 23_naive_allreduce.cu
//
// The naive baseline: reduce every device's buffer to a host
// accumulator (Chapter 8, Section 8.2), write the result back onto
// the root device, then broadcast it out to every other device
// (Chapter 8, Section 8.1). Genuinely compiled with a real nvcc and
// genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int ROOT = 0;
    const size_t N_INTS = 4;
    const size_t BYTES = N_INTS * sizeof(int);

    // Phase 1: reduce every device's buffer to a host accumulator,
    // exactly like Chapter 8, Section 8.2.
    long long accumulator[N_INTS] = {0, 0, 0, 0};
    int hostStaging[N_INTS];
    int devicesReduced = 0;
    for (int src = 0; src < deviceCount; ++src) {
        void* devPtr = nullptr;
        cudaError_t eAlloc = cudaMalloc(&devPtr, BYTES);
        cudaError_t eCopy = cudaMemcpy(hostStaging, devPtr, BYTES, cudaMemcpyDeviceToHost);
        printf("Reduce phase, cudaMemcpy(device %d -> host): %s (code %d)\n",
               src, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) {
            for (size_t i = 0; i < N_INTS; ++i) accumulator[i] += hostStaging[i];
            ++devicesReduced;
        }
        if (eAlloc == cudaSuccess) cudaFree(devPtr);
    }

    // Phase 2: write the reduced result back onto the root device, then
    // broadcast it to every other device -- exactly like Chapter 8,
    // Section 8.1. Only after this phase does every device hold the
    // SAME reduced value; this two-phase, root-centered round trip is
    // the "naive" all-reduce this chapter improves on.
    int hostResult[N_INTS];
    for (size_t i = 0; i < N_INTS; ++i) hostResult[i] = (int)accumulator[i];

    void* rootPtr = nullptr;
    cudaError_t eAllocRoot = cudaMalloc(&rootPtr, BYTES);
    cudaError_t eWriteRoot = cudaMemcpy(rootPtr, hostResult, BYTES, cudaMemcpyHostToDevice);
    printf("Broadcast phase, cudaMemcpy(host result -> root device): %s (code %d)\n",
           cudaGetErrorString(eWriteRoot), (int)eWriteRoot);

    int broadcastCount = 0;
    for (int dst = 0; dst < deviceCount; ++dst) {
        if (dst == ROOT) continue;
        void* dstPtr = nullptr;
        cudaError_t eAllocDst = cudaMalloc(&dstPtr, BYTES);
        cudaError_t eCopy = cudaMemcpyPeer(dstPtr, dst, rootPtr, ROOT, BYTES);
        printf("Broadcast phase, cudaMemcpyPeer(dst=dev%d, src=root): %s (code %d)\n",
               dst, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) ++broadcastCount;
        if (eAllocDst == cudaSuccess) cudaFree(dstPtr);
    }

    printf("\nReduced %d device(s), then broadcast reached %d non-root "
           "device(s). Every byte of this all-reduce passed through the\n"
           "root device TWICE -- once inbound during the reduce phase, "
           "once outbound during the broadcast phase -- while every other\n"
           "device only ever sent or received once. Section 9.3 makes this\n"
           "asymmetry concrete with real counted data-volume numbers.\n",
           devicesReduced, broadcastCount);

    if (eAllocRoot == cudaSuccess) cudaFree(rootPtr);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).
Broadcast phase, cudaMemcpy(host result -> root device): no CUDA-capable device is detected (code 100)

Reduced 0 device(s), then broadcast reached 0 non-root device(s). Every byte of this all-reduce passed through the
root device TWICE -- once inbound during the reduce phase, once outbound during the broadcast phase -- while every other
device only ever sent or received once. Section 9.3 makes this
asymmetry concrete with real counted data-volume numbers.
```

With zero devices reported, both phases have nothing real to move, and the one call that *can* still be attempted -- writing the host's (all-zero) result onto a root device that doesn't exist -- fails with the same honest `cudaErrorNoDevice` this book has reported since Chapter 1. What this section verifies genuinely is the real, compiled call sequence the naive approach actually uses; Section 9.3 is where this book can put a real number on exactly how lopsided that approach's workload is.

!!! warning "[COMMON TRAP] Assuming the root's extra work is 'fine' because it's still just one loop iteration more per call site"
    Reading Section 9.1's code, the root device's special handling looks like a small, local detail -- one extra `cudaMemcpy()` call, one `continue` statement skipping it in the broadcast loop. What that framing hides is that the root's *total* data movement scales with the number of *other* devices in the group: it receives from every one of them during the reduce phase, and sends to every one of them during the broadcast phase. As the device count grows, the root's workload grows right along with it, while every other device's workload stays flat at "one buffer sent, one buffer received." Section 9.3 turns this into an exact ratio rather than an intuition.

## 9.2 Ring All-Reduce: Every Device Talks Only to Its Neighbor

### Intuition

Ring all-reduce solves Section 9.1's imbalance by removing the root entirely. Arrange every device in a ring, so each one has exactly one "next" neighbor and one "previous" neighbor, and split each device's buffer into as many equal chunks as there are devices. The algorithm then runs in two phases, using the *identical* ring communication pattern for both: a **scatter-reduce** phase, where each chunk is passed around the ring, picking up one more device's contribution at every hop, until it has been summed across every device; and an **all-gather** phase, which uses that same ring pattern to forward each now-fully-summed chunk the rest of the way around, so every device ends up holding every chunk. As Andrew Gibiansky's widely-cited description of Baidu's ring-allreduce implementation puts it plainly, "the algorithm proceeds in two steps: first, a scatter-reduce, and then, an allgather" -- and because every device does the same small amount of work at every step, "given the right choice of neighbors for every GPU, this algorithm is bandwidth-optimal."

### Background

One ring step, on its own, is nothing more than a single `cudaMemcpyPeer()` call per device, each one sending its current chunk to `(i+1) % N` -- there is no root, no special case, and every device runs the exact same line of code. The full algorithm (built in Section 9.3) is just this one step, repeated `2*(N-1)` times: `N-1` repetitions for the scatter-reduce phase, then `N-1` more for the all-gather phase.

```text
ONE ring step, all N devices simultaneously:

    dev0 --chunk--> dev1        dev1 --chunk--> dev2
    dev2 --chunk--> dev3        dev3 --chunk--> dev0

Every device: exactly one cudaMemcpyPeer() call, to exactly one
neighbor, moving exactly one chunk. No device does more work than
any other, at any step -- this is the entire structural difference
from Section 9.1's star pattern.
```

```cpp
// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 24_ring_step_by_hand.cu
//
// ONE ring step: every device, symmetrically, sends its current chunk
// to its ring-neighbor (i+1)%N. Unlike Section 9.1's reduce/broadcast
// loops, there is no root here -- every device runs the identical
// operation. Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    int stepTransfers = 0;
    for (int i = 0; i < deviceCount; ++i) {
        int next = (i + 1) % deviceCount;
        void* srcPtr = nullptr;
        void* dstPtr = nullptr;
        cudaError_t eAllocSrc = cudaMalloc(&srcPtr, CHUNK_BYTES);
        cudaError_t eAllocDst = cudaMalloc(&dstPtr, CHUNK_BYTES);
        cudaError_t eCopy = cudaMemcpyPeer(dstPtr, next, srcPtr, i, CHUNK_BYTES);
        printf("Ring step, cudaMemcpyPeer(dst=dev%d, src=dev%d): %s (code %d)\n",
               next, i, cudaGetErrorString(eCopy), (int)eCopy);
        if (eCopy == cudaSuccess) ++stepTransfers;
        if (eAllocSrc == cudaSuccess) cudaFree(srcPtr);
        if (eAllocDst == cudaSuccess) cudaFree(dstPtr);
    }

    printf("\n%d of %d device(s) completed this ring step. On real hardware,\n"
           "every device performs exactly one cudaMemcpyPeer of exactly one\n"
           "chunk per step -- the same amount of work, every step, no matter\n"
           "how many devices are in the ring, and no device singled out.\n",
           stepTransfers, deviceCount);

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

0 of 0 device(s) completed this ring step. On real hardware,
every device performs exactly one cudaMemcpyPeer of exactly one
chunk per step -- the same amount of work, every step, no matter
how many devices are in the ring, and no device singled out.
```

With zero devices, the loop has nothing to iterate over, and the honest result is zero completed transfers -- the same correct degradation this book has shown for every loop-over-devices pattern since Chapter 4. What this section verifies is the real, single-step call pattern; Section 9.3 runs that same pattern `2*(N-1)` times in simulation and checks the result it actually produces.

!!! warning "[COMMON TRAP] Thinking one ring step alone constitutes a complete all-reduce"
    A single step like the one in this section's code only moves each device's chunk one hop around the ring -- it does not, by itself, sum anything across every device, and it certainly doesn't leave every device with the same final answer. The scatter-reduce phase needs `N-1` such steps just to get one fully-reduced chunk to its "owner" device, and the all-gather phase needs `N-1` more to spread each owner's completed chunk back out to everyone. One step is the algorithm's atomic building block, not the algorithm.

## 9.3 Verifying Correctness and the Real Data-Volume Advantage

### Intuition

Neither Section 9.1 nor 9.2 could check a result on this machine -- both, correctly, ran down to zero real devices. The actual claims worth checking are algorithmic: does the two-phase ring pattern really produce the correct sum, identically, at every device? And is the workload really spread evenly, in a way that can be stated as an exact number rather than just "seems more balanced"? Both are checkable with this book's host-side simulation technique, exactly as Chapter 8 already established as Part 2's primary verification tool.

### Background

Tracking a single chunk's actual journey around the ring makes the two-phase structure concrete. Using this section's own simulation, with device `i`'s chunk 0 initialized to `(i+1)*10` (so device 0 holds 10, device 1 holds 20, and so on):

```text
Tracking just CHUNK 0 as it travels around the ring (N=4 devices):

SCATTER-REDUCE phase (each hop ADDS the local value in):

  dev0(10) --chunk0=10--> dev1: 20+10 = 30
  dev1(30) --chunk0=30--> dev2: 30+30 = 60
  dev2(60) --chunk0=60--> dev3: 40+60 = 100   <-- dev3 now owns the FULL
                                                    reduced sum for chunk 0

ALL-GATHER phase (each hop OVERWRITES/forwards, no more adding):

  dev3(100) --chunk0=100--> dev0: now 100
  dev0(100) --chunk0=100--> dev1: now 100
  dev1(100) --chunk0=100--> dev2: now 100

  Every device now holds 100 for chunk 0 -- matching this book's own
  independently-computed reference (10+20+30+40 = 100).
```

The full simulation below runs this same pattern for *every* chunk simultaneously, across all `2*(N-1)` steps, using a single formula -- `sendChunk = (deviceIndex - step) mod N` -- for which chunk moves at every step of both phases; only whether that step *adds* or *overwrites* changes between the two phases. After confirming correctness, the code computes the real, counted data-volume comparison against Section 9.1's naive baseline for this exact problem size, using the same closed-form-arithmetic style Chapter 5's cost model established: no fabricated timing, only real counts.

```cpp
// Chapter 9: Ring All-Reduce -- The Algorithm Behind Every Multi-GPU
// Training Job
// 25_ring_allreduce_simulation.cpp
//
// Plain host C++ -- verifying the real ring all-reduce algorithm's
// correctness, and computing its real, counted data-volume advantage
// over Section 9.1's naive baseline. N real in-memory buffers stand in
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
    for (int step = 0; step < totalSteps; ++step) {
        std::vector<Chunks> snapshot = devices;
        bool scatterReducePhase = (step < N - 1);
        for (int j = 0; j < N; ++j) {
            int sendChunk = ((j - step) % N + N) % N;
            int next = (j + 1) % N;
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

    // Real, counted (not fabricated) data-volume comparison against
    // Section 9.1's naive reduce-then-broadcast baseline, for this
    // exact same problem size. "Traffic" below counts BOTH directions
    // (sent + received) for every device, so the two algorithms are
    // compared on the same basis.
    const int bufferSize = N; // elements per device (N chunks of 1 element each)

    // Naive: root receives (N-1)*bufferSize during reduce, then sends
    // (N-1)*bufferSize during broadcast -- both directions through
    // the SAME device.
    long long naiveRootTraffic = 2LL * (N - 1) * bufferSize;
    // Every other device just sends once and receives once, each of
    // one full buffer.
    long long naiveOtherTraffic = 2LL * bufferSize;

    // Ring: every device sends exactly 1 chunk AND receives exactly 1
    // chunk at every one of the 2*(N-1) steps.
    long long ringPerDeviceTraffic = 2LL * totalSteps * 1;

    printf("\nData-volume comparison for this exact problem (N=%d devices, "
           "%d elements/device):\n", N, bufferSize);
    printf("  Naive root device's traffic:      %lld elements (bottleneck)\n", naiveRootTraffic);
    printf("  Naive non-root device's traffic:  %lld elements\n", naiveOtherTraffic);
    printf("  Ring: EVERY device's traffic:      %lld elements\n", ringPerDeviceTraffic);
    printf("  Ratio (naive root / ring device):  %.2fx -- equals N/2 exactly.\n",
           (double)naiveRootTraffic / (double)ringPerDeviceTraffic);

    return allMatch ? 0 : 1;
}
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
Independent reference (sum of all 4 devices' chunks): 100 104 108 112 

After 6 ring steps (scatter-reduce + all-gather), all 4 devices hold the reference sum: PASS

Data-volume comparison for this exact problem (N=4 devices, 4 elements/device):
  Naive root device's traffic:      24 elements (bottleneck)
  Naive non-root device's traffic:  8 elements
  Ring: EVERY device's traffic:      12 elements
  Ratio (naive root / ring device):  2.00x -- equals N/2 exactly.
```

The correctness check passes: every device genuinely ends up holding the same, correctly-summed buffer, verified against a reference computed completely independently of the ring algorithm's own logic. The data-volume comparison then makes Section 9.1's Common Trap concrete: for this N=4 example, the naive approach's root device moves exactly 2x as much data as *any single device* in the ring does -- and per Gibiansky's own description, "the total amount of data transferred to and from every GPU is `2(N−1)K/N` which, crucially, is independent of the number of GPUs" (with `K` as the full buffer size) -- meaning the ring's *per-device* load stays bounded as `N` grows, while the naive approach's root keeps absorbing more and more of the total traffic as more devices join.

!!! warning "[COMMON TRAP] Assuming the '2.00x' ratio measured here is the whole story as N grows"
    This section's ratio (`naiveRootTraffic / ringPerDeviceTraffic`) works out to exactly `N/2` for the buffer size used here -- 2.00x at N=4, but 5x at N=10, and 50x at N=100. The gap between the two approaches doesn't stay fixed; it widens linearly with the number of devices in the group, because the naive root's traffic scales with `N` while the ring's per-device traffic (per Gibiansky's cited formula) is asymptotically *independent* of `N`. This section's PASS results establish correctness at N=4; the formula, not the specific number 2.00x, is what generalizes to real training runs with far more than 4 devices.

## Chapter Summary

All-reduce asks for every device to hold an identical, correctly-combined result -- a stronger requirement than Chapter 8's reduce, which only guaranteed that result on one root. The most direct way to get there chains Chapter 8's two collectives together, but concentrates all the real work on the root device, whose total traffic grows with the number of other devices in the group (Section 9.1). Ring all-reduce removes the root entirely: every device talks only to its immediate ring-neighbor, running the identical single-chunk transfer at every step (Section 9.2), across two phases -- scatter-reduce, then all-gather -- that use the exact same communication pattern for fundamentally different purposes, first accumulating each chunk's sum, then spreading each completed chunk back out. Section 9.3 verified both properties genuinely: the algorithm produces the correct, identical result at every simulated device, and its data-volume advantage over the naive baseline is a real, closed-form ratio (`N/2` for this section's buffer size) that widens as the device count grows, cited directly to the standard description of Baidu's ring-allreduce implementation. Chapter 10 extends this same ring-based thinking to the remaining core collectives -- all-gather, reduce-scatter, and all-to-all -- each one either a piece of what this chapter already built, or a close relative of it.

## Self-Check Questions

1. Explain, in your own words, the difference between what Chapter 8's reduce guarantees and what this chapter's all-reduce guarantees.
2. Section 9.1's Common Trap says the root's extra call "looks like a small, local detail" but isn't. Using the naive algorithm's two phases, explain specifically what makes the root's total workload scale with the device count while every other device's does not.
3. Section 9.2 states that one ring step is "the algorithm's atomic building block, not the algorithm." How many total ring steps does the full two-phase algorithm need, for a group of N devices, and how are they split between the two phases?
4. Using Section 9.3's chunk-0 trace, explain why device 3 (not device 0) is the device that ends up holding the fully-reduced sum for chunk 0 at the end of the scatter-reduce phase, for this specific N=4 example.
5. Section 9.3's code uses a single formula, `sendChunk = (deviceIndex - step) mod N`, for every step of both phases. What is the one thing that actually changes between the scatter-reduce phase and the all-gather phase, if the chunk-selection formula itself doesn't?
6. Quote the specific real-world claim this chapter cites to justify that the ring algorithm's per-device data volume does not grow with the number of devices, unlike the naive approach's root.
7. Suppose this chapter's exact simulation were re-run with N=8 devices instead of N=4, keeping one element per chunk. Using the formulas in Section 9.3's code, compute the new naive root traffic, the new ring per-device traffic, and their ratio. Does the ratio match what Section 9.3's Common Trap predicts for N=8?
8. Why does Section 9.3 check the ring algorithm's result against an independently-computed `reference`, rather than simply checking that the naive algorithm and the ring algorithm produce the same result as each other?

## Where We Go Next

Chapter 10 builds the remaining core collectives -- all-gather, reduce-scatter, and all-to-all -- on the same ring-based foundation this chapter established: all-gather is exactly this chapter's second phase used on its own, reduce-scatter is exactly this chapter's first phase used on its own, and all-to-all is a related but genuinely different pattern this book builds from scratch using the same verification discipline.

## Worked Solutions

**1.** Chapter 8's reduce combines every device's data into a single correct result that exists on exactly one device -- the root. This chapter's all-reduce combines the same data into a single correct result, but requires that result to exist, identically, on *every* device in the group, not just one. All-reduce is strictly the stronger guarantee; reduce is one building block toward it (as Section 9.1's naive approach shows directly), but reduce alone does not satisfy all-reduce's requirement.

**2.** During the reduce phase, the root receives one full buffer from each of the other `N-1` devices -- its inbound traffic is `(N-1)` buffers. During the broadcast phase, the root sends one full buffer to each of those same `N-1` devices -- its outbound traffic is another `(N-1)` buffers. Every other device, by contrast, sends exactly one buffer (during the reduce phase) and receives exactly one buffer (during the broadcast phase), regardless of how many total devices are in the group. The root's traffic is tied to `N-1`; everyone else's is a fixed, small constant.

**3.** The full algorithm needs `2*(N-1)` total ring steps: `N-1` steps for the scatter-reduce phase, and `N-1` more steps for the all-gather phase. Section 9.3's code computes this directly as `totalSteps = 2 * (N - 1)`, and for the N=4 example used throughout this chapter, that's 6 total steps (3 scatter-reduce, 3 all-gather), matching the locked output's "After 6 ring steps."

**4.** Following Section 9.3's chunk-0 trace: chunk 0 starts at device 0 (value 10), and at each scatter-reduce step it moves one hop forward around the ring (`(i+1) % N`), picking up the receiving device's own value by addition. Device 0 -> device 1 (30) -> device 2 (60) -> device 3 (100) is exactly `N-1 = 3` hops, which is precisely enough hops to have picked up all `N = 4` devices' original contributions (device 0's own value was already included when it was first sent). Device 3 is simply the device that chunk 0 happens to land on after traveling exactly `N-1` hops forward from its starting device, device 0 -- for a different chunk index, this "owner" device would be different, following the same `N-1`-hops-forward rule from that chunk's own starting device.

**5.** The single actual difference is whether the receiving device *adds* the incoming chunk to its own existing value (scatter-reduce) or *overwrites* its own value with the incoming one (all-gather) -- in the code, this is the `if (scatterReducePhase) { ... += ... } else { ... = ... }` branch. The chunk-selection formula, and the sender/receiver ring structure, are completely identical between the two phases; only this one arithmetic choice changes.

**6.** "The total amount of data transferred to and from every GPU is `2(N−1)K/N` which, crucially, is independent of the number of GPUs" -- this is the specific claim, cited to the standard description of Baidu's ring-allreduce implementation, that this chapter uses to establish the ring algorithm's per-device traffic does not scale with `N`, unlike the naive approach's root.

**7.** With N=8 and bufferSize=8 (one element per chunk, matching this chapter's convention): naive root traffic = `2*(N-1)*bufferSize = 2*7*8 = 112` elements. Ring per-device traffic = `2*totalSteps*1 = 2*(2*(N-1))*1 = 2*14*1 = 28` elements. The ratio is `112/28 = 4.00`, which is exactly `N/2 = 8/2 = 4` -- confirming Section 9.3's Common Trap's prediction that the ratio equals `N/2` and grows linearly with the device count (2.00 at N=4, 4.00 at N=8).

**8.** Checking the ring algorithm only against the naive algorithm's result would only prove the two approaches agree with *each other* -- which could still pass even if both happened to share an identical, unrelated bug (for instance, both silently using the wrong data type and overflowing identically). Checking against a `reference` computed by an entirely separate, independent calculation (a plain nested loop with no relationship to either algorithm's own internal logic) rules that out: the only way the ring algorithm's PASS result is meaningful is if it independently arrives at the mathematically correct answer, not merely an answer that happens to match a different algorithm built on related assumptions.

---

**Sources cited in this chapter:**

- [Bringing HPC Techniques to Deep Learning — Andrew Gibiansky](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/) -- the two-phase (scatter-reduce, then allgather) structure of ring all-reduce; the bandwidth-optimality claim; and the exact `2(N-1)K/N` per-device data-volume formula, independent of the number of GPUs.
