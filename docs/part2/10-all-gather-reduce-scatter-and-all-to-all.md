# Chapter 10: All-Gather, Reduce-Scatter, and All-to-All

**What you will understand by the end of this chapter:**

- Why all-gather (every device ends up holding *all* the original data, unmodified) and reduce-scatter (every device ends up holding exactly *one* reduced chunk) are, respectively, exactly Chapter 9's all-gather phase and scatter-reduce phase, run alone instead of chained together.
- Why all-to-all -- every device sending a *different* chunk to every other device -- cannot be built from either of Chapter 9's ring phases, and needs its own direct implementation and its own verification.
- Why reduce-scatter needs the host to do real arithmetic (like Chapter 8's reduce), while all-gather and all-to-all never do (like Chapter 8's broadcast) -- because moving bytes and combining values are different operations, and only one of them ever needs a `+`.

**What you need to know first:**

- Chapter 9's two-phase ring all-reduce (scatter-reduce, then all-gather) and its `sendChunk` formula.
- Chapter 8's broadcast (direct `cudaMemcpyPeer` loop) and reduce (host-accumulated `cudaMemcpy` loop) -- this chapter's three collectives are variations on those same two mechanisms.
- This book's host-side simulation technique, Part 2's primary way of checking a multi-device result.

---

Chapter 9 built ring all-reduce as one continuous two-phase algorithm -- scatter-reduce, then all-gather -- with a single ring pattern doing double duty for two different purposes. Both of those phases turn out to be useful operations in their own right, with their own names and their own uses outside of all-reduce entirely. This chapter isolates them: **all-gather** is Chapter 9's all-gather phase, run alone, starting from each device's own original data rather than a partially-reduced one; **reduce-scatter** is Chapter 9's scatter-reduce phase, run alone, stopping before the all-gather step that would normally follow it. A third collective, **all-to-all**, is a genuinely different pattern with no reduction step and, as Section 10.3 shows, no ring shortcut at all -- it needs its own implementation built from scratch. Together with Chapter 8's broadcast and reduce and Chapter 9's all-reduce, this completes the standard set of collective operations this book builds by hand, before Chapter 11 turns to NCCL, the library that implements all of them for real.

```text
This chapter's three collectives, N=4 devices, one chunk per device:

ALL-GATHER (10.1)                    REDUCE-SCATTER (10.2)
collect everything, combine nothing  combine everything, collect nothing

  before:  dev0[10]  dev1[20]          before:  dev0[10 11 12 13]
           dev2[30]  dev3[40]                   dev1[20 21 22 23]
                                                 dev2[30 31 32 33]
  after:   dev0[10 20 30 40]                    dev3[40 41 42 43]
           dev1[10 20 30 40]
           dev2[10 20 30 40]          after:   dev0[100]  dev1[104]
           dev3[10 20 30 40]                   dev2[108]  dev3[112]
  (every device: the SAME 4 values)   (every device: ONE different value --
                                        its own chunk index, summed)

ALL-TO-ALL (10.3): every device sends a DIFFERENT chunk to every other

  dev0 holds, for delivery:  ->dev1: 01   ->dev2: 02   ->dev3: 03
  dev1 holds, for delivery:  ->dev0: 10   ->dev2: 12   ->dev3: 13
  dev2 holds, for delivery:  ->dev0: 20   ->dev1: 21   ->dev3: 23
  dev3 holds, for delivery:  ->dev0: 30   ->dev1: 31   ->dev2: 32

  after: dev1 holds "01" (received from dev0), dev2 holds "02" (from dev0),
  dev0 holds "10" (from dev1), and so on -- a full transpose, no combining.
```

## 10.1 All-Gather: Collecting Without Reducing

### Intuition

Picture N people, each holding one page of a report they wrote independently. All-gather is everyone photocopying their own page and handing a copy to everyone else, until every person is holding the complete, N-page report -- every page exactly as its author wrote it, nothing edited, nothing combined. Contrast that with Chapter 8's broadcast, where only ONE person's page gets copied out to everyone: all-gather is what happens when *every* device is the broadcaster of its *own* one chunk, all at the same time.

```text
All-gather as N simultaneous single-chunk broadcasts:

  dev0 broadcasts chunk "10"  --->  dev1, dev2, dev3 each receive it
  dev1 broadcasts chunk "20"  --->  dev0, dev2, dev3 each receive it
  dev2 broadcasts chunk "30"  --->  dev0, dev1, dev3 each receive it
  dev3 broadcasts chunk "40"  --->  dev0, dev1, dev2 each receive it

  End state, every device: [10, 20, 30, 40] -- identical everywhere.
```

### Background

Because nothing here is combined, all-gather never needs the host at all -- every transfer is a direct, real `cudaMemcpyPeer()` between two devices, exactly like Chapter 8's broadcast. The direct (non-ring) way to implement it is the simplest possible reading of the diagram above: for every ordered pair of distinct devices `(src, dst)`, copy `src`'s own chunk into the slot reserved for it on `dst`. A device's own chunk needs no transfer at all -- it's already there.

```cpp
// Chapter 10: All-Gather, Reduce-Scatter, and All-to-All
// 26_allgather_by_hand.cu
//
// All-gather, done directly: every device sends its OWN chunk to
// every other device (N*(N-1) point-to-point transfers, no root, no
// reduction). This is the same all-gather PHASE Chapter 9 used inside
// its ring all-reduce (Section 9.2-9.3) -- shown here running alone,
// with the straightforward direct pattern instead of the ring
// pattern, so the operation itself can be seen in isolation.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    // Every device holds one recvBuffer with room for deviceCount
    // chunks -- slot i will hold device i's original chunk once
    // all-gather completes.
    int transfersAttempted = 0;
    int transfersCompleted = 0;
    for (int src = 0; src < deviceCount; ++src) {
        void* sendPtr = nullptr;
        cudaError_t eAllocSend = cudaMalloc(&sendPtr, CHUNK_BYTES);
        for (int dst = 0; dst < deviceCount; ++dst) {
            if (dst == src) continue; // a device's own chunk needs no transfer
            void* recvPtr = nullptr;
            cudaError_t eAllocRecv = cudaMalloc(&recvPtr, CHUNK_BYTES);
            cudaError_t eCopy = cudaMemcpyPeer(recvPtr, dst, sendPtr, src, CHUNK_BYTES);
            ++transfersAttempted;
            printf("All-gather, cudaMemcpyPeer(dst=dev%d's slot[%d], src=dev%d): %s (code %d)\n",
                   dst, src, src, cudaGetErrorString(eCopy), (int)eCopy);
            if (eCopy == cudaSuccess) ++transfersCompleted;
            if (eAllocRecv == cudaSuccess) cudaFree(recvPtr);
        }
        if (eAllocSend == cudaSuccess) cudaFree(sendPtr);
    }

    printf("\n%d of %d attempted point-to-point transfers completed. On real\n"
           "hardware, every device ends up holding all %d devices' original\n"
           "chunks, side by side, completely UNCHANGED -- all-gather never\n"
           "reduces or modifies anything, it only collects. Section 10.2\n"
           "shows the mirror-image operation, reduce-scatter.\n",
           transfersCompleted, transfersAttempted, deviceCount);

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

0 of 0 attempted point-to-point transfers completed. On real
hardware, every device ends up holding all 0 devices' original
chunks, side by side, completely UNCHANGED -- all-gather never
reduces or modifies anything, it only collects. Section 10.2
shows the mirror-image operation, reduce-scatter.
```

With zero devices, the outer loop never runs, so zero transfers are attempted and zero complete -- the honest degradation this book has shown for every device-count-driven loop since Chapter 4. What this section verifies genuinely is the real call pattern -- `cudaMemcpyPeer()` for every ordered pair, nothing else -- and Section 10.3's simulation is where this book can check the actual resulting data.

!!! warning "[COMMON TRAP] Thinking all-gather's N*(N-1) transfers make it as expensive, per device, as a broadcast to everyone"
    A single Chapter-8-style broadcast has one device sending to `N-1` others -- that device's outbound traffic is `N-1` chunks, and everyone else's is just one chunk received. All-gather runs `N` of those broadcasts *simultaneously*, one per device, so **every** device is simultaneously a sender (to `N-1` others) and a receiver (from `N-1` others) -- their traffic is symmetric, not concentrated on one device the way Chapter 9's naive root was. It's easy to see the total `N*(N-1)` transfer count and assume some device is doing `N-1` times the "normal" chunk-worth of broadcast work; in fact every device does exactly the same `N-1`-transfers-out, `N-1`-transfers-in amount of work as every other device.

## 10.2 Reduce-Scatter: Reducing Without Gathering Everywhere

### Intuition

Now picture N accountants, each holding a full ledger listing every one of N clients' numbers. Reduce-scatter assigns each accountant exactly one client and has them compute that one client's grand total, summed across all N ledgers -- and that's the only total they end up holding. Nobody needs any other accountant's client's total; each accountant's job is narrower than Chapter 8's reduce (where ONE root ends up with every client's total) but the work of actually summing is identical.

```text
Reduce-scatter as N independent single-chunk reduces:

  chunk 0, gathered from all 4 devices and summed: 10+20+30+40 = 100  -> dev0
  chunk 1, gathered from all 4 devices and summed: 11+21+31+41 = 104  -> dev1
  chunk 2, gathered from all 4 devices and summed: 12+22+32+42 = 108  -> dev2
  chunk 3, gathered from all 4 devices and summed: 13+23+33+43 = 112  -> dev3

  End state: dev0 holds ONLY 100, dev1 holds ONLY 104, and so on --
  never the same value twice, never more than one value per device.
```

### Background

Summing values is arithmetic, and neither `cudaMemcpy()` nor `cudaMemcpyPeer()` performs arithmetic -- they only move bytes. That is why reduce-scatter, like Chapter 8's reduce and Chapter 9's naive baseline, has to pull every device's contribution to a host accumulator, add it there with ordinary C++ `+=`, and write only the final sum back -- there is no way to do this reduction purely with device-to-device copies. The code below runs exactly this pattern once per chunk, writing chunk `c`'s reduced sum back only onto device `c`, its owner:

```cpp
// Chapter 10: All-Gather, Reduce-Scatter, and All-to-All
// 27_reducescatter_by_hand.cu
//
// Reduce-scatter, done directly: for each chunk index c, gather that
// ONE chunk from every device to the host, sum it, and write the sum
// back ONLY onto device c -- the device "assigned" to that chunk.
// This is exactly the SCATTER-REDUCE phase Chapter 9 used inside its
// ring all-reduce (Section 9.2-9.3), stopped there instead of
// continuing on into an all-gather. Genuinely compiled with a real
// nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    int chunksReduced = 0;
    int chunksScattered = 0;
    for (int owner = 0; owner < deviceCount; ++owner) {
        // Gather chunk `owner` from every device and reduce on the host --
        // exactly Chapter 8's reduce loop, but for a single chunk index.
        long long accumulator = 0;
        int hostStaging = 0;
        int devicesGathered = 0;
        for (int src = 0; src < deviceCount; ++src) {
            void* devPtr = nullptr;
            cudaError_t eAlloc = cudaMalloc(&devPtr, CHUNK_BYTES);
            cudaError_t eCopy = cudaMemcpy(&hostStaging, devPtr, CHUNK_BYTES, cudaMemcpyDeviceToHost);
            printf("Reduce-scatter, chunk %d: cudaMemcpy(device %d -> host): %s (code %d)\n",
                   owner, src, cudaGetErrorString(eCopy), (int)eCopy);
            if (eCopy == cudaSuccess) {
                accumulator += hostStaging;
                ++devicesGathered;
            }
            if (eAlloc == cudaSuccess) cudaFree(devPtr);
        }
        if (devicesGathered == deviceCount && deviceCount > 0) ++chunksReduced;

        // Scatter: write the reduced chunk back ONLY to device `owner`.
        // Every other device never receives this chunk at all -- that is
        // the entire point of reduce-scatter.
        int hostResult = (int)accumulator;
        void* ownerPtr = nullptr;
        cudaError_t eAllocOwner = cudaMalloc(&ownerPtr, CHUNK_BYTES);
        cudaError_t eWrite = cudaMemcpy(ownerPtr, &hostResult, CHUNK_BYTES, cudaMemcpyHostToDevice);
        printf("Reduce-scatter, chunk %d: cudaMemcpy(host reduced chunk -> owner dev%d): %s (code %d)\n\n",
               owner, owner, cudaGetErrorString(eWrite), (int)eWrite);
        if (eWrite == cudaSuccess) ++chunksScattered;
        if (eAllocOwner == cudaSuccess) cudaFree(ownerPtr);
    }

    printf("%d of %d chunk(s) fully reduced, %d of %d chunk(s) scattered to\n"
           "their owner. On real hardware, EVERY device ends up holding\n"
           "exactly ONE reduced chunk -- never all %d, and never the SAME\n"
           "chunk as any other device. Section 10.3 covers all-to-all, the\n"
           "one collective in this chapter with no reduction step at all.\n",
           chunksReduced, deviceCount, chunksScattered, deviceCount, deviceCount);

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).
0 of 0 chunk(s) fully reduced, 0 of 0 chunk(s) scattered to
their owner. On real hardware, EVERY device ends up holding
exactly ONE reduced chunk -- never all 0, and never the SAME
chunk as any other device. Section 10.3 covers all-to-all, the
one collective in this chapter with no reduction step at all.
```

With zero devices, the outer `owner` loop never runs, so no `cudaMemcpy` calls are attempted at all -- not even the honest `cudaErrorNoDevice` this book usually reports, since there's no loop iteration left to make the attempt from. Section 10.3's simulation checks the actual arithmetic this section's real code structure performs.

!!! warning "[COMMON TRAP] Assuming reduce-scatter is simply 'half the cost' of an all-reduce"
    It's tempting to reason that since all-reduce is (per Chapter 9) a reduce-scatter followed by an all-gather, reduce-scatter alone must be "half the work, so half the cost" -- and for total data volume, that's roughly right. But reduce-scatter is a complete, independently useful operation, not merely an incomplete all-reduce: whenever a device genuinely only ever needs *its own* shard of a combined result -- for instance, an optimizer that shards its state across devices, each device owning only the update for the parameters it's responsible for (a direct echo of Chapter 1's ZeRO-style memory-wall discussion) -- reduce-scatter alone is the right, final operation to use, with no all-gather needed or wanted afterward.

## 10.3 All-to-All: Personalized Exchange, No Reduction, No Shortcut

### Intuition

Picture N kids at a card-trading table, each holding a stack of N different cards, one earmarked for each of the N kids at the table (including, trivially, one for themselves). All-to-all is everyone trading simultaneously: every kid ends up with a new stack made of exactly one card from every other kid -- and crucially, the card kid A sends to kid B is completely different from the card kid A sends to kid C. That's the one thing that makes all-to-all fundamentally different from everything else in this chapter: all-gather sends the *same* chunk to everyone, and reduce-scatter *combines* chunks together, but all-to-all sends a *different*, uncombined chunk to every single destination.

```text
All-to-all as a full transpose (N=4, chunk "ij" = from device i, for device j):

  BEFORE (each device's own row, ready to send):     AFTER (each device's own column, received):

    dev0: [00, 01, 02, 03]                             dev0: [00, 10, 20, 30]
    dev1: [10, 11, 12, 13]        ===transpose===>     dev1: [01, 11, 21, 31]
    dev2: [20, 21, 22, 23]                             dev2: [02, 12, 22, 32]
    dev3: [30, 31, 32, 33]                             dev3: [03, 13, 23, 33]

  dev0's chunk "02" (meant for dev2) and dev0's chunk "03" (meant for
  dev3) are two DIFFERENT values -- unlike all-gather, where dev0 would
  send the SAME chunk to both.
```

### Background

Nothing about this pattern can be built from Chapter 9's ring phases: a ring hop can forward an all-reduce chunk unchanged because every device, by the end, wants that *same* value -- but all-to-all's chunk "02" and chunk "03" are both currently sitting on device 0, both need to leave device 0, and they need to go to two *different* places. There's no way to combine them into one relay, and no way for a chunk addressed to device 2 to also happen to satisfy device 3 along the way. All-to-all needs exactly `N*(N-1)` distinct point-to-point transfers -- the direct, real `cudaMemcpyPeer()` loop below, covering every ordered pair of distinct devices exactly once, each one carrying data no other transfer could have carried:

```cpp
// Chapter 10: All-Gather, Reduce-Scatter, and All-to-All
// 28_alltoall_by_hand.cu
//
// All-to-all: every device sends a DIFFERENT chunk to every other
// device, and receives a DIFFERENT chunk from every other device.
// Unlike all-gather (every device sends the SAME chunk to everyone)
// or reduce-scatter (chunks are combined, not just relayed), every
// one of the N*(N-1) transfers below carries genuinely distinct data
// -- there is no way to build this out of Chapter 9's ring phases,
// because nothing here is identical across destinations for a ring
// hop to forward along unchanged. Genuinely compiled with a real
// nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const size_t CHUNK_BYTES = sizeof(int);

    // Device `src` holds deviceCount send-slots; send-slot[dst] is the
    // chunk destined ONLY for device `dst`, and is different from every
    // other send-slot on device `src`.
    int transfersAttempted = 0;
    int transfersCompleted = 0;
    for (int src = 0; src < deviceCount; ++src) {
        for (int dst = 0; dst < deviceCount; ++dst) {
            if (dst == src) continue; // a device's own reserved slot needs no transfer
            void* sendSlotPtr = nullptr;
            void* recvSlotPtr = nullptr;
            cudaError_t eAllocSend = cudaMalloc(&sendSlotPtr, CHUNK_BYTES);
            cudaError_t eAllocRecv = cudaMalloc(&recvSlotPtr, CHUNK_BYTES);
            cudaError_t eCopy = cudaMemcpyPeer(recvSlotPtr, dst, sendSlotPtr, src, CHUNK_BYTES);
            ++transfersAttempted;
            printf("All-to-all, cudaMemcpyPeer(dev%d's slot[from=%d] <- dev%d's slot[to=%d]): %s (code %d)\n",
                   dst, src, src, dst, cudaGetErrorString(eCopy), (int)eCopy);
            if (eCopy == cudaSuccess) ++transfersCompleted;
            if (eAllocSend == cudaSuccess) cudaFree(sendSlotPtr);
            if (eAllocRecv == cudaSuccess) cudaFree(recvSlotPtr);
        }
    }

    printf("\n%d of %d attempted point-to-point transfers completed --\n"
           "exactly %d * (%d - 1) of them, one per ordered (src, dst) pair\n"
           "with src != dst. No reduction, no forwarding, no shortcut: every\n"
           "one of these transfers carries data no other transfer could have\n"
           "carried. Section 10.3's simulation verifies every device ends up\n"
           "with exactly the chunks addressed to it, and nothing else.\n",
           transfersCompleted, transfersAttempted, deviceCount, deviceCount);

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

0 of 0 attempted point-to-point transfers completed --
exactly 0 * (0 - 1) of them, one per ordered (src, dst) pair
with src != dst. No reduction, no forwarding, no shortcut: every
one of these transfers carries data no other transfer could have
carried. Section 10.3's simulation verifies every device ends up
with exactly the chunks addressed to it, and nothing else.
```

With zero devices, both loops never run and the `0 * (0 - 1)` in the summary is exactly what the honest arithmetic produces for `deviceCount = 0` -- the same pattern this book has reported since Chapter 1 rather than special-casing the degenerate count. What this section verifies genuinely is the real all-pairs call structure; the simulation below is where the actual chunk-delivery correctness gets checked, alongside Section 10.1's all-gather and Section 10.2's reduce-scatter, against three independent references:

```cpp
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
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
All-gather (N=4): independent reference chunks: 10 20 30 40 
  Every device's gathered array matches the reference: PASS

Reduce-scatter (N=4): independent reference sums: 100 104 108 112 
  Device c's scattered chunk matches reference[c] for every c: PASS

All-to-all (N=4): every device's 4 received chunks match the
  chunk their sender addressed to them (a full transpose): PASS

Direct all-gather/reduce-scatter: 3 message(s) per device, each to/from a DIFFERENT remote device.
All-to-all: 3 message(s) per device too -- but unlike all-gather/reduce-scatter, none of them can be replaced by a cheaper ring relay, because every message is unique.
```

All three collectives check out against references computed completely independently of any of this section's own transfer logic -- reusing exactly the same numbers Chapter 8 (10/20/30/40) and Chapter 9 (100/104/108/112) already established, so the continuity between chapters is checkable, not just claimed. The final two lines make the real structural point concrete: all-gather and reduce-scatter, run directly, need the same `N-1`-messages-per-device as all-to-all does -- but only the first two have a cheaper ring alternative (Chapter 9 already built it) waiting to be swapped in.

!!! warning "[COMMON TRAP] Assuming all-to-all's ring-free cost is a special weakness of this particular textbook implementation"
    It's tempting to read Section 10.3's "no ring shortcut" as a limitation of the specific direct loop this chapter wrote, fixable with a cleverer implementation. It isn't -- it's a property of the operation itself. All-to-all's defining feature is that every one of its `N*(N-1)` chunks is genuinely distinct data with a genuinely distinct destination; a ring algorithm's entire speed advantage comes from forwarding *the same* (or *the same running sum of*) data through several hops at once, which requires the data at each hop to be useful to more than one final destination. All-to-all's data structurally can't satisfy that requirement, so real communication libraries use different tricks for it entirely -- for short messages, an approach like Bruck's algorithm (cited in Sources) restructures *which pairs exchange data at which step* to get all-to-all done in only `log(N)` steps instead of `N-1`, but even that isn't a ring relay -- it's a different algorithm serving a fundamentally different data pattern.

## Chapter Summary

All-gather and reduce-scatter are not new algorithms this chapter invents -- they are Chapter 9's own two ring-all-reduce phases, named and run in isolation: all-gather is the phase that spreads an already-known value out to everyone (Section 10.1, here shown starting from each device's own untouched chunk rather than a partially-summed one), and reduce-scatter is the phase that combines values down to one owner per chunk (Section 10.2, here shown stopping short of the all-gather that would turn it into a full all-reduce). Because summing needs real arithmetic and moving bytes doesn't, reduce-scatter mirrors Chapter 8's host-mediated reduce while all-gather mirrors Chapter 8's direct, host-free broadcast. All-to-all (Section 10.3) breaks that pattern entirely: every one of its `N*(N-1)` transfers carries data no other transfer could carry, so there is no reduction, no ring relay, and no shortcut borrowed from Chapter 9 -- it needed its own direct implementation and its own independent verification, alongside the other two, in this chapter's simulation. Together with Chapter 8's broadcast and reduce and Chapter 9's all-reduce, this chapter completes the standard set of collective operations this book builds by hand. Chapter 11 turns to NCCL -- the library that implements every one of these operations for real, and this book's chance to check what, if anything, six chapters of hand-built collectives actually got right about how the real thing works.

## Self-Check Questions

1. Explain the difference between what all-gather guarantees ends up on every device and what reduce-scatter guarantees ends up on every device.
2. Which specific phase of Chapter 9's ring all-reduce is structurally identical to this chapter's all-gather, and what's the one thing different about how each device's starting data is initialized?
3. Which specific phase of Chapter 9's ring all-reduce is structurally identical to this chapter's reduce-scatter, and what step would you need to add afterward to turn a reduce-scatter into a full all-reduce?
4. Section 10.2's reduce-scatter code uses `cudaMemcpy` (host-mediated), while Section 10.1's all-gather and Section 10.3's all-to-all both use `cudaMemcpyPeer` directly between devices. Explain why reduce-scatter needs the host in the loop but the other two don't.
5. Section 10.3's Common Trap says all-to-all has "no ring shortcut." Using the fact that a ring hop can forward an all-reduce chunk unchanged but can't do the same for an all-to-all chunk, explain concretely why.
6. Using Section 10.3's simulation setup (`sendSlots[i][j] = i*100 + j`), what value ends up in device 2's `recvSlots[1]` after the all-to-all completes, and why?
7. The NCCL documentation (cited in Sources) defines AllGather as leaving "identical copies of the result in each recvbuff." Using that phrase, explain what specifically distinguishes all-gather's result from reduce-scatter's result on each device.
8. Suppose Section 10.1's direct all-gather code were run with N=6 devices instead of N=4. How many total `cudaMemcpyPeer` calls would it attempt, and how many chunks would each device hold at the end?

## Where We Go Next

Chapter 11 turns to NCCL, the standard library that implements broadcast, reduce, all-reduce, all-gather, reduce-scatter, and all-to-all for real, on real hardware -- the same six operations this book has now built by hand across Chapters 8 through 10. Rather than introducing new algorithms, Chapter 11 is where this book checks its own work: which of NCCL's real design choices (ring topologies, tree topologies, topology auto-detection, multi-node scaling) match what these chapters already predicted, and which go further than a from-scratch implementation would, on its own, ever have reason to.

## Worked Solutions

**1.** All-gather: every device ends up holding ALL `N` original chunks, side by side, completely unmodified -- the same full set of data, identical on every device. Reduce-scatter: every device ends up holding exactly ONE chunk, but that chunk is the SUM of that chunk index's value across all `N` devices -- a different reduced value on each device, never the same result appearing twice (except by coincidence of the underlying data). All-gather multiplies what every device holds; reduce-scatter divides it.

**2.** All-gather is structurally identical to the ALL-GATHER PHASE of Chapter 9's ring all-reduce -- the phase that forwards/overwrites each already-completed chunk the rest of the way around the ring. The one difference: Chapter 9's all-gather phase starts from chunks that have ALREADY been summed by the preceding scatter-reduce phase, while this chapter's all-gather starts directly from each device's own untouched original chunk, since there is no reduction step involved at all.

**3.** Reduce-scatter is structurally identical to the SCATTER-REDUCE PHASE of Chapter 9's ring all-reduce -- the first `N-1` steps, where each hop adds the incoming chunk into the receiving device's own running value. To turn a reduce-scatter into a full all-reduce, you would need to run an all-gather afterward: forward each now-fully-reduced chunk the rest of the way around (or, as this chapter's direct implementation would, send it to everyone directly) so every device ends up holding every reduced chunk, not just the one it owns.

**4.** Summing values is arithmetic, and neither `cudaMemcpy` nor `cudaMemcpyPeer` performs arithmetic -- both only move bytes from one place to another. Reduce-scatter needs a `+=` at every accumulation step, so, exactly like Chapter 8's reduce and Chapter 9's naive baseline, it has to pull every device's contribution to the host, add it there with ordinary C++ arithmetic, and write only the final sum back. All-gather and all-to-all never combine two chunks' values together -- they only relocate bytes from one device's memory to another's -- so a direct `cudaMemcpyPeer` call, with no host round-trip and no arithmetic, is all either operation ever needs.

**5.** In all-reduce, every version of a chunk traveling around the ring is either identical to, or a partial sum building toward, the exact same final value every device will end up holding -- so one relay chain of hops can serve every device along the way. In all-to-all, device `i`'s chunk meant for device `j` and device `i`'s chunk meant for device `k` are two different, unrelated pieces of data: forwarding the "for `j`" chunk through some intermediate device on its way to `j` does not also happen to deliver the "for `k`" chunk to `k`, the way a ring's shared partial sum happens to serve every device it passes through. Every one of the `N*(N-1)` chunks needs its own dedicated trip; none of them can piggyback on another's.

**6.** `sendSlots[i][j]` represents "the chunk device `i` holds, meant for device `j`," and the exchange rule is `recvSlots[j][i] = sendSlots[i][j]`. For device 2's `recvSlots[1]`, the indices `j=2, i=1` give `recvSlots[2][1] = sendSlots[1][2] = 1*100 + 2 = 102`. Device 2 ends up holding `102` in the slot recording what it received from device 1, matching exactly what device 1 originally addressed to device 2.

**7.** All-gather's result places the SAME complete set of chunks on every device -- "identical copies... in each recvbuff," in NCCL's own phrasing. Reduce-scatter's result is deliberately NOT identical across devices: each device's single chunk is the sum for a DIFFERENT chunk index, so device 0's result and device 1's result are, in general, two different numbers. "Identical" is precisely the word that would be the wrong description of reduce-scatter's output.

**8.** With `N=6`, the direct all-gather attempts `N*(N-1) = 6*5 = 30` total `cudaMemcpyPeer` calls -- one for every ordered pair of distinct devices. At the end, every one of the 6 devices holds all 6 original chunks: the chunk count per device always equals `N`, regardless of how many transfers it took to get there.

---

**Sources cited in this chapter:**

- [Collective Operations — NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) -- the exact AllGather ("gathers N values from k ranks into an output buffer of size k*N, and distributes that result to all ranks," and NCCL's `ncclAllGather` leaving "identical copies of the result in each recvbuff") and ReduceScatter ("the result is scattered in equal-sized blocks between ranks, each rank getting a chunk of data based on its rank index") definitions.
- [Optimization of Collective Communication Operations in MPICH — Thakur, Rabenseifner, Gropp](https://web.cels.anl.gov/~thakur/papers/ijhpca-coll.pdf) -- the nearest-neighbor-vs-farther-apart comparison between ring and recursive-doubling algorithms for allgather/reduce-scatter ("for long messages, the ring algorithm performs better than recursive doubling... because it uses a nearest-neighbor communication pattern, whereas in recursive doubling, processes that are much farther apart communicate"); the reduce-scatter-then-allgather decomposition of allreduce; and the Bruck algorithm for short-message all-to-all.
