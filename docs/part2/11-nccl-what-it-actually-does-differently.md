# Chapter 11: NCCL: What It Actually Does Differently From What You Just Built

<!-- mgpu:opener:start -->
[![Chapter 11: NCCL: What It Actually Does Differently From What You Just Built](../images/chapter-openers/11.svg)](../images/chapter-openers/11.svg)
<!-- mgpu:opener:end -->

**What you will understand by the end of this chapter:**

- How a real `ncclComm_t` communicator relates to the `deviceCount` loops this book has used since Chapter 3, and the two genuinely different ways a real `ncclCommInitAll()` call can honestly fail -- a malformed argument, rejected up front, versus a well-formed request that fails inside CUDA.
- That five of this book's six hand-built collectives (Chapters 8-10) have a real, one-call NCCL equivalent in the NCCL 2.26.2 this book links against, while all-to-all there is built exactly the way Chapter 10 did: from grouped point-to-point transfers. (Newer NCCL, from 2.28, adds a dedicated `ncclAlltoAll()`.)
- The one genuine algorithmic gap between what this book built and what NCCL does for real: ring all-reduce's round count grows linearly with the number of devices, and at large enough scale that stops being good enough -- which is exactly why NCCL added a second algorithm.

**What you need to know first:**

- All six of Chapters 8-10's hand-built collectives: broadcast, reduce, all-reduce (ring), all-gather, reduce-scatter, all-to-all.
- Chapter 9's ring all-reduce round count, `2*(N-1)`.
- This book's closed-form cost/round-count modeling technique (Chapter 2, Chapter 5), used again in Section 11.3.

---

This chapter introduces communicator construction, expected-error API probes and NCCL algorithm selection. Later initialized-buffer examples exercise the numerical collectives.

The [2026-11-17 H100 execution record](../validation-h100.md) covers two GPUs connected by local NV18. It distinguishes numerical checks, API probes and other executions; a successful exit alone does not establish GPU algorithm correctness.

```text
This book's hand-built collectives (Ch8-10)         NCCL's real API (this chapter)

  broadcast:      cudaMemcpyPeer loop, from root  -->  ncclBroadcast()
  reduce:         cudaMemcpy + host accumulator   -->  ncclReduce()
  all-reduce:     two-phase ring, by hand          -->  ncclAllReduce()  (ring OR
                                                          tree, chosen automatically
                                                          -- Section 11.3)
  all-gather:     direct cudaMemcpyPeer loop       -->  ncclAllGather()
  reduce-scatter: host-mediated, by hand           -->  ncclReduceScatter()
  all-to-all:     direct cudaMemcpyPeer all-pairs  -->  ncclSend()/ncclRecv(),
                                                          all-pairs, grouped --
                                                          no dedicated call in
                                                          2.26.2 (ncclAlltoAll()
                                                          arrived in NCCL 2.28)
```

<!-- mgpu:worked:start -->
## Worked example — Communicator rank is not a device ordinal

One host thread controls three GPUs. Map communicator ranks 0, 1, 2 to CUDA devices 2, 0, 1 respectively. Each rank contributes four FP32 values (16 bytes).

[![Numeric worked example: Communicator rank is not a device ordinal](../images/worked/ch11.svg)](../images/worked/ch11.svg)

1. Rank 0/device 2 contributes [1, 2, 3, 4]; rank 1/device 0 contributes [10, 20, 30, 40]; rank 2/device 1 contributes [100, 200, 300, 400].

2. Select the appropriate current device before its stream and NCCL call. Inside the multi-device launch group, each rank calls ncclAllReduce with count 4, ncclFloat and ncclSum.

3. End the group before waiting for stream completion. Group submission alone is not permission to read an unfinished device result.

4. After each relevant stream completes, every rank holds [111, 222, 333, 444]. In a rooted operation, root = 1 would refer to device 0 under this mapping.

**Expected result and pitfall.** The expected result follows communicator rank membership, independently of the numeric CUDA ordinals. Compatible operation, datatype, counts and call order are required across participants. Check CUDA and NCCL return values and asynchronous errors using the chapter’s full examples.

**Exercise.** Does root = 2 select CUDA device 2 in this mapping?

**Worked answer.** No. Root rank 2 maps to CUDA device 1. Keep an explicit rank-to-device mapping rather than conflating the two spaces.
<!-- mgpu:worked:end -->

## 11.1 A Real Communicator: `ncclCommInitAll` and the World This Book Has Called `deviceCount`

<!-- mgpu:figure-1:start -->
[![A communicator establishes membership](../images/technical/ch11/1-flow.svg)](../images/technical/ch11/1-flow.svg)
<!-- mgpu:figure-1:end -->

### Intuition

Every real NCCL collective call needs a communicator -- a single object that names which devices are in the group and how they're numbered, so that a call like "broadcast this buffer" has an actual group to broadcast across. Think of it like a team roster handed out before a group project starts: nobody on the team can be assigned a task until the roster itself exists, with everyone's name and role settled. This book's `deviceCount`-driven loops have quietly done the roster's job themselves, informally, since Chapter 3; NCCL just makes that roster a real, explicit object you create once, up front.

```text
Before any collective call, a real NCCL program needs a communicator:

  cudaGetDeviceCount()  -->  ncclCommInitAll(comms, ndev, devlist)  -->  ncclComm_t[ndev]
        |                              |
        +--> "how many devices         +--> "bind each of these ndev communicators
              do I actually have?"           to a real CUDA device, as one group"

  This chapter's own deviceCount is 0 -- so ndev=0 is exactly what an
  honest program, built the way every prior chapter has been, would
  actually pass here.
```

### Background

`ncclGetVersion()` reads the linked library version. Communicator initialization additionally binds ranks to CUDA devices. File 30 checks valid setup and deliberately invalid arguments as separate cases; NCCL warning text identifies the reason for each rejected call.


The [2026-11-17 H100 execution record](../validation-h100.md) covers two GPUs connected by local NV18. It distinguishes numerical checks, API probes and other executions; a successful exit alone does not establish GPU algorithm correctness.

```cpp
// 30_nccl_comm_init.cu
// API probe: valid communicator lifetimes and rejected malformed arguments.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
int main() {
 int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count<1) return 2;
 std::vector<ncclComm_t> comms(count,nullptr);
 ncclResult_t result=ncclCommInitAll(comms.data(),count,nullptr);
 if(result!=ncclSuccess) { fprintf(stderr,"init: %s\n",ncclGetErrorString(result)); return 1; }
 int errors=0;
 for(auto comm:comms) if(ncclCommDestroy(comm)!=ncclSuccess) ++errors;
 ncclComm_t single=nullptr;
 result=ncclCommInitAll(&single,1,nullptr);
 if(result!=ncclSuccess) ++errors;
 else if(ncclCommDestroy(single)!=ncclSuccess) ++errors;
 ncclComm_t unused=nullptr;
 if(ncclCommInitAll(nullptr,1,nullptr)!=ncclInvalidArgument) ++errors;
 if(ncclCommInitAll(&unused,-1,nullptr)!=ncclInvalidArgument) ++errors;
 printf("API PROBE %s: %d devices, valid initialization/destruction and invalid argument checks, errors=%d\n",errors?"FAIL":"PASS",count,errors);
 return errors?1:0;
}
```

<!-- execution:start:h100-rerun:30_nccl_comm_init.cu -->
> **Compiled and executed — 2026-11-18-h100: API validation (recorded scope).** Hardware: 2 H100 GPUs. Executed UTC: `2026-11-18T11:21:17.040890+00:00`.

<!-- execution:metadata Source SHA-256: 8727f9eeaaba6ddea198f70216798f22ba552ebeaf625453234128773ef4074b; recorded UTC (program start): 2026-11-18T11:21:15.874665+00:00 -->

[Exact tested source](../validation/2026-11-18-h100/30_nccl_comm_init/30_nccl_comm_init.cu) · [Compiler log](../validation/2026-11-18-h100/30_nccl_comm_init/compile.log) · [Execution log](../validation/2026-11-18-h100/30_nccl_comm_init/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
nvcc -std=c++17 -O2 -arch=sm_90 30_nccl_comm_init.cu -o ./example -lnccl
./example
```

**Sample output**

```text
API PROBE PASS: 2 devices, valid initialization/destruction and invalid argument checks, errors=0
```

<!-- execution:end:h100-rerun:30_nccl_comm_init.cu -->


<!-- execution:start:a100:30_nccl_comm_init.cu -->
> **Compiled and executed — 2026-11-17-a100: API validation (recorded scope).** Hardware: 8 A100 GPUs. Executed UTC: `2026-11-18T00:01:50.025564+00:00`.

<!-- execution:metadata Source SHA-256: 8727f9eeaaba6ddea198f70216798f22ba552ebeaf625453234128773ef4074b; recorded UTC (program start): 2026-11-18T00:01:48.006326+00:00 -->

[Exact tested source](../validation/2026-11-17-a100/30_nccl_comm_init/30_nccl_comm_init.cu) · [Compiler log](../validation/2026-11-17-a100/30_nccl_comm_init/compile.log) · [Execution log](../validation/2026-11-17-a100/30_nccl_comm_init/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
nvcc -std=c++17 -O2 -arch=sm_80 30_nccl_comm_init.cu -o ./example -lnccl
./example
```

**Sample output**

```text
API PROBE PASS: 8 devices, valid initialization/destruction and invalid argument checks, errors=0
```


<!-- execution:end:a100:30_nccl_comm_init.cu -->


<!-- execution:start:h100:30_nccl_comm_init.cu -->
> **Compiled and executed — 2026-11-17-h100: API validation (recorded scope).** Hardware: 2 H100 GPUs. Execution record UTC (run-batch start; per-program time unrecorded): `2026-11-17T23:48:18.967387+00:00`.

<!-- execution:metadata Source SHA-256: 8727f9eeaaba6ddea198f70216798f22ba552ebeaf625453234128773ef4074b; recorded UTC (run-batch start; per-program time unrecorded): 2026-11-17T23:48:18.967387+00:00 -->

[Exact tested source](../validation/2026-11-17-h100/30_nccl_comm_init/30_nccl_comm_init.cu) · [Compiler log](../validation/2026-11-17-h100/30_nccl_comm_init/compile.log) · [Execution log](../validation/2026-11-17-h100/30_nccl_comm_init/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
nvcc -std=c++17 -O2 -arch=sm_90 30_nccl_comm_init.cu -o ./example -lnccl
./example
```

<details markdown="1"><summary>Captured current-source output</summary>

```text
API PROBE PASS: 2 devices, valid initialization/destruction and invalid argument checks, errors=0
```

</details>
<!-- execution:end:h100:30_nccl_comm_init.cu -->


File 30 checks communicator initialization and destruction against the detected device count, then checks invalid arguments as separate expected-error cases.

The [2026-11-17 H100 execution record](../validation-h100.md) covers two GPUs connected by local NV18. It distinguishes numerical checks, API probes and other executions; a successful exit alone does not establish GPU algorithm correctness.

!!! warning "[COMMON TRAP] Inferring an exact root cause from a result code alone"
    `ncclInvalidArgument` identifies an argument error; `ncclUnhandledCudaError` reports a CUDA failure within NCCL. Neither numeric code alone identifies the failed argument or CUDA operation. Read `NCCL_DEBUG=WARN` diagnostics and the checked call context. Expected-error probes are API checks, not successful collective computations.

The [2026-11-17 H100 execution record](../validation-h100.md) covers two GPUs connected by local NV18. It distinguishes numerical checks, API probes and other executions; a successful exit alone does not establish GPU algorithm correctness.

## 11.2 Six Collectives, Six Real Calls (Mostly)

<!-- mgpu:figure-2:start -->
[![Match the mathematical operation to the call](../images/technical/ch11/2-mapping.svg)](../images/technical/ch11/2-mapping.svg)
<!-- mgpu:figure-2:end -->

### Intuition

Chapters 8 through 10 built six collectives by hand, one loop of `cudaMemcpyPeer()` or `cudaMemcpy()` calls at a time. Real NCCL packages five of those six as a single function call each -- the entire loop this book wrote out explicitly, hidden behind one name. The sixth, all-to-all, doesn't get that treatment in the NCCL 2.26.2 this book links against: there it's built from the same kind of point-to-point calls this book used, just NCCL's own `ncclSend()`/`ncclRecv()` instead of `cudaMemcpyPeer()`. That is a version-specific fact, not a permanent one: NCCL 2.28.3's release notes record "Added new host collective APIs: ncclAlltoAll, ncclScatter, and ncclGather," and the current collective-operations documentation includes `ncclAlltoAll`.

```text
Chapter 8-10's six collectives, and their real NCCL 2.26.2 call:

  Ch8 8.1   --> ncclBroadcast()          Ch10 10.1 --> ncclAllGather()
  Ch8 8.2   --> ncclReduce()             Ch10 10.2 --> ncclReduceScatter()
  Ch9       --> ncclAllReduce()          Ch10 10.3 --> ncclSend() + ncclRecv()
                                                       (grouped; no dedicated
                                                        call until NCCL 2.28's
                                                        ncclAlltoAll())
```

### Background

File 31 deliberately passes a NULL communicator to check argument rejection. These expected-error cases are distinct from the initialized-buffer collective examples in Chapters 12, 14 and 17.


The [2026-11-17 H100 execution record](../validation-h100.md) covers two GPUs connected by local NV18. It distinguishes numerical checks, API probes and other executions; a successful exit alone does not establish GPU algorithm correctness.

```cpp
// 31_nccl_collectives_mapped.cu
// Expected-error API probe; no collective is claimed to execute successfully.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>
int main() {
 ncclComm_t comm=nullptr; cudaStream_t stream=nullptr; float *buf=nullptr;
 int errors=0;
 auto check=[&](const char *name,ncclResult_t value) {
  printf("%s: %s (expected invalid argument)\n",name,ncclGetErrorString(value));
  if(value!=ncclInvalidArgument) ++errors;
 };
 check("Broadcast",ncclBroadcast(buf,buf,4,ncclFloat,0,comm,stream));
 check("Reduce",ncclReduce(buf,buf,4,ncclFloat,ncclSum,0,comm,stream));
 check("AllReduce",ncclAllReduce(buf,buf,4,ncclFloat,ncclSum,comm,stream));
 check("AllGather",ncclAllGather(buf,buf,4,ncclFloat,comm,stream));
 check("ReduceScatter",ncclReduceScatter(buf,buf,4,ncclFloat,ncclSum,comm,stream));
 check("Send",ncclSend(buf,1,ncclFloat,0,comm,stream));
 check("Recv",ncclRecv(buf,1,ncclFloat,0,comm,stream));
 printf("API PROBE %s: NULL-communicator rejection only, errors=%d\n",errors?"FAIL":"PASS",errors);
 return errors?1:0;
}
```

<!-- execution:start:h100-rerun:31_nccl_collectives_mapped.cu -->
> **Compiled and executed — 2026-11-18-h100: API validation (recorded scope).** Hardware: 2 H100 GPUs. Executed UTC: `2026-11-18T11:21:20.326830+00:00`.

<!-- execution:metadata Source SHA-256: 1d7c8fdc5703d2568fad0a308233cb39f8245aa0f25d25c709d493e216f63a2c; recorded UTC (program start): 2026-11-18T11:21:19.310597+00:00 -->

[Exact tested source](../validation/2026-11-18-h100/31_nccl_collectives_mapped/31_nccl_collectives_mapped.cu) · [Compiler log](../validation/2026-11-18-h100/31_nccl_collectives_mapped/compile.log) · [Execution log](../validation/2026-11-18-h100/31_nccl_collectives_mapped/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
nvcc -std=c++17 -O2 -arch=sm_90 31_nccl_collectives_mapped.cu -o ./example -lnccl
./example
```

**Sample output**

```text
Broadcast: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Reduce: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
AllReduce: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
AllGather: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
ReduceScatter: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Send: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Recv: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
API PROBE PASS: NULL-communicator rejection only, errors=0
```

<!-- execution:end:h100-rerun:31_nccl_collectives_mapped.cu -->


<!-- execution:start:a100:31_nccl_collectives_mapped.cu -->
> **Compiled and executed — 2026-11-17-a100: API validation (recorded scope).** Hardware: 8 A100 GPUs. Executed UTC: `2026-11-18T00:01:59.278843+00:00`.

<!-- execution:metadata Source SHA-256: 1d7c8fdc5703d2568fad0a308233cb39f8245aa0f25d25c709d493e216f63a2c; recorded UTC (program start): 2026-11-18T00:01:57.510122+00:00 -->

[Exact tested source](../validation/2026-11-17-a100/31_nccl_collectives_mapped/31_nccl_collectives_mapped.cu) · [Compiler log](../validation/2026-11-17-a100/31_nccl_collectives_mapped/compile.log) · [Execution log](../validation/2026-11-17-a100/31_nccl_collectives_mapped/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
nvcc -std=c++17 -O2 -arch=sm_80 31_nccl_collectives_mapped.cu -o ./example -lnccl
./example
```

**Sample output**

```text
Broadcast: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Reduce: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
AllReduce: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
AllGather: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
ReduceScatter: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Send: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Recv: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
API PROBE PASS: NULL-communicator rejection only, errors=0
```


<!-- execution:end:a100:31_nccl_collectives_mapped.cu -->


<!-- execution:start:h100:31_nccl_collectives_mapped.cu -->
> **Compiled and executed — 2026-11-17-h100: API validation (recorded scope).** Hardware: 2 H100 GPUs. Execution record UTC (run-batch start; per-program time unrecorded): `2026-11-17T23:48:18.967387+00:00`.

<!-- execution:metadata Source SHA-256: 1d7c8fdc5703d2568fad0a308233cb39f8245aa0f25d25c709d493e216f63a2c; recorded UTC (run-batch start; per-program time unrecorded): 2026-11-17T23:48:18.967387+00:00 -->

[Exact tested source](../validation/2026-11-17-h100/31_nccl_collectives_mapped/31_nccl_collectives_mapped.cu) · [Compiler log](../validation/2026-11-17-h100/31_nccl_collectives_mapped/compile.log) · [Execution log](../validation/2026-11-17-h100/31_nccl_collectives_mapped/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
nvcc -std=c++17 -O2 -arch=sm_90 31_nccl_collectives_mapped.cu -o ./example -lnccl
./example
```

<details markdown="1"><summary>Captured current-source output</summary>

```text
Broadcast: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Reduce: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
AllReduce: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
AllGather: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
ReduceScatter: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Send: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
Recv: invalid argument (run with NCCL_DEBUG=WARN for details) (expected invalid argument)
API PROBE PASS: NULL-communicator rejection only, errors=0
```

</details>
<!-- execution:end:h100:31_nccl_collectives_mapped.cu -->


The API probe checks the documented rejection of invalid communicator handles. Real All-to-All communication uses matching sends and receives within a group, with every rank participating.

!!! warning "[COMMON TRAP] Assuming the missing all-to-all call (in NCCL 2.26.2) means all-to-all is somehow a second-class operation in NCCL"
    It would be easy to read "no dedicated call" as "NCCL doesn't really support this." Two things say otherwise. First, the gap is version-specific: NCCL 2.28 added a dedicated `ncclAlltoAll()`, so on a current NCCL the one-call form exists. Second, even in the 2.26.2 this book links against, NCCL's own documentation says otherwise: point-to-point `ncclSend()`/`ncclRecv()` calls "can be fused together with `ncclGroupStart()` and `ncclGroupEnd()` to form more complex communication patterns," and the documentation's own example for exactly this fusion is an all-to-all loop, line for line the same shape as this section's code. All-to-all is fully supported in 2.26.2 -- it's just supported there as a composition of two more primitive calls, the same way this book's own Chapter 10 built it as a composition of `cudaMemcpyPeer()` calls, rather than as one specially-named function.

## 11.3 Ring vs. Tree: The Latency Problem Chapter 9 Never Had to Solve

### Intuition

Chapter 9's ring all-reduce runs in `2*(N-1)` rounds, and at `N=4` that's a perfectly reasonable 6 rounds -- nobody would notice. But `2*(N-1)` is a straight line: double the device count, and you roughly double the number of rounds. At the scale real large training clusters actually reach -- thousands, sometimes tens of thousands of devices -- that straight line stops being a minor detail and starts being the whole bottleneck. NCCL's answer, since version 2.4, is a second algorithm for exactly this situation: instead of a ring, arrange the devices as a tree, where a value only has to climb up to a root and back down, a number of hops set by the tree's *height* rather than its device *count* -- and a balanced tree's height barely grows at all as you add more devices.

```text
Ring (Ch9): every device touches only its 2 neighbors, but a chunk
needs N-1 hops to cross the whole ring once:

  dev0 -> dev1 -> dev2 -> dev3 -> ... -> dev(N-1)     (N-1 hops)

Tree: every device is a node in a tree of height ~log2(N); a value
climbs UP to the root in ~log2(N) hops, then comes back DOWN in
~log2(N) more:

                 root
                   |
          +--------+--------+
          |                 |
         node              node
          |                 |
      +---+---+         +---+---+
      |       |         |       |
     leaf    leaf      leaf    leaf

  N = 4 leaves: height log2(4) = 2 hops from any leaf up to the root
  (3 levels of nodes, counting the leaves themselves). In general a
  balanced binary tree over N leaves has height ceil(log2(N)).
```

### Background

NVIDIA's own announcement of this feature states the ring's problem plainly: "latency scales linearly with the number of GPUs, preventing scaling above hundreds of GPUs." Its fix, the double binary tree, is described as offering "full bandwidth and a logarithmic latency" -- and, importantly, NCCL doesn't force a choice between the two: "NCCL automatically switches back to rings when that pattern results in greater bandwidth," picking whichever algorithm actually wins for the real topology and message size in front of it. The model below doesn't measure any real timing -- it counts, in the same closed-form spirit as Chapter 2's topology model and Chapter 5's cost model, exactly how many communication *rounds* each shape needs, reusing Chapter 9's own `2*(N-1)` ring formula unchanged and adding the matching round count for a tree of height `ceil(log2(N))`.

```cpp
// Chapter 11: NCCL -- What It Actually Does Differently From What You
// Just Built
// 32_ring_vs_tree_model.cpp
//
// Plain host C++ -- a closed-form ROUND-COUNT model, in the same
// spirit as Chapter 2's topology model and Chapter 5's cost model:
// no fabricated timings, only real, defensible structural counts.
// This compares Chapter 9's own ring all-reduce round count against
// a double binary tree's round count, for growing device counts, to
// make the NCCL 2.4 blog post's "latency scales linearly... versus
// logarithmic latency" claim (cited in Sources) concrete as numbers.
#include <cstdio>
#include <cmath>

int main() {
    printf("%-10s %-22s %-22s %-10s\n", "N", "Ring rounds (Ch9)", "Tree rounds (up+down)", "Ratio");
    printf("%-10s %-22s %-22s %-10s\n", "-", "2*(N-1)", "2*ceil(log2(N))", "ring/tree");

    int Ns[] = {4, 8, 16, 64, 256, 1024, 8192, 24576};
    for (int N : Ns) {
        // Ring: Chapter 9's own totalSteps formula, unchanged.
        long long ringRounds = 2LL * (N - 1);

        // Double binary tree: reducing up to a root takes one round per
        // level of the tree, and broadcasting back down takes another
        // round per level -- a balanced binary tree over N leaves has
        // ceil(log2(N)) levels.
        int levels = (int)std::ceil(std::log2((double)N));
        long long treeRounds = 2LL * levels;

        double ratio = (double)ringRounds / (double)treeRounds;
        printf("%-10d %-22lld %-22lld %-10.2fx\n", N, ringRounds, treeRounds, ratio);
    }

    printf("\nRing's round count grows linearly with N (doubling N roughly\n"
           "doubles the rounds needed); the tree's grows logarithmically\n"
           "(doubling N adds only one more level in each direction). This\n"
           "is a structural count of communication ROUNDS, not a measured\n"
           "time -- it makes concrete exactly the shape of the claim this\n"
           "chapter cites from NVIDIA's own NCCL 2.4 announcement, without\n"
           "fabricating a single timing number of its own.\n");

    return 0;
}
```

<!-- execution:start:h100-rerun:32_ring_vs_tree_model.cpp -->
> **Compiled and executed — 2026-11-18-h100: Host program/model (exit-only).** Hardware: 2 H100 GPUs. Executed UTC: `2026-11-18T11:21:20.496018+00:00`.

<!-- execution:metadata Source SHA-256: 26cb373d652c1985ee1c42d2aa5edb5bee5a47ca5900dc938ca2e110e76967e5; recorded UTC (program start): 2026-11-18T11:21:20.331710+00:00 -->

[Exact tested source](../validation/2026-11-18-h100/32_ring_vs_tree_model/32_ring_vs_tree_model.cpp) · [Compiler log](../validation/2026-11-18-h100/32_ring_vs_tree_model/compile.log) · [Execution log](../validation/2026-11-18-h100/32_ring_vs_tree_model/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
g++ -std=c++17 -O2 32_ring_vs_tree_model.cpp -o ./example
./example
```

**Sample output**

```text
N          Ring rounds (Ch9)      Tree rounds (up+down)  Ratio
-          2*(N-1)                2*ceil(log2(N))        ring/tree
4          6                      4                      1.50      x
8          14                     6                      2.33      x
16         30                     8                      3.75      x
64         126                    12                     10.50     x
256        510                    16                     31.88     x
1024       2046                   20                     102.30    x
8192       16382                  26                     630.08    x
24576      49150                  30                     1638.33   x

Ring's round count grows linearly with N (doubling N roughly
doubles the rounds needed); the tree's grows logarithmically
(doubling N adds only one more level in each direction). This
is a structural count of communication ROUNDS, not a measured
time -- it makes concrete exactly the shape of the claim this
chapter cites from NVIDIA's own NCCL 2.4 announcement, without
fabricating a single timing number of its own.
```

<!-- execution:end:h100-rerun:32_ring_vs_tree_model.cpp -->


<!-- execution:start:a100:32_ring_vs_tree_model.cpp -->
> **Compiled and executed — 2026-11-17-a100: Host program/model (exit-only).** Hardware: 8 A100 GPUs. Executed UTC: `2026-11-18T00:01:59.500551+00:00`.

<!-- execution:metadata Source SHA-256: 26cb373d652c1985ee1c42d2aa5edb5bee5a47ca5900dc938ca2e110e76967e5; recorded UTC (program start): 2026-11-18T00:01:59.285330+00:00 -->

[Exact tested source](../validation/2026-11-17-a100/32_ring_vs_tree_model/32_ring_vs_tree_model.cpp) · [Compiler log](../validation/2026-11-17-a100/32_ring_vs_tree_model/compile.log) · [Execution log](../validation/2026-11-17-a100/32_ring_vs_tree_model/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
g++ -std=c++17 -O2 32_ring_vs_tree_model.cpp -o ./example
./example
```

**Sample output**

```text
N          Ring rounds (Ch9)      Tree rounds (up+down)  Ratio
-          2*(N-1)                2*ceil(log2(N))        ring/tree
4          6                      4                      1.50      x
8          14                     6                      2.33      x
16         30                     8                      3.75      x
64         126                    12                     10.50     x
256        510                    16                     31.88     x
1024       2046                   20                     102.30    x
8192       16382                  26                     630.08    x
24576      49150                  30                     1638.33   x

Ring's round count grows linearly with N (doubling N roughly
doubles the rounds needed); the tree's grows logarithmically
(doubling N adds only one more level in each direction). This
is a structural count of communication ROUNDS, not a measured
time -- it makes concrete exactly the shape of the claim this
chapter cites from NVIDIA's own NCCL 2.4 announcement, without
fabricating a single timing number of its own.
```


<!-- execution:end:a100:32_ring_vs_tree_model.cpp -->


<!-- execution:start:h100:32_ring_vs_tree_model.cpp -->
> **Compiled and executed — 2026-11-17-h100: Host program/model (exit-only).** Hardware: 2 H100 GPUs. Execution record UTC (run-batch start; per-program time unrecorded): `2026-11-17T23:48:18.967387+00:00`.

<!-- execution:metadata Source SHA-256: 26cb373d652c1985ee1c42d2aa5edb5bee5a47ca5900dc938ca2e110e76967e5; recorded UTC (run-batch start; per-program time unrecorded): 2026-11-17T23:48:18.967387+00:00 -->

[Exact tested source](../validation/2026-11-17-h100/32_ring_vs_tree_model/32_ring_vs_tree_model.cpp) · [Compiler log](../validation/2026-11-17-h100/32_ring_vs_tree_model/compile.log) · [Execution log](../validation/2026-11-17-h100/32_ring_vs_tree_model/run.log). Compile exit: `0`; run exit: `0`.

Commands below use local source/binary names; raw logs retain host paths. Execute in a scratch directory containing the downloaded source.

```bash
g++ -std=c++17 -O2 32_ring_vs_tree_model.cpp -o ./example
./example
```

<details markdown="1"><summary>Captured current-source output</summary>

```text
N          Ring rounds (Ch9)      Tree rounds (up+down)  Ratio
-          2*(N-1)                2*ceil(log2(N))        ring/tree
4          6                      4                      1.50      x
8          14                     6                      2.33      x
16         30                     8                      3.75      x
64         126                    12                     10.50     x
256        510                    16                     31.88     x
1024       2046                   20                     102.30    x
8192       16382                  26                     630.08    x
24576      49150                  30                     1638.33   x

Ring's round count grows linearly with N (doubling N roughly
doubles the rounds needed); the tree's grows logarithmically
(doubling N adds only one more level in each direction). This
is a structural count of communication ROUNDS, not a measured
time -- it makes concrete exactly the shape of the claim this
chapter cites from NVIDIA's own NCCL 2.4 announcement, without
fabricating a single timing number of its own.
```

</details>
<!-- execution:end:h100:32_ring_vs_tree_model.cpp -->


The last row uses `N=24576` deliberately: NVIDIA's own announcement cites testing this exact feature at 24,576 GPUs on the Summit supercomputer, reporting "up to 180x" latency improvement at that scale. This model's own round-count ratio at that same `N` -- 1638.33x -- is not that measured number, and shouldn't be read as if it were; a round-count ratio and a measured latency-improvement ratio are different quantities, computed differently, and this book has no real timing data to produce the second one. What this table *does* honestly establish is the shape both numbers share: at small `N` the two algorithms are close enough that either one is fine, and the gap between them widens dramatically, in the same direction, as `N` grows into the thousands.

!!! warning "[COMMON TRAP] Treating this section's round-count ratio as a measured performance number"
    Section 11.3's table looks, superficially, like a benchmark result -- it has an N column and a ratio column, the same shape a real measured speedup table would have. It is not one. Every number in it is a pure count of communication *rounds*, computed from two closed-form formulas (`2*(N-1)` and `2*ceil(log2(N))`), with no notion of how long any single round actually takes on any real network. The real, measured "up to 180x" figure cited from NVIDIA's own Summit testing is a genuinely different quantity -- and this chapter is careful to cite that number as NVIDIA's own measurement, not to reproduce or approximate it with a model that was never designed to predict real timings in the first place.

## Chapter Summary

NCCL supplies the collective semantics built by hand in Chapters 8–10 and chooses implementations for the available topology and payload. Ring and tree schedules trade bandwidth and latency differently. The version-specific API probes, numerical examples and round-count model address these separate concerns.

## Self-Check Questions

1. What does `ncclGetVersion()` need that `ncclCommInitAll()` does, and why is a version query insufficient to establish successful communicator initialization?
2. Current File 30 tests successful communicator construction/destruction for all visible devices and for one device, then deliberately passes a `NULL` communicator-array pointer and a negative device count. Which calls should succeed, which should return `ncclInvalidArgument`, and why does the successful initialization not establish numerical collective correctness?
3. Match each of this book's six hand-built collectives (Chapters 8-10) to its one real NCCL call name from Section 11.2 -- for the one collective with no single matching call in NCCL 2.26.2, explain what it's built from instead, and what changed in NCCL 2.28.
4. Section 11.2's all-to-all code wraps `ncclSend()`/`ncclRecv()` in `ncclGroupStart()`/`ncclGroupEnd()`. Per NCCL's own documented ordering guarantee, what's different about two calls inside that group targeting the SAME peer versus two calls targeting DIFFERENT peers?
5. Quote the specific claim this chapter cites from NVIDIA's NCCL 2.4 announcement explaining why ring all-reduce alone isn't good enough at very large scale.
6. Using Section 11.3's model, compute the ring-to-tree round-count ratio for N=1024, and explain in your own words why that ratio keeps growing as N grows, using the formulas the code actually computes.
7. NCCL automatically switches between ring and tree rather than requiring the caller to pick one. Using this chapter's cited quote about when NCCL "switches back to rings," explain in what circumstance the bandwidth-optimal ring would still win over the lower-latency tree.
8. Chapter 9 never needed to reduce its round count below O(N) -- why not? What's different about the N=4 example Chapter 9 used throughout versus the N=24,576 Summit example this chapter cites?

## Where We Go Next

Part 2 is complete. Part 3 turns to parallelization strategies -- starting with Chapter 12, data parallelism, where every device holds a full replica of the same model and this book's own all-reduce (not hand-built this time, but the real `ncclAllReduce()` this chapter just linked against) is what keeps every replica's gradients in sync.

## Worked Solutions

**1.** `ncclGetVersion()` reads library version information without performing a collective. A valid communicator additionally requires the requested CUDA devices and successful initialization. The current probe checks this lifecycle and handles expected invalid arguments separately.

The [2026-11-17 H100 execution record](../validation-h100.md) covers two GPUs connected by local NV18. It distinguishes numerical checks, API probes and other executions; a successful exit alone does not establish GPU algorithm correctness.

**2.** With the required runtime and visible devices, both valid initialization cases should succeed and their communicators must be destroyed. The null output-array pointer and negative count are intentionally malformed and should return `ncclInvalidArgument`. These lifecycle checks do not transfer and independently verify collective payloads; numerical checks require initialized buffers, completed operations and reference comparisons.

The [2026-11-17 H100 execution record](../validation-h100.md) covers two GPUs connected by local NV18. It distinguishes numerical checks, API probes and other executions; a successful exit alone does not establish GPU algorithm correctness.

**3.** `ncclBroadcast()` <- Chapter 8, Section 8.1 (broadcast); `ncclReduce()` <- Chapter 8, Section 8.2 (reduce); `ncclAllReduce()` <- Chapter 9 (ring all-reduce); `ncclAllGather()` <- Chapter 10, Section 10.1 (all-gather); `ncclReduceScatter()` <- Chapter 10, Section 10.2 (reduce-scatter). All-to-all (Chapter 10, Section 10.3) has no single matching call in this installed NCCL version (2.26.2) -- it's built the same way this book built it, from repeated point-to-point transfers (`ncclSend()`/`ncclRecv()`) covering every ordered pair, fused into one operation with `ncclGroupStart()`/`ncclGroupEnd()`. NCCL 2.28 added a dedicated `ncclAlltoAll()`, so on a current NCCL it does have a one-call form.

**4.** Per NCCL's own documented ordering guarantee, point-to-point calls within a group targeting the SAME peer execute in order -- so two sends to the same peer inside one group would still need to be issued in the correct relative order, or that peer would receive them in the wrong sequence. Calls targeting DIFFERENT peers have no such ordering requirement and can progress concurrently -- which is exactly why Section 11.2's all-to-all loop, issuing one send and one receive per peer, can safely fuse all of them into a single group without worrying about cross-peer ordering.

**5.** "Latency scales linearly with the number of GPUs, preventing scaling above hundreds of GPUs" -- this is the specific limitation of the ring algorithm that motivated NCCL 2.4's double binary tree addition, cited directly from NVIDIA's own announcement.

**6.** From Section 11.3's printed table, N=1024 gives ring rounds = `2*(1024-1) = 2046` and tree rounds = `2*ceil(log2(1024)) = 2*10 = 20`, for a ratio of `2046/20 = 102.30x`. The ratio keeps growing because the numerator (ring rounds) grows linearly in N while the denominator (tree rounds) grows only logarithmically in N -- doubling N roughly doubles the ring's round count but adds only one more level (two more rounds) to the tree's.

**7.** The chapter's cited quote is that "NCCL automatically switches back to rings when that pattern results in greater bandwidth." Both algorithms are described as capable of achieving full bandwidth, but for smaller device counts or particular message-size/topology combinations, the ring's simpler, fully-pipelined pattern can still deliver more actual bandwidth than the tree's; NCCL's automatic selection exists precisely because neither algorithm dominates the other in every situation.

**8.** Chapter 9 never needed to go below O(N) round count because its running example used only N=4 devices, where a linearly-growing round count is a trivial 6 rounds -- nowhere near large enough for the ring's latency problem to matter. NVIDIA's own Summit example, at N=24,576 GPUs, is precisely the regime where a ring's O(N) round count (nearly 50,000 rounds, per Section 11.3's table) becomes genuinely prohibitive, and where a tree's O(log N) round count (only 30 rounds) becomes essential rather than a nice-to-have.

---

**Sources cited in this chapter:**

- [Massively Scale Your Deep Learning Training with NCCL 2.4 — NVIDIA Developer Blog](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4) -- the ring algorithm's linear latency scaling limitation, the double binary tree's logarithmic-latency design and "each rank is at most a node in one tree and a leaf in the other" structure, NCCL's automatic ring/tree selection ("automatically switches back to rings when that pattern results in greater bandwidth"), and the Summit supercomputer test at 24,576 GPUs with "up to 180x" latency improvement.
- [NCCL User Guide — Communicator API (comms.html)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html) -- the exact `ncclGetVersion()`, `ncclCommInitAll()`, and `ncclCommInitRank()` signatures and descriptions, including how rank count and device list determine a communicator clique's size.
- [NCCL User Guide — Point-to-Point Communication (p2p.html)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html) -- the documented `ncclSend()`/`ncclRecv()` pairing requirement, the `ncclGroupStart()`/`ncclGroupEnd()` fusion mechanism for building "more complex communication patterns," its own all-to-all code example, and the same-peer-vs-different-peer ordering guarantee.
- [NCCL User Guide — Collective Operations (collectives.html)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) -- the real `ncclBroadcast()`, `ncclReduce()`, `ncclAllReduce()`, `ncclAllGather()`, and `ncclReduceScatter()` definitions matched against this book's own Chapters 8-10, and (in current versions) the dedicated `ncclAlltoAll`.
- [NCCL Release Notes — NVIDIA](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-28-3.html) -- NCCL 2.28.3: "Added new host collective APIs: ncclAlltoAll, ncclScatter, and ncclGather," which dates the end of the no-dedicated-all-to-all situation this chapter describes for 2.26.2.
- [Recorded H100 stack and execution evidence](../validation-h100-rerun.md) -- NCCL 2.26.2 headers, the successful communicator-lifecycle probe and the deliberately invalid-argument probe. The earlier no-device diagnostics describe their historical environment, not a current communicator-initialization failure.
