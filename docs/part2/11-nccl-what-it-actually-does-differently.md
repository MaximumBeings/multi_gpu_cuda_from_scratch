# Chapter 11: NCCL: What It Actually Does Differently From What You Just Built

**What you will understand by the end of this chapter:**

- How a real `ncclComm_t` communicator relates to the `deviceCount` loops this book has used since Chapter 3, and the two genuinely different ways a real `ncclCommInitAll()` call can honestly fail -- a malformed argument, rejected up front, versus a well-formed request that fails inside CUDA.
- That five of this book's six hand-built collectives (Chapters 8-10) have a real, one-call NCCL equivalent in the NCCL 2.18.3 this book links against, while all-to-all there is built exactly the way Chapter 10 did: from grouped point-to-point transfers. (Newer NCCL, from 2.28, adds a dedicated `ncclAlltoAll()`.)
- The one genuine algorithmic gap between what this book built and what NCCL does for real: ring all-reduce's round count grows linearly with the number of devices, and at large enough scale that stops being good enough -- which is exactly why NCCL added a second algorithm.

**What you need to know first:**

- All six of Chapters 8-10's hand-built collectives: broadcast, reduce, all-reduce (ring), all-gather, reduce-scatter, all-to-all.
- Chapter 9's ring all-reduce round count, `2*(N-1)`.
- This book's closed-form cost/round-count modeling technique (Chapter 2, Chapter 5), used again in Section 11.3.

---

This chapter is different from every one before it in one specific way: for the first time, this book genuinely links against and genuinely calls a real, installed NCCL library -- not a hand-built substitute standing in for something unavailable. That changes what "honest failure" means here. Every call below is a real NCCL API call, checked against this book's own actual installed NCCL headers, and every locked output is what that real library genuinely reports on a machine with zero CUDA devices -- not a simulation of what it might say. Sections 11.1 and 11.2 check this book's own hand-built work against the real thing: a real communicator, and a real one-call equivalent for (almost) every collective Chapters 8-10 built by hand. Section 11.3 covers the one place this book's own ring all-reduce genuinely falls short at scale, and what NCCL does differently to fix it.

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
                                                          2.18.3 (ncclAlltoAll()
                                                          arrived in NCCL 2.28)
```

## 11.1 A Real Communicator: `ncclCommInitAll` and the World This Book Has Called `deviceCount`

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

`ncclGetVersion()` is the one call in this section that needs nothing from the machine at all -- it reads a version number baked into the library at build time, so it can genuinely succeed even here. `ncclCommInitAll()` is different: it has to actually attach communicators to real CUDA devices, and it can fail in two genuinely different ways depending on *which argument* is wrong. NCCL first runs argument checks: a `NULL` `comms` pointer or a negative `ndev` is rejected immediately as `ncclInvalidArgument` (code 4), before any device is touched. A request that passes those checks -- and on NCCL 2.18.3 that includes `ndev=0` as well as `ndev=1`, as long as `comms` points at real storage -- proceeds into NCCL's CUDA setup and fails there as `ncclUnhandledCudaError` (code 1). The code below makes all four cases explicit, and every explanation in this section was confirmed by running it with `NCCL_DEBUG=WARN`, which makes NCCL print its own reason for each failure on stderr.

```cpp
// Chapter 11: NCCL -- What It Actually Does Differently From What You
// Just Built
// 30_nccl_comm_init.cu
//
// Every chapter since Chapter 3 has looped over `deviceCount` and
// called CUDA Runtime API functions directly, device by device. Real
// NCCL adds one concept in front of all of that: a communicator, a
// single object representing the whole group of devices a collective
// call will run across. This section genuinely links against a real,
// installed NCCL library and genuinely calls its real API -- no
// hand-rolled substitute -- to see exactly how a communicator gets
// created, and what happens when it can't be. Genuinely compiled
// with a real nvcc, genuinely linked against a real libnccl, and
// genuinely run (each failure's cause confirmed with NCCL_DEBUG=WARN).
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // ncclGetVersion() reads the linked library's own version -- it
    // needs no device at all, so unlike almost every other call in
    // this book, it can genuinely succeed even here.
    int ncclRuntimeVersion = 0;
    ncclResult_t eVer = ncclGetVersion(&ncclRuntimeVersion);
    printf("ncclGetVersion(): %s (code %d), reports version %d "
           "(major.minor.patch = %d.%d.%d)\n",
           ncclGetErrorString(eVer), (int)eVer, ncclRuntimeVersion,
           ncclRuntimeVersion / 10000, (ncclRuntimeVersion / 100) % 100,
           ncclRuntimeVersion % 100);

    // ncclCommInitAll() fails in two genuinely different ways, and
    // which one you get depends on WHICH argument is bad -- not on
    // how many devices were asked for. Run with NCCL_DEBUG=WARN to see
    // NCCL's own explanation of each failure on stderr.
    ncclComm_t comms[1];

    // (a) This book's own real deviceCount (0), with a real comms array.
    // ndev=0 is NOT rejected by NCCL 2.18.3's argument checks: the call
    // proceeds until NCCL's first CUDA call, which finds no device.
    ncclResult_t eInitReal = ncclCommInitAll(comms, deviceCount, nullptr);
    printf("\n(a) ncclCommInitAll(comms, ndev=%d [this book's real deviceCount]): "
           "%s (code %d)\n", deviceCount, ncclGetErrorString(eInitReal), (int)eInitReal);

    // (b) Exactly ONE device, with a real comms array: also a well-formed
    // request, and it fails at exactly the same point, for the same
    // reason -- NCCL's first CUDA call finds no device.
    ncclResult_t eInitOne = ncclCommInitAll(comms, 1, nullptr);
    printf("(b) ncclCommInitAll(comms, ndev=1): %s (code %d)\n",
           ncclGetErrorString(eInitOne), (int)eInitOne);

    // (c) A NULL comms pointer: NCCL's argument check rejects this
    // before touching any device ("comms argument is NULL").
    ncclResult_t eInitNull = ncclCommInitAll(nullptr, 1, nullptr);
    printf("(c) ncclCommInitAll(comms=nullptr, ndev=1): %s (code %d)\n",
           ncclGetErrorString(eInitNull), (int)eInitNull);

    // (d) A negative device count: also rejected up front
    // ("Invalid device count requested").
    ncclResult_t eInitNeg = ncclCommInitAll(comms, -1, nullptr);
    printf("(d) ncclCommInitAll(comms, ndev=-1): %s (code %d)\n",
           ncclGetErrorString(eInitNeg), (int)eInitNeg);

    printf("\nTwo genuinely different failure modes: a malformed ARGUMENT\n"
           "(a NULL comms pointer, or a negative device count) is rejected\n"
           "immediately as ncclInvalidArgument (code 4); a well-formed request\n"
           "-- including ndev=0 -- gets past argument checking and fails as\n"
           "ncclUnhandledCudaError (code 1) once NCCL's first CUDA call finds\n"
           "no device. No communicator was created by any of these calls;\n"
           "Section 11.2 shows what every collective reports when handed a\n"
           "comm that is still NULL.\n");

    return 0;
}
```

Genuinely compiled with a real `nvcc`, genuinely linked against a real, installed `libnccl` (version 2.18.3), and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).
ncclGetVersion(): no error (code 0), reports version 21803 (major.minor.patch = 2.18.3)

(a) ncclCommInitAll(comms, ndev=0 [this book's real deviceCount]): unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)
(b) ncclCommInitAll(comms, ndev=1): unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)
(c) ncclCommInitAll(comms=nullptr, ndev=1): invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)
(d) ncclCommInitAll(comms, ndev=-1): invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)

Two genuinely different failure modes: a malformed ARGUMENT
(a NULL comms pointer, or a negative device count) is rejected
immediately as ncclInvalidArgument (code 4); a well-formed request
-- including ndev=0 -- gets past argument checking and fails as
ncclUnhandledCudaError (code 1) once NCCL's first CUDA call finds
no device. No communicator was created by any of these calls;
Section 11.2 shows what every collective reports when handed a
comm that is still NULL.
```

This is the first chapter where a call in this book has genuinely succeeded without a device at all -- `ncclGetVersion()`'s "no error" is real, not a special case carved out for this environment. Everything after it fails with one of two genuinely different, genuinely informative codes, and `NCCL_DEBUG=WARN` shows exactly why. Cases (a) and (b) each print `NCCL WARN Cuda failure 'no CUDA-capable device is detected'` -- `ndev=0` is *not* rejected as malformed; it fails in the same place, for the same reason, as `ndev=1`. Case (c) prints `NCCL WARN CommInitAll : comms argument is NULL`, and case (d) prints `NCCL WARN Invalid device count requested : -1`. An earlier draft of this chapter passed `nullptr` for `comms` together with `ndev=0` and attributed the resulting code 4 to `ndev=0`; the warning text shows the `NULL` pointer was the actual cause, which is why the code now separates the two.

!!! warning "[COMMON TRAP] Assuming NCCL's error codes distinguish 'no devices' from 'not enough devices' the way this book's own code always has"
    Every prior chapter's honest failures have used a single, uniform signal -- `cudaErrorNoDevice`, code 100 -- regardless of how the loop was structured. NCCL's own error taxonomy is richer: `ncclInvalidArgument` (code 4) means an argument itself was malformed (a `NULL` pointer, a negative count) before any hardware was even considered, while `ncclUnhandledCudaError` (code 1) means the request passed argument checking but a lower-level CUDA call inside NCCL's own implementation failed. Only code 1 traces back to this machine having no GPU; code 4 would come back identically on a fully-populated 8-GPU server, because it is a bug in the *calling code*. And the numeric code alone doesn't say *which* argument was bad -- that is what `NCCL_DEBUG=WARN` is for, and reading its output (rather than guessing from the code) is the only reliable way to attribute an NCCL failure.

## 11.2 Six Collectives, Six Real Calls (Mostly)

### Intuition

Chapters 8 through 10 built six collectives by hand, one loop of `cudaMemcpyPeer()` or `cudaMemcpy()` calls at a time. Real NCCL packages five of those six as a single function call each -- the entire loop this book wrote out explicitly, hidden behind one name. The sixth, all-to-all, doesn't get that treatment in the NCCL 2.18.3 this book links against: there it's built from the same kind of point-to-point calls this book used, just NCCL's own `ncclSend()`/`ncclRecv()` instead of `cudaMemcpyPeer()`. That is a version-specific fact, not a permanent one: NCCL 2.28.3's release notes record "Added new host collective APIs: ncclAlltoAll, ncclScatter, and ncclGather," and the current collective-operations documentation includes `ncclAlltoAll`.

```text
Chapter 8-10's six collectives, and their real NCCL 2.18.3 call:

  Ch8 8.1   --> ncclBroadcast()          Ch10 10.1 --> ncclAllGather()
  Ch8 8.2   --> ncclReduce()             Ch10 10.2 --> ncclReduceScatter()
  Ch9       --> ncclAllReduce()          Ch10 10.3 --> ncclSend() + ncclRecv()
                                                       (grouped; no dedicated
                                                        call until NCCL 2.28's
                                                        ncclAlltoAll())
```

### Background

No communicator can be created on this machine (Section 11.1), so every call below is handed a `comm` handle that is literally `nullptr`. NCCL's own argument check rejects that `NULL` handle before any of these calls could do anything with real device memory, so every one of the five collectives reports `ncclInvalidArgument` (code 4) -- and `NCCL_DEBUG=WARN` names the cause precisely, once per call: `NCCL WARN Broadcast : comm argument is NULL`, `Reduce : comm argument is NULL`, and likewise for `AllReduce`, `AllGather`, and `ReduceScatter`. This is the same `NULL`-pointer argument check as Section 11.1's case (c), not anything to do with how many devices were requested -- which also means a real program that ignored a failed `ncclCommInitAll()` and carried on would not necessarily see this particular error, since NCCL makes no promise that a failed init leaves the handle `NULL`.

```cpp
// Chapter 11: NCCL -- What It Actually Does Differently From What You
// Just Built
// 31_nccl_collectives_mapped.cu
//
// Every collective this book built by hand in Chapters 8-10 has a
// literal, real NCCL API call behind it in the installed NCCL 2.18.3
// -- except all-to-all (a dedicated ncclAlltoAll() only arrived in
// NCCL 2.28). This section genuinely calls each real function with a
// comm handle that is still NULL -- no communicator can be created on
// this machine (Section 11.1) -- and honestly reports whatever
// ncclResult_t comes back. Genuinely compiled with a real nvcc,
// genuinely linked against a real libnccl, and genuinely run (the
// failure's cause confirmed with NCCL_DEBUG=WARN).
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // No communicator exists on this machine (Section 11.1), so the
    // handle passed to every call below is literally nullptr. NCCL's
    // argument check rejects it before doing anything else -- with
    // NCCL_DEBUG=WARN it prints "<Collective> : comm argument is NULL"
    // and returns ncclInvalidArgument (code 4). This is a NULL-handle
    // check, not a check on how many devices were asked for.
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;
    float* buf = nullptr;
    const size_t COUNT = 4;

    printf("\n-- Five of this book's six hand-built collectives, as one\n"
           "   real NCCL call each: --\n\n");

    ncclResult_t eBcast = ncclBroadcast(buf, buf, COUNT, ncclFloat, 0, comm, stream);
    printf("ncclBroadcast()      [Chapter 8, Section 8.1's broadcast]:      %s (code %d)\n",
           ncclGetErrorString(eBcast), (int)eBcast);

    ncclResult_t eReduce = ncclReduce(buf, buf, COUNT, ncclFloat, ncclSum, 0, comm, stream);
    printf("ncclReduce()         [Chapter 8, Section 8.2's reduce]:         %s (code %d)\n",
           ncclGetErrorString(eReduce), (int)eReduce);

    ncclResult_t eAllReduce = ncclAllReduce(buf, buf, COUNT, ncclFloat, ncclSum, comm, stream);
    printf("ncclAllReduce()      [Chapter 9's ring all-reduce]:             %s (code %d)\n",
           ncclGetErrorString(eAllReduce), (int)eAllReduce);

    ncclResult_t eAllGather = ncclAllGather(buf, buf, COUNT, ncclFloat, comm, stream);
    printf("ncclAllGather()      [Chapter 10, Section 10.1's all-gather]:   %s (code %d)\n",
           ncclGetErrorString(eAllGather), (int)eAllGather);

    ncclResult_t eReduceScatter = ncclReduceScatter(buf, buf, COUNT, ncclFloat, ncclSum, comm, stream);
    printf("ncclReduceScatter()  [Chapter 10, Section 10.2's reduce-scatter]: %s (code %d)\n",
           ncclGetErrorString(eReduceScatter), (int)eReduceScatter);

    // All-to-all has no single dedicated call in this installed NCCL
    // version (2.18.3; ncclAlltoAll() was added in 2.28) -- it is built the same way this book built it in
    // Chapter 10, Section 10.3: one point-to-point transfer per
    // ordered pair, just using ncclSend()/ncclRecv() instead of
    // cudaMemcpyPeer(), fused together with ncclGroupStart()/
    // ncclGroupEnd() into one coordinated operation.
    printf("\n-- All-to-all [Chapter 10, Section 10.3]: no dedicated call in 2.18.3 --\n"
           "   built from ncclSend()/ncclRecv(), exactly like this book's\n"
           "   own direct implementation: --\n\n");

    ncclResult_t eGroupStart = ncclGroupStart();
    printf("ncclGroupStart(): %s (code %d)\n", ncclGetErrorString(eGroupStart), (int)eGroupStart);

    int p2pAttempted = 0;
    for (int peer = 0; peer < deviceCount; ++peer) {
        ncclResult_t eSend = ncclSend(buf, 1, ncclFloat, peer, comm, stream);
        ncclResult_t eRecv = ncclRecv(buf, 1, ncclFloat, peer, comm, stream);
        ++p2pAttempted;
        printf("ncclSend()/ncclRecv(peer=%d): %s / %s\n", peer,
               ncclGetErrorString(eSend), ncclGetErrorString(eRecv));
    }

    ncclResult_t eGroupEnd = ncclGroupEnd();
    printf("ncclGroupEnd(): %s (code %d)\n", ncclGetErrorString(eGroupEnd), (int)eGroupEnd);

    printf("\n%d device(s) worth of ncclSend()/ncclRecv() pairs attempted\n"
           "inside the group, so ncclSend()/ncclRecv() were never actually\n"
           "called. All five collective calls above report the same\n"
           "failure: ncclInvalidArgument, because the comm handle they were\n"
           "given is NULL (NCCL_DEBUG=WARN: \"comm argument is NULL\").\n"
           "ncclGroupStart()/ncclGroupEnd() succeed: they touch no comm.\n"
           "Section 11.3 turns to the one real difference that would\n"
           "matter if a comm HAD been created: how NCCL chooses an\n"
           "algorithm for ncclAllReduce().\n",
           p2pAttempted);

    return 0;
}
```

Genuinely compiled with a real `nvcc`, genuinely linked against a real `libnccl`, and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

-- Five of this book's six hand-built collectives, as one
   real NCCL call each: --

ncclBroadcast()      [Chapter 8, Section 8.1's broadcast]:      invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)
ncclReduce()         [Chapter 8, Section 8.2's reduce]:         invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)
ncclAllReduce()      [Chapter 9's ring all-reduce]:             invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)
ncclAllGather()      [Chapter 10, Section 10.1's all-gather]:   invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)
ncclReduceScatter()  [Chapter 10, Section 10.2's reduce-scatter]: invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)

-- All-to-all [Chapter 10, Section 10.3]: no dedicated call in 2.18.3 --
   built from ncclSend()/ncclRecv(), exactly like this book's
   own direct implementation: --

ncclGroupStart(): no error (code 0)
ncclGroupEnd(): no error (code 0)

0 device(s) worth of ncclSend()/ncclRecv() pairs attempted
inside the group, so ncclSend()/ncclRecv() were never actually
called. All five collective calls above report the same
failure: ncclInvalidArgument, because the comm handle they were
given is NULL (NCCL_DEBUG=WARN: "comm argument is NULL").
ncclGroupStart()/ncclGroupEnd() succeed: they touch no comm.
Section 11.3 turns to the one real difference that would
matter if a comm HAD been created: how NCCL chooses an
algorithm for ncclAllReduce().
```

`ncclGroupStart()` and `ncclGroupEnd()` genuinely succeed -- like `ncclGetVersion()`, they only manage process-local grouping state, not device state, and take no communicator, so they need no real hardware at all. With `deviceCount = 0`, the loop between them never runs, so `ncclSend()`/`ncclRecv()` are never actually called -- the same honest degradation this book has shown for every device-count-driven loop since Chapter 4. So exactly five real NCCL calls fail here, not six, and all five fail for the reason `NCCL_DEBUG=WARN` reports: a `NULL` comm argument.

!!! warning "[COMMON TRAP] Assuming the missing all-to-all call (in NCCL 2.18.3) means all-to-all is somehow a second-class operation in NCCL"
    It would be easy to read "no dedicated call" as "NCCL doesn't really support this." Two things say otherwise. First, the gap is version-specific: NCCL 2.28 added a dedicated `ncclAlltoAll()`, so on a current NCCL the one-call form exists. Second, even in the 2.18.3 this book links against, NCCL's own documentation says otherwise: point-to-point `ncclSend()`/`ncclRecv()` calls "can be fused together with `ncclGroupStart()` and `ncclGroupEnd()` to form more complex communication patterns," and the documentation's own example for exactly this fusion is an all-to-all loop, line for line the same shape as this section's code. All-to-all is fully supported in 2.18.3 -- it's just supported there as a composition of two more primitive calls, the same way this book's own Chapter 10 built it as a composition of `cudaMemcpyPeer()` calls, rather than as one specially-named function.

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

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
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

The last row uses `N=24576` deliberately: NVIDIA's own announcement cites testing this exact feature at 24,576 GPUs on the Summit supercomputer, reporting "up to 180x" latency improvement at that scale. This model's own round-count ratio at that same `N` -- 1638.33x -- is not that measured number, and shouldn't be read as if it were; a round-count ratio and a measured latency-improvement ratio are different quantities, computed differently, and this book has no real timing data to produce the second one. What this table *does* honestly establish is the shape both numbers share: at small `N` the two algorithms are close enough that either one is fine, and the gap between them widens dramatically, in the same direction, as `N` grows into the thousands.

!!! warning "[COMMON TRAP] Treating this section's round-count ratio as a measured performance number"
    Section 11.3's table looks, superficially, like a benchmark result -- it has an N column and a ratio column, the same shape a real measured speedup table would have. It is not one. Every number in it is a pure count of communication *rounds*, computed from two closed-form formulas (`2*(N-1)` and `2*ceil(log2(N))`), with no notion of how long any single round actually takes on any real network. The real, measured "up to 180x" figure cited from NVIDIA's own Summit testing is a genuinely different quantity -- and this chapter is careful to cite that number as NVIDIA's own measurement, not to reproduce or approximate it with a model that was never designed to predict real timings in the first place.

## Chapter Summary

This chapter checked this book's own hand-built work in Chapters 8 through 10 against a real, genuinely-linked NCCL library, rather than building a new hand-rolled substitute. A real communicator (Section 11.1) plays exactly the role this book's own `deviceCount` loops have played informally since Chapter 3, and genuinely fails in two different, genuinely informative ways: a malformed argument (a `NULL` `comms` pointer, a negative device count) is rejected up front as `ncclInvalidArgument`, while any well-formed request -- `ndev=0` included -- fails inside CUDA as `ncclUnhandledCudaError`, with `NCCL_DEBUG=WARN` naming the cause in each case. Five of this book's six hand-built collectives turn out to have a literal, one-call NCCL equivalent (Section 11.2), and each one, handed a `NULL` comm, reports "comm argument is NULL"; the sixth, all-to-all, has no dedicated call in the NCCL 2.18.3 this book links against (NCCL 2.28 later added `ncclAlltoAll()`), and there is built the same way Chapter 10 built it -- from repeated point-to-point transfers, just using `ncclSend()`/`ncclRecv()` instead of `cudaMemcpyPeer()`. The one place this book's own algorithm genuinely falls short of what NCCL does for real is scale: Chapter 9's ring all-reduce needs a round count that grows linearly with the number of devices, which is fine at the `N=4` this book has used throughout but becomes a genuine bottleneck at the thousands-of-devices scale real training clusters reach -- which is exactly why NCCL, since version 2.4, automatically chooses between the ring this book built and a second, tree-based algorithm with logarithmic round growth (Section 11.3). With this, **Part 2 is complete**: Chapters 8 through 11 took this book from broadcast and reduce, by hand, all the way to checking that hand-built work against the real library every real multi-GPU training job actually uses. Part 3 turns to what all of this communication infrastructure is actually *for*: the parallelization strategies -- data, model, tensor, and pipeline parallelism -- that decide who computes what, and when they need to talk to each other at all.

## Self-Check Questions

1. What does `ncclGetVersion()` need that `ncclCommInitAll()` does, and why can the first genuinely succeed on this machine while the second cannot?
2. Section 11.1 shows two different NCCL error codes across four requests. Explain why `ndev=0` and `ndev=1` (each with a real `comms` array) fail with the *same* code, while a `NULL` `comms` pointer or `ndev=-1` fail with a different one -- and how `NCCL_DEBUG=WARN` lets you tell which argument was at fault.
3. Match each of this book's six hand-built collectives (Chapters 8-10) to its one real NCCL call name from Section 11.2 -- for the one collective with no single matching call in NCCL 2.18.3, explain what it's built from instead, and what changed in NCCL 2.28.
4. Section 11.2's all-to-all code wraps `ncclSend()`/`ncclRecv()` in `ncclGroupStart()`/`ncclGroupEnd()`. Per NCCL's own documented ordering guarantee, what's different about two calls inside that group targeting the SAME peer versus two calls targeting DIFFERENT peers?
5. Quote the specific claim this chapter cites from NVIDIA's NCCL 2.4 announcement explaining why ring all-reduce alone isn't good enough at very large scale.
6. Using Section 11.3's model, compute the ring-to-tree round-count ratio for N=1024, and explain in your own words why that ratio keeps growing as N grows, using the formulas the code actually computes.
7. NCCL automatically switches between ring and tree rather than requiring the caller to pick one. Using this chapter's cited quote about when NCCL "switches back to rings," explain in what circumstance the bandwidth-optimal ring would still win over the lower-latency tree.
8. Chapter 9 never needed to reduce its round count below O(N) -- why not? What's different about the N=4 example Chapter 9 used throughout versus the N=24,576 Summit example this chapter cites?

## Where We Go Next

Part 2 is complete. Part 3 turns to parallelization strategies -- starting with Chapter 12, data parallelism, where every device holds a full replica of the same model and this book's own all-reduce (not hand-built this time, but the real `ncclAllReduce()` this chapter just linked against) is what keeps every replica's gradients in sync.

## Worked Solutions

**1.** `ncclGetVersion()` only reads a version number baked into the linked library at build time -- it needs no CUDA device, no context, nothing about the actual machine's hardware. `ncclCommInitAll()` has to actually attach each requested communicator to a real CUDA device, so it can only succeed once a real device genuinely exists. That's exactly why the first call in Section 11.1's code reports "no error" while every later call reports a real failure.

**2.** NCCL's argument check looks at whether the arguments are *usable*, not at whether the device count is zero. With a real `comms` array, both `ndev=0` and `ndev=1` pass that check, proceed into NCCL's CUDA setup, and fail there as `ncclUnhandledCudaError` (code 1) -- `NCCL_DEBUG=WARN` prints `Cuda failure 'no CUDA-capable device is detected'` for both. A `NULL` `comms` pointer (`comms argument is NULL`) or a negative count (`Invalid device count requested : -1`) is rejected immediately as `ncclInvalidArgument` (code 4), before any device is considered. One failure is about the *calling code* passing an unusable argument -- it would happen identically on a machine full of GPUs; the other is about the *hardware* being absent. The numeric code tells you which category; only the `NCCL_DEBUG=WARN` text tells you which argument or which CUDA call.

**3.** `ncclBroadcast()` <- Chapter 8, Section 8.1 (broadcast); `ncclReduce()` <- Chapter 8, Section 8.2 (reduce); `ncclAllReduce()` <- Chapter 9 (ring all-reduce); `ncclAllGather()` <- Chapter 10, Section 10.1 (all-gather); `ncclReduceScatter()` <- Chapter 10, Section 10.2 (reduce-scatter). All-to-all (Chapter 10, Section 10.3) has no single matching call in this installed NCCL version (2.18.3) -- it's built the same way this book built it, from repeated point-to-point transfers (`ncclSend()`/`ncclRecv()`) covering every ordered pair, fused into one operation with `ncclGroupStart()`/`ncclGroupEnd()`. NCCL 2.28 added a dedicated `ncclAlltoAll()`, so on a current NCCL it does have a one-call form.

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
- [NCCL Release Notes — NVIDIA](https://docs.nvidia.com/deeplearning/nccl/release-notes/index.html) -- NCCL 2.28.3: "Added new host collective APIs: ncclAlltoAll, ncclScatter, and ncclGather," which dates the end of the no-dedicated-all-to-all situation this chapter describes for 2.18.3.
- Real `NCCL_DEBUG=WARN` output from the installed NCCL 2.18.3 (`misc/argcheck.cc`: "comms argument is NULL" / "comm argument is NULL"; `init.cc`: "Invalid device count requested" and "Cuda failure 'no CUDA-capable device is detected'") -- the evidence for every failure attribution in Sections 11.1 and 11.2.
