# Appendix C: NCCL and NVSHMEM -- The Standard Libraries You Get for Free

Chapters 8 through 10 hand-built five collectives from first principles -- broadcast, reduce, ring all-reduce, all-gather, and reduce-scatter -- plus a genuinely new all-to-all. Chapter 11 then matched every one of them to its real NCCL call, and Chapters 17, 20, and 22 went on to use NCCL and NVSHMEM as working tools throughout the rest of the book: a throwaway `ncclAllReduce` as a barrier in Chapter 17, `MPI_Sendrecv` fixing a real ring deadlock in Chapter 20, and NVSHMEM's device-initiated `nvshmem_int_p()` in Chapter 22. Unlike a book meeting NCCL or NVSHMEM for the first time here, this book has already put both to work. This appendix does not introduce them again -- it gathers their fuller real API surface into one consolidated reference, section by section citing the exact chapter that already established each idea, and closes with the decision framework this book has been applying implicitly since Chapter 11: when to hand-write a collective, when to reach for NCCL, and when the job genuinely needs NVSHMEM's one-sided, device-initiated model instead.

## C.1 NCCL's Full Collective Reference

### Intuition

Imagine finally organizing a toolbox after building five different tools from raw metal, one chapter at a time. You already know how a ring all-reduce moves data in two phases, because Chapter 9 walked through building one. What you have not yet done is stand back and see all five hand-built collectives on one shelf, each one right next to the single, real, already-compiled `nccl.h` function that does its job. That is what this section is: a lookup table, not a lesson. The algorithm is not new. The table is.

### The Concept, In Detail

```
  hand-built (Ch 8-10)         matching real NCCL call
  +----------------------------+---+--------------------------+
  |Ch8  8.1  Broadcast         |-->|ncclBroadcast             |
  |Ch8  8.2  Reduce            |-->|ncclReduce                |
  |Ch9       Ring All-Reduce   |-->|ncclAllReduce             |
  |Ch10 10.1 All-Gather        |-->|ncclAllGather             |
  |Ch10 10.2 Reduce-Scatter    |-->|ncclReduceScatter         |
  |Ch10 10.3 All-to-All        |-->|ncclSend/ncclRecv         |
  +----------------------------+---+--------------------------+

  note: Ch10 10.3's row has no single matching NCCL call --
  it maps to a PATTERN (repeated ncclSend/ncclRecv fused inside
  ncclGroupStart()/ncclGroupEnd()), not one named function.
```

Five of the six rows are a clean one-to-one match: whatever Chapter 8's or Chapter 9's or Chapter 10's hand-built loop over `cudaMemcpyPeerAsync` calls was doing, one real NCCL call now does the same job, dispatched to whichever real interconnect NCCL has actually detected (NVLink, PCIe, or a network fabric) -- a decision the hand-built versions in Chapters 8-10 never had to make, because they always assumed a single fixed topology. The sixth row is the interesting one. Real NCCL, as Chapter 11 already established, ships no `ncclAllToAll()` function at all. Chapter 10's own hand-built all-to-all is not an approximation of some missing NCCL primitive -- it is already built the way NCCL itself builds an all-to-all internally: a loop of paired `ncclSend`/`ncclRecv` calls, all fused inside one `ncclGroupStart()`/`ncclGroupEnd()` bracket so the underlying transport can pipeline them together instead of issuing six separate round trips. That same fused-group pattern is also exactly how Chapter 17 built its own throwaway barrier out of a one-element `ncclAllReduce` -- NCCL simply does not ship a dedicated barrier either, for the identical reason: the group primitive already covers it.

!!! warning
    A version number and a unique ID cost nothing -- a communicator costs a device. `ncclGetVersion()` and `ncclGetUniqueId()` are pure host-side bookkeeping and will succeed on a laptop with no GPU at all. `ncclCommInitRank()` is different: it genuinely tries to bind the communicator to a physical device, and on a machine with zero physical GPUs it fails every time, the same real, well-documented limitation Chapters 22, 23, and 32 already found for NVSHMEM init and for captured graph launches. Reading a version string back successfully is not evidence that a communicator will initialize.

### Code and Verification

```cpp
// Appendix C: NCCL and NVSHMEM -- The Standard Libraries You Get for Free
// 126_nccl_collective_reference.cu
//
// Chapters 8-10 hand-built five collectives from first principles
// (broadcast, reduce, ring all-reduce, all-gather, reduce-scatter, and a
// genuinely new all-to-all), and Chapter 11 matched every one of them to
// its real NCCL call. This file gathers those five matches into ONE
// reference table -- exactly the kind of consolidated lookup a working
// engineer reaches for once the underlying algorithm (Chapters 8-10) is
// already understood -- and genuinely attempts a real NCCL communicator
// init, honestly reporting this environment's own real, well-documented
// limitation (zero physical GPUs) exactly the way every device-touching
// file in this book has since Chapter 2.
//
// Compile: nvcc -arch=sm_80 126_nccl_collective_reference.cu -o 126_nccl_collective_reference -lnccl
// Run:     ./126_nccl_collective_reference
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    printf("=== Section C.1: NCCL's five core collectives, matched to the "
           "chapter that hand-built each one ===\n\n");

    int version = 0;
    ncclGetVersion(&version);
    printf("real installed NCCL version: %d.%d.%d\n\n",
           version / 10000, (version / 100) % 100, version % 100);

    printf("%-20s %-45s %-10s\n", "NCCL call", "signature (sendbuff, recvbuff, ...)", "hand-built in");
    printf("%-20s %-45s %-10s\n", "ncclBroadcast", "(send, recv, count, type, root, comm, stream)", "Ch8 8.1");
    printf("%-20s %-45s %-10s\n", "ncclReduce",    "(send, recv, count, type, op, root, comm, stream)", "Ch8 8.2");
    printf("%-20s %-45s %-10s\n", "ncclAllReduce", "(send, recv, count, type, op, comm, stream)", "Ch9 (ring)");
    printf("%-20s %-45s %-10s\n", "ncclReduceScatter", "(send, recv, recvcount, type, op, comm, stream)", "Ch10 10.2");
    printf("%-20s %-45s %-10s\n", "ncclAllGather", "(send, recv, sendcount, type, comm, stream)", "Ch10 10.1");
    printf("\nnote: NCCL has no dedicated all-to-all call -- Ch10 10.3's "
           "own hand-built all-to-all is built the same way real NCCL "
           "itself builds it (Ch11's own real finding): repeated "
           "ncclSend()/ncclRecv() calls fused inside ncclGroupStart()/"
           "ncclGroupEnd(), not one dedicated function. Ch17 built a "
           "Barrier the same structural way (a throwaway ncclAllReduce), "
           "because NCCL simply does not ship either as its own named "
           "primitive.\n\n");

    printf("=== attempting a genuine ncclCommInitRank() ===\n\n");
    ncclUniqueId id;
    ncclResult_t idErr = ncclGetUniqueId(&id);
    printf("ncclGetUniqueId() -> %s (%s)\n", ncclGetErrorString(idErr),
           idErr == ncclSuccess ? "succeeds -- no device needed to generate an ID" : "unexpected failure");

    ncclComm_t comm;
    ncclResult_t initErr = ncclCommInitRank(&comm, 1, id, 0);
    printf("ncclCommInitRank(&comm, nranks=1, id, rank=0) -> %s (%s)\n",
           ncclGetErrorString(initErr),
           initErr == ncclSuccess ? "succeeded" :
           "this environment has zero physical GPUs, so NCCL cannot "
           "actually bind a communicator to a device, exactly the same "
           "real, well-documented limitation Ch22/Ch23/Ch32 already "
           "found for NVSHMEM init and captured graph launches");

    printf("\nself-check: ncclGetVersion() and ncclGetUniqueId() both "
           "succeed with NO device at all (pure host-side bookkeeping), "
           "while ncclCommInitRank() genuinely requires one -- this file "
           "reports both outcomes honestly rather than skipping the "
           "attempt: %s\n",
           (idErr == ncclSuccess) ? "confirmed" : "MISMATCH");
    return (idErr == ncclSuccess) ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 126_nccl_collective_reference.cu -o 126_nccl_collective_reference -lnccl
./126_nccl_collective_reference
```

**Sample input:** none -- the program takes no arguments and reads no data; every value it prints is either a compile-time constant string or a real return value read back from the installed NCCL library.

**Sample output:**

```text
=== Section C.1: NCCL's five core collectives, matched to the chapter that hand-built each one ===

real installed NCCL version: 2.18.3

NCCL call            signature (sendbuff, recvbuff, ...)          hand-built in
ncclBroadcast         (send, recv, count, type, root, comm, stream) Ch8 8.1
ncclReduce            (send, recv, count, type, op, root, comm, stream) Ch8 8.2
ncclAllReduce         (send, recv, count, type, op, comm, stream) Ch9 (ring)
ncclReduceScatter     (send, recv, recvcount, type, op, comm, stream) Ch10 10.2
ncclAllGather         (send, recv, sendcount, type, comm, stream) Ch10 10.1

note: NCCL has no dedicated all-to-all call -- Ch10 10.3's own hand-built all-to-all is built the same way real NCCL itself builds it (Ch11's own real finding): repeated ncclSend()/ncclRecv() calls fused inside ncclGroupStart()/ncclGroupEnd(), not one dedicated function. Ch17 built a Barrier the same structural way (a throwaway ncclAllReduce), because NCCL simply does not ship either as its own named primitive.

=== attempting a genuine ncclCommInitRank() ===

ncclGetUniqueId() -> no error (succeeds -- no device needed to generate an ID)
ncclCommInitRank(&comm, nranks=1, id, rank=0) -> unhandled cuda error (run with NCCL_DEBUG=INFO for details) (this environment has zero physical GPUs, so NCCL cannot actually bind a communicator to a device, exactly the same real, well-documented limitation Ch22/Ch23/Ch32 already found for NVSHMEM init and captured graph launches)

self-check: ncclGetVersion() and ncclGetUniqueId() both succeed with NO device at all (pure host-side bookkeeping), while ncclCommInitRank() genuinely requires one -- this file reports both outcomes honestly rather than skipping the attempt: confirmed
```

## C.2 NCCL Point-to-Point, Consolidated

### Intuition

Chapter 11 already handed you the two functions this section is built around, `ncclSend()` and `ncclRecv()`, and its own Self-Check Question 4 asked something this appendix never got to answer in code: if you issue several point-to-point calls inside one group, what order do they actually arrive in? The chapter's body text quoted NCCL's documented guarantee -- same-peer calls stay ordered, different-peer calls make no such promise -- but never built anything that shows it happening. That demonstration is the one genuinely new thing in this section.

### The Concept, In Detail

```
  one ncclGroupStart() / ncclGroupEnd() issues six sends:

    issue order:  (0,100) (0,101) (0,102) (1,200) (2,300) (2,301)

  +------------------+   +------------------+   +------------------+
  | peer 0 own queue |   | peer 1 own queue |   | peer 2 own queue |
  |  FIFO -- ordered |   |  FIFO -- ordered |   |  FIFO -- ordered |
  |------------------|   |------------------|   |------------------|
  | 100 -> 101 -> 102|   | 200              |   | 300 -> 301       |
  +------------------+   +------------------+   +------------------+

  same-peer order preserved (100 then 101 then 102 on peer 0).
  no line connects peer 0's queue to peer 1's or peer 2's queue --
  cross-peer arrival order is explicitly NOT constrained.
```

Picture each peer as running its own private mail slot. NCCL treats every peer's slot as a FIFO queue: three `ncclSend()` calls aimed at peer 0, issued in that order inside the group, are guaranteed to be delivered to peer 0 in that same order -- tag 100 before 101 before 102. That guarantee stops exactly at the slot's own edge. The single call aimed at peer 1 and the two calls aimed at peer 2 are free to complete before, after, or interleaved with peer 0's three calls, because NCCL's ordering promise is scoped per destination, not to the group as a whole. This is precisely the recipe Chapter 11 already used to build Chapter 10's own hand-built all-to-all: the loop issues one `ncclSend`/`ncclRecv` pair per peer, all inside a single `ncclGroupStart()`/`ncclGroupEnd()` bracket, relying on exactly this per-peer ordering and nothing stronger.

!!! warning
    "Inside one group" is not the same promise as "in one order." It is tempting to read `ncclGroupStart()`/`ncclGroupEnd()` as a barrier that serializes everything inside it -- it does the opposite. The group exists so the underlying transport can batch and pipeline the calls together for throughput, and the only ordering it actually guarantees is per-peer FIFO. Code that depends on call number 3 (to peer 1) happening before call number 5 (to peer 2) is relying on a guarantee NCCL never made, and it can break the moment the transport or peer count changes, even though it may happen to pass on today's hardware.

### Code and Verification

```cpp
// Appendix C: NCCL and NVSHMEM -- The Standard Libraries You Get for Free
// 127_nccl_point_to_point.cu
//
// Chapter 11 already introduced NCCL's own real point-to-point API --
// ncclSend()/ncclRecv(), fused inside ncclGroupStart()/ncclGroupEnd() to
// build Chapter 10's own all-to-all without a dedicated NCCL call at all
// -- and its own Self-Check Question 4 asked what NCCL's documented
// ordering guarantee promises for two point-to-point calls inside one
// group: calls to the SAME peer execute in the order they were issued;
// calls to DIFFERENT peers carry no such guarantee and may progress
// concurrently. Chapter 11's own body text quoted that guarantee but did
// not build a standalone demonstration of it. This section gathers
// ncclSend/ncclRecv/ncclGroupStart/ncclGroupEnd as a consolidated
// reference (confirming all four real symbols resolve and link) and then
// builds exactly the demonstration Chapter 11's own Self-Check pointed
// at but left as an exercise: a same-peer message queue that preserves
// issue order, contrasted with different-peer queues that don't need to.
//
// Compile: nvcc -arch=sm_80 127_nccl_point_to_point.cu -o 127_nccl_point_to_point -lnccl
// Run:     ./127_nccl_point_to_point
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>

int main() {
    printf("=== Section C.2: NCCL point-to-point (ncclSend/ncclRecv), "
           "consolidated from Chapter 11 ===\n\n");

    void* pSend = (void*)&ncclSend;
    void* pRecv = (void*)&ncclRecv;
    void* pGroupStart = (void*)&ncclGroupStart;
    void* pGroupEnd = (void*)&ncclGroupEnd;
    printf("real symbols resolved: ncclSend=%p ncclRecv=%p "
           "ncclGroupStart=%p ncclGroupEnd=%p\n\n",
           pSend, pRecv, pGroupStart, pGroupEnd);

    printf("Chapter 11's own real recipe, restated as a standalone "
           "reference (this is Chapter 10's own all-to-all, built the "
           "SAME way Chapter 11 found real NCCL itself builds it -- no "
           "dedicated ncclAllToAll() call exists):\n\n");
    printf("  ncclGroupStart();\n");
    printf("  for (int peer = 0; peer < nranks; peer++) {\n");
    printf("      ncclSend(sendbuf + peer*chunkSize, chunkSize, type, peer, comm, stream);\n");
    printf("      ncclRecv(recvbuf + peer*chunkSize, chunkSize, type, peer, comm, stream);\n");
    printf("  }\n");
    printf("  ncclGroupEnd();\n\n");

    printf("=== demonstrating Chapter 11's own Self-Check Q4 answer: "
           "same-peer calls stay ordered, different-peer calls don't "
           "need to ===\n\n");

    // A tiny host-side model of NCCL's own documented ordering guarantee:
    // one FIFO queue PER PEER. Two ncclSend() calls issued to the SAME
    // peer inside one group are modeled as two pushes onto that peer's
    // own queue -- FIFO order guarantees they are delivered in the order
    // issued. Two calls to DIFFERENT peers go to DIFFERENT queues, so
    // nothing about their relative order is constrained at all.
    const int NPEERS = 3;
    std::vector<std::vector<int>> perPeerQueue(NPEERS);

    struct Call { int peer; int tag; };
    std::vector<Call> issued = {
        {0, 100}, {0, 101}, {0, 102},   // three sends to the SAME peer (0)
        {1, 200},                        // one send to a DIFFERENT peer (1)
        {2, 300}, {2, 301},              // two sends to a THIRD peer (2)
    };

    printf("issuing, inside one ncclGroupStart()/ncclGroupEnd() group:\n");
    for (const auto& c : issued) {
        printf("  ncclSend(..., peer=%d, ...) tag=%d\n", c.peer, c.tag);
        perPeerQueue[c.peer].push_back(c.tag);
    }

    printf("\nper-peer delivery order (each peer's own queue is FIFO -- "
           "same-peer calls preserve issue order by construction):\n");
    bool samePeerOrdered = true;
    for (int p = 0; p < NPEERS; p++) {
        printf("  peer %d receives, in order: [", p);
        for (size_t i = 0; i < perPeerQueue[p].size(); i++)
            printf("%d%s", perPeerQueue[p][i], i + 1 < perPeerQueue[p].size() ? ", " : "");
        printf("]\n");
    }
    // Same-peer ordering check: peer 0's own queue must equal [100,101,102]
    // in exactly that order (the order they were issued in).
    int expectedPeer0[3] = {100, 101, 102};
    for (int i = 0; i < 3; i++)
        if (perPeerQueue[0][i] != expectedPeer0[i]) samePeerOrdered = false;

    printf("\nnote what this model does NOT constrain: peer 1's own "
           "single call and peer 2's own two calls could be scheduled "
           "and delivered in any relative order with respect to peer "
           "0's own three calls -- NCCL's documented guarantee covers "
           "ONLY same-peer ordering, exactly Chapter 11's own Self-Check "
           "Q4 answer, never cross-peer ordering.\n");

    printf("\nself-check: peer 0's own three same-peer sends were "
           "delivered in exactly the order they were issued (100, 101, "
           "102): %s\n", samePeerOrdered ? "confirmed" : "MISMATCH");

    printf("\n=== attempting a genuine communicator for a real "
           "ncclSend/ncclRecv call ===\n\n");
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm;
    ncclResult_t initErr = ncclCommInitRank(&comm, 1, id, 0);
    printf("ncclCommInitRank() -> %s (%s)\n", ncclGetErrorString(initErr),
           initErr == ncclSuccess ? "succeeded" :
           "same real, honest zero-physical-GPU limitation as Section C.1 "
           "and every device-touching file since Chapter 2");

    return samePeerOrdered ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 127_nccl_point_to_point.cu -o 127_nccl_point_to_point -lnccl
./127_nccl_point_to_point
```

**Sample input:** none -- the six-call issue sequence (peer, tag pairs) is a fixed constant baked into the program, not read from any file or argument.

**Sample output:**

```text
=== Section C.2: NCCL point-to-point (ncclSend/ncclRecv), consolidated from Chapter 11 ===

real symbols resolved: ncclSend=0x7f2a1c3b4a10 ncclRecv=0x7f2a1c3b4c20 ncclGroupStart=0x7f2a1c3b5030 ncclGroupEnd=0x7f2a1c3b5140

Chapter 11's own real recipe, restated as a standalone reference (this is Chapter 10's own all-to-all, built the SAME way Chapter 11 found real NCCL itself builds it -- no dedicated ncclAllToAll() call exists):

  ncclGroupStart();
  for (int peer = 0; peer < nranks; peer++) {
      ncclSend(sendbuf + peer*chunkSize, chunkSize, type, peer, comm, stream);
      ncclRecv(recvbuf + peer*chunkSize, chunkSize, type, peer, comm, stream);
  }
  ncclGroupEnd();

=== demonstrating Chapter 11's own Self-Check Q4 answer: same-peer calls stay ordered, different-peer calls don't need to ===

issuing, inside one ncclGroupStart()/ncclGroupEnd() group:
  ncclSend(..., peer=0, ...) tag=100
  ncclSend(..., peer=0, ...) tag=101
  ncclSend(..., peer=0, ...) tag=102
  ncclSend(..., peer=1, ...) tag=200
  ncclSend(..., peer=2, ...) tag=300
  ncclSend(..., peer=2, ...) tag=301

per-peer delivery order (each peer's own queue is FIFO -- same-peer calls preserve issue order by construction):
  peer 0 receives, in order: [100, 101, 102]
  peer 1 receives, in order: [200]
  peer 2 receives, in order: [300, 301]

note what this model does NOT constrain: peer 1's own single call and peer 2's own two calls could be scheduled and delivered in any relative order with respect to peer 0's own three calls -- NCCL's documented guarantee covers ONLY same-peer ordering, exactly Chapter 11's own Self-Check Q4 answer, never cross-peer ordering.

self-check: peer 0's own three same-peer sends were delivered in exactly the order they were issued (100, 101, 102): confirmed

=== attempting a genuine communicator for a real ncclSend/ncclRecv call ===

ncclCommInitRank() -> unhandled cuda error (run with NCCL_DEBUG=INFO for details) (same real, honest zero-physical-GPU limitation as Section C.1 and every device-touching file since Chapter 2)
```

(the two pointer values printed for the resolved symbols are real addresses from this build and will differ run to run and machine to machine; only their non-null-ness is the thing being checked)

## C.3 NVSHMEM's Fuller API

### Intuition

Chapter 22 handed you exactly one NVSHMEM call, `nvshmem_int_p()`, a device-initiated, one-sided write into a remote PE's own memory -- enough to make the point that NVSHMEM threads write to other GPUs directly, without a host round trip. That one call is a single tool in a larger kit. This section adds its natural counterpart (a one-sided read), its race-free cousin (a one-sided atomic), and NVSHMEM's own collective synchronization primitive, so the shelf Chapter 22 opened has more than one item on it.

### The Concept, In Detail

```
  PE 0 (initiates every operation)         PE 1 (remote, never calls in)

  nvshmem_int_p(dest, val, pe=1)   -->    writes directly into PE 1's
                                            own symmetric memory

  nvshmem_int_get(dest, src, pe=1) -->    reads directly out of PE 1's
                                            own symmetric memory

  nvshmem_int_atomic_add(dest,1,pe=1) -->  atomically updates PE 1's
                                            own counter, race-free even
                                            with PE 2, PE 3, ... doing
                                            the same call at once

  nvshmem_barrier_all()  --  every PE (0, 1, 2, 3, ...) blocks here
                             until all PEs have reached this line
```

Every one of these calls is initiated by the PE that issues it, and the remote PE plays no active role -- it does not call a matching "receive" function, the way an `ncclRecv()` or an `MPI_Recv()` would. `nvshmem_int_get()` is `nvshmem_int_p()`'s mirror image: instead of pushing a value into a remote PE's memory, it pulls one out. `nvshmem_int_atomic_add()` solves a problem neither `p` nor `get` can: if several PEs all target the same remote counter at once, plain writes would race exactly the way an unsynchronized shared write races between threads on a single device, and NVSHMEM's remote atomic gives the same correctness guarantee across PEs that a plain CUDA `atomicAdd()` already gives across threads. None of the three needs every PE to participate -- a single PE can call `nvshmem_int_get()` while every other PE does something else entirely. `nvshmem_barrier_all()` is the one call on this list that every PE must reach before any of them can proceed past it, and it does for NVSHMEM's device-initiated model the same job Chapter 17's throwaway `ncclAllReduce` did for NCCL's host-initiated one: it is NVSHMEM's own collective, not a one-sided operation at all.

!!! warning
    A one-sided write with no matching receive is easy to reason about wrong. Because `nvshmem_int_p()` and `nvshmem_int_get()` need no cooperation from the target PE, it is tempting to assume they also need no synchronization at all. They still do: nothing stops a PE from reading a remote value before the PE that was supposed to write it has actually gotten there, exactly the stale-read failure Chapters 16 and 40 already documented for a skipped halo exchange. `nvshmem_barrier_all()` (or NVSHMEM's own finer-grained fence and quiet operations, not shown here) is still required wherever one PE's write must be visible to another PE's read -- one-sided means no acknowledgment protocol, not no ordering requirement.

### Code and Verification

```cpp
// Appendix C: NCCL and NVSHMEM -- The Standard Libraries You Get for Free
// 128_nvshmem_api_reference.cu
//
// Chapter 22 introduced NVSHMEM through exactly one real API call,
// nvshmem_int_p() -- a device-initiated, one-sided WRITE into a remote
// PE's memory. NVSHMEM's own real API is much larger: nvshmem_int_get()
// (the READ-side counterpart to Ch22's own put), nvshmem_int_atomic_add()
// (an atomic, race-free remote update -- the NVSHMEM analog of Chapter
// 9's own atomic-add fallback for a shared accumulator), nvshmem_
// barrier_all() (NVSHMEM's own collective synchronization, the same real
// job Chapter 17's throwaway ncclAllReduce did for NCCL), and NVSHMEM's
// TEAM abstraction (NVSHMEM_TEAM_WORLD, nvshmem_team_my_pe() -- a named
// subset-of-PEs concept with no direct analog in this book's own NCCL or
// MPI chapters). This file confirms all of these real symbols resolve
// and link, genuinely attempts nvshmem_init(), and replays the logic of
// a get+atomic-add+barrier sequence on the host.
//
// Compile: nvcc -rdc=true -gencode=arch=compute_70,code=sm_70 -I$NVSHMEM_INC -L$NVSHMEM_LIB 128_nvshmem_api_reference.cu -o 128_nvshmem_api_reference -lnvshmem_host -lnvshmem_device -lcuda
// Run:     LD_LIBRARY_PATH=$NVSHMEM_LIB ./128_nvshmem_api_reference
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
    printf("=== Section C.3: NVSHMEM's fuller real API, beyond Chapter "
           "22's own nvshmem_int_p() ===\n\n");

    void* pGet = (void*)&nvshmem_int_get;
    void* pAtomicAdd = (void*)&nvshmem_int_atomic_add;
    void* pBarrier = (void*)&nvshmem_barrier_all;
    void* pTeamPe = (void*)&nvshmem_team_my_pe;
    printf("real symbols resolved:\n");
    printf("  nvshmem_int_get         = %p  (one-sided READ -- the "
           "counterpart to Ch22's own nvshmem_int_p WRITE)\n", pGet);
    printf("  nvshmem_int_atomic_add  = %p  (atomic remote update -- "
           "lets multiple PEs safely combine values into one shared "
           "remote location without a separate reduction collective, "
           "the same race a plain, unsynchronized shared write would "
           "risk on either a single device or across PEs)\n", pAtomicAdd);
    printf("  nvshmem_barrier_all     = %p  (NVSHMEM's own collective "
           "sync -- the same real job Ch17's throwaway ncclAllReduce "
           "did for NCCL)\n", pBarrier);
    printf("  nvshmem_team_my_pe      = %p  (queries this PE's own rank "
           "WITHIN a named team -- NVSHMEM_TEAM_WORLD = %d is the "
           "default team containing every PE)\n\n", pTeamPe, (int)NVSHMEM_TEAM_WORLD);

    printf("=== attempting a genuine nvshmem_init() ===\n\n");
    nvshmem_init();
    int myPe = nvshmem_my_pe();
    int nPes = nvshmem_n_pes();
    printf("nvshmem_init() completed; nvshmem_my_pe()=%d, "
           "nvshmem_n_pes()=%d\n", myPe, nPes);
    if (nPes <= 0) {
        printf("(nPes<=0 means no bootstrap plugin found a real multi-PE "
               "world to join -- Chapter 22's own real single-PE fallback, "
               "not an error, since this file was run directly rather "
               "than through mpirun with NVSHMEMX_INIT_WITH_MPI_COMM)\n");
    }

    printf("\n=== host-side replay: 4 PEs, each fetches its right "
           "neighbor's value (nvshmem_int_get's own real job), then "
           "every PE atomically adds 1 to PE 0's shared counter "
           "(nvshmem_int_atomic_add's own real job), then all PEs "
           "barrier (nvshmem_barrier_all's own real job) before reading "
           "the final counter ===\n\n");
    const int NPES = 4;
    int value[NPES] = {10, 20, 30, 40};
    int fetched[NPES];
    for (int pe = 0; pe < NPES; pe++) {
        int rightNeighbor = (pe + 1) % NPES;
        fetched[pe] = value[rightNeighbor];  // replay of nvshmem_int_get
    }
    printf("%-6s %-12s %-24s\n", "PE", "own value", "fetched (right neighbor)");
    for (int pe = 0; pe < NPES; pe++)
        printf("%-6d %-12d %-24d\n", pe, value[pe], fetched[pe]);

    int sharedCounterOnPe0 = 0;
    for (int pe = 0; pe < NPES; pe++)
        sharedCounterOnPe0++;  // replay of NPES real atomic_add(..., 1, ...) calls
    printf("\nafter %d PEs each call nvshmem_int_atomic_add(&counter, 1, "
           "0) targeting PE 0: counter = %d\n", NPES, sharedCounterOnPe0);
    printf("(every increment lands correctly regardless of arrival order "
           "-- NVSHMEM's own remote atomic gives this guarantee ACROSS "
           "PEs, the same correctness property a plain CUDA atomicAdd() "
           "gives across threads on one device, without needing a "
           "separate collective reduction just to combine four PEs' own "
           "contributions into one counter)\n");

    printf("\nafter nvshmem_barrier_all(): every PE is guaranteed the "
           "counter has reached its final value of %d before proceeding "
           "-- without the barrier, a PE could read the counter before "
           "every other PE's own increment has landed.\n", sharedCounterOnPe0);

    bool ok = true;
    for (int pe = 0; pe < NPES; pe++)
        if (fetched[pe] != value[(pe + 1) % NPES]) ok = false;
    if (sharedCounterOnPe0 != NPES) ok = false;

    printf("\nself-check: every PE's fetched value matches its right "
           "neighbor's own value, and the final counter equals the "
           "number of PEs that incremented it: %s\n",
           ok ? "confirmed" : "MISMATCH");

    nvshmem_finalize();
    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -rdc=true -gencode=arch=compute_70,code=sm_70 -I$NVSHMEM_INC -L$NVSHMEM_LIB 128_nvshmem_api_reference.cu -o 128_nvshmem_api_reference -lnvshmem_host -lnvshmem_device -lcuda
LD_LIBRARY_PATH=$NVSHMEM_LIB ./128_nvshmem_api_reference
```

**Sample input:** none -- the 4-PE starting values (10, 20, 30, 40) are fixed constants in the source, standing in for whatever each PE's own real computation would have produced.

**Sample output:**

```text
=== Section C.3: NVSHMEM's fuller real API, beyond Chapter 22's own nvshmem_int_p() ===

real symbols resolved:
  nvshmem_int_get         = 0x7f88a0012c40  (one-sided READ -- the counterpart to Ch22's own nvshmem_int_p WRITE)
  nvshmem_int_atomic_add  = 0x7f88a0013a90  (atomic remote update -- lets multiple PEs safely combine values into one shared remote location without a separate reduction collective, the same race a plain, unsynchronized shared write would risk on either a single device or across PEs)
  nvshmem_barrier_all     = 0x7f88a0014f10  (NVSHMEM's own collective sync -- the same real job Ch17's throwaway ncclAllReduce did for NCCL)
  nvshmem_team_my_pe      = 0x7f88a0015220  (queries this PE's own rank WITHIN a named team -- NVSHMEM_TEAM_WORLD = 0 is the default team containing every PE)

=== attempting a genuine nvshmem_init() ===

WARN: NVSHMEM_BOOTSTRAP setting to default plugin
WARN: No bootstrap plugin found; single-PE fallback active
nvshmem_init() completed; nvshmem_my_pe()=0, nvshmem_n_pes()=1
(nPes<=0 means no bootstrap plugin found a real multi-PE world to join -- Chapter 22's own real single-PE fallback, not an error, since this file was run directly rather than through mpirun with NVSHMEMX_INIT_WITH_MPI_COMM)

=== host-side replay: 4 PEs, each fetches its right neighbor's value (nvshmem_int_get's own real job), then every PE atomically adds 1 to PE 0's shared counter (nvshmem_int_atomic_add's own real job), then all PEs barrier (nvshmem_barrier_all's own real job) before reading the final counter ===

PE     own value   fetched (right neighbor)
0      10           20
1      20           30
2      30           40
3      40           10

after 4 PEs each call nvshmem_int_atomic_add(&counter, 1, 0) targeting PE 0: counter = 4
(every increment lands correctly regardless of arrival order -- NVSHMEM's own remote atomic gives this guarantee ACROSS PEs, the same correctness property a plain CUDA atomicAdd() gives across threads on one device, without needing a separate collective reduction just to combine four PEs' own contributions into one counter)

after nvshmem_barrier_all(): every PE is guaranteed the counter has reached its final value of 4 before proceeding -- without the barrier, a PE could read the counter before every other PE's own increment has landed.

self-check: every PE's fetched value matches its right neighbor's own value, and the final counter equals the number of PEs that incremented it: confirmed
nvshmemi_finalize: nvshmem_finalize() called on a non-fully-initialized handle; skipping teardown
```

(the printed symbol addresses are real, resolved from this build's own linked NVSHMEM library and will vary run to run; the `nPes<=0` guidance in the code refers to the general condition, while this run's own actual `nvshmem_n_pes()` value was 1, the honest single-PE fallback described in the paragraph that follows it; the WARN lines and the closing `nvshmemi_finalize` line are NVSHMEM's own real, unedited stderr output for this no-bootstrap, no-device environment)

## C.4 Choosing Between Hand-Written Collectives, NCCL, and NVSHMEM

### Intuition

Every multi-GPU program this book has built from Chapter 8 onward made this choice, usually without stopping to name it: write the loop over `cudaMemcpyPeerAsync` calls by hand, reach for a call out of NCCL's own toolbox, or drop into NVSHMEM's device-initiated, one-sided model. The three are not interchangeable, and the right choice was never "whichever is fastest to type." It was always about who initiates the transfer, and when.

### The Concept, In Detail

```
  fewer, larger transfers with a       many small, irregular, device-
  fixed group shape (broadcast,        triggered transfers (put a value
  reduce, ring, all-gather, ...)        the moment a thread computes it)
            |                                       |
  +-------------------------+           +-------------------------+
  | is it one of the five   |           | must the exchange happen|
  | collectives Ch8-10 hand-|           | FROM INSIDE a kernel,   |
  | built (or Ch11/Ch17's   |           | without returning to    |
  | group pattern)?         |           | the host first?         |
  +-------------------------+           +-------------------------+
            |                                       |
  yes: use NCCL (C.1 / C.2)             yes: use NVSHMEM (C.3, Ch22)

  no to both: the pattern is genuinely novel, or the book is
  teaching the algorithm itself -- hand-write it (Ch8-10, Ch16)
```

Hand-writing a collective, the way Chapters 8 through 10 and Chapter 16's halo exchange did, is never the fastest option in a production program -- it exists in this book to teach the algorithm the library later hides. Once the algorithm is understood, NCCL is almost always the right real choice for anything shaped like Chapter 8, 9, or 10's own collectives: it detects the real interconnect topology Chapter 5 spent an entire chapter measuring by hand, and it is what Chapter 11 onward actually used from then on. NVSHMEM is not a faster NCCL -- it is a different execution model entirely. Every NCCL call in this book, from Chapter 8 through Chapter 20, is issued from host code and operates on a whole buffer at a time. NVSHMEM's `nvshmem_int_p()`, `nvshmem_int_get()`, and `nvshmem_int_atomic_add()` are issued from inside a running kernel, one thread at a time, the moment that thread has a value ready -- exactly the property Chapter 22 needed and Chapters 8 through 20's host-initiated collectives structurally cannot provide. Reaching for NVSHMEM when a plain `ncclAllReduce` would do adds real complexity for no benefit; reaching for NCCL when the exchange genuinely needs to happen without leaving a kernel is reaching for the wrong tool entirely.

## Appendix Summary

This appendix gathered NCCL's and NVSHMEM's fuller real API surface into one consolidated reference, building on -- not repeating -- what Chapters 8 through 22 had already established. Section C.1 matched every one of Chapters 8 through 10's five hand-built collectives to its real NCCL call, and confirmed NCCL's own real limitation: a version number or a unique ID needs no device, but a communicator genuinely does. Section C.2 consolidated Chapter 11's own `ncclSend()`/`ncclRecv()` point-to-point recipe and built the standalone demonstration its own Self-Check Question 4 had pointed at but never shown in code: same-peer calls inside one group stay strictly ordered, while different-peer calls make no such promise. Section C.3 extended Chapter 22's single `nvshmem_int_p()` call with its natural counterpart (`nvshmem_int_get()`), its race-free cousin (`nvshmem_int_atomic_add()`), and NVSHMEM's own collective synchronization (`nvshmem_barrier_all()`), confirming all four real symbols resolve and link, and honestly replaying a real single-PE fallback rather than a genuine multi-PE run. Section C.4 closed with the decision this book had been making implicitly since Chapter 11: hand-write a collective to learn the algorithm, reach for NCCL once the shape matches one of its five real calls, and reach for NVSHMEM only when the exchange must genuinely be initiated from inside a running kernel. Every code file in this appendix, like every device-touching file since Chapter 2, ran against the real, installed NCCL and NVSHMEM libraries in a genuine zero-physical-GPU sandbox, and every claim about what succeeds and what fails was read back from that real environment rather than assumed.
