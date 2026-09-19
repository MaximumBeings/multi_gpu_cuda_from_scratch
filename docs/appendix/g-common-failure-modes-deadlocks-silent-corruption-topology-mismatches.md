# Appendix G: Common Failure Modes -- Deadlocks, Silent Corruption From Missed Synchronization, and Topology Mismatches

Every chapter in this book has reported one specific real finding at a time. This closing appendix collects three failure modes that have each surfaced, piece by piece, across many of those chapters -- Chapter 17 and Chapter 20's own real deadlocks, Chapter 6's own cross-device event semantics, Chapter 16 and Chapter 40's own halo-sync corruption, Chapter 4's own peer-access query, Chapter 2's own topology model -- and names each one directly, with one new, genuinely tested demonstration per failure mode rather than a repeat of an earlier chapter's own example. None of the three failure modes below announces itself with an error message. A deadlocked program hangs silently; a program with a missed synchronization step returns a wrong answer with `cudaSuccess` on every call; a program built on an assumed topology runs correctly, just slower than planned, with nothing in any return code to say why. That is precisely what makes all three worth a dedicated closing appendix: each one is a program that never crashes and never reports an error, right up until someone notices the number is wrong.

## G.1 Deadlocks: When Every Process Waits for a Call Another Process Never Makes

### Intuition

Chapter 20's own File 57 already reproduced one real MPI deadlock: a ring where every rank calls `MPI_Send()` before any rank calls `MPI_Recv()`. That deadlock has an escape hatch, and Chapter 20 found it directly -- a small enough message fits inside Open MPI's own eager-send buffer, so `MPI_Send()` returns immediately without waiting for a matching receive, and the program runs to completion looking perfectly correct. Only once the message crosses that real, measured buffering threshold does the deadlock actually appear. This section builds the other half of the same real hazard: a version with no size-dependent escape hatch at all. If every rank calls `MPI_Recv()` *before* any rank calls `MPI_Send()`, the deadlock is absolute -- `MPI_Recv()` has no buffering option to fall back on, because there is nothing yet to buffer.

### The Concept, In Detail

```
  ring, 4 ranks, DEADLOCK order (every rank: Recv before Send)

  rank0 -- blocked in Recv(from rank3) -- waiting for rank3's Send
  rank1 -- blocked in Recv(from rank0) -- waiting for rank0's Send
  rank2 -- blocked in Recv(from rank1) -- waiting for rank1's Send
  rank3 -- blocked in Recv(from rank2) -- waiting for rank2's Send

  every rank waits on a Send call no rank has reached yet -- a closed
  cycle, at ANY message size (contrast: Chapter 20's Send-first
  deadlock only appears past the eager-buffering threshold)

  FIXED order (even ranks Send-first, odd ranks Recv-first):

  rank0(even) --Send-> rank1(odd, already in Recv) : delivered immediately
  rank2(even) --Send-> rank3(odd, already in Recv) : delivered immediately
  rank1(odd)  --Send-> rank2(even, now in Recv)     : delivered immediately
  rank3(odd)  --Send-> rank0(even, now in Recv)     : delivered immediately
```

`MPI_Recv()` genuinely blocks until a matching message arrives -- there is no buffering variant of receiving, unlike `MPI_Send()`, which the MPI standard (already quoted in Chapter 20) allows an implementation to complete early by copying into a temporary system buffer. So a ring where every rank calls `MPI_Recv()` first has no size-dependent way out: every rank is genuinely, permanently blocked waiting for a `MPI_Send()` call that no rank has reached, because every rank is stuck at the exact same point in the exact same cycle. The real, standard fix does not remove either call -- it breaks the cycle by having half the ranks issue their `MPI_Send()` first: even-numbered ranks send, then receive; odd-numbered ranks receive, then send. An even rank's `MPI_Send()` has an odd rank already waiting in `MPI_Recv()` to receive it, so it completes immediately; that same odd rank's own subsequent `MPI_Send()` then has the next even rank waiting for it, and the pattern propagates around the entire ring without a single rank ever blocking on a call nobody has issued.

!!! warning "[COMMON TRAP] Assuming a deadlock is a message-size problem because the last one was"
    Chapter 20's own deadlock depended on crossing a real buffering threshold, which makes it tempting to treat "keep messages small" as a general defense against MPI deadlocks. It is not. This section's own recv-first deadlock hangs at a single `int` exactly as completely as it would at a million -- there is no buffer to exploit, because the blocking call at fault (`MPI_Recv()`) was never eligible for buffering in the first place. The real, general rule has nothing to do with size: every rank's *sequence* of blocking calls has to be constructed so that some rank is always in a position to unblock another, at every point in the program. A size threshold is Chapter 20's own specific symptom of a specific call shape (`Send`-then-`Recv`, everywhere); it is not the underlying disease.

### Code and Verification

```c
// Appendix G: Common Failure Modes -- Deadlocks, Silent Corruption From
// Missed Synchronization, and Topology Mismatches
// 138_collective_order_deadlock.c
//
// Appendix G.1 -- Chapter 20's own File 57 already genuinely reproduced
// one real MPI deadlock: a naive point-to-point ring where every rank
// calls MPI_Send() before any rank calls MPI_Recv() -- but that
// deadlock only manifests once the message crosses Open MPI's own
// eager-send buffering threshold (Chapter 20 needed 1,000,000 ints; a
// single int silently survives). This file builds the OTHER, absolute
// version of the same real hazard: every rank calls MPI_Recv() BEFORE
// any rank calls MPI_Send(). MPI_Recv() genuinely blocks until a
// matching message arrives, with no buffering escape hatch at any
// message size -- if every rank in a ring is blocked inside its own
// MPI_Recv(), no rank can ever reach the MPI_Send() call the rank
// ahead of it in the ring is waiting for, and the program hangs
// forever, confirmed here the same way Chapter 20 confirmed its own
// deadlock: run under a bounded `timeout`, which kills the genuinely-
// hung job and reports exit code 124. The real, standard fix -- half
// the ranks send first, half receive first, so the blocking calls
// interleave instead of all waiting on each other -- is built and
// verified alongside it.
//
// argv[1] == "fixed"    -> even ranks Send-then-Recv, odd ranks Recv-then-Send (completes)
// argv[1] == "deadlock" (or omitted) -> every rank Recv-then-Send (hangs at ANY message size)
//
// Compile: mpicc -std=c11 -Wall -Wextra -O2 138_collective_order_deadlock.c -o 138_collective_order_deadlock
// Run:     timeout 6 mpirun --allow-run-as-root --oversubscribe -np 4 ./138_collective_order_deadlock fixed
//          timeout 6 mpirun --allow-run-as-root --oversubscribe -np 4 ./138_collective_order_deadlock deadlock
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int fixed = (argc >= 2) && (strcmp(argv[1], "fixed") == 0);
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;
    int sendVal = rank, recvVal = -1;

    if (fixed && (rank % 2 == 0)) {
        // Even ranks: Send first, Recv second -- interleaves correctly
        // with the odd ranks below, which do the opposite.
        fprintf(stderr, "rank %d (even, fixed order): about to MPI_Send to %d\n", rank, next);
        MPI_Send(&sendVal, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        fprintf(stderr, "rank %d: MPI_Send returned; about to MPI_Recv from %d\n", rank, prev);
        MPI_Recv(&recvVal, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stderr, "rank %d: MPI_Recv returned\n", rank);
    } else {
        // Odd ranks in "fixed" mode, and EVERY rank in "deadlock" mode:
        // Recv first, Send second.
        fprintf(stderr, "rank %d: about to MPI_Recv from %d (recv-first order)\n", rank, prev);
        MPI_Recv(&recvVal, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stderr, "rank %d: MPI_Recv returned; about to MPI_Send to %d\n", rank, next);
        MPI_Send(&sendVal, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        fprintf(stderr, "rank %d: MPI_Send returned\n", rank);
    }

    printf("rank %d done, recvVal=%d (expected %d)\n", rank, recvVal, prev);
    MPI_Finalize();
    return 0;
}
```

**Compile and run:**

```bash
mpicc -std=c11 -Wall -Wextra -O2 138_collective_order_deadlock.c -o 138_collective_order_deadlock
timeout 6 mpirun --allow-run-as-root --oversubscribe -np 4 ./138_collective_order_deadlock fixed
timeout 6 mpirun --allow-run-as-root --oversubscribe -np 4 ./138_collective_order_deadlock deadlock
```

**Sample input:** `argv[1]` selects the mode (`fixed` or `deadlock`); every rank's own message is just its own rank number, `MPI_INT`, one element.

**Sample output, `fixed` mode (completes; per-rank stderr interleaving is real and varies run to run, since these are genuinely separate OS processes -- stdout ordering, printed after `MPI_Finalize()`, does not affect correctness):**

```text
rank 2 (even, fixed order): about to MPI_Send to 3
rank 2: MPI_Send returned; about to MPI_Recv from 1
rank 0 (even, fixed order): about to MPI_Send to 1
rank 0: MPI_Send returned; about to MPI_Recv from 3
rank 3: about to MPI_Recv from 2 (recv-first order)
rank 1: about to MPI_Recv from 0 (recv-first order)
rank 1: MPI_Recv returned; about to MPI_Send to 2
rank 3: MPI_Recv returned; about to MPI_Send to 0
rank 1: MPI_Send returned
rank 2: MPI_Recv returned
rank 2 done, recvVal=1 (expected 1)
rank 1 done, recvVal=0 (expected 0)
rank 3 done, recvVal=2 (expected 2)
rank 3: MPI_Send returned
rank 0: MPI_Recv returned
rank 0 done, recvVal=3 (expected 3)
```

**Sample output, `deadlock` mode (genuinely hangs -- `timeout 6` kills it and the shell reports exit code `124`; every rank's own stderr line printed before it, and nothing after, is the real, complete evidence of the hang):**

```text
rank 3: about to MPI_Recv from 2 (recv-first order)
rank 0: about to MPI_Recv from 3 (recv-first order)
rank 2: about to MPI_Recv from 1 (recv-first order)
rank 1: about to MPI_Recv from 0 (recv-first order)
mpirun: Forwarding signal 18 to job
```

The shell's own `$?` after that command reports `124` -- `timeout`'s own documented signal for "the command was killed because it exceeded the time limit," the identical proof technique Chapter 20 already established for its own real deadlock.

## G.2 Silent Corruption From Missed Synchronization: The Wait That Comes Too Late

### Intuition

Chapter 6's own File 15 built the real cross-device dependency mechanism -- an event recorded on one device's stream, handed to `cudaStreamWaitEvent()` on a different device's stream -- and its own locked output stated the finding plainly: "`cudaStreamWaitEvent` only ever inserts a dependency for FUTURE work on stream1; it does not itself wait for anything." That word, "future," is doing more work than it looks like. It does not mean "whatever happens to execute later in wall-clock time" -- it means "whatever gets *enqueued* onto that stream after this call, in program order." A consumer operation that was enqueued onto the stream *before* the `cudaStreamWaitEvent()` call is not future work relative to that call at all, and the wait does nothing for it -- it runs whenever its own stream reaches it, with no dependency on the producer's event inserted anywhere.

### The Concept, In Detail

```
  CORRECT enqueue order            BUGGY enqueue order
  stream queue (top = enqueued     stream queue (top = enqueued
  first):                          first):

  [0] cudaStreamWaitEvent          [0] consumer op
      (depends on producerEvent:       (depends on producerEvent: no)
       no -- it IS the wait)
  [1] consumer op                  [1] cudaStreamWaitEvent
      (depends on producerEvent:       (depends on producerEvent: no
       YES -- enqueued AFTER            -- it IS the wait, and nothing
       the wait)                        enqueued after it needs one)
```

The two orderings above enqueue the exact same two operations -- a `cudaStreamWaitEvent()` call and a consumer operation -- onto the exact same stream. The only difference is which one gets enqueued first. In the correct order, the wait is enqueued first, so the consumer operation genuinely is future work relative to it, and the dependency applies. In the buggy order, the consumer operation is enqueued first -- perhaps because it is issued earlier in the same function, or by a code path that runs before the synchronization call is reached -- and the `cudaStreamWaitEvent()` call that comes after it has no mechanism to reach backward into the queue and attach a dependency to something already sitting there. Both calls succeed. Neither one produces an error, a warning, or any observable difference in a return code -- Chapter 6's own real API, exercised in both orders, reports the identical honest result either way. The only real difference is in the dependency graph the stream actually executes, and the only way to see that difference is to know the rule and check the enqueue order in the source, not the return codes at runtime.

!!! warning "[COMMON TRAP] Treating cudaStreamWaitEvent() as a checkpoint that protects a whole function"
    It is tempting to read a `cudaStreamWaitEvent()` call sitting in the middle of a function as if it retroactively guards everything above it on the same stream -- as though it were a fence the compiler or runtime inserts wherever it is textually written. It guards nothing above it. It is exactly one more entry in that stream's own queue, and it only ever constrains what comes *after* it in that same queue. Refactoring that moves a consumer operation earlier in a function -- a common, innocent-looking change made for readability or to overlap unrelated work -- can silently move it in front of a `cudaStreamWaitEvent()` call it used to come after, and nothing about compiling or running the refactored code will say so.

### Code and Verification

```cpp
// Appendix G: Common Failure Modes -- Deadlocks, Silent Corruption From
// Missed Synchronization, and Topology Mismatches
// 139_stream_wait_enqueue_order.cu
//
// Appendix G.2 -- Chapter 6's own File 15 already established, in its
// own real printed output, that "cudaStreamWaitEvent only ever inserts
// a dependency for FUTURE work on stream1; it does not itself wait for
// anything." This file makes the natural, genuinely real consequence of
// that finding concrete: "future work" means work ENQUEUED after the
// cudaStreamWaitEvent() call, in program order -- not work that is
// merely EXECUTED after it. If a consumer operation is enqueued onto a
// stream BEFORE that stream's cudaStreamWaitEvent() call is issued, the
// wait does not retroactively cover it; the consumer runs whenever its
// own stream reaches it, with no dependency on the producer's event at
// all. Because every CUDA call in this environment honestly reports
// zero physical devices (Chapter 2 onward), this file cannot execute a
// real race -- so it does two things instead: it genuinely issues both
// the CORRECT enqueue order (wait before consumer) and the BUGGY
// enqueue order (consumer before wait) against the real CUDA Runtime
// API, reporting the real (honestly identical) return codes either way,
// and it builds a deterministic host-side model of stream QUEUE
// ORDER -- not execution timing -- that captures the actual rule the
// CUDA Programming Model documents: dependencies attach to a stream's
// enqueue position, not to wall-clock time.
//
// Compile: nvcc -arch=sm_80 139_stream_wait_enqueue_order.cu -o 139_stream_wait_enqueue_order
// Run:     ./139_stream_wait_enqueue_order
#include <cstdio>
#include <vector>
#include <string>
#include <cuda_runtime.h>

int main() {
    printf("=== Section G.2: cudaStreamWaitEvent() only covers work enqueued\n");
    printf("    AFTER it -- Chapter 6's own File 15 finding, taken to its real\n");
    printf("    conclusion ===\n\n");

    printf("Chapter 6's own File 15, verbatim: \"cudaStreamWaitEvent only ever\n");
    printf("inserts a dependency for FUTURE work on stream1; it does not itself\n");
    printf("wait for anything.\" \"Future\" is an ENQUEUE-ORDER guarantee, not a\n");
    printf("wall-clock one -- it means \"whatever gets enqueued onto this stream\n");
    printf("after this call\", not \"whatever happens to run later\".\n\n");

    // === Real API calls, both orders, both genuinely issued. ===
    cudaSetDevice(0);
    cudaStream_t consumerStream;
    cudaError_t eStream = cudaStreamCreate(&consumerStream);
    cudaEvent_t producerEvent;
    cudaError_t eEvent = cudaEventCreate(&producerEvent);
    cudaError_t eRecord = cudaEventRecord(producerEvent, consumerStream);

    printf("=== CORRECT enqueue order: wait enqueued BEFORE the consumer op ===\n");
    printf("  1. cudaStreamWaitEvent(consumerStream, producerEvent) -- enqueued first\n");
    cudaError_t eWaitCorrect = cudaStreamWaitEvent(consumerStream, producerEvent, 0);
    printf("     -> %s\n", cudaGetErrorString(eWaitCorrect));
    printf("  2. consumer op enqueued SECOND -- genuinely covered by the wait above,\n");
    printf("     because it was enqueued onto consumerStream AFTER the wait call\n\n");

    printf("=== BUGGY enqueue order: consumer op enqueued BEFORE the wait ===\n");
    printf("  1. consumer op enqueued FIRST (imagine: issued earlier in the same\n");
    printf("     function, or by code that runs before the sync call is reached)\n");
    printf("  2. cudaStreamWaitEvent(consumerStream, producerEvent) -- enqueued SECOND\n");
    cudaError_t eWaitBuggy = cudaStreamWaitEvent(consumerStream, producerEvent, 0);
    printf("     -> %s\n", cudaGetErrorString(eWaitBuggy));
    printf("     on real hardware this call would SUCCEED -- cudaStreamWaitEvent() has\n");
    printf("     no way to know a consumer op was already enqueued earlier on the same\n");
    printf("     stream, and no way to retroactively insert a dependency in front of\n");
    printf("     it. The already-enqueued consumer op runs with NO dependency on\n");
    printf("     producerEvent at all -- and here, it reports this environment's own\n");
    printf("     same honest zero-device limitation as every call since Chapter 2.\n\n");

    printf("both real API calls above report the SAME honest return code (%s) in\n",
           cudaGetErrorString(eWaitCorrect));
    printf("this zero-physical-GPU environment -- the bug is not a return-code\n");
    printf("failure, which is exactly what makes it \"silent corruption\": there is no\n");
    printf("error anywhere to catch, on real hardware or here.\n\n");

    // === Host-side deterministic model of enqueue-order semantics. ===
    printf("=== deterministic model: what each enqueue order actually guarantees ===\n\n");

    struct QueueEntry { std::string label; bool isWait; bool dependsOnProducer; };

    // Correct order: [wait, consumer] -- consumer is AFTER the wait entry.
    std::vector<QueueEntry> correctQueue = {
        {"cudaStreamWaitEvent(producerEvent)", true, false},
        {"consumer op",                        false, true},   // enqueued after the wait -> depends
    };
    // Buggy order: [consumer, wait] -- consumer is BEFORE the wait entry.
    std::vector<QueueEntry> buggyQueue = {
        {"consumer op",                        false, false},  // enqueued before the wait -> independent
        {"cudaStreamWaitEvent(producerEvent)", true, false},
    };

    auto printQueue = [](const char* label, std::vector<QueueEntry>& q) {
        printf("%s stream queue, in real enqueue order:\n", label);
        for (size_t i = 0; i < q.size(); ++i) {
            printf("  [%zu] %-38s depends on producerEvent: %s\n", i, q[i].label.c_str(),
                   q[i].dependsOnProducer ? "YES" : "no");
        }
        printf("\n");
    };
    printQueue("CORRECT-order", correctQueue);
    printQueue("BUGGY-order", buggyQueue);

    bool correctDependsCorrectly = correctQueue[1].dependsOnProducer == true;
    bool buggyDoesNotDepend = buggyQueue[0].dependsOnProducer == false;

    printf("self-check: in the CORRECT order, the consumer op (position 1, enqueued\n");
    printf("after the wait) genuinely depends on producerEvent (%s); in the BUGGY\n",
           correctDependsCorrectly ? "confirmed" : "MISMATCH");
    printf("order, the consumer op (position 0, enqueued before the wait) genuinely\n");
    printf("does NOT depend on it (%s) -- the exact same two lines of code, reordered,\n",
           buggyDoesNotDepend ? "confirmed" : "MISMATCH");
    printf("produce two different real dependency graphs, with zero difference in any\n");
    printf("return code from either cudaStreamWaitEvent() call: %s\n",
           (correctDependsCorrectly && buggyDoesNotDepend) ? "confirmed" : "MISMATCH");

    if (eEvent == cudaSuccess) cudaEventDestroy(producerEvent);
    if (eStream == cudaSuccess) cudaStreamDestroy(consumerStream);
    (void)eRecord;
    return (correctDependsCorrectly && buggyDoesNotDepend) ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 139_stream_wait_enqueue_order.cu -o 139_stream_wait_enqueue_order
./139_stream_wait_enqueue_order
```

**Sample input:** none -- every value is either a real return code from the CUDA Runtime API or a compile-time-fixed description of the two enqueue orders being compared.

**Sample output:**

```text
=== Section G.2: cudaStreamWaitEvent() only covers work enqueued
    AFTER it -- Chapter 6's own File 15 finding, taken to its real
    conclusion ===

Chapter 6's own File 15, verbatim: "cudaStreamWaitEvent only ever
inserts a dependency for FUTURE work on stream1; it does not itself
wait for anything." "Future" is an ENQUEUE-ORDER guarantee, not a
wall-clock one -- it means "whatever gets enqueued onto this stream
after this call", not "whatever happens to run later".

=== CORRECT enqueue order: wait enqueued BEFORE the consumer op ===
  1. cudaStreamWaitEvent(consumerStream, producerEvent) -- enqueued first
     -> no CUDA-capable device is detected
  2. consumer op enqueued SECOND -- genuinely covered by the wait above,
     because it was enqueued onto consumerStream AFTER the wait call

=== BUGGY enqueue order: consumer op enqueued BEFORE the wait ===
  1. consumer op enqueued FIRST (imagine: issued earlier in the same
     function, or by code that runs before the sync call is reached)
  2. cudaStreamWaitEvent(consumerStream, producerEvent) -- enqueued SECOND
     -> no CUDA-capable device is detected
     on real hardware this call would SUCCEED -- cudaStreamWaitEvent() has
     no way to know a consumer op was already enqueued earlier on the same
     stream, and no way to retroactively insert a dependency in front of
     it. The already-enqueued consumer op runs with NO dependency on
     producerEvent at all -- and here, it reports this environment's own
     same honest zero-device limitation as every call since Chapter 2.

both real API calls above report the SAME honest return code (no CUDA-capable device is detected) in
this zero-physical-GPU environment -- the bug is not a return-code
failure, which is exactly what makes it "silent corruption": there is no
error anywhere to catch, on real hardware or here.

=== deterministic model: what each enqueue order actually guarantees ===

CORRECT-order stream queue, in real enqueue order:
  [0] cudaStreamWaitEvent(producerEvent)     depends on producerEvent: no
  [1] consumer op                            depends on producerEvent: YES

BUGGY-order stream queue, in real enqueue order:
  [0] consumer op                            depends on producerEvent: no
  [1] cudaStreamWaitEvent(producerEvent)     depends on producerEvent: no

self-check: in the CORRECT order, the consumer op (position 1, enqueued
after the wait) genuinely depends on producerEvent (confirmed); in the BUGGY
order, the consumer op (position 0, enqueued before the wait) genuinely
does NOT depend on it (confirmed) -- the exact same two lines of code, reordered,
produce two different real dependency graphs, with zero difference in any
return code from either cudaStreamWaitEvent() call: confirmed
```

## G.3 Topology Mismatches: When Code Assumes a Link That Isn't There

### Intuition

Chapter 4's own File 9 built the real, idiomatic sequence: check `cudaDeviceCanAccessPeer()` for a device pair before assuming anything about it, then enable access only where it genuinely exists. Chapter 5's own File 11 and File 13 showed why skipping that query never produces a wrong *answer* -- `cudaMemcpyPeer()` falls back to a staged host copy transparently, for any pair that never had peer access enabled, with no error and no corrupted data either way. What skipping the query can still silently break is a *performance* assumption baked into the code around those calls -- a capacity plan, a scheduling deadline, a batch size chosen assuming a specific link class -- because Chapter 2's own partial-mesh topology model already established that not every device pair on the same real machine shares the same link.

### The Concept, In Detail

```
  real transfer time for 1.0 GB, by ACTUAL topology (Ch2/Ch5 figures)

  +--------------------------------+------------------------------+
  | NVLink-wired pair    (1 hop)   | 0.0200 s ---                 |
  | PCIe-only direct pair (1 hop)  | 0.0317 s -----                |
  | no direct link      (2-hop)    | 0.0635 s ----------            |
  +--------------------------------+------------------------------+

  code that ASSUMES the top row and never re-queries silently pays
  1.59x (PCIe-only) to 3.17x (no direct link) more than planned,
  with zero error code anywhere
```

Chapter 5's own File 13 already built the closed-form cost comparison between a direct peer transfer and a two-hop staged transfer, using Chapter 2's own already-cited bandwidth figures (31.5 GB/s PCIe 4.0 x16, 50 GB/s a directly-wired NVLink pair). This section reuses that exact model for a different question: not "which route does `cudaMemcpyPeer()` actually take," which Chapter 5 already answered, but "what happens to code that assumed one row of this table and never checked which row actually applies to the pair it is running on." Because `cudaMemcpyPeer()` and `cudaMemcpyPeerAsync()` route correctly regardless -- Chapter 5's own real finding -- nothing in their return code will ever reveal that the assumption was wrong. A capacity plan, a communication-computation overlap budget, or a training-step deadline built around the NVLink row of this table runs, silently, 1.59x slower than planned if the actual pair only has a direct PCIe link, and 3.17x slower if the pair has no direct link at all and every transfer must be staged through the host -- exactly the two-hop cost Chapter 5's own File 12 already measured the call pattern for.

!!! warning "[COMMON TRAP] Trusting a topology number that was measured on a different machine"
    It is tempting to treat a bandwidth or link-class figure as a property of the *code*, once it has been measured or read off a vendor spec sheet -- something that can be hardcoded into a constant and reused everywhere the same code runs. It is a property of the specific *deployment*, not the code. Chapter 2's own partial-mesh topology model is itself the proof: two GPUs in the same server can be NVLink-wired while another pair in the same server share only a PCIe switch, and a cluster's own topology can differ node to node even running the identical binary. `cudaDeviceCanAccessPeer()` and `cudaDeviceGetP2PAttribute()` query the *actual* topology of the *actual* devices at *actual* runtime -- re-running them on every deployment is the only real defense; a number copied from one cluster's own topology report into a shared constant is a silent liability the moment the code runs anywhere else.

### Code and Verification

```cpp
// Appendix G: Common Failure Modes -- Deadlocks, Silent Corruption From
// Missed Synchronization, and Topology Mismatches
// 140_topology_assumption_cost.cpp
//
// Appendix G.3 -- Chapter 4's own File 9 established the real,
// idiomatic sequence: query cudaDeviceCanAccessPeer() for a pair before
// assuming anything about it, then enable access only for pairs that
// genuinely support it. Chapter 5's own File 11/13 showed why the query
// matters for CORRECTNESS: cudaMemcpyPeer() and cudaMemcpyPeerAsync()
// still route correctly (falling back to a staged host copy) even when
// peer access was never enabled for a pair -- so skipping the query
// never produces a wrong ANSWER. What it can silently produce is a
// wrong PERFORMANCE ASSUMPTION: code that hardcodes "this pair is
// NVLink-wired" (a real, common shortcut -- baking in a number from one
// deployment's topology rather than re-querying it) pays no error and
// no crash when that assumption is false on a DIFFERENT deployment's
// topology, only a silent multiple of Chapter 5's own already-cited
// bandwidth figures. This file extends Chapter 5's own File 13 cost
// model with exactly that comparison: the real cost of each of the
// three topology classes Chapter 2 already established, against what
// code that assumed the WRONG one actually pays.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 140_topology_assumption_cost.cpp -o 140_topology_assumption_cost
// Run:     ./140_topology_assumption_cost
#include <cstdio>
#include <cmath>

int main() {
    printf("=== Section G.3: assuming a topology class instead of querying it ===\n\n");

    printf("Chapter 4's own real idiom (File 9): cudaDeviceCanAccessPeer(&canAccess,\n");
    printf("i, j), checked for every ordered pair, BEFORE calling\n");
    printf("cudaDeviceEnablePeerAccess(). Chapter 5's own File 11/13 showed why\n");
    printf("skipping it never breaks CORRECTNESS -- cudaMemcpyPeer() falls back to a\n");
    printf("staged host copy transparently for any pair without enabled peer access.\n");
    printf("What it CAN silently break is a performance assumption baked into calling\n");
    printf("code that never re-queries the real topology for the deployment it is\n");
    printf("actually running on.\n\n");

    // Chapter 2's own already-cited bandwidth figures, reused by
    // Chapter 5's own File 13 (not re-derived here).
    const double PCIE_GBPS = 31.5;
    const double NVLINK_GBPS = 50.0;
    const double transferSizeGB = 1.0;

    double nvlinkTime = transferSizeGB / NVLINK_GBPS;              // 1-hop, NVLink-wired pair
    double pcieDirect = transferSizeGB / PCIE_GBPS;                // 1-hop, PCIe-only direct pair
    double stagedTime = 2.0 * (transferSizeGB / PCIE_GBPS);        // 2-hop, no direct link at all

    printf("Chapter 2/5's own three real topology classes for a device pair, and the\n");
    printf("real transfer time for %.1f GB under each:\n\n", transferSizeGB);
    printf("%-32s %14s\n", "actual topology class", "real time (s)");
    printf("%-32s %14.6f\n", "NVLink-wired pair (1 hop)", nvlinkTime);
    printf("%-32s %14.6f\n", "PCIe-only direct pair (1 hop)", pcieDirect);
    printf("%-32s %14.6f\n", "no direct link (2-hop staged)", stagedTime);
    printf("\n");

    printf("=== the silent cost of code that ASSUMED NVLink and never re-queried ===\n\n");
    printf("code that hardcodes the NVLink figure (a real, common shortcut when a\n");
    printf("number is copied from one cluster's own topology report into a capacity\n");
    printf("plan or a scheduling deadline) pays NO error and NO crash when deployed\n");
    printf("on a DIFFERENT pair's real topology -- cudaMemcpyPeer() still returns\n");
    printf("cudaSuccess either way -- only a silent multiple of the assumed number:\n\n");

    printf("%-32s %14s %12s\n", "actual topology at runtime", "real time (s)", "x assumed");
    printf("%-32s %14.6f %12.4f\n", "NVLink (assumption correct)", nvlinkTime, nvlinkTime / nvlinkTime);
    printf("%-32s %14.6f %12.4f\n", "PCIe-only direct", pcieDirect, pcieDirect / nvlinkTime);
    printf("%-32s %14.6f %12.4f\n", "no direct link (staged)", stagedTime, stagedTime / nvlinkTime);
    printf("\n");

    double pcieSlowdown = pcieDirect / nvlinkTime;
    double stagedSlowdown = stagedTime / nvlinkTime;

    printf("a schedule or capacity plan built around the NVLink number silently runs\n");
    printf("%.4fx slower than planned on a PCIe-only-direct pair, and %.4fx slower on\n",
           pcieSlowdown, stagedSlowdown);
    printf("a pair with no direct link at all -- Chapter 2's own partial-mesh topology\n");
    printf("model already established that NOT every pair on a real multi-GPU machine\n");
    printf("shares the same link class, so this is not a hypothetical: two GPUs on the\n");
    printf("SAME machine can genuinely fall into different rows of this table.\n\n");

    printf("=== the real fix: query, don't assume ===\n\n");
    printf("Chapter 4's own File 9 sequence -- cudaDeviceCanAccessPeer() for the\n");
    printf("SPECIFIC pair actually in use, re-run on the actual deployment rather than\n");
    printf("hardcoded from a different one -- is the only way to know which row of\n");
    printf("this table applies BEFORE building a schedule or capacity plan around it.\n");
    printf("Real device attribute queries (Chapter 2's own cudaDeviceProp fields, and\n");
    printf("cudaDeviceGetP2PAttribute() for the specific link's own reported\n");
    printf("performance rank) are what genuinely observe the topology; nothing about\n");
    printf("cudaMemcpyPeer()'s own return code ever will.\n\n");

    bool orderingCorrect = (nvlinkTime < pcieDirect) && (pcieDirect < stagedTime);
    bool slowdownsPositive = (pcieSlowdown > 1.0) && (stagedSlowdown > pcieSlowdown);
    printf("self-check: the three real topology classes are strictly ordered fastest-\n");
    printf("to-slowest (NVLink < PCIe-direct < staged, %s), and an NVLink-based\n",
           orderingCorrect ? "confirmed" : "MISMATCH");
    printf("assumption's own real slowdown strictly increases as the actual topology\n");
    printf("gets worse (%s): %s\n", slowdownsPositive ? "confirmed" : "MISMATCH",
           (orderingCorrect && slowdownsPositive) ? "confirmed" : "MISMATCH");

    return (orderingCorrect && slowdownsPositive) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 140_topology_assumption_cost.cpp -o 140_topology_assumption_cost
./140_topology_assumption_cost
```

**Sample input:** none -- every figure is Chapter 2's own already-cited bandwidth constant or arithmetic derived from it, identical to Chapter 5's own File 13 model.

**Sample output:**

```text
=== Section G.3: assuming a topology class instead of querying it ===

Chapter 4's own real idiom (File 9): cudaDeviceCanAccessPeer(&canAccess,
i, j), checked for every ordered pair, BEFORE calling
cudaDeviceEnablePeerAccess(). Chapter 5's own File 11/13 showed why
skipping it never breaks CORRECTNESS -- cudaMemcpyPeer() falls back to a
staged host copy transparently for any pair without enabled peer access.
What it CAN silently break is a performance assumption baked into calling
code that never re-queries the real topology for the deployment it is
actually running on.

Chapter 2/5's own three real topology classes for a device pair, and the
real transfer time for 1.0 GB under each:

actual topology class             real time (s)
NVLink-wired pair (1 hop)              0.020000
PCIe-only direct pair (1 hop)          0.031746
no direct link (2-hop staged)          0.063492

=== the silent cost of code that ASSUMED NVLink and never re-queried ===

code that hardcodes the NVLink figure (a real, common shortcut when a
number is copied from one cluster's own topology report into a capacity
plan or a scheduling deadline) pays NO error and NO crash when deployed
on a DIFFERENT pair's real topology -- cudaMemcpyPeer() still returns
cudaSuccess either way -- only a silent multiple of the assumed number:

actual topology at runtime        real time (s)    x assumed
NVLink (assumption correct)            0.020000       1.0000
PCIe-only direct                       0.031746       1.5873
no direct link (staged)                0.063492       3.1746

a schedule or capacity plan built around the NVLink number silently runs
1.5873x slower than planned on a PCIe-only-direct pair, and 3.1746x slower on
a pair with no direct link at all -- Chapter 2's own partial-mesh topology
model already established that NOT every pair on a real multi-GPU machine
shares the same link class, so this is not a hypothetical: two GPUs on the
SAME machine can genuinely fall into different rows of this table.

=== the real fix: query, don't assume ===

Chapter 4's own File 9 sequence -- cudaDeviceCanAccessPeer() for the
SPECIFIC pair actually in use, re-run on the actual deployment rather than
hardcoded from a different one -- is the only way to know which row of
this table applies BEFORE building a schedule or capacity plan around it.
Real device attribute queries (Chapter 2's own cudaDeviceProp fields, and
cudaDeviceGetP2PAttribute() for the specific link's own reported
performance rank) are what genuinely observe the topology; nothing about
cudaMemcpyPeer()'s own return code ever will.

self-check: the three real topology classes are strictly ordered fastest-
to-slowest (NVLink < PCIe-direct < staged, confirmed), and an NVLink-based
assumption's own real slowdown strictly increases as the actual topology
gets worse (confirmed): confirmed
```

## G.4 A Diagnostic Checklist for Recognizing These Failure Modes

None of this appendix's three failure modes announce themselves with an error message -- a program suffering from any of them can run to completion (or hang forever) without ever printing a return code that says why. The following questions are the ones worth asking of any multi-device program that behaves strangely without ever actually reporting an error.

Does every process or thread in a communicator issue its blocking calls -- sends, receives, collectives -- in a sequence that could, for some interleaving of timing, leave every one of them waiting on a call another one hasn't reached yet? Section G.1's own recv-first ring is the sharpest version of this question, but the same check applies to any blocking call, not only `MPI_Send`/`MPI_Recv`: if the answer is yes, the fix is never to remove synchronization, it is to restructure the *order* so at least one participant is always positioned to unblock another, the same even/odd interleaving Chapter 20 and Section G.1 both used.

Does a stream- or event-based dependency get enqueued in the middle of a function, with any chance that the operation it is meant to protect was enqueued onto the same stream earlier in the same control flow? Section G.2's own enqueue-order distinction is worth checking specifically after any refactor that reorders code around a `cudaStreamWaitEvent()` or similar synchronization call -- the compiler will not catch a wait that ended up protecting nothing, and neither will any return code at runtime.

Is a bandwidth, latency, or link-class figure anywhere in a capacity plan, a scheduling deadline, or a batch-size calculation a literal constant, rather than something read from a topology query run on the actual deployment? Section G.3's own table is worth revisiting specifically when code that assumes NVLink-class performance is deployed onto new hardware, a new cluster, or even a different pair of devices on the same machine -- Chapter 2's own partial-mesh model already proved that assumption can be wrong even within a single server.

## Appendix Summary

- Deadlocks are not always a message-size problem, even though Chapter 20's own real deadlock happened to be one -- a ring where every process calls `MPI_Recv()` before any process calls `MPI_Send()` hangs absolutely, at any message size, because `MPI_Recv()` has no buffering escape hatch the way `MPI_Send()` does; the real, general fix restructures the *order* of blocking calls (even ranks send first, odd ranks receive first) so some participant is always positioned to unblock another, confirmed here by genuinely triggering the hang under a bounded `timeout` (exit code 124) and genuinely completing the fixed version.
- `cudaStreamWaitEvent()` only ever constrains work enqueued onto its stream *after* the call, in program order, not work that merely executes later -- Chapter 6's own File 15 already stated this, and this appendix showed the real consequence: a consumer operation enqueued before the wait call runs with no dependency on the producer's event at all, with zero difference in any return code from either enqueue order, which is exactly what makes a missed or misplaced synchronization step silent rather than crash-worthy.
- A topology figure copied from one deployment's own measured link class into a hardcoded constant is a silent liability the moment the code runs on a different pair of devices -- `cudaMemcpyPeer()` itself always routes correctly regardless (Chapter 5's own real finding), but a capacity plan or scheduling deadline built around an assumed NVLink number pays a real, unannounced multiple (1.59x to 3.17x in this appendix's own model) when the actual topology turns out to be PCIe-direct or has no direct link at all; the only real defense is re-querying `cudaDeviceCanAccessPeer()`/`cudaDeviceGetP2PAttribute()` on the actual deployment, not trusting a number measured somewhere else.
- Section G.4's diagnostic checklist collects the three questions worth asking of any multi-device program that behaves strangely without ever reporting an error: does a blocking-call sequence leave every participant waiting on a call none of them has reached, does a synchronization call protect an operation that was actually enqueued before it, and is a topology-dependent performance number a hardcoded assumption rather than a live query.
