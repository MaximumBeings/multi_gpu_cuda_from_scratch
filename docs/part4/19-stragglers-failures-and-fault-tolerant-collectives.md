# Chapter 19: Stragglers, Failures, and Fault-Tolerant Collectives

**What you will understand by the end of this chapter:**

- Why every real NCCL call this book has made since Chapter 11 has returned an instant, honest error code -- and why that is *not* how a real cluster's real peer failure behaves: NVIDIA's own documentation states plainly that even healthy ranks should expect NCCL to either error *or hang* on a collective.
- How to build a non-blocking NCCL communicator (`ncclConfig_t`, `config.blocking = 0`) and poll `ncclCommGetAsyncError()` in a bounded loop, so a hang becomes something you can detect instead of something you endure.
- Why `ncclCommAbort()` only stops a hang -- it does not repair or replace the communicator -- and what a real application still has to do afterward.
- How Open MPI's ULFM extension offers a genuinely richer recovery model, `MPIX_Comm_revoke()` and `MPIX_Comm_shrink()`, that NCCL has no equivalent for.
- The real "backup workers" technique (Chen et al., 2016) for tolerating stragglers with *no failure detection at all*, and the one invariant it guarantees.

**What you need to know first:**

- Chapter 11's real `ncclCommInitAll()` communicator and the honest error codes it returns in this environment.
- Chapter 17's real barrier (a throwaway `ncclAllReduce()` plus a required `cudaStreamSynchronize()`) -- this chapter's failures happen at exactly that kind of waiting point.
- Chapter 18's makespan framing: what a barrier costs when one rank is slower than the rest.

---

Chapter 18 asked what happens when every rank finishes, just not at the same time. This chapter asks the harder question Chapter 18 left alone: what happens when a rank doesn't finish at all. Every real CUDA and NCCL call this book has made, since Chapter 3, has returned an honest, *immediate* error code, because this environment has never had a real device to begin with -- `cudaErrorNoDevice`, `ncclUnhandledCudaError`, `ncclInvalidArgument`, always instant, always clean. It would be easy to let that instant-error pattern quietly stand in for what a real cluster does when a rank genuinely dies mid-collective. It's the wrong model. NVIDIA's own documentation on building fault-tolerant NCCL applications states the real failure mode plainly, and it is worse than an error: "even healthy ranks should expect NCCL to either return an error or hang on any collective operation." A rank that dies doesn't make the other ranks see an error -- it makes them wait, forever, at whatever collective or barrier they were sharing with it. This chapter builds the real detect-and-recover pattern NCCL actually offers for that, names what it doesn't offer (contrasted with MPI's own ULFM extension, previewing Chapter 20), and then covers a real, different, milder technique -- backup workers -- for the far more common case where a rank isn't dead, just slow.

```text
This book's own environment            A real cluster, one peer dies
(no device, ever):                     mid-collective:

  rank0 -> ncclAllReduce()                 rank0 -> ncclAllReduce() ... waiting
  rank0 -> ncclUnhandledCudaError          rank1 -> ncclAllReduce() ... waiting
           (instant, honest, code 1)       rank2 -> ncclAllReduce() ... waiting
                                            rank3 -> dead, never returns
                                            -----------------------------------
                                            rank0, rank1, rank2 HANG. Forever.

  Every chapter's own call, through        Section 19.1 names this real gap;
  Ch18, has looked like the LEFT           19.2 builds the real escape; 19.3
  column above.                            covers the milder case: SLOW, not dead.
```

## 19.1 The Problem: NCCL Hangs, Doesn't Error, When a Peer Genuinely Dies

### Intuition

Look back at every real NCCL call this book has made since Chapter 11: `ncclCommInitAll()` in Chapter 11 itself, `ncclAllReduce()` for gradient averaging in Chapter 12, for tensor-parallel combines in Chapter 14, for Chapter 17's own barrier. Every single one of them returned an honest, *immediate* `ncclResult_t` -- `ncclInvalidArgument`, `ncclUnhandledCudaError` -- because this environment has never had a real device for any of them to succeed on. That pattern is this environment's own honest failure mode, and it is worth naming precisely so it isn't mistaken for a real cluster's failure mode, which is genuinely different and, for an application trying to stay correct, genuinely worse. NVIDIA's own fault-tolerance documentation draws this exact distinction, contrasting an ordinary bad NCCL call (which does error out) with what happens when a peer in an otherwise-correct program actually dies: "even healthy ranks should expect NCCL to either return an error or hang on any collective operation." A rank calling `ncclAllReduce()` with a dead peer doesn't get `ncclUnhandledCudaError` back and move on -- it sits inside that call, waiting for a peer that will never respond, with no way to tell the difference between "still in progress" and "never coming back," because a blocking communicator's calls, by construction, don't return until they're done.

```text
Blocking communicator (every real NCCL call this book has made, Ch11-18):

  ncclCommInitRankConfig(blocking=1)  or  ncclAllReduce(...)  or  Ch17's barrier
  +----------------------------------------+
  | waiting ... waiting ... waiting ...    |   no way to poll, no way to time out
  +----------------------------------------+
  returns ... eventually. Or, with a genuinely dead peer, never.

Non-blocking communicator (this section names it; 19.2 builds the real escape):

  ncclCommInitRankConfig(blocking=0)
  -> returns IMMEDIATELY
     -> poll ncclCommGetAsyncError() in a loop,
        bounded by YOUR OWN timeout, not NCCL's.
```

### Background

Every communicator this book has built through Chapter 18 -- Chapter 11's `ncclCommInitAll()`, and every direct `ncclCommInitRank()`-style call implicit since -- is *blocking* by default: `config.blocking = 1`, whether or not any code ever set that field explicitly. This section builds both the default blocking communicator every earlier chapter used, and the real, documented alternative -- a non-blocking one, `config.blocking = 0` -- side by side, using NCCL's real `ncclConfig_t` struct and `NCCL_CONFIG_INITIALIZER` macro, confirmed present in the actual installed NCCL header (libnccl 2.18.3). Section 19.2 is what a non-blocking communicator like the second one below actually makes possible.

```cpp
// Chapter 19: Stragglers, Failures, and Fault-Tolerant Collectives
// 54_blocking_vs_nonblocking_comm.cu
//
// Every real NCCL call this book has made since Chapter 11 has
// returned an honest, IMMEDIATE error code -- ncclInvalidArgument,
// ncclUnhandledCudaError -- because this environment has never had a
// real device to begin with. A real cluster's real failure mode is
// different, and worse: NVIDIA's own documentation on building
// fault-tolerant NCCL applications states it plainly -- "even healthy
// ranks should expect NCCL to either return an error OR HANG on any
// collective operation." A rank that genuinely dies mid-collective
// doesn't make every other rank see an error; it makes every other
// rank hang, forever, at whatever collective (or Chapter 17's own
// barrier) they were waiting on with it. This section builds the real
// fix: a NON-BLOCKING communicator, configured with a real
// ncclConfig_t, so a hang can be DETECTED instead of endured.
// Genuinely compiled with a real nvcc against the real installed
// libnccl.
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    ncclUniqueId uniqueId;
    ncclResult_t eUniqueId = ncclGetUniqueId(&uniqueId);
    printf("\nncclGetUniqueId(): %s (code %d)\n",
           ncclGetErrorString(eUniqueId), (int)eUniqueId);

    // The DEFAULT communicator every earlier chapter's ncclCommInitAll()
    // implicitly built: blocking. Every NCCL call on it either finishes
    // or, on a real cluster with a genuinely dead peer, never returns
    // at all -- there is no way to poll it, no way to time it out.
    ncclConfig_t blockingConfig = NCCL_CONFIG_INITIALIZER;
    blockingConfig.blocking = 1;
    ncclComm_t blockingComm = nullptr;
    ncclResult_t eBlocking = ncclCommInitRankConfig(&blockingComm, 1, uniqueId, 0, &blockingConfig);
    printf("ncclCommInitRankConfig(blocking=1): %s (code %d)\n",
           ncclGetErrorString(eBlocking), (int)eBlocking);

    // The FIX: a real, documented NCCL feature -- config.blocking = 0.
    // NVIDIA's own words: "NCCL communicators can be configured to be
    // non-blocking so that initialization functions may continue in
    // the background," specifically "so that we may detect and react
    // to timeouts." Section 19.2 uses exactly this communicator.
    ncclConfig_t nonBlockingConfig = NCCL_CONFIG_INITIALIZER;
    nonBlockingConfig.blocking = 0;
    ncclComm_t nonBlockingComm = nullptr;
    ncclResult_t eNonBlocking = ncclCommInitRankConfig(&nonBlockingComm, 1, uniqueId, 0, &nonBlockingConfig);
    printf("ncclCommInitRankConfig(blocking=0): %s (code %d)\n",
           ncclGetErrorString(eNonBlocking), (int)eNonBlocking);

    printf("\nBoth calls above return immediately in THIS environment, "
           "honestly reporting that initialization on a never-successful "
           "communicator failed to progress. That's this environment's "
           "own honest failure mode (no device, ever) -- it is NOT the "
           "same as a real cluster's hang-on-peer-failure mode, which "
           "this section's whole point is to name correctly rather than "
           "let this book's own always-instant errors quietly stand in "
           "for it. Section 19.2 builds the real detect-and-recover "
           "sequence a non-blocking communicator like the second one "
           "above makes possible.\n");

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

ncclGetUniqueId(): no error (code 0)
ncclCommInitRankConfig(blocking=1): unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)
ncclCommInitRankConfig(blocking=0): unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)

Both calls above return immediately in THIS environment, honestly reporting that initialization on a never-successful communicator failed to progress. That's this environment's own honest failure mode (no device, ever) -- it is NOT the same as a real cluster's hang-on-peer-failure mode, which this section's whole point is to name correctly rather than let this book's own always-instant errors quietly stand in for it. Section 19.2 builds the real detect-and-recover sequence a non-blocking communicator like the second one above makes possible.
```

`ncclGetUniqueId()` succeeds outright -- it needs no device at all, just a host-side handle to distribute to every rank before any communicator exists. Both `ncclCommInitRankConfig()` calls fail identically here, `blocking=1` and `blocking=0` alike, and that's the honest, important detail: the `blocking` field changes what a caller is *allowed to do while waiting*, not whether initialization can succeed without a device. In this environment, with no device to ever succeed on, that distinction is invisible -- both calls report the same instant `code 1`. On a real cluster with a real, live peer that later dies, the same two configurations diverge sharply: the blocking one leaves every healthy rank stuck inside the call with nothing to do but wait; the non-blocking one returns control immediately, so the caller can go poll for the answer on its own terms. That divergence is Section 19.2's whole subject.

!!! warning "[COMMON TRAP] Assuming a caught NCCL error means the failure has been handled"
    Every earlier chapter's own honest error codes -- `ncclInvalidArgument`, `ncclUnhandledCudaError` -- made it easy to build a habit: call an NCCL function, check its `ncclResult_t`, print or handle the error, move on. That habit works perfectly in this book's own environment, where every real call fails instantly, every single time. It does not generalize to a real cluster. NVIDIA's own documentation is explicit that a genuinely failed peer can make an *otherwise-correct* collective call simply never return, on a *blocking* communicator -- there is no `ncclResult_t` to check, caught or otherwise, because the call that would produce one hasn't come back. Checking a return code handles the errors this book has shown you throughout; it says nothing at all about the hangs a real dead peer causes, which is exactly why Section 19.2's non-blocking-plus-polling pattern exists as a genuinely separate mechanism, not a refinement of error-checking.

## 19.2 Detecting and Escaping a Hang: Non-Blocking Communicators and `ncclCommAbort()`

### Intuition

A non-blocking communicator on its own only solves half the problem Section 19.1 named: it stops a caller from being trapped *inside* a call, but it doesn't yet tell the caller whether initialization -- or any later collective -- actually succeeded, failed, or is still working. NVIDIA's own documentation names the missing piece directly: a non-blocking communicator lets an application "enable non-blocking NCCL communicator so that we may detect and react to timeouts," and the mechanism for that detection is a second real call, `ncclCommGetAsyncError()`, checked in a loop. Their own example code shows exactly this pattern -- `NCCL_CHECK(ncclCommGetAsyncError(comm, &asyncError)); while (asyncError == ncclInProgress)` -- polling until the result stops being "still working" and becomes something definite. Once it's definite and bad, NVIDIA's documentation names the one way out in equally direct terms: "typical recovery for the healthy ranks starts with `ncclCommAbort` on the existing communicator," which "will exit any operation currently in progress, and destroy the communicator." Abort is not repair. It stops the hang and nothing more -- a real application still has to build a fresh communicator afterward, almost always excluding the rank that caused the trouble. Open MPI's own ULFM (User-Level Failure Mitigation) extension, previewed here and built for real in Chapter 20, offers a strictly richer version of this same idea: `MPIX_Comm_revoke()`, which "interrupts any communication pending on the communicator at all ranks" in one call, and `MPIX_Comm_shrink()`, which then "creates a new communicator where dead processes ... were removed" -- survivors keep going in a smaller world, with no full rebuild required. NCCL has nothing like `shrink()`; its own abort-and-rebuild cycle involves every surviving rank from scratch.

```text
NCCL's real escape (this section):          MPI ULFM's real escape (Ch20 preview):

  detect: poll ncclCommGetAsyncError()          MPIX_Comm_revoke()
  -> ncclCommAbort()                            -> interrupts pending comm on
     stops the hang, nothing more                  EVERY surviving rank at once
  -> rebuild: ncclCommInitRankConfig()          MPIX_Comm_shrink()
     again, from scratch, ALL ranks             -> builds a SMALLER communicator,
                                                    dead rank excluded, survivors
                                                    keep going -- no full rebuild
```

### Background

`pollUntilSettled()` below is this book's own version of NVIDIA's documented polling loop, bounded by a maximum poll count standing in for a real wall-clock timeout, exactly the way Section 19.1 named it. It calls `ncclCommGetAsyncError()` on the non-blocking communicator Section 19.1 introduced, then, once settled, calls the real `ncclCommAbort()` escape.

```cpp
// Chapter 19: Stragglers, Failures, and Fault-Tolerant Collectives
// 55_nccl_async_detect_and_recover.cu
//
// Section 19.1 named the real problem: a blocking NCCL communicator
// gives an application no way to notice a hung collective, because
// the call that would tell it so is itself the thing that's hanging.
// This file builds NVIDIA's own real, documented fix in full: create
// the communicator NON-BLOCKING (config.blocking = 0), then instead
// of blocking on the init/collective call itself, POLL
// ncclCommGetAsyncError() in a bounded loop -- NVIDIA's own example
// code shows exactly this pattern: "NCCL_CHECK(ncclCommGetAsyncError(
// comm, &asyncError)); while (asyncError == ncclInProgress)". Once a
// real problem is detected, NVIDIA's docs describe the escape
// directly: "Typical recovery for the healthy ranks starts with
// ncclCommAbort on the existing communicator, followed by
// ncclCommInit" -- ncclCommAbort() is not recovery on its own; it
// "will exit any operation currently in progress, and destroy the
// communicator." The application still has to rebuild a fresh
// communicator afterward (not shown here -- that's just
// ncclCommInitRankConfig() again, already built in 19.1).
// Genuinely compiled with a real nvcc against the real installed
// libnccl.
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

// The real polling pattern NVIDIA's fault-tolerance docs describe:
// call ncclCommGetAsyncError() in a loop, bounded by a max iteration
// count standing in for a real wall-clock timeout, until the
// communicator reports something other than "still in progress."
ncclResult_t pollUntilSettled(ncclComm_t comm, int maxPolls, int* pollsUsed) {
    ncclResult_t asyncErr = ncclInProgress;
    int i = 0;
    for (; i < maxPolls; ++i) {
        ncclResult_t queryErr = ncclCommGetAsyncError(comm, &asyncErr);
        if (queryErr != ncclSuccess) {
            // The query call itself failed -- e.g. comm was never a
            // valid handle to begin with, this environment's own
            // honest case, since ncclCommInitRankConfig() below never
            // completes without a device.
            *pollsUsed = i + 1;
            return queryErr;
        }
        if (asyncErr != ncclInProgress) {
            *pollsUsed = i + 1;
            return asyncErr;
        }
        // A real caller would sleep briefly here before polling
        // again; this book's own honest environment never reaches a
        // second iteration, so there's nothing to genuinely sleep on.
    }
    *pollsUsed = i;
    return ncclInProgress; // exhausted the poll budget -- a real timeout
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    ncclUniqueId uniqueId;
    ncclGetUniqueId(&uniqueId);

    // Build the non-blocking communicator 19.1 introduced.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    ncclComm_t comm = nullptr;
    ncclResult_t eInit = ncclCommInitRankConfig(&comm, 1, uniqueId, 0, &config);
    printf("\nncclCommInitRankConfig(blocking=0): %s (code %d)\n",
           ncclGetErrorString(eInit), (int)eInit);
    printf("On a non-blocking communicator this call is allowed to return "
           "BEFORE initialization has actually finished -- that's the whole "
           "point, so the caller gets control back immediately instead of "
           "blocking. Whether init is really done yet is a separate "
           "question, answered only by polling ncclCommGetAsyncError(), "
           "never by this return value alone.\n");

    // The real detect step: poll for up to MAX_POLLS iterations, the
    // way a real application would poll for up to some wall-clock
    // timeout while a peer might be hanging.
    const int MAX_POLLS = 1000;
    int pollsUsed = 0;
    ncclResult_t settled = pollUntilSettled(comm, MAX_POLLS, &pollsUsed);
    printf("\npollUntilSettled(): settled after %d/%d poll(s) with result "
           "%s (code %d)\n",
           pollsUsed, MAX_POLLS, ncclGetErrorString(settled), (int)settled);

    if (pollsUsed < MAX_POLLS) {
        printf("The very first poll already saw a settled (non-"
               "ncclInProgress) result -- this environment's own honest "
               "failure mode is immediate, not a hang, so there was never "
               "anything for this loop to genuinely wait out. On a real "
               "cluster with a genuinely dead peer, ncclCommGetAsyncError() "
               "keeps returning ncclInProgress for as long as the peer "
               "stays dead, and THIS loop is what turns that into a bounded "
               "wait instead of an unbounded hang.\n");
    } else {
        printf("Exhausted the poll budget without settling -- on a real "
               "cluster this is the real timeout firing: a peer that "
               "never comes back. Either way, settled or timed out, the "
               "next step is the same.\n");
    }

    // The real escape: NVIDIA's own docs state it directly -- "Typical
    // recovery for the healthy ranks starts with ncclCommAbort on the
    // existing communicator, followed by ncclCommInit." Abort is not
    // recovery; it just stops the hang so the process can move on.
    ncclResult_t eAbort = ncclCommAbort(comm);
    printf("\nncclCommAbort(): %s (code %d)\n",
           ncclGetErrorString(eAbort), (int)eAbort);
    printf("ncclCommAbort() only guarantees that this call returns -- it "
           "does NOT rebuild a working communicator. A real application "
           "still has to call ncclCommInitRankConfig() again afterward "
           "(19.1's own call, unchanged) to get back into a workable "
           "state, almost always with the dead rank excluded from the new "
           "membership list.\n");

    printf("\nOpen MPI's ULFM (User-Level Failure Mitigation) extension "
           "offers a real, richer alternative to this abort-and-rebuild "
           "cycle: MPIX_Comm_revoke() invalidates a communicator across "
           "every surviving rank at once (so nobody is left waiting on a "
           "peer that already knows to give up), and MPIX_Comm_shrink() "
           "then produces a NEW communicator that simply excludes the "
           "dead rank -- survivors keep going with a smaller world, no "
           "full rebuild required. NCCL has no equivalent to shrink(): "
           "ncclCommAbort() only stops the hang, and rebuilding an NCCL "
           "communicator to exclude one dead rank means calling "
           "ncclCommInitRankConfig() again with an entirely new rank "
           "count and a new ncclUniqueId, involving every surviving rank "
           "from scratch. Chapter 20 opens Part 5 with MPI's own first "
           "dedicated chapter, where ULFM gets built and run directly, "
           "not just described.\n");

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

ncclCommInitRankConfig(blocking=0): unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)
On a non-blocking communicator this call is allowed to return BEFORE initialization has actually finished -- that's the whole point, so the caller gets control back immediately instead of blocking. Whether init is really done yet is a separate question, answered only by polling ncclCommGetAsyncError(), never by this return value alone.

pollUntilSettled(): settled after 1/1000 poll(s) with result invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)
The very first poll already saw a settled (non-ncclInProgress) result -- this environment's own honest failure mode is immediate, not a hang, so there was never anything for this loop to genuinely wait out. On a real cluster with a genuinely dead peer, ncclCommGetAsyncError() keeps returning ncclInProgress for as long as the peer stays dead, and THIS loop is what turns that into a bounded wait instead of an unbounded hang.

ncclCommAbort(): no error (code 0)
ncclCommAbort() only guarantees that this call returns -- it does NOT rebuild a working communicator. A real application still has to call ncclCommInitRankConfig() again afterward (19.1's own call, unchanged) to get back into a workable state, almost always with the dead rank excluded from the new membership list.

Open MPI's ULFM (User-Level Failure Mitigation) extension offers a real, richer alternative to this abort-and-rebuild cycle: MPIX_Comm_revoke() invalidates a communicator across every surviving rank at once (so nobody is left waiting on a peer that already knows to give up), and MPIX_Comm_shrink() then produces a NEW communicator that simply excludes the dead rank -- survivors keep going with a smaller world, no full rebuild required. NCCL has no equivalent to shrink(): ncclCommAbort() only stops the hang, and rebuilding an NCCL communicator to exclude one dead rank means calling ncclCommInitRankConfig() again with an entirely new rank count and a new ncclUniqueId, involving every surviving rank from scratch. Chapter 20 opens Part 5 with MPI's own first dedicated chapter, where ULFM gets built and run directly, not just described.
```

`pollUntilSettled()` settles on its very first iteration here -- `ncclCommGetAsyncError()` reports `ncclInvalidArgument` (code 4) immediately, because `comm` was never a genuinely valid handle to begin with in an environment with no device. That is, once again, this environment's own honest failure mode standing in the same spot a real hang would occupy: the loop itself is real and correct either way, and on a real cluster with a real hanging peer it is precisely this loop -- not this book's own instant result -- that would keep running, `ncclInProgress`, iteration after iteration, until either the peer recovers or the poll budget runs out. The most interesting single line in this output is `ncclCommAbort(): no error (code 0)`: NCCL genuinely reports success at aborting a communicator that never finished initializing, which is consistent with its own documented behavior -- `ncclCommAbort()`'s job is only to make sure nothing is left in progress and the handle is cleaned up, not to judge whether that communicator was ever healthy to begin with.

!!! warning "[COMMON TRAP] Treating `ncclCommAbort()` as recovery instead of just an escape"
    It's tempting to read `ncclCommAbort()`'s clean `ncclSuccess` return and treat the problem as solved. It isn't. NVIDIA's own documentation is explicit that abort's whole job is narrower: it "will exit any operation currently in progress, and destroy the communicator" -- stopping the hang, and nothing more. After `ncclCommAbort()` returns, there is no working communicator at all; every rank that was part of it needs a brand-new `ncclCommInitRankConfig()` call, Section 19.1's own call, run again from scratch, almost always with the dead rank's slot removed from the membership list and a fresh `ncclUniqueId`. Skipping that rebuild and trying to keep using the aborted handle, or assuming the abort itself somehow restored connectivity, leaves an application with no communicator, not a repaired one -- exactly the gap MPI's ULFM `Comm_shrink()` closes, by producing the smaller, still-working communicator directly instead of leaving that rebuild step to the caller.

## 19.3 Stragglers Are Not Failures: Backup Workers

### Intuition

Sections 19.1 and 19.2 dealt with a peer that is genuinely, permanently dead. That's the rarer, more catastrophic case. The far more common real failure mode in a large cluster is milder: a worker that's merely *slow* on this particular step -- a noisy neighbor process stealing CPU time, a transient network hiccup, one slow disk read -- and then back to normal speed on the very next step. Chen et al.'s "Revisiting Distributed Synchronous SGD" studies exactly this case, and their fix needs no failure detector, no polling loop, and no `ncclCommAbort()` at all. Their own description of it is direct: "instead of having only N workers, we add b extra workers, but as soon as the parameter servers receive gradients from any N workers, they stop waiting and update their parameters using the N gradients." Whichever workers (up to b of them) haven't reported yet are simply dropped, for that step only -- no error, no abort, no rebuild. Their own real experiments used this at real scale, reporting `N = 96, b = 4` as a strong configuration out of 100 total machines. The technique's own real guarantee, checked concretely in this section, is an order-statistic one: pooling N+b candidates and keeping the fastest N of them can only ever finish sooner (or, in the worst case, at the same time) than being stuck with one fixed set of N candidates -- for any timings at all, straggler or no straggler.

```text
N=4 required + b=2 backup workers, one step:

  required: [ 10 ][ 11 ][ 9 ][ 1000000 ]   one of these is a catastrophic straggler
  backup:   [ 12 ][ 13 ]

  Pool all 6, sort by finish time:
    9 -- 10 -- 11 -- 12 -- 13 -- 1000000
                 ^ first 4 finishers -- proceed HERE, drop the rest

  without backups: must wait for 1000000
  with backups:    proceeds at 12
```

### Background

`withoutBackupsWait()` below reproduces every earlier chapter's own implicit assumption -- exactly N required workers, wait for all of them. `withBackupsWait()` implements Chen et al.'s real technique: pool N required and b backup finish times together, sort them, and take the N-th smallest -- the finish time of whichever worker arrives N-th out of all N+b candidates. Three fixed, named scenarios check the invariant directly rather than assuming it.

```cpp
// Chapter 19: Stragglers, Failures, and Fault-Tolerant Collectives
// 56_backup_workers_simulation.cpp
//
// Plain host C++ -- this chapter's real verification for Section
// 19.3. Sections 19.1-19.2 dealt with a peer that's fully DEAD.
// Chen et al.'s "Revisiting Distributed Synchronous SGD" (arXiv
// 1604.00981) studies a milder, far more common real failure mode
// instead: a peer that's merely SLOW -- a straggler -- on any given
// step, for reasons (a noisy neighbor VM, a transient network hiccup,
// a slow disk read) that mostly don't repeat next step. Their real
// fix needs no failure detector at all: launch N+b workers for N
// required results, and proceed the moment the FIRST N of them
// report, dropping whichever b (or fewer) stragglers haven't finished
// yet. This file simulates that rule directly on fixed, named
// timings (no randomness -- every number below is chosen to make one
// concrete point) and checks, across every scenario, the one
// invariant Chen et al.'s own technique promises: waiting for backups
// is NEVER slower than waiting for the fixed required set alone.
#include <cstdio>
#include <vector>
#include <algorithm>

// The baseline every earlier chapter's own collectives implicitly
// assumed: exactly N required workers, no backups. One slow (or
// effectively hung) worker among the N holds up the whole step --
// this is precisely Chapter 17's own barrier cost, paid again here
// per-step instead of once at a global sync.
double withoutBackupsWait(const std::vector<double>& requiredTimes) {
    double worst = 0.0;
    for (double t : requiredTimes) worst = std::max(worst, t);
    return worst;
}

// Chen et al.'s real technique: launch requiredTimes.size() + backup-
// Times.size() workers, and proceed once the FIRST requiredTimes.size()
// of ALL of them (required and backup, pooled together) have reported
// -- the n-th order statistic of the pooled finishing times. Whichever
// workers haven't finished yet (up to backupTimes.size() of them) are
// simply dropped for this step.
double withBackupsWait(const std::vector<double>& requiredTimes,
                        const std::vector<double>& backupTimes) {
    std::vector<double> pooled = requiredTimes;
    pooled.insert(pooled.end(), backupTimes.begin(), backupTimes.end());
    std::sort(pooled.begin(), pooled.end());
    size_t n = requiredTimes.size();
    return pooled[n - 1]; // n-th smallest, 0-indexed as [n-1]
}

struct Scenario {
    const char* name;
    std::vector<double> requiredTimes; // N=4 required workers' finish times
    std::vector<double> backupTimes;   // b=2 backup workers' finish times
    const char* explanation;
};

int main() {
    std::vector<Scenario> scenarios = {
        {
            "no true straggler",
            {10.0, 11.0, 9.0, 12.0},
            {10.5, 11.5},
            "Nothing is actually hung -- every one of the 6 workers "
            "finishes in ordinary time. Even here, backups still help: "
            "the 4th-fastest of all 6 (11.0) beats the slowest of the "
            "4 originally-required ones (12.0), purely from having two "
            "extra chances at the draw."
        },
        {
            "a required worker catastrophically straggles",
            {10.0, 11.0, 9.0, 1000000.0},
            {12.0, 13.0},
            "One of the 4 ORIGINALLY-REQUIRED workers is the real "
            "target case -- effectively hung (a straggler so slow it "
            "may as well be Section 19.1's dead peer). Without "
            "backups, the whole step is held hostage to it. With 2 "
            "backups in flight, the pool has 4 OTHER finishers well "
            "under the straggler's time, so the step proceeds without "
            "ever waiting on it at all."
        },
        {
            "slow backups, no downside",
            {10.0, 11.0, 9.0, 12.0},
            {500.0, 600.0},
            "The backups themselves are the slow ones this time, far "
            "slower than any required worker. That costs nothing: the "
            "first 4 finishers are still the 4 required workers "
            "(9,10,11,12), so the backups' own results are simply "
            "dropped, unused, exactly as designed."
        },
    };

    printf("Section 19.3: backup workers, N=4 required + b=2 backup, "
           "3 fixed scenarios.\n\n");

    bool allInvariantsHold = true;
    for (const Scenario& s : scenarios) {
        double without = withoutBackupsWait(s.requiredTimes);
        double with = withBackupsWait(s.requiredTimes, s.backupTimes);
        bool invariantHolds = (with <= without);
        allInvariantsHold = allInvariantsHold && invariantHolds;

        printf("Scenario: %s\n", s.name);
        printf("  required times: [%.1f, %.1f, %.1f, %.1f]   backup times: [%.1f, %.1f]\n",
               s.requiredTimes[0], s.requiredTimes[1], s.requiredTimes[2], s.requiredTimes[3],
               s.backupTimes[0], s.backupTimes[1]);
        printf("  without backups (max of the 4 required):        %.1f\n", without);
        printf("  with backups (4th-fastest of all 6, pooled):     %.1f\n", with);
        printf("  with <= without: %s\n", invariantHolds ? "PASS" : "FAIL");
        printf("  %s\n\n", s.explanation);
    }

    printf("Invariant across every scenario above -- waiting for backups "
           "is never slower than waiting for the fixed required set "
           "alone -- holds in all %zu/%zu cases: %s\n",
           scenarios.size(), scenarios.size(), allInvariantsHold ? "PASS" : "FAIL");

    if (allInvariantsHold) {
        printf("\nThis is exactly Chen et al.'s own real claim, checked "
               "here on fixed numbers rather than assumed: taking the "
               "n-th order statistic of a LARGER pool can only ever be "
               "less than or equal to the max of one FIXED subset of "
               "that same pool's own n members, for any numbers at all "
               "-- adding more candidates and picking the best n of them "
               "never makes the n-th-best candidate's time worse.\n");
    }

    return allInvariantsHold ? 0 : 1;
}
```

Genuinely compiled with a real `g++` and genuinely run, re-verified identical on both this book's real toolchains (no floating-point exact-bit comparison here -- only `<=` comparisons and a `max` -- so no `-ffp-contract=off` flag was needed). Locked output, deterministic across repeated runs:

```
Section 19.3: backup workers, N=4 required + b=2 backup, 3 fixed scenarios.

Scenario: no true straggler
  required times: [10.0, 11.0, 9.0, 12.0]   backup times: [10.5, 11.5]
  without backups (max of the 4 required):        12.0
  with backups (4th-fastest of all 6, pooled):     11.0
  with <= without: PASS
  Nothing is actually hung -- every one of the 6 workers finishes in ordinary time. Even here, backups still help: the 4th-fastest of all 6 (11.0) beats the slowest of the 4 originally-required ones (12.0), purely from having two extra chances at the draw.

Scenario: a required worker catastrophically straggles
  required times: [10.0, 11.0, 9.0, 1000000.0]   backup times: [12.0, 13.0]
  without backups (max of the 4 required):        1000000.0
  with backups (4th-fastest of all 6, pooled):     12.0
  with <= without: PASS
  One of the 4 ORIGINALLY-REQUIRED workers is the real target case -- effectively hung (a straggler so slow it may as well be Section 19.1's dead peer). Without backups, the whole step is held hostage to it. With 2 backups in flight, the pool has 4 OTHER finishers well under the straggler's time, so the step proceeds without ever waiting on it at all.

Scenario: slow backups, no downside
  required times: [10.0, 11.0, 9.0, 12.0]   backup times: [500.0, 600.0]
  without backups (max of the 4 required):        12.0
  with backups (4th-fastest of all 6, pooled):     12.0
  with <= without: PASS
  The backups themselves are the slow ones this time, far slower than any required worker. That costs nothing: the first 4 finishers are still the 4 required workers (9,10,11,12), so the backups' own results are simply dropped, unused, exactly as designed.

Invariant across every scenario above -- waiting for backups is never slower than waiting for the fixed required set alone -- holds in all 3/3 cases: PASS

This is exactly Chen et al.'s own real claim, checked here on fixed numbers rather than assumed: taking the n-th order statistic of a LARGER pool can only ever be less than or equal to the max of one FIXED subset of that same pool's own n members, for any numbers at all -- adding more candidates and picking the best n of them never makes the n-th-best candidate's time worse.
```

The middle scenario is the one worth sitting with: without backups, the step's makespan is `1000000.0` -- Chapter 18's own worst-case makespan, taken to an extreme. With two backups in flight, the very same catastrophic straggler costs the step nothing at all; the pool's 4th-fastest finisher is `12.0`, and the straggler is simply never waited on. That's the real payoff Chen et al. report at scale (`N=96, b=4`, out of 100 total machines): a small, fixed number of extra workers absorbs almost any single straggler, without ever having to detect, classify, or recover from anything -- the technique's whole mechanism is "don't wait past the N-th finisher," full stop.

!!! warning "[COMMON TRAP] Treating backup workers as a fault-tolerance technique, not a straggler-mitigation one"
    Backup workers and Sections 19.1-19.2's detect-and-recover pattern solve two genuinely different problems, and it's easy to conflate them because both involve "extra" workers and both involve *not waiting* for someone. Chen et al.'s technique tolerates a worker being *slow on this particular step* -- it assumes, implicitly, that a straggler this step is very likely to be a normal, healthy worker again next step, which is exactly why dropping its result just this once is safe. It does not detect, and cannot recover from, a worker that is genuinely and permanently dead: with a fixed budget of `b` backups, a required worker that never comes back consumes one unit of that budget every single step, forever, and the moment the number of permanently-dead workers exceeds `b`, this technique degrades right back into Section 19.1's own hang -- there simply aren't enough live candidates left to reach `N`. A real system needs both: backup workers to absorb the common case (transient slowness) cheaply, and Section 19.2's real detect-and-abort-and-rebuild machinery (or MPI ULFM's richer revoke-and-shrink) for the rarer case backup workers was never designed to solve.

## Chapter Summary

This chapter completed Part 4's arc from "every rank is equally fast and always present" (the silent assumption behind every partitioning function since Chapter 12) to "a rank can be absent entirely." Section 19.1 named the real, and worse, failure mode a genuinely dead peer causes on a real cluster: not the instant, honest error this book's every device-less call has produced since Chapter 3, but a hang, on any collective, at any rank still healthy enough to be waiting -- NVIDIA's own documentation states this plainly for NCCL. Section 19.2 built the real fix NCCL actually offers: a non-blocking communicator (`ncclConfig_t`, `config.blocking = 0`), polled via `ncclCommGetAsyncError()` in a bounded loop instead of blocked on directly, escaped via `ncclCommAbort()` once a real problem is detected -- while being explicit that abort only stops the hang, it doesn't repair anything, and a full communicator rebuild (all ranks, from scratch) is still required afterward. That section also named a real, richer alternative this book hasn't built yet: MPI's ULFM extension, whose `MPIX_Comm_revoke()` and `MPIX_Comm_shrink()` let survivors continue in a smaller communicator without a full rebuild, previewing Chapter 20. Section 19.3 then covered a genuinely different, milder, and far more common case -- a worker that's merely slow, not dead -- via Chen et al.'s real backup-workers technique, verified directly against a fixed, checked invariant: pooling more candidates and keeping the fastest N of them is never slower than waiting on one fixed set of N, for any timings at all. This completes Part 4 (Chapters 17-19: barriers, load balancing, and fault tolerance). Chapter 20 opens Part 5 by leaving single-node NCCL behind for MPI's own first dedicated chapter -- built, not just referenced, and including a real, working look at the ULFM extension this chapter could only preview.

## Self-Check Questions

1. Quote NVIDIA's own documented distinction between an ordinary NCCL error and what a genuinely dead peer causes on a *blocking* communicator, and explain in your own words why checking a call's `ncclResult_t` cannot catch the second case.
2. What does setting `config.blocking = 0` actually change about when `ncclCommInitRankConfig()` returns, and what does it *not* change about whether initialization has actually finished?
3. In Section 19.2's own locked output, `ncclCommAbort()` reports `no error (code 0)` even though the communicator it aborted never successfully initialized. Explain why that's consistent with `ncclCommAbort()`'s own documented job, rather than a bug.
4. Name the one real capability Open MPI's ULFM extension offers (via `MPIX_Comm_shrink()`) that NCCL's own `ncclCommAbort()` + rebuild cycle does not.
5. Using Section 19.3's own `withBackupsWait()` function, explain in your own words why pooling N+b candidates and taking the N-th smallest can never produce a *worse* (larger) result than `withoutBackupsWait()`'s max of one fixed set of N, for any input timings at all.
6. Section 19.3's third scenario makes the backup workers themselves the slowest of all 6 candidates, and the "with backups" result is unchanged from the baseline. Why does that happen, and what does it say about the actual cost of adding backup workers when no one straggles?
7. Explain the COMMON TRAP in Section 19.3 in your own words: why can't a fixed budget of `b` backup workers substitute for Section 19.2's detect-and-recover machinery when a required worker is genuinely, permanently dead rather than merely slow?
8. This chapter cites Chen et al.'s own real configuration, `N=96, b=4` out of 100 total machines. If 5 of those 100 workers die permanently (not just straggle), what happens to the system's ability to reach 96 reports on any future step, and why does that outcome resemble Section 19.1's hang rather than the designed, harmless straggler-dropping this section demonstrated?

## Where We Go Next

Chapter 20 opens Part 5 by leaving single-node NCCL behind for the first time in this book: MPI, built from scratch and then made CUDA-aware, for scaling a job beyond one node entirely. It also returns to two things this chapter could only preview -- MPI's own dedicated `MPI_Barrier()` (contrasted with Chapter 17's throwaway-`ncclAllReduce()` workaround) and, most directly following from this chapter, ULFM's real `MPIX_Comm_revoke()` and `MPIX_Comm_shrink()`, built and run for real rather than just quoted.

## Worked Solutions

**1.** NVIDIA's documentation states: "even healthy ranks should expect NCCL to either return an error or hang on any collective operation." Checking a call's `ncclResult_t` only catches the first branch of that either/or -- a call that *returns* with a bad code. A hang, by definition, is a call that has not returned at all; there is no `ncclResult_t` value to inspect yet, caught or otherwise, because the call itself hasn't produced one. Error-checking and hang-detection are two genuinely separate mechanisms for exactly this reason.

**2.** Setting `config.blocking = 0` changes *when the call returns*: a non-blocking `ncclCommInitRankConfig()` is allowed to return control to the caller before initialization has actually finished, instead of blocking until it's done (or never returning, if a peer is dead). It does *not* change whether initialization has actually completed by the time the call returns -- that's a separate fact, which the caller has to check afterward, in a loop, via `ncclCommGetAsyncError()`.

**3.** `ncclCommAbort()`'s own documented job is narrow: it "will exit any operation currently in progress, and destroy the communicator." That job doesn't require the communicator to have ever been healthy or fully initialized -- aborting a handle that never finished setting up is still a well-defined, successful operation (there's nothing in progress to exit, and the handle can still be destroyed cleanly), which is exactly why NCCL reports `no error` for it here rather than treating the earlier failed initialization as somehow making abort itself fail too.

**4.** ULFM's `MPIX_Comm_shrink()` produces a new, smaller communicator that excludes the dead process directly, letting survivors continue without a full rebuild. NCCL's own `ncclCommAbort()` only stops the hang -- it destroys the communicator entirely, requiring every surviving rank to call `ncclCommInitRankConfig()` again from scratch (a new rank count, a new `ncclUniqueId`) to get back to a workable state, which is a strictly heavier-weight recovery than ULFM's shrink-in-place.

**5.** `withBackupsWait()` computes the N-th smallest value out of a pool of N+b candidates (the required workers plus the backups), while `withoutBackupsWait()` computes the max (the N-th, and last, smallest value) of one *fixed* subset of exactly N of those same kinds of candidates. Adding more candidates to choose from and then keeping the best (smallest) N of the *larger* pool can only ever match or improve on the max of any one fixed subset of size N drawn from a similar distribution, because the larger pool's own N-th-smallest value is, at worst, equal to the smaller pool's N-th-smallest (its max) -- extra candidates can only ever displace a slower one from the "first N to finish" group, never force a faster one out.

**6.** Because the four required workers alone already finish in `9.0, 10.0, 11.0, 12.0` -- ordinary time -- and the two (very slow) backups at `500.0, 600.0` never rank among the fastest four out of all six. `withBackupsWait()`'s pooled sort places both backups at the very end, past the four required finishers, so they're the ones "dropped" for this step, not the required workers. This shows the actual cost of adding backup workers, when nothing goes wrong, is exactly zero to the step's own wait time -- the backups' work is simply wasted (extra compute spent, unused), never a wasted step.

**7.** Chen et al.'s technique tolerates a worker being *slow on this specific step*, implicitly assuming it's likely to be a normal, healthy worker again on the next step -- that assumption is what makes dropping its result, just this once, safe. A permanently dead worker breaks that assumption: it consumes one unit of the fixed `b`-sized backup budget on every single step, forever, with no chance of "catching up" next time. Section 19.2's detect-and-recover machinery (or ULFM's richer revoke-and-shrink) exists for exactly this different case, where the problem isn't transient and dropping-and-moving-on stops being a complete solution.

**8.** Chen et al.'s 100 machines are symmetric -- any 96 of them reporting first is enough, not a fixed 96 "required" plus a fixed 4 "backup" roster. With 5 of the 100 permanently dead, only 95 machines are ever live again, on every future step, and `withBackupsWait()`-style pooling can only ever draw from however many candidates are actually still alive: 95 is one short of the 96 needed to settle the step at all. No amount of waiting produces a 96th report, because the 5 that are gone are gone for every step, not just this one -- that's exactly Section 19.1's hang, not Section 19.3's designed drop-a-straggler case, because dropping assumes the dropped worker rejoins next step, and these five never will. The only way forward is to shrink the requirement itself (redefine N below 95, or exclude the dead ranks and rebuild the group), which is Section 19.2's rebuild-the-communicator move (or ULFM's `Comm_shrink()`), not anything Section 19.3's own technique does on its own.

---

**Sources cited in this chapter:**

- ["Building Scalable and Fault-Tolerant NCCL Applications" (NVIDIA Technical Blog)](https://developer.nvidia.com/blog/building-scalable-and-fault-tolerant-nccl-applications/) -- the exact quotes "even healthy ranks should expect NCCL to either return an error or hang on any collective operation," "enable non-blocking NCCL communicator so that we may detect and react to timeouts," the `ncclCommGetAsyncError()` polling pattern (`while (asyncError == ncclInProgress)`), and "typical recovery for the healthy ranks starts with `ncclCommAbort` on the existing communicator," "`ncclCommAbort` will exit any operation currently in progress, and destroy the communicator."
- NCCL's own installed header (`/usr/include/nccl.h`, libnccl 2.18.3) -- `ncclConfig_t`, `NCCL_CONFIG_INITIALIZER`, `ncclCommInitRankConfig()`, `ncclCommGetAsyncError()`, `ncclCommAbort()`, `ncclGetUniqueId()`, confirmed present and used genuinely in this chapter's own code.
- Chen, J. et al., ["Revisiting Distributed Synchronous SGD"](https://arxiv.org/abs/1604.00981) (arXiv 1604.00981, 2016) -- the exact quote "instead of having only N workers, we add b extra workers, but as soon as the parameter servers receive gradients from any N workers, they stop waiting and update their parameters using the N gradients," and their own real configuration, `N=96, b=4` out of 100 total machines.
- [Open MPI's ULFM (User-Level Failure Mitigation) documentation](https://docs.open-mpi.org/en/v5.0.2/features/ulfm.html) -- the exact quotes describing `MPIX_Comm_revoke()` ("interrupts any communication pending on the communicator at all ranks") and `MPIX_Comm_shrink()` ("creates a new communicator where dead processes in comm were removed, and the remaining processes are renamed to cover all the gaps"), previewing Chapter 20.
- This book's own Chapter 11 (`ncclCommInitAll()`, the real communicator this chapter's `ncclCommInitRankConfig()` extends), Chapter 17 (the real barrier this chapter's hangs would occur at, and MPI's `MPI_Barrier()` as a first real MPI contrast), and Chapter 18 (the makespan framing this chapter's "catastrophic straggler" scenario deliberately pushes to an extreme).
