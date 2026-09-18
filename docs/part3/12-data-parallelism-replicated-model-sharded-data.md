# Chapter 12: Data Parallelism: Replicated Model, Sharded Data

**What you will understand by the end of this chapter:**

- Why data parallelism replicates the *entire* model on every device but splits only the *data* -- and why PyTorch's own DistributedDataParallel paper states this book's whole Part 2 exists to guarantee exactly one thing: "all model replicas start from the exact same model state, and see the same parameter gradients after every backward pass."
- Why gradients must be *averaged* across every replica before the optimizer step, never summed and never skipped -- and how this book's real `ncclAllReduce()` does the averaging in one call, using a reduction op this chapter introduces: `ncclAvg`.
- The exact mechanism by which skipping that averaging step -- even once -- silently turns N replicas of one model into N different models, verified against an independent reference.

**What you need to know first:**

- Chapter 8's broadcast (used here to give every replica the same initial weights) and Chapter 9/11's all-reduce (used here every training step).
- Chapter 11's real, linked `ncclComm_t`/`ncclAllReduce()` and the two honest failure modes it can report.
- No new CUDA concepts beyond `ncclAvg`, a reduction op this chapter introduces alongside the `ncclSum` this book has used since Chapter 8.

---

Part 2 spent four chapters building the machinery to move and combine data across devices. Part 3 starts putting that machinery to work. Data parallelism is the simplest, most common way multi-GPU training actually happens: every device holds a complete, identical copy of the model, and the training data -- not the model -- gets split across them. Each device runs its own forward and backward pass on its own slice of data, producing its own local gradients; those gradients then have to be combined back into one shared update, or "replica" stops meaning anything. This chapter builds exactly that pipeline -- replicate, shard, compute, average, update -- and verifies, with a small checkable example, that averaging is not an optional refinement but the one thing standing between "one model, replicated" and "N different models that happen to share a name."

```text
Data parallelism, N=4 devices, ONE global batch of 16 samples:

  SAME weights, broadcast to every device:      DIFFERENT data, sharded across devices:

    dev0: w = [ . . . . ]                         dev0: samples [0, 4)
    dev1: w = [ . . . . ]   <- identical           dev1: samples [4, 8)
    dev2: w = [ . . . . ]      copies              dev2: samples [8, 12)
    dev3: w = [ . . . . ]                          dev3: samples [12, 16)

Every step: each device computes its OWN gradient from its OWN
shard, then all four gradients are AVERAGED (ncclAllReduce, ncclAvg)
before anyone updates -- so every device's weights stay identical.
```

## 12.1 Same Model, Different Data: What "Replica" Actually Means

### Intuition

Picture four students given the exact same textbook and the exact same starting answer key, but each assigned a different quarter of the homework problems to practice on. For "replica" to mean anything, two things have to be true before anyone starts: every student's answer key has to start out genuinely identical -- not "close," identical -- and every student's practice problems have to be genuinely different from everyone else's, or there was no point splitting the homework up in the first place. Splitting the problems is just arithmetic -- dividing a stack into four piles needs no coordination. Making sure every answer key starts identical is not arithmetic -- it needs someone to actually hand out identical copies.

```text
Setting up one training step's replicas:

  cudaGetDeviceCount()  ------------------>  "how many replicas?"
        |
        v
  computeShard(globalBatch, worldSize, rank)  -->  "which samples are MINE?"
        (pure host arithmetic -- no device needed)
        |
        v
  ncclBroadcast(root=0, weights)  --------->  "make sure everyone starts
        (a real collective -- needs a device)      from the SAME weights"
```

### Background

Splitting a global batch into per-device shards is pure index arithmetic -- it genuinely needs no device at all, so unlike almost every other call in this book, it can genuinely succeed here regardless of `deviceCount`. Broadcasting the initial weights is different: it's a real collective, using the real `ncclBroadcast()` this book linked against in Chapter 11, and Horovod's own documentation names this exact step -- `hvd.BroadcastGlobalVariablesHook(0)` -- as what makes "every replica starts identical" an enforced fact rather than an assumption.

```cpp
// Chapter 12: Data Parallelism -- Replicated Model, Sharded Data
// 33_replica_setup.cu
//
// Data parallelism needs exactly two things set up before any
// training step can run: every device holding an IDENTICAL copy of
// the model's initial weights, and every device holding a DIFFERENT
// slice of the global batch. The first needs a real collective
// (ncclBroadcast, genuinely called below); the second needs no
// device at all -- it's just arithmetic over indices, so this
// section can genuinely succeed even here. Genuinely compiled with
// a real nvcc, genuinely linked against a real libnccl, and
// genuinely run.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

// Splits a global batch into `worldSize` equal, non-overlapping
// shards, and returns device `rank`'s own [start, end) range. This
// is pure host-side index arithmetic -- no device, no CUDA call, no
// NCCL call. It genuinely succeeds no matter how many real devices
// exist, because it doesn't need any.
void computeShard(int globalBatchSize, int worldSize, int rank, int* start, int* end) {
    int perDevice = globalBatchSize / worldSize; // assumes an even split
    *start = rank * perDevice;
    *end = *start + perDevice;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int GLOBAL_BATCH = 16;

    // This book's own real deviceCount is 0, so there is genuinely
    // nothing to shard across -- but the shard formula itself runs
    // and reports that honestly, exactly like every device-count loop
    // in this book since Chapter 4.
    printf("\nSharding a global batch of %d across this book's own real "
           "deviceCount (%d):\n", GLOBAL_BATCH, deviceCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        int start, end;
        computeShard(GLOBAL_BATCH, deviceCount, rank, &start, &end);
        printf("  rank %d: samples [%d, %d)\n", rank, start, end);
    }
    printf("  (%d device(s) -- nothing to print above.)\n", deviceCount);

    // Hypothetically, if there WERE 4 devices: every rank gets a
    // different, non-overlapping, equal-sized slice of the same
    // global batch -- this is genuinely computed, not illustrative
    // pseudocode.
    const int HYPOTHETICAL_WORLD_SIZE = 4;
    printf("\nThe same formula, hypothetically, with world size %d:\n", HYPOTHETICAL_WORLD_SIZE);
    for (int rank = 0; rank < HYPOTHETICAL_WORLD_SIZE; ++rank) {
        int start, end;
        computeShard(GLOBAL_BATCH, HYPOTHETICAL_WORLD_SIZE, rank, &start, &end);
        printf("  rank %d: samples [%d, %d)\n", rank, start, end);
    }

    // Splitting the DATA needs no device. Making every replica start
    // from the SAME weights does -- that's a real collective, exactly
    // the way Horovod's own hvd.BroadcastGlobalVariablesHook or this
    // book's own Chapter 8 broadcast enforces it, rather than assuming
    // it. Attempted here with the real ncclBroadcast() this book
    // linked against in Chapter 11.
    ncclComm_t comm = nullptr; // never successfully created, as in Chapter 11
    cudaStream_t stream = nullptr;
    float* weights = nullptr;
    const size_t WEIGHT_COUNT = 4;

    ncclResult_t eBcast = ncclBroadcast(weights, weights, WEIGHT_COUNT, ncclFloat, 0, comm, stream);
    printf("\nncclBroadcast(root=0, initial weights -> every replica): %s (code %d)\n",
           ncclGetErrorString(eBcast), (int)eBcast);

    printf("\nSharding data is arithmetic this chapter can genuinely run;\n"
           "broadcasting weights is a real collective this chapter can only\n"
           "genuinely attempt. Section 12.2 covers the other real collective\n"
           "data parallelism needs every step after that: averaging gradients.\n");

    return 0;
}
```

Genuinely compiled with a real `nvcc`, genuinely linked against a real `libnccl`, and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

Sharding a global batch of 16 across this book's own real deviceCount (0):
  (0 device(s) -- nothing to print above.)

The same formula, hypothetically, with world size 4:
  rank 0: samples [0, 4)
  rank 1: samples [4, 8)
  rank 2: samples [8, 12)
  rank 3: samples [12, 16)

ncclBroadcast(root=0, initial weights -> every replica): invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)

Sharding data is arithmetic this chapter can genuinely run;
broadcasting weights is a real collective this chapter can only
genuinely attempt. Section 12.2 covers the other real collective
data parallelism needs every step after that: averaging gradients.
```

The shard formula is the first calculation in this section of the book that hasn't needed a single device to genuinely succeed -- it ran, honestly, twice: once against this book's own real `deviceCount` (reporting, honestly, that there's nothing to shard across), and once against a hypothetical world size, producing four genuinely correct, non-overlapping ranges. `ncclBroadcast()` is a different kind of call entirely -- it needs a real communicator attached to real devices, so it reports exactly the honest `ncclInvalidArgument` this book's Chapter 11 already established for a communicator that was never successfully created.

!!! warning "[COMMON TRAP] Assuming a replica's weights start identical just because the training script initializes them the same way"
    Every replica running the *same initialization code* -- the same random seed, the same formula -- can still end up with genuinely different starting weights, because random number generators, floating-point rounding, and even library versions can differ subtly across processes and devices. PyTorch's own DDP paper is explicit that correctness rests on replicas provably starting from "the exact same model state," not merely similar initialization logic -- which is exactly why real frameworks broadcast the actual weight values from one root, the way this section's code attempts, rather than trusting every replica to compute the same values independently.

## 12.2 Averaging Gradients: One Real Call, `ncclAvg`

### Intuition

Once every replica has computed its own gradient from its own slice of data, the four students from Section 12.1's analogy need to compare notes before anyone updates their answer key. But comparing notes and simply *adding up* everyone's suggested correction would overcorrect by a factor of four -- what the group actually wants is the *average* suggested correction, applied identically to every copy of the answer key. That's a genuinely different operation from every all-reduce this book has built or called so far: Chapters 8 through 11 only ever needed a sum.

```text
Before this chapter: ncclSum only              This chapter: ncclAvg too

  ncclAllReduce(g, ..., ncclSum, ...)            ncclAllReduce(g, ..., ncclAvg, ...)
  every replica ends with SUM(g0..g3)            every replica ends with
  -- N times too large for a gradient step         MEAN(g0..g3) -- exactly the
                                                     step size ONE model training
                                                     on the whole global batch
                                                     would have taken
```

### Background

NCCL's own documentation describes `ncclAvg` plainly: "an average operation, i.e. a sum across all ranks, divided by the number of ranks." That division is the entire difference between this section and every prior all-reduce in this book -- the call shape is identical to Chapter 9's and Chapter 11's `ncclAllReduce()`, only the reduction operation passed in changes.

```cpp
// Chapter 12: Data Parallelism -- Replicated Model, Sharded Data
// 34_gradient_allreduce.cu
//
// Every training step, after every replica computes its own local
// gradients from its own data shard, those gradients have to be
// averaged across every replica before anyone updates their weights
// -- otherwise "replica" stops meaning what Section 12.1 set it up
// to mean. NCCL's ncclAvg reduction op (new in this chapter) does
// the averaging as part of the same real ncclAllReduce() call this
// book linked against in Chapter 11 -- no separate divide-by-N step
// required. Genuinely compiled with a real nvcc, genuinely linked
// against a real libnccl, and genuinely run.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    ncclComm_t comm = nullptr; // never successfully created, as in Chapter 11
    cudaStream_t stream = nullptr;
    float* localGradients = nullptr;
    const size_t WEIGHT_COUNT = 4;

    // ncclSum, used in Chapters 8-11, would leave every replica with
    // the SUM of every other replica's gradient -- exactly N times
    // too large an update. ncclAvg divides by the number of ranks in
    // the communicator as part of the same reduction, so the result
    // is already the correct per-parameter average gradient.
    ncclResult_t eAllReduceAvg = ncclAllReduce(localGradients, localGradients, WEIGHT_COUNT,
                                                ncclFloat, ncclAvg, comm, stream);
    printf("ncclAllReduce(localGradients, ..., ncclAvg, ...): %s (code %d)\n",
           ncclGetErrorString(eAllReduceAvg), (int)eAllReduceAvg);

    // For contrast: the same call with ncclSum, this book's only
    // reduction op through Chapter 11 -- genuinely a different real
    // call, not a relabeling of the one above.
    ncclResult_t eAllReduceSum = ncclAllReduce(localGradients, localGradients, WEIGHT_COUNT,
                                                ncclFloat, ncclSum, comm, stream);
    printf("ncclAllReduce(localGradients, ..., ncclSum, ...): %s (code %d)\n",
           ncclGetErrorString(eAllReduceSum), (int)eAllReduceSum);

    printf("\nBoth calls report the same honest failure this book's\n"
           "never-successfully-created communicator has reported since\n"
           "Chapter 11 -- the real difference between ncclAvg and ncclSum\n"
           "only shows up in what value ends up in the buffer, which needs\n"
           "a real device to observe. Section 12.3's simulation checks that\n"
           "difference the way this book always has when hardware can't:\n"
           "against an independent reference, computed on the host.\n");

    return 0;
}
```

Genuinely compiled with a real `nvcc`, genuinely linked against a real `libnccl`, and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).
ncclAllReduce(localGradients, ..., ncclAvg, ...): invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)
ncclAllReduce(localGradients, ..., ncclSum, ...): invalid argument (run with NCCL_DEBUG=WARN for details) (code 4)

Both calls report the same honest failure this book's
never-successfully-created communicator has reported since
Chapter 11 -- the real difference between ncclAvg and ncclSum
only shows up in what value ends up in the buffer, which needs
a real device to observe. Section 12.3's simulation checks that
difference the way this book always has when hardware can't:
against an independent reference, computed on the host.
```

Both calls genuinely fail identically here, for the same reason every real collective call has failed since Chapter 11 -- the communicator was never successfully created, so NCCL's own argument validation catches both before either reduction operation would ever run. That's honest, but it's also exactly why this section can't be where the `ncclAvg`-vs-`ncclSum` distinction actually gets checked -- Section 12.3 is.

!!! warning "[COMMON TRAP] Assuming you can just divide by N yourself after an ncclSum, instead of using ncclAvg"
    Manually dividing a summed gradient by `N` after an `ncclSum`-based all-reduce produces the mathematically identical number to `ncclAvg` -- so it's tempting to treat the two as interchangeable, just one extra line of code either way. They aren't equivalent as a matter of what actually executes: `ncclAvg` performs the division inside the same reduction NCCL already has to do, on whatever internal precision NCCL's own implementation uses for the reduction itself, while a separate manual division is a distinct floating-point operation performed afterward, by your own code, on the already-reduced result. For plain sums of well-scaled floating-point gradients the difference is usually invisible; the two are not, however, defined to be bit-identical, and reaching for `ncclAvg` when it exists avoids ever needing to reason about whether that difference matters for a given training run.

## 12.3 Verifying Consistency: Why Skipping the All-Reduce Silently Breaks Everything

### Intuition

Suppose the four students from Section 12.1 stop comparing notes -- each one just keeps privately correcting their own answer key using only their own quarter of the homework. Nobody makes an obviously wrong move at any single step; each correction, on its own, looks completely reasonable. But because each student is nudging their answer key toward *their own* quarter's answers, and those answers aren't the same across students, the four answer keys quietly stop agreeing with each other -- a little after the first correction, more after the second, and so on. Nothing crashes, nothing errors, and nothing about any single step looks wrong in isolation; the only thing that reveals the problem is comparing all four answer keys against each other after the fact.

```text
After just ONE training step, starting all four replicas at w=0:

  WITH all-reduce (average, then same update):    WITHOUT all-reduce (each on its own):

    dev0: 1.4      dev1: 1.4                        dev0: 1.2      dev1: 0.8
    dev2: 1.4      dev3: 1.4                         dev2: 2.0      dev3: 1.6

    still ONE model, replicated four times          four DIFFERENT models already --
                                                       and the gap only grows from here
```

### Background

This is the section where this chapter's actual correctness claim gets checked, since neither real collective call in Sections 12.1-12.2 could run on this machine. The simulation below uses the smallest model this book could reasonably check by hand: a single scalar weight per replica, and a toy gradient, `grad = weight - target`, standing in for a real backward pass. Four devices, four different per-shard targets, matching this book's own established `N=4` convention. It runs the training loop two ways -- once with a real average all-reduce every step (Section 12.2's `ncclAvg`, computed here on the host since no device can run it for real), and once with no synchronization at all -- and checks the first against an independent single-model reference, and the second for the divergence Section 12.3's intuition predicts.

```cpp
// Chapter 12: Data Parallelism -- Replicated Model, Sharded Data
// 35_data_parallel_consistency_simulation.cpp
//
// Plain host C++ -- verifying the one property PyTorch's own
// DistributedDataParallel paper states as DDP's actual correctness
// guarantee: "all model replicas start from the exact same model
// state, and see the same parameter gradients after every backward
// pass" (cited in Sources). N real in-memory scalars stand in for N
// real devices' weight replicas; a tiny synthetic linear-regression
// gradient (grad = w - target) stands in for a real backward pass,
// small enough to check by hand.
#include <cstdio>

int main() {
    const int N = 4;
    const double targets[N] = {12.0, 8.0, 20.0, 16.0}; // each device's own data-shard "target"
    const double LR = 0.1;
    const int STEPS = 5;

    double meanTarget = 0.0;
    for (int i = 0; i < N; ++i) meanTarget += targets[i];
    meanTarget /= N;

    // ---------------------------------------------------------------
    // (a) WITH synchronization: every step, average all N replicas'
    // local gradients (this book's real ncclAllReduce(..., ncclAvg,
    // ...) from Section 12.2), then apply that SAME averaged update
    // to every replica -- exactly the mechanism this chapter's cited
    // sources describe.
    // ---------------------------------------------------------------
    double synced[N];
    for (int i = 0; i < N; ++i) synced[i] = 0.0; // Section 12.1's broadcast initial weight

    // Independent reference: a single, non-replicated model trained
    // directly on the mean target -- mathematically what training on
    // the true GLOBAL batch (the union of every shard) would do,
    // since the shards are equal-sized. Computed completely
    // separately from the loop below.
    double reference = 0.0;

    printf("WITH synchronization (average gradients via ncclAvg every step):\n");
    printf("step 0: ");
    for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, synced[i]);
    printf(" | reference=%.4f\n", reference);

    for (int step = 1; step <= STEPS; ++step) {
        // Every replica computes its OWN local gradient from its OWN
        // data shard -- and since every replica's weight is still
        // identical going into this step, this is genuinely N
        // separate (but, this step, numerically equal-to-each-other's
        // starting point) local computations, not one computation
        // copied N times.
        double localGrad[N];
        for (int i = 0; i < N; ++i) localGrad[i] = synced[i] - targets[i];

        // ncclAllReduce(..., ncclAvg, ...): average across all N.
        double avgGrad = 0.0;
        for (int i = 0; i < N; ++i) avgGrad += localGrad[i];
        avgGrad /= N;

        // Every replica applies the SAME averaged gradient.
        for (int i = 0; i < N; ++i) synced[i] = synced[i] - LR * avgGrad;

        reference = reference - LR * (reference - meanTarget);

        printf("step %d: ", step);
        for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, synced[i]);
        printf(" | reference=%.4f\n", reference);
    }

    bool allSyncedEqual = true;
    bool allMatchReference = true;
    for (int i = 0; i < N; ++i) {
        if (synced[i] != synced[0]) allSyncedEqual = false;
        if (synced[i] != reference) allMatchReference = false;
    }
    printf("After %d step(s): every replica identical to every other: %s; "
           "every replica matches the independent single-model reference: %s\n\n",
           STEPS, allSyncedEqual ? "PASS" : "FAIL", allMatchReference ? "PASS" : "FAIL");

    // ---------------------------------------------------------------
    // (b) WITHOUT synchronization: every replica applies its OWN
    // local gradient, with no all-reduce at all -- the mistake this
    // book's entire Part 2 exists to prevent.
    // ---------------------------------------------------------------
    double unsynced[N];
    for (int i = 0; i < N; ++i) unsynced[i] = 0.0;

    printf("WITHOUT synchronization (each replica updates from its own local gradient only):\n");
    printf("step 0: ");
    for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, unsynced[i]);
    printf("\n");

    for (int step = 1; step <= STEPS; ++step) {
        double localGrad[N];
        for (int i = 0; i < N; ++i) localGrad[i] = unsynced[i] - targets[i];
        for (int i = 0; i < N; ++i) unsynced[i] = unsynced[i] - LR * localGrad[i]; // no averaging

        printf("step %d: ", step);
        for (int i = 0; i < N; ++i) printf("replica%d=%.4f ", i, unsynced[i]);
        printf("\n");
    }

    bool anyDivergence = false;
    for (int i = 1; i < N; ++i) {
        if (unsynced[i] != unsynced[0]) anyDivergence = true;
    }
    printf("After %d step(s) with no synchronization: replicas have "
           "diverged from each other: %s\n", STEPS,
           anyDivergence ? "PASS (divergence correctly demonstrated)" : "FAIL (unexpectedly still identical)");

    return (allSyncedEqual && allMatchReference && anyDivergence) ? 0 : 1;
}
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
WITH synchronization (average gradients via ncclAvg every step):
step 0: replica0=0.0000 replica1=0.0000 replica2=0.0000 replica3=0.0000  | reference=0.0000
step 1: replica0=1.4000 replica1=1.4000 replica2=1.4000 replica3=1.4000  | reference=1.4000
step 2: replica0=2.6600 replica1=2.6600 replica2=2.6600 replica3=2.6600  | reference=2.6600
step 3: replica0=3.7940 replica1=3.7940 replica2=3.7940 replica3=3.7940  | reference=3.7940
step 4: replica0=4.8146 replica1=4.8146 replica2=4.8146 replica3=4.8146  | reference=4.8146
step 5: replica0=5.7331 replica1=5.7331 replica2=5.7331 replica3=5.7331  | reference=5.7331
After 5 step(s): every replica identical to every other: PASS; every replica matches the independent single-model reference: PASS

WITHOUT synchronization (each replica updates from its own local gradient only):
step 0: replica0=0.0000 replica1=0.0000 replica2=0.0000 replica3=0.0000 
step 1: replica0=1.2000 replica1=0.8000 replica2=2.0000 replica3=1.6000 
step 2: replica0=2.2800 replica1=1.5200 replica2=3.8000 replica3=3.0400 
step 3: replica0=3.2520 replica1=2.1680 replica2=5.4200 replica3=4.3360 
step 4: replica0=4.1268 replica1=2.7512 replica2=6.8780 replica3=5.5024 
step 5: replica0=4.9141 replica1=3.2761 replica2=8.1902 replica3=6.5522 
After 5 step(s) with no synchronization: replicas have diverged from each other: PASS (divergence correctly demonstrated)
```

With synchronization, all four replicas stay bit-identical through all five steps, and match a reference model trained directly on the mean of all four targets -- the mathematical stand-in for "the true global batch" -- exactly, at every single step, not just at the end. This is checkable by hand: since every replica's weight is identical entering a step, each replica's local gradient is `w - target_i`, their average is `w - mean(target_i)`, and applying that average update to `w` is *exactly* the recurrence the single-model reference runs directly -- the two are the same computation, performed two different ways, and this section's PASS confirms they produce the same number every time. Without synchronization, the four replicas are already different after step 1 (`1.2`, `0.8`, `2.0`, `1.6` instead of one shared `1.4`), and the gap only widens with every further step -- by step 5, replica 2's weight (`8.19`) is more than double replica 1's (`3.28`). Nothing in the loop that produced this divergence looks wrong on its own; the only thing that reveals it is exactly the kind of independent, cross-replica check this section ran.

!!! warning "[COMMON TRAP] Assuming a bug that skips synchronization will be obvious quickly"
    Section 12.3's WITHOUT-synchronization output doesn't crash, doesn't print an error, and doesn't look obviously broken at any single step -- every replica's own weight update is completely ordinary gradient descent, correctly computed from its own data. The only way to notice anything is wrong is to compare replicas against *each other*, which a training script that never does this check has no reason to do until, potentially, results at evaluation time look inexplicably inconsistent between runs. A missing or misconfigured all-reduce call is exactly this kind of bug: syntactically fine, locally correct at every step, and silently catastrophic in aggregate -- which is precisely why this book spent four chapters (8 through 11) building and then linking against the real collective that exists to prevent it.

## Chapter Summary

Data parallelism replicates a full model across every device and splits only the training data, and this chapter showed that "replica" is a claim that has to be actively maintained, not a label that stays true on its own. Section 12.1 split this setup into the two things it actually requires: sharding data, which is pure arithmetic that needs no device at all, and broadcasting the initial weights, a real collective (`ncclBroadcast()`) that enforces every replica starting genuinely identical, the same guarantee Horovod's own `hvd.BroadcastGlobalVariablesHook` and PyTorch DDP's own stated correctness property both describe directly. Section 12.2 introduced `ncclAvg`, a second real reduction op alongside the `ncclSum` this book has used since Chapter 8, needed because gradient synchronization requires the *mean* of every replica's local gradient, not their sum. Section 12.3 -- the chapter's actual verification, since neither real collective call could run on this machine -- checked both halves of the claim with a small, checkable simulation: with synchronization, every replica stays bit-identical and matches an independently-computed single-model reference at every step; without it, replicas silently diverge starting at step 1, with no error, no crash, and no local sign anything is wrong. Chapter 13 turns to model parallelism -- what to do when replicating the *entire* model on every device isn't an option in the first place, because no single device can hold it.

## Self-Check Questions

1. Explain, using Section 12.1's setup, why sharding the data needs no device at all while broadcasting the weights does.
2. Section 12.2 introduces `ncclAvg`. What real reduction op has this book used for every all-reduce before this chapter, and what result would using it here, uncorrected, produce?
3. Using Section 12.3's simulation, explain in your own words why the WITH-synchronization replicas' local gradients are identical to each other at the start of every step, even though each replica supposedly computes its own gradient from its own, different data shard.
4. Quote the specific claim this chapter cites from PyTorch's own DistributedDataParallel paper describing what property makes replicas correct.
5. Using Section 12.3's numbers, compute by hand what replica 2's weight would be after step 1 WITHOUT synchronization, and explain why it differs from every other replica's step-1 weight.
6. Why does Section 12.3's WITH-synchronization result match the independent single-model `reference` exactly, rather than merely approximately? What mathematical property of averaging makes this an exact equality rather than a coincidence?
7. Section 12.3's Common Trap says a missing all-reduce call is "syntactically fine, locally correct at every step." Explain what checking WOULD have caught the bug in Section 12.3's WITHOUT-synchronization run, and why a normal single-replica training script would never happen to run that check.
8. Using this chapter's own cited quote about Horovod's `hvd.BroadcastGlobalVariablesHook(0)`, explain what specific failure that call is designed to prevent, and how it relates to Section 12.1's own `ncclBroadcast()` call.

## Where We Go Next

Chapter 13 turns to model parallelism: what happens when the model itself doesn't fit on one device, so replication -- this chapter's entire premise -- isn't available at all, and the model has to be split the way this chapter split the data instead.

## Worked Solutions

**1.** Sharding the data is pure host-side index arithmetic -- dividing a known total (`globalBatchSize`) by a known count (`worldSize`) and computing offsets. Nothing about that calculation touches a device, a memory address, or any hardware state, so it can be computed correctly regardless of how many real devices exist. Broadcasting the weights, by contrast, has to move real values into real device memory on every replica -- an actual data transfer across devices, which is exactly the kind of operation this book has needed a real device for since Chapter 4.

**2.** This book has used `ncclSum` for every all-reduce before this chapter (Chapters 8, 9, and 11). Using `ncclSum`, uncorrected, on N replicas' local gradients would leave every replica holding the SUM of all N gradients -- N times larger than the correct average gradient -- which would make every weight update N times too large.

**3.** At the start of every step in the WITH-synchronization loop, every replica's weight is identical, because the previous step ended by applying the exact same averaged update to every replica (or, at step 1, every replica started from the exact same broadcast initial weight). Each replica does compute its own gradient from its own different target (`localGrad[i] = synced[i] - targets[i]`), and since the weight `synced[i]` is the same number for every `i` at that point, only the different `targets[i]` values make each replica's own gradient genuinely different from the others -- it's the subsequent averaging step, not identical inputs, that produces the shared result every replica then applies.

**4.** "DDP guarantees the correctness by making sure that all model replicas start from the exact same model state, and see the same parameter gradients after every backward pass" -- this is the specific claim this chapter cites from PyTorch's own DistributedDataParallel paper.

**5.** Without synchronization, replica 2's local gradient at step 1 is `localGrad[2] = synced[2] - targets[2] = 0 - 20 = -20`. Its step-1 weight is `unsynced[2] = 0 - LR * (-20) = 0 - 0.1 * (-20) = 2.0`, matching the locked output's `replica2=2.0000`. It differs from every other replica's step-1 weight because each replica used its OWN `targets[i]` value with no averaging step to reconcile them -- replica 2's larger target (`20`, the largest of the four) produces a larger gradient magnitude and therefore a larger step-1 update than replicas with smaller targets.

**6.** Averaging is a linear operation: the average of `(w - target_i)` across all `i` equals `w - mean(target_i)` exactly, by simple algebraic rearrangement, not by approximation. Since every WITH-synchronization replica enters each step holding the identical `w` (established by Q3), the averaged gradient every replica receives is mathematically identical to the gradient a single reference model would compute directly against `meanTarget`, and applying that identical update to both `w` and `reference` keeps them exactly equal at every single step -- there is no numerical coincidence involved, just the same computation performed two different ways.

**7.** The check that would have caught the bug is exactly Section 12.3's own cross-replica comparison -- comparing every replica's weight against every other replica's weight (or against an independent reference) after a training step. A normal single-replica training script has no reason to run this check because it only ever has one copy of the model to begin with; the check only becomes meaningful, and only becomes necessary, once a training setup deliberately introduces multiple replicas that are supposed to stay identical -- exactly the situation data parallelism creates and exactly the situation a missing all-reduce call quietly breaks.

**8.** `hvd.BroadcastGlobalVariablesHook(0)` is designed to prevent replicas from starting training with genuinely different initial weights -- the same failure mode this chapter's own Common Trap in Section 12.1 describes, where identical initialization *code* run independently on each replica does not guarantee identical initialization *values*. It relates directly to Section 12.1's `ncclBroadcast()` call: both are the same fix for the same problem, one from a real, widely-used framework and one from this book's own from-scratch NCCL code -- take one root's actual weight values and copy them, verbatim, to every other replica, rather than trusting every replica to compute the same values on its own.

---

**Sources cited in this chapter:**

- [PyTorch Distributed: Experiences on Accelerating Data Parallel Training — Li et al., VLDB 2020](https://www.vldb.org/pvldb/vol13/p3005-li.pdf) -- the exact correctness guarantee ("all model replicas start from the exact same model state, and see the same parameter gradients after every backward pass") and the statement that gradient communication happens "before the optimizer step to make sure that parameters of all model replicas are updated using exactly the same set of gradients."
- [PyTorch DDP Tutorial: DDP Theory](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html) -- "the model is replicated on all the devices," `DistributedSampler` ensuring "each device gets a non-overlapping input batch," and gradient synchronization via "the ring all-reduce algorithm."
- [Horovod: fast and easy distributed deep learning in TensorFlow — Sergeev & Del Balso, 2018](https://arxiv.org/pdf/1802.05799) -- the "average gradients among those multiple copies" description of data-parallel synchronization, and `hvd.BroadcastGlobalVariablesHook(0)` for consistent initialization across workers.
- [NCCL User Guide — Types (ncclRedOp_t)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html) -- the exact `ncclAvg` definition: "an average operation, i.e. a sum across all ranks, divided by the number of ranks."
