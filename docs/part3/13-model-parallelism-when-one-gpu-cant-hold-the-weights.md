# Chapter 13: Model Parallelism: When One GPU Can't Hold the Weights

**What you will understand by the end of this chapter:**

- Why data parallelism (Chapter 12) cannot fix the memory-capacity wall Chapter 1 opened this book with -- replicating a model that doesn't fit on one device onto *more* devices still doesn't make it fit on any *one* of them.
- How to partition a model's own layers across devices using the same host-arithmetic pattern Chapter 12 used to partition data, and what real transfer -- not a new API, the same `cudaMemcpyPeer()` from Chapter 5 -- has to move between devices at every layer boundary.
- Why a forward pass split across devices this way must produce the *exact same*, bit-identical output as an unsplit one -- and why that guarantee is strictly stronger than anything this book's collectives (Chapters 8-11) could promise about floating-point order.
- The real cost naive model parallelism doesn't pay for free: GPipe's own documented "severe under-utilization due to the sequential dependency of the network" -- named here, quantified in Chapter 15.

**What you need to know first:**

- Chapter 1's memory-capacity wall and its real ZeRO 16-bytes-per-parameter mixed-precision Adam formula.
- Chapter 5's real `cudaMemcpyPeer()` call.
- Chapter 12's host-side sharding arithmetic (`computeShard()`) and its replicated-model premise, which this chapter abandons.

---

Chapter 12 assumed something this chapter can no longer assume: that the model fits, whole, on every device. Data parallelism's entire premise is replication -- every device holds a complete copy of the model and a different slice of the data. That premise has a hard floor: it requires the model to fit on *one* device in the first place, and Chapter 1 already showed, in real bytes, that plenty of real models don't. Model parallelism is the other half of the split this book's Part 3 is built around -- instead of splitting the *data* and replicating the *model*, split the *model itself* and let the *data* flow through the pieces, sequentially, one device at a time. This chapter builds that split: how much a model's own size forces you to split it into, how to assign layers to devices and move activations between them, and the one property -- exact correctness -- that a split forward pass has to preserve no matter how many pieces the model gets cut into.

```text
Data parallelism (Chapter 12):        Model parallelism (this chapter):
  every device holds the WHOLE model    every device holds a PART of the model
  a DIFFERENT slice of data each          the SAME data flows through all of them

  dev0: [ALL layers] <- data[0:4)         dev0: [layers 0-23]  --activation-->
  dev1: [ALL layers] <- data[4:8)         dev1: [layers 24-47] --activation-->
  dev2: [ALL layers] <- data[8:12)        dev2: [layers 48-71] --activation-->
  dev3: [ALL layers] <- data[12:16)       dev3: [layers 72-95]

  Requires the model to FIT on one       Requires the ACTIVATION at each
  device in the first place.             layer boundary to cross a real
                                          device-to-device transfer.
```

## 13.1 Why Split the Model Itself: Memory Capacity, Revisited

### Intuition

Chapter 1 measured GPT-3's real training-state footprint in bytes and found it needs the equivalent of 35 real H100 SXM GPUs' worth of memory just to exist -- before a single training step runs. Data parallelism, as built in Chapter 12, doesn't touch that number at all: giving a model 8 identical replicas across a cluster means the cluster now needs 8 times as much *total* memory, but each individual replica still, on its own, needs that same 35-GPUs' worth to hold its own copy. Replication multiplies how many times a thing exists; it does nothing to shrink the thing itself. If no single device can hold the whole model, having more devices each try to hold the *whole* model doesn't help -- the only way to make one device's job possible is to give it less of the model to hold.

```text
8 data-parallel replicas of GPT-3:        Model parallelism, 35-way split:

  dev  0- 4: replica 0 (needs 35 GPUs'      dev 0: layers  0- k  (1/35th of
             worth of memory ON ITS OWN)         the training state)
  dev  5- 9: replica 1 (needs 35 GPUs'      dev 1: layers  k-2k  (1/35th)
             worth of memory ON ITS OWN)     ...
  ...        (x8, 280 GPUs total)           dev34: layers ... -end (1/35th)

  Total memory: 8x bigger.                  Total memory: SAME 35 GPUs'
  Per-replica requirement: UNCHANGED.       worth -- but now split so each
                                             device only needs its 1/35th.
```

### Background

Chapter 1's own real formula still applies unchanged: mixed-precision Adam training needs 16 bytes of memory per parameter (Rajbhandari et al.'s ZeRO paper -- 2 bytes each for fp16 parameters and gradients, 4 bytes each for the fp32 parameter copy, momentum, and variance Adam tracks). What this section adds is the other half of that same arithmetic: given a fixed per-device memory budget, how many devices does a model's training state have to be split across just to exist at all. The code below re-derives Chapter 1's own already-locked GPT-3 numbers as a self-consistency check, then asks the question Chapter 1 didn't: what does data parallelism's own replication do to that number (nothing), and what does model parallelism do to it (divides it by the number of shards).

```cpp
// Chapter 13: Model Parallelism -- When One GPU Can't Hold the Weights
// 36_model_shard_memory_model.cpp
//
// Plain host C++ closed-form model, reusing Chapter 1's own already-cited
// numbers exactly (ZeRO's real 16-bytes-per-parameter mixed-precision Adam
// breakdown, and the NVIDIA H100 SXM/NVL memory capacities) rather than
// inventing anything new. This section asks a different question than
// Chapter 1 did: not "how much memory does training this model need in
// total," but "given a fixed per-device memory budget, what is the
// MINIMUM number of model-parallel shards required for the model's
// training state to exist at all."
#include <cstdio>
#include <cmath>

struct Model {
    const char* name;
    double params; // parameter count
};

int main() {
    // Same two models Chapter 1 used, plus the ZeRO paper's own worked
    // example (GPT-2, 1.5B parameters) -- a third real, independently
    // cited figure, not one this book picked itself.
    const Model models[] = {
        {"GPT-2 (1.5B)",    1.5e9},
        {"7B-class model",  7.0e9},
        {"GPT-3 (175B)",  175.0e9},
    };

    const double H100_SXM_GB = 80.0; // NVIDIA H100 SXM (Chapter 1, Chapter 2)
    const double H100_NVL_GB = 94.0; // NVIDIA H100 NVL (Chapter 1, Chapter 2)
    const double BYTES_PER_GB = 1.0e9;

    printf("%-16s %10s %18s %14s %14s\n",
           "Model", "Params", "Training(Adam,mp)", "Min SXM x", "Min NVL x");

    double gpt3TrainingGB = 0.0;
    double gpt3ShardsSXM = 0.0, gpt3ShardsNVL = 0.0;

    for (const auto& m : models) {
        // Chapter 1's own real formula: 16 bytes/parameter for
        // mixed-precision Adam (Rajbhandari et al., ZeRO, SC'20).
        double trainingGB = (m.params * 16.0) / BYTES_PER_GB;

        // The question this chapter asks, that Chapter 1 didn't: given a
        // fixed per-device budget, what is the minimum number of
        // model-parallel shards needed for the training state to exist
        // at all, i.e. for it to fit across that many devices at once.
        double shardsSXM = std::ceil(trainingGB / H100_SXM_GB);
        double shardsNVL = std::ceil(trainingGB / H100_NVL_GB);

        printf("%-16s %8.1fB %15.1f GB %13.0f %13.0f\n",
               m.name, m.params / 1e9, trainingGB, shardsSXM, shardsNVL);

        if (m.params == 175.0e9) {
            gpt3TrainingGB = trainingGB;
            gpt3ShardsSXM = shardsSXM;
            gpt3ShardsNVL = shardsNVL;
        }
    }

    // Self-consistency check: this chapter's own formula, applied to
    // GPT-3, must reproduce Chapter 1's own already-locked numbers
    // exactly -- 35 H100 SXM GPUs' worth of memory, 30 H100 NVL GPUs'
    // worth. If it doesn't, one of the two chapters has a bug.
    bool matchesChapter1 = (gpt3ShardsSXM == 35.0) && (gpt3ShardsNVL == 30.0);
    printf("\nSelf-consistency check against Chapter 1's own locked GPT-3 "
           "figures (35 H100 SXM, 30 H100 NVL): %s\n",
           matchesChapter1 ? "PASS" : "FAIL");

    // Data parallelism (Chapter 12) does not change any number in the
    // table above -- it changes how many TIMES the model exists, never
    // how big any single copy is. Every one of the REPLICAS Chapter 12
    // built still, individually, needs the same number of shards.
    const int REPLICAS = 8;
    printf("\nRunning %d data-parallel replicas of GPT-3 (Chapter 12's "
           "own technique) needs %.1f GB of TOTAL cluster memory (%d x "
           "%.1f GB) -- but EACH replica still individually needs the "
           "same %.0f H100 SXM GPUs' worth of memory to exist at all.\n"
           "Data parallelism multiplies how many times the model exists;\n"
           "it does not reduce how big any one copy has to be. Only "
           "model\nparallelism -- splitting the model ITSELF, this "
           "chapter's subject -- changes that number.\n",
           REPLICAS, gpt3TrainingGB * REPLICAS, REPLICAS, gpt3TrainingGB,
           gpt3ShardsSXM);

    return matchesChapter1 ? 0 : 1;
}
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
Model                Params  Training(Adam,mp)      Min SXM x      Min NVL x
GPT-2 (1.5B)          1.5B            24.0 GB             1             1
7B-class model        7.0B           112.0 GB             2             2
GPT-3 (175B)        175.0B          2800.0 GB            35            30

Self-consistency check against Chapter 1's own locked GPT-3 figures (35 H100 SXM, 30 H100 NVL): PASS

Running 8 data-parallel replicas of GPT-3 (Chapter 12's own technique) needs 22400.0 GB of TOTAL cluster memory (8 x 2800.0 GB) -- but EACH replica still individually needs the same 35 H100 SXM GPUs' worth of memory to exist at all.
Data parallelism multiplies how many times the model exists;
it does not reduce how big any one copy has to be. Only model
parallelism -- splitting the model ITSELF, this chapter's subject -- changes that number.
```

GPT-2's own 1.5-billion-parameter model needs only 24 GB of training-state memory -- Rajbhandari et al.'s own ZeRO paper cites this exact figure as their motivating example, and it fits on a single modern H100 with room to spare, which is exactly why GPT-2-scale models never needed model parallelism to train. The 7B-class model already needs 2 shards; GPT-3 needs 35, reproducing Chapter 1's own already-locked number exactly, confirmed by this section's own self-consistency check. The closing calculation is the point of this whole section: multiplying replicas multiplies total cluster memory linearly, but does nothing at all to the 35-GPU figure any single replica is stuck with -- that number only moves when the model itself gets divided, which is what the rest of this chapter builds.

!!! warning "[COMMON TRAP] Assuming more GPUs always means more capacity, regardless of how they're used"
    A cluster with 280 GPUs (8 data-parallel replicas x 35 GPUs' worth of memory each, from this section's own numbers) has 8 times the memory of Chapter 1's minimum -- but every one of those 280 GPUs is still organized into groups of 35 that each, individually, face exactly the same capacity wall a single 35-GPU model-parallel group would. Adding GPUs only relieves a capacity wall when the *way they're used* actually divides the thing that didn't fit -- data parallelism adds GPUs by dividing the *data*, which never touches the per-device memory requirement at all. It's a completely reasonable-sounding mistake to assume "more GPUs" and "more capacity" are the same statement; Section 13.1's own table is the concrete case where they aren't.

## 13.2 Partitioning Layers Across Devices, and the Real Transfer Between Them

### Intuition

Once splitting the model is the plan, something has to decide *which* device holds *which* part of it. The simplest possible answer -- and the one this section builds -- is to treat the model's layers the way Chapter 12 treated a batch of data: cut them into equal, contiguous, non-overlapping ranges, and hand one range to each device. That's pure bookkeeping, identical in shape to Chapter 12's own batch-sharding arithmetic, just partitioning layer indices instead of sample indices. But a model split this way isn't independent pieces the way data-parallel replicas were -- Megatron-LM's own paper describes exactly this shape of split as "layer-wise pipeline parallelism," where "groups of operations are performed on one device before the outputs are passed to the next device in the pipeline." Every device downstream needs the *output* of the device before it, which means something has to physically carry that output -- the activation tensor -- across the real device boundary. That's not bookkeeping. That's a real transfer, and this book already has the API for it.

```text
Assigning layers (pure arithmetic,          Moving activations between them
needs no device):                          (a REAL transfer, needs a device):

  computeLayerRange(96, 4, rank)              dev0 finishes layer 23's output
       |                                            |
       v                                            v cudaMemcpyPeer()
  rank 0: layers [ 0, 24)                     dev1 receives it as layer 24's input
  rank 1: layers [24, 48)                            |
  rank 2: layers [48, 72)                            v cudaMemcpyPeer()
  rank 3: layers [72, 96)                     dev2 receives it as layer 48's input
                                                      ...
```

### Background

Assigning layer ranges is the exact same kind of arithmetic Chapter 12's `computeShard()` did for batches -- divide a known total by a known device count, compute offsets, and it genuinely succeeds regardless of how many real devices exist, because it doesn't need any. Moving the activation tensor produced at each layer boundary is a different kind of call entirely: it's the same real `cudaMemcpyPeer()` this book introduced in Chapter 5 for exactly this reason -- a real device-to-device transfer -- now carrying activations instead of the raw application data Chapter 5 first demonstrated it with. GPT-3's own real dimensions (Brown et al. 2020, Table 2.1: 96 layers, `d_model` = 12288, and a context window `n_ctx` = 2048 tokens applied uniformly across every model size in that paper) size a real, concrete activation tensor below, rather than an arbitrary placeholder buffer.

```cpp
// Chapter 13: Model Parallelism -- When One GPU Can't Hold the Weights
// 37_layer_partition_activation_handoff.cu
//
// Model parallelism needs exactly two things: an assignment of which
// device holds which LAYERS (pure host arithmetic, needs no device --
// structurally identical to Chapter 12's own batch-sharding formula,
// just partitioning layer indices instead of sample indices), and a
// real transfer that hands the ACTIVATION tensor produced by one
// device's last layer to the next device's first layer. That second
// part is not a new API -- it's the same real cudaMemcpyPeer() this
// book introduced in Chapter 5 and has used ever since, now moving a
// different kind of data across the same real device boundary.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

// Splits TOTAL_LAYERS into `worldSize` equal, non-overlapping,
// contiguous ranges, and returns device `rank`'s own [start, end)
// range of layer indices. Structurally identical to Chapter 12's
// computeShard() -- only what's being partitioned (layers, not
// samples) has changed. Pure host arithmetic; needs no device.
void computeLayerRange(int totalLayers, int worldSize, int rank, int* start, int* end) {
    int perDevice = totalLayers / worldSize; // assumes an even split
    *start = rank * perDevice;
    *end = *start + perDevice;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // GPT-3 175B's own real depth: 96 transformer layers (Brown et al.
    // 2020, Table 2.1).
    const int TOTAL_LAYERS = 96;

    printf("\nPartitioning %d layers across this book's own real "
           "deviceCount (%d):\n", TOTAL_LAYERS, deviceCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        int start, end;
        computeLayerRange(TOTAL_LAYERS, deviceCount, rank, &start, &end);
        printf("  rank %d: layers [%d, %d)\n", rank, start, end);
    }
    printf("  (%d device(s) -- nothing to print above.)\n", deviceCount);

    // Hypothetically, if there WERE 4 devices: every rank gets a
    // different, non-overlapping, equal-sized run of consecutive
    // layers -- genuinely computed, not illustrative.
    const int HYPOTHETICAL_WORLD_SIZE = 4;
    printf("\nThe same formula, hypothetically, with world size %d:\n", HYPOTHETICAL_WORLD_SIZE);
    for (int rank = 0; rank < HYPOTHETICAL_WORLD_SIZE; ++rank) {
        int start, end;
        computeLayerRange(TOTAL_LAYERS, HYPOTHETICAL_WORLD_SIZE, rank, &start, &end);
        printf("  rank %d: layers [%d, %d)\n", rank, start, end);
    }

    // Partitioning layer INDICES needs no device, same as Chapter 12's
    // batch sharding. Moving the ACTIVATION TENSOR that rank 0's last
    // layer produced into rank 1's first layer's input is different --
    // it is a real, unavoidable device-to-device transfer, sized by
    // GPT-3's own real dimensions: a sequence of 2048 tokens (Brown et
    // al.'s own context length, nctx) x 12288 features per token
    // (d_model), stored fp16 (2 bytes/element) the way Chapter 1's own
    // inference-footprint row already did.
    const size_t SEQ_LEN = 2048;
    const size_t D_MODEL = 12288;
    const size_t ACTIVATION_ELEMS = SEQ_LEN * D_MODEL;
    const size_t ACTIVATION_BYTES = ACTIVATION_ELEMS * 2; // fp16
    printf("\nOne layer-boundary activation tensor: %zu tokens x %zu "
           "features, fp16 -> %.2f MB that must genuinely cross the "
           "device boundary between rank 0 and rank 1 at every forward "
           "pass.\n", SEQ_LEN, D_MODEL, ACTIVATION_BYTES / 1.0e6);

    float* activation = nullptr; // never successfully allocated -- deviceCount is 0
    cudaError_t eMemcpyPeer = cudaMemcpyPeer(activation, 1, activation, 0, ACTIVATION_BYTES);
    printf("\ncudaMemcpyPeer(activation, dstDevice=1, activation, "
           "srcDevice=0, %.2f MB): %s (code %d)\n",
           ACTIVATION_BYTES / 1.0e6, cudaGetErrorString(eMemcpyPeer), (int)eMemcpyPeer);

    printf("\nPartitioning layer indices is arithmetic this chapter can\n"
           "genuinely run, exactly like Chapter 12's batch sharding. "
           "Handing\nthe activation tensor across the device boundary "
           "is the same real\ntransfer call this book has used since "
           "Chapter 5 -- it is not a new\nprimitive, only a new reason "
           "to call it. Section 13.3 checks the one\nthing that "
           "actually matters about doing this: whether a SPLIT forward\n"
           "pass computes the exact same thing an UNSPLIT one would.\n");

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).

Partitioning 96 layers across this book's own real deviceCount (0):
  (0 device(s) -- nothing to print above.)

The same formula, hypothetically, with world size 4:
  rank 0: layers [0, 24)
  rank 1: layers [24, 48)
  rank 2: layers [48, 72)
  rank 3: layers [72, 96)

One layer-boundary activation tensor: 2048 tokens x 12288 features, fp16 -> 50.33 MB that must genuinely cross the device boundary between rank 0 and rank 1 at every forward pass.

cudaMemcpyPeer(activation, dstDevice=1, activation, srcDevice=0, 50.33 MB): no CUDA-capable device is detected (code 100)

Partitioning layer indices is arithmetic this chapter can
genuinely run, exactly like Chapter 12's batch sharding. Handing
the activation tensor across the device boundary is the same real
transfer call this book has used since Chapter 5 -- it is not a new
primitive, only a new reason to call it. Section 13.3 checks the one
thing that actually matters about doing this: whether a SPLIT forward
pass computes the exact same thing an UNSPLIT one would.
```

Four ranks split GPT-3's 96 real layers into four genuinely computed, equal, 24-layer ranges -- the same host arithmetic Chapter 12 used, now partitioning depth instead of batch. The activation tensor at each of those three boundaries is not a token placeholder: 2048 tokens by 12288 features at 2 bytes each is a real 50.33 MB that would have to physically leave one device and arrive at another, every single forward pass, for every layer boundary this partition creates. `cudaMemcpyPeer()` reports the same honest `cudaErrorNoDevice` (code 100) this book's every real device-touching call has reported since Chapter 3 -- this section attempted a real transfer, and got a real, honest answer about why it can't happen here.

!!! warning "[COMMON TRAP] Thinking model parallelism replaces the peer-to-peer machinery Part 1 built"
    It's tempting to think of model parallelism as a fundamentally different technique from the P2P transfers and IPC this book spent Chapters 4 through 7 building -- after all, those chapters were about moving raw data, and this chapter is about running a model. But Section 13.2's own code shows the opposite: partitioning layers doesn't introduce a new transfer mechanism, it introduces a new *reason to call the same one*. The `cudaMemcpyPeer()` call above is byte-for-byte the same API Chapter 5 introduced, now carrying an activation tensor instead of Chapter 5's own example buffer. Every real multi-GPU technique this book builds -- collectives, data parallelism, and now model parallelism -- is built out of the same small set of real transfer primitives from Part 1, not a new one per technique.

## 13.3 Correctness: A Split Forward Pass Has to Match an Unsplit One, Exactly

### Intuition

Here is the one property that actually has to be true for Sections 13.1 and 13.2's partitioning to be worth doing at all: computing a model's forward pass split across four devices has to produce the *exact same result* as computing it, unsplit, on one device that happened to be big enough. That's a stronger claim than it might sound. This book's own collectives already showed a case where splitting a computation across devices genuinely can change a floating-point result -- Chapter 8 explicitly declined to verify reduce bit-identical across summation orders, citing real IEEE 754 non-associativity. Model parallelism doesn't have that risk, and the reason is structural: a model's layers already execute in one strict, unavoidable sequence -- layer 24 cannot start until layer 23 has finished, on one device or four. Partitioning that sequence across devices doesn't reorder a single operation; it only relocates *where* each already-ordered step happens to run. Nothing about the arithmetic itself changes, which is exactly why this section's check can demand bit-identical equality, not "close enough."

```text
Unsplit (one device, all 8 layers):        Split (4 devices, 2 layers each):

  x -> L0 -> L1 -> L2 -> L3 -> L4 ->         dev0: x -> L0 -> L1 -----+
       L5 -> L6 -> L7 -> out                                          |
                                              dev1:    L2 -> L3 <-----+
  SAME 8 operations, SAME order.                            |
                                              dev2:    L4 -> L5 <-----+
  Splitting only changes WHERE each                              |
  step runs -- never the SEQUENCE.           dev3:    L6 -> L7 <-----+ -> out

                                              SAME 8 operations, SAME order,
                                              just relocated across 4 devices.
```

### Background

The check below applies the exact same function, in the exact same order, two different ways: once in one unbroken loop over all 8 of a toy network's layers (the unsplit reference), and once across four simulated devices, each applying only its own 2-layer range (Section 13.2's own layer-partition formula, reused) before handing the running result to the next simulated device -- standing in for the real `cudaMemcpyPeer()` handoff Section 13.2 attempted. Because both paths call the identical layer function in the identical sequence, the two outputs are not merely expected to be close -- they are expected to be bit-for-bit equal, checked with exact `==`, not an epsilon tolerance.

```cpp
// Chapter 13: Model Parallelism -- When One GPU Can't Hold the Weights
// 38_model_parallel_forward_simulation.cpp
//
// Plain host C++ -- this chapter's real verification, since no real
// forward pass can run without a real device. A tiny 8-layer toy
// network (2-element vectors, one small affine transform + ReLU per
// layer) stands in for a real model deep enough that no single
// device could hold every layer. Two host arrays stand in for two
// simulated "devices" worth of memory holding different layer
// ranges (Section 13.2's own layer-partition formula, reused here),
// exchanging the SAME buffer instead of a real cudaMemcpyPeer() --
// the point being checked is whether the ANSWER changes when the
// computation is split across a partition boundary, not how the
// bytes physically move.
#include <cstdio>
#include <cmath>

const int LAYERS = 8;

// One layer's transform: y = ReLU(W_i * x + b_i), for a 2-element
// vector. Weights are small, deterministic, and distinct per layer
// index -- there is nothing to "hand-check" about the specific
// numbers; what this section checks is that calling this SAME
// function, in the SAME order, produces the SAME output whether it
// is called from one unbroken loop or from four separate simulated
// devices' loops.
void applyLayer(int layerIdx, double x0, double x1, double* y0, double* y1) {
    double w00 = 0.50 + 0.01 * layerIdx, w01 = 0.10;
    double w10 = 0.15,                   w11 = 0.55 + 0.01 * layerIdx;
    double b0 = 0.02 * layerIdx, b1 = 0.01 * layerIdx;

    double r0 = w00 * x0 + w01 * x1 + b0;
    double r1 = w10 * x0 + w11 * x1 + b1;

    *y0 = r0 > 0.0 ? r0 : 0.0; // ReLU
    *y1 = r1 > 0.0 ? r1 : 0.0; // ReLU
}

// Structurally identical to Section 13.2's computeLayerRange() (and
// Chapter 12's computeShard() before that) -- pure host arithmetic,
// needs no device.
void computeLayerRange(int totalLayers, int worldSize, int rank, int* start, int* end) {
    int perDevice = totalLayers / worldSize;
    *start = rank * perDevice;
    *end = *start + perDevice;
}

// The UNSPLIT reference: every layer applied in one unbroken loop,
// as if a single device held the entire 8-layer model.
void referenceForward(double x0, double x1, double* out0, double* out1) {
    double cur0 = x0, cur1 = x1;
    for (int layer = 0; layer < LAYERS; ++layer) {
        double n0, n1;
        applyLayer(layer, cur0, cur1, &n0, &n1);
        cur0 = n0; cur1 = n1;
    }
    *out0 = cur0; *out1 = cur1;
}

// The SPLIT forward pass: WORLD_SIZE simulated devices, each holding
// a different contiguous range of layers (Section 13.2's formula).
// Every rank applies only ITS OWN layers, in order, to whatever
// buffer the previous rank produced -- the same real dependency a
// real cudaMemcpyPeer() handoff would enforce, just without a real
// transfer call moving the bytes.
void splitForward(double x0, double x1, int worldSize, double* out0, double* out1) {
    double cur0 = x0, cur1 = x1;
    for (int rank = 0; rank < worldSize; ++rank) {
        int start, end;
        computeLayerRange(LAYERS, worldSize, rank, &start, &end);
        for (int layer = start; layer < end; ++layer) {
            double n0, n1;
            applyLayer(layer, cur0, cur1, &n0, &n1);
            cur0 = n0; cur1 = n1;
        }
        // Section 13.2's cudaMemcpyPeer() call is what would carry
        // (cur0, cur1) from rank's device to (rank+1)'s device here,
        // on real hardware. The buffer itself does not change value
        // when it crosses that boundary -- only where it physically
        // lives does.
    }
    *out0 = cur0; *out1 = cur1;
}

int main() {
    const int WORLD_SIZE = 4; // 8 layers / 4 devices = 2 layers/device, matching Section 13.2

    struct Case { double x0, x1; };
    const Case cases[] = { {1.0, 1.0}, {2.0, -1.0}, {0.5, 3.0} };

    bool allMatch = true;
    for (const auto& c : cases) {
        double refOut0, refOut1, splitOut0, splitOut1;
        referenceForward(c.x0, c.x1, &refOut0, &refOut1);
        splitForward(c.x0, c.x1, WORLD_SIZE, &splitOut0, &splitOut1);

        bool matches = (refOut0 == splitOut0) && (refOut1 == splitOut1);
        allMatch = allMatch && matches;

        printf("input (%.2f, %.2f): unsplit reference = (%.10f, %.10f), "
               "split across %d devices = (%.10f, %.10f) -- %s\n",
               c.x0, c.x1, refOut0, refOut1, WORLD_SIZE, splitOut0, splitOut1,
               matches ? "PASS (bit-identical)" : "FAIL");
    }

    printf("\nAll %zu input(s) bit-identical between the unsplit reference "
           "and the %d-device split forward pass: %s\n",
           sizeof(cases) / sizeof(cases[0]), WORLD_SIZE,
           allMatch ? "PASS" : "FAIL");

    printf("\nUnlike Chapter 8's floating-point reduce (order-sensitive --\n"
           "explicitly NOT verified bit-identical there), splitting a\n"
           "strictly SEQUENTIAL computation across devices never reorders\n"
           "any operation -- it only relocates WHERE each already-ordered\n"
           "step runs. That is why this section's match is bit-identical,\n"
           "not merely equal up to floating-point rounding.\n");

    return allMatch ? 0 : 1;
}
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
input (1.00, 1.00): unsplit reference = (0.3285223992, 0.2628870847), split across 4 devices = (0.3285223992, 0.2628870847) -- PASS (bit-identical)
input (2.00, -1.00): unsplit reference = (0.3181013191, 0.2436102732), split across 4 devices = (0.3181013191, 0.2436102732) -- PASS (bit-identical)
input (0.50, 3.00): unsplit reference = (0.3560831617, 0.3077149331), split across 4 devices = (0.3560831617, 0.3077149331) -- PASS (bit-identical)

All 3 input(s) bit-identical between the unsplit reference and the 4-device split forward pass: PASS

Unlike Chapter 8's floating-point reduce (order-sensitive --
explicitly NOT verified bit-identical there), splitting a
strictly SEQUENTIAL computation across devices never reorders
any operation -- it only relocates WHERE each already-ordered
step runs. That is why this section's match is bit-identical,
not merely equal up to floating-point rounding.
```

All three test inputs match bit-for-bit, exactly as the structural argument in this section's Intuition predicted -- there was never a floating-point-ordering risk to check for in the first place, since splitting a strictly sequential computation never changes the order any operation happens in. That correctness comes with a real cost this chapter has deliberately not measured: GPipe's own paper names the naive version of exactly this split -- "consecutive groups of layers... partitioned into cells... placed on a separate accelerator" -- and states plainly that "the naive model parallelism strategy leads to severe under-utilization due to the sequential dependency of the network." The reason is visible right in this section's own diagram: while device 1 is running layers 2 and 3, device 0 has nothing left to do until the *next* input arrives -- it already finished layers 0 and 1 and handed off the result. Every device downstream of the first one sits idle waiting for its input, and the first device sits idle waiting for the *next* forward pass to start. This chapter's correctness guarantee is real and unconditional; the idle time it creates is also real, and Chapter 15 -- whose own title already promises "the bubble that costs you" -- is where this book quantifies and then fixes it.

!!! warning "[COMMON TRAP] Assuming model parallelism parallelizes computation"
    The name is misleading. Model parallelism does not make a model's *compute* run in parallel across devices the way data parallelism's replicas genuinely do -- Chapter 12's four replicas really were computing four independent forward passes at the same time. Section 13.3's own diagram shows the opposite is true here: layer 24 cannot start until layer 23's output has arrived, no matter which device holds which layer, so at any given instant only *one* device in the whole partition is doing useful work on any single input. What model parallelism parallelizes is *memory* -- it lets a model that doesn't fit on one device exist across several -- not compute. Confusing the two leads to expecting a speedup that a naive model-parallel split, on its own, was never going to provide.

## Chapter Summary

Data parallelism's entire premise -- replicate the model, split the data -- has a hard floor: the model has to fit on one device to begin with, and Section 13.1 showed in real bytes (reusing Chapter 1's own already-locked GPT-3 figures as a self-consistency check) that replicating a model which doesn't fit does nothing to fix that; only dividing the model itself changes the per-device memory requirement. Section 13.2 built that division the same way Chapter 12 built its data shards -- pure host arithmetic partitioning layer indices instead of sample indices -- and showed that the real cost of this split isn't a new API at all: it's the same `cudaMemcpyPeer()` call this book introduced in Chapter 5, now carrying a real, GPT-3-sized activation tensor across a real device boundary at every layer partition. Section 13.3 -- this chapter's actual verification, since no real forward pass could run on this machine -- proved the one property that makes any of this worth doing: a forward pass split across devices produces a bit-identical result to an unsplit one, because partitioning a strictly sequential computation only relocates *where* each step runs, never the *order* it runs in, a strictly stronger guarantee than this book's own collectives could offer about floating-point summation order. That correctness has a real, named, currently-unquantified cost -- GPipe's own documented "severe under-utilization due to the sequential dependency of the network" -- which this chapter deliberately left unmeasured. Chapter 14 splits a *single layer* across devices instead of splitting *between* layers; Chapter 15 comes back to this chapter's idle-time cost directly and quantifies it.

## Self-Check Questions

1. Using Section 13.1's own numbers, explain why running 8 data-parallel replicas of GPT-3 does not reduce the 35-H100-SXM-GPU figure any single replica needs.
2. What real API call does Section 13.2 use to move an activation tensor between devices, and which earlier chapter introduced it?
3. Section 13.2 computes a real activation tensor size of 50.33 MB. Using the code's own numbers, explain what each of the three factors multiplied together represents.
4. Section 13.3 checks its result with exact `==`, not an epsilon tolerance. Explain, using the chapter's own structural argument, why that is a valid check here but was explicitly NOT used for Chapter 8's reduce.
5. Quote GPipe's own description of the problem with naive model parallelism, and explain, using Section 13.3's own diagram, which device is idle and when.
6. Why does this chapter say model parallelism parallelizes memory, not compute? Contrast this directly with what Chapter 12's data parallelism actually parallelizes.
7. Megatron-LM's paper distinguishes "layer-wise pipeline parallelism" from "distributed tensor computation." Which of those two does this chapter build, and which is Chapter 14's subject?
8. Suppose a model has 50 real layers and is split 5 ways using Section 13.2's `computeLayerRange()` formula. Compute, by hand, which layer indices rank 3 would hold.

## Where We Go Next

Chapter 14 stays inside a single layer instead of splitting between layers -- tensor parallelism divides one large matrix multiply itself across devices, the "distributed tensor computation" Megatron-LM's own paper contrasts with this chapter's layer-wise split. Chapter 15 then returns directly to this chapter's own unresolved cost: the idle time a naive layer-wise split creates, quantified and fixed with the micro-batch pipelining GPipe's paper introduces.

## Worked Solutions

**1.** Section 13.1's own closed-form model shows that replication multiplies *total cluster memory* (8 replicas x 2800 GB = 22400 GB) but leaves the *per-replica* requirement completely unchanged -- each of the 8 replicas is still, individually, a full 2800 GB copy of GPT-3's training state, needing the same 35 H100 SXM GPUs' worth of memory it always needed. Adding more replicas adds more copies of the same unsolved problem; it never divides the problem itself.

**2.** Section 13.2 uses `cudaMemcpyPeer()`, the same real device-to-device transfer call Chapter 5 introduced for explicit peer transfers.

**3.** The three factors are `SEQ_LEN` (2048, GPT-3's own context window, `n_ctx`, from Brown et al.'s Table 2.1), `D_MODEL` (12288, GPT-3's own per-token feature width from the same table), and 2 bytes per element for fp16 storage. Together they represent the size of one full activation tensor -- every token in a 2048-token sequence, each represented by a 12288-element feature vector, stored at 2 bytes per value -- that has to physically exist at a layer boundary and cross a device-to-device transfer.

**4.** Chapter 8 explicitly declined to verify its reduce bit-identical because summing floating-point values in a different order can produce a genuinely different result, due to real IEEE 754 non-associativity -- splitting a *reduction* across devices can change the order values are combined in. Section 13.3's split forward pass never combines values from different sources at all; it only ever applies one layer's function to the single running result of the layer before it, in the exact same sequence an unsplit run would use. Since no operation's order changes, there is no floating-point-ordering risk to guard against, so exact `==` is a valid, not merely convenient, check.

**5.** GPipe's paper states: "the naive model parallelism strategy leads to severe under-utilization due to the sequential dependency of the network." Using Section 13.3's own diagram (8 layers split across 4 devices, 2 layers each): while device 1 is computing layers 2 and 3 on the current input, device 0 has already finished layers 0 and 1 for that same input and has nothing left to do until the *next* input begins its own forward pass -- so device 0 sits idle during the time devices 1 through 3 are still working on the input it already finished processing.

**6.** Model parallelism lets a model that is too large to fit in one device's memory exist across several devices' memory at once -- it solves a capacity problem. It does not make the model's own computation run any faster or more in parallel, because Section 13.3 established that the layers still execute in one strict, unavoidable sequence regardless of which device holds which layer -- at any instant, only one device is doing useful work on a given input. Chapter 12's data parallelism is the opposite: its replicas genuinely do run their forward and backward passes on different data at the same time, so it parallelizes compute (throughput on independent data), not capacity (it explicitly requires the model to already fit on one device).

**7.** This chapter builds "layer-wise pipeline parallelism" -- partitioning the model's own layers, as whole units, across devices. Chapter 14 is Megatron-LM's other category, "distributed tensor computation" -- splitting a single large matrix multiply's own computation across devices, rather than splitting between whole layers.

**8.** `perDevice = 50 / 5 = 10`. For rank 3: `start = rank * perDevice = 3 * 10 = 30`, `end = start + perDevice = 30 + 10 = 40`. Rank 3 would hold layers `[30, 40)` -- layers 30 through 39.

---

**Sources cited in this chapter:**

- Rajbhandari, S. et al., ["ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054) (SC'20) -- the real 16-bytes-per-parameter mixed-precision Adam breakdown (already cited in Chapter 1), reused here, and the paper's own GPT-2 1.5B / 24 GB worked example, cited independently for this chapter's own memory-shard model.
- Huang, Y. et al., ["GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"](https://arxiv.org/abs/1811.06965) -- the exact quotes "the naive model parallelism strategy leads to severe under-utilization due to the sequential dependency of the network" and "consecutive groups of layers... partitioned into cells... placed on a separate accelerator."
- Shoeybi, M. et al., ["Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"](https://arxiv.org/abs/1909.08053) -- the memory-constraint motivation for model parallelism, and the "layer-wise pipeline parallelism" vs. "distributed tensor computation" distinction this chapter uses to preview Chapter 14.
- Brown, T. et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (2020) -- GPT-3's real architecture from Table 2.1: 96 layers, `d_model` = 12288, and the paper's own uniform context window, `n_ctx` = 2048 tokens.
- [NVIDIA H100 GPU product page](https://www.nvidia.com/en-us/data-center/h100/) -- H100 SXM (80 GB) and H100 NVL (94 GB) memory capacities, already cited in Chapters 1 and 2, reused here unchanged.
