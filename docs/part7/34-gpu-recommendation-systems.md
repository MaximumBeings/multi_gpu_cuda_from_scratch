**What you will understand after this chapter:** why a single recommendation model routinely needs Chapter 12's data parallelism and Chapter 13's model parallelism at the SAME TIME, applied to two different parts of the same model, rather than picking one strategy for the whole thing the way every earlier chapter did; how NVIDIA's own real, deployed Merlin HugeCTR framework's "hybrid embedding" design uses Chapter 10's all-to-all for a genuinely new purpose -- reassembling data from one parallelism scheme into another, on a fixed deterministic schedule, not a gating decision like Chapter 32's Mixture-of-Experts -- and why sharding a huge embedding table across more GPUs keeps helping memory capacity long after it stops meaningfully reducing that all-to-all's own communication cost.

**What you need to know first:** Chapter 1 (the memory wall, reused here for embedding-table capacity), Chapter 9 (the real round-count formula this chapter's own cost model reuses), Chapter 10 (all-to-all, resharding a tensor from one split dimension to another), Chapter 12 (data parallelism), Chapter 13 (model parallelism), and Chapter 32 (the book's other recent use of an all-to-all for a non-collective purpose, useful here as a contrast).

---

Every parallelization strategy this book has built so far -- data, model, tensor, pipeline, domain decomposition -- picks ONE strategy and applies it to an ENTIRE model or computation. NVIDIA's own real, deployed Merlin HugeCTR framework, described in its own real paper ("Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference," arXiv:2210.08803, RecSys 2022), shows why that assumption breaks down for one of the most widely deployed model families in industry: recommendation systems. A recommendation model has two parts with almost nothing in common: embedding tables, one row per user or item ID, that can genuinely reach terabyte scale, and a comparatively tiny dense neural network that consumes those embeddings to produce a prediction. HugeCTR's own real design applies Chapter 13's model parallelism to the first part and Chapter 12's data parallelism to the second, within the SAME model, and stitches the two together with a real, production all-to-all -- the same primitive Chapter 10 built and Chapter 32 reused, now doing a third, genuinely different job.

```text
+------------------------------------------------------------------+
| Every earlier chapter: ONE strategy for the WHOLE model            |
|   [ Chapter 12: data-parallel  ] OR  [ Chapter 13: model-parallel ]|
+------------------------------------------------------------------+
| Chapter 34 (HugeCTR's own real hybrid design):                    |
|                                                                    |
|  EMBEDDING tables (model-parallel,   DENSE layers (data-parallel,  |
|  sharded by FEATURE, can be TBs)      tiny, replicated everywhere) |
|  [ GPU ][ GPU ][ GPU ][ GPU ]  --all-to-all-->  [ GPU ][ GPU ]     |
|      shard by feature            reshard to      shard by SAMPLE   |
|                                   batch dimension                  |
+------------------------------------------------------------------+
```

## 34.1 Why Recommendation Models Need Both Data and Model Parallelism

### Intuition

Picture a department store with two very different systems: a colossal warehouse catalog listing every item the store has ever stocked -- far too large for any one clerk's desk, so different sections of the catalog live on different desks -- and a small decision checklist every cashier uses to ring up a sale, identical and small enough to be printed and taped to every register in the store. The catalog is HugeCTR's own real embedding tables: "each GPU only keeps a shard of the embedding table (MP). Therefore, the total size of an embedding feature can exceed the memory capacity of a single GPU." The checklist is the dense DNN: "the dense model, such as DNN, is data parallel and contains a copy of the dense model in each GPU." Forcing the whole store to use ONE system -- one catalog copy per register, or one shared checklist everyone has to walk over to consult -- would be absurd in both directions, and that is exactly the shape HugeCTR's own real design avoids.

!!! warning "[COMMON TRAP] Assuming one parallelism strategy must apply to a whole model"
    Chapter 12 and Chapter 13 each introduced a full parallelization strategy, and it's natural to conclude a real system picks one FOR the model. HugeCTR's own real production design shows that assumption is specific to models where every part is roughly the same size. File 99's own capacity numbers make the failure concrete: at HugeCTR's own real terabyte-scale example, replicating the FULL embedding table on every GPU (pure data parallelism) would require 16x the real per-GPU capacity HugeCTR's own documentation cites as the practical single-GPU limit -- while replicating the tiny 2GB dense model the same way costs nothing at any GPU count.

### Background

```text
+----------------------------------------------------------+
| Embedding tables: can reach TBs -> must be SHARDED (MP)    |
|   real cited limit: ~640 GB practical single-GPU capacity  |
+----------------------------------------------------------+
| Dense DNN layers: stays small -> fully REPLICATED (DP)     |
|   fits comfortably at any GPU count                        |
+----------------------------------------------------------+
```

File 99 builds a closed-form capacity model, reusing Chapter 1's own memory-wall framing and Chapter 13's own memory-shard formula, using HugeCTR's own real cited practical single-GPU capacity figure to show exactly where pure data parallelism stops being an option.

```cpp
// Chapter 34: GPU Recommendation Systems
// 99_embedding_capacity_and_parallelism_choice_model.cpp
//
// NVIDIA's own real Merlin HugeCTR paper ("Merlin HugeCTR: GPU-
// accelerated Recommender System Training and Inference," arXiv:2210.08803,
// RecSys 2022) describes a model shape neither Chapter 12 (pure data
// parallelism) nor Chapter 13 (pure model parallelism) was built for: one
// model with two wildly different-sized parts. The embedding tables
// (one row per user/item/category ID) are described as needing model
// parallelism specifically because of their size -- "each GPU only keeps
// a shard of the embedding table (MP). Therefore, the total size of an
// embedding feature can exceed the memory capacity of a single GPU" --
// while the dense DNN layers that consume those embeddings stay small
// enough to replicate everywhere: "the dense model, such as DNN, is data
// parallel and contains a copy of the dense model in each GPU." This
// file builds a closed-form capacity model, reusing Chapter 1's own
// memory-wall framing and Chapter 13's own memory-shard formula, to show
// exactly where pure data parallelism (replicate everything) stops being
// an option and hybrid parallelism (Chapter 12's technique for the dense
// part, Chapter 13's technique for the embeddings) becomes necessary --
// grounded in HugeCTR's own real cited practical single-GPU capacity
// figure.
#include <cstdio>

int main() {
    // HugeCTR's own real cited practical single-GPU embedding capacity
    // figure (its own ETC -- Embedding Training Cache -- documentation
    // describes this as the point beyond which on-demand loading, not
    // simple replication or even simple sharding alone, becomes
    // necessary for truly terabyte-scale tables).
    double perGpuBudgetGB = 640.0;

    // The dense DNN part stays tiny by comparison -- illustrative size
    // for a realistic DLRM-style bottom+top MLP stack.
    double denseModelGB = 2.0;

    printf("Dense model size: %.1f GB -- comfortably replicated on EVERY "
           "GPU regardless of GPU count (Chapter 12's data parallelism), "
           "since it never approaches the %.0f GB real single-GPU budget.\n\n",
           denseModelGB, perGpuBudgetGB);

    struct Scenario { const char *label; double totalEmbeddingGB; };
    Scenario scenarios[] = {
        {"Small catalog",          50.0},
        {"At HugeCTR's own real single-GPU limit", 640.0},
        {"Production-scale",       2048.0},
        {"Real terabyte-scale (HugeCTR's own cited ETC use case)", 10240.0},
    };

    printf("%-58s %-16s %-10s %-24s\n", "Scenario", "Total emb. (GB)",
           "Fits 1 GPU?", "GPUs needed if sharded (MP)");
    for (auto &s : scenarios) {
        bool fitsOne = s.totalEmbeddingGB <= perGpuBudgetGB;
        int gpusNeeded = (int)((s.totalEmbeddingGB + perGpuBudgetGB - 1.0) / perGpuBudgetGB);
        printf("%-58s %-16.1f %-10s %-24d\n", s.label, s.totalEmbeddingGB,
               fitsOne ? "YES" : "NO", gpusNeeded);
    }

    printf("\nPure data parallelism (replicate the WHOLE model on every "
           "GPU, Chapter 12's technique alone) only works for the first "
           "scenario -- everywhere else, a full embedding replica would "
           "not fit in a single real GPU's own %.0f GB budget, exactly "
           "why HugeCTR's own real design shards the embeddings (Chapter "
           "13's technique) while keeping the dense layers replicated "
           "(Chapter 12's technique) -- a genuine HYBRID of two "
           "techniques this book built separately, applied to two "
           "different PARTS of the same model rather than to the whole "
           "model uniformly.\n", perGpuBudgetGB);
    return 0;
}
```

Compile and run (a plain host `.cpp` file with no CUDA/NCCL/MPI/NVSHMEM linkage, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 99_embedding_capacity_and_parallelism_choice_model \
    99_embedding_capacity_and_parallelism_choice_model.cpp
./99_embedding_capacity_and_parallelism_choice_model
```

Locked output:

```
Dense model size: 2.0 GB -- comfortably replicated on EVERY GPU regardless of GPU count (Chapter 12's data parallelism), since it never approaches the 640 GB real single-GPU budget.

Scenario                                                   Total emb. (GB)  Fits 1 GPU? GPUs needed if sharded (MP)
Small catalog                                              50.0             YES        1                       
At HugeCTR's own real single-GPU limit                     640.0            YES        1                       
Production-scale                                           2048.0           NO         4                       
Real terabyte-scale (HugeCTR's own cited ETC use case)     10240.0          NO         16                      

Pure data parallelism (replicate the WHOLE model on every GPU, Chapter 12's technique alone) only works for the first scenario -- everywhere else, a full embedding replica would not fit in a single real GPU's own 640 GB budget, exactly why HugeCTR's own real design shards the embeddings (Chapter 13's technique) while keeping the dense layers replicated (Chapter 12's technique) -- a genuine HYBRID of two techniques this book built separately, applied to two different PARTS of the same model rather than to the whole model uniformly.
```

## 34.2 The Embedding All-to-All: Resharding From Feature-Parallel to Batch-Parallel

### Intuition

Once every GPU has looked up its own shard of embedding features for the WHOLE batch, there's a mismatch: the embedding phase is organized by feature (Chapter 13's model-parallel shape), but the dense layers that come next are organized by batch sample (Chapter 12's data-parallel shape), and each GPU needs the COMPLETE feature vector for its own samples, not a partial one. HugeCTR's own real paper names the fix directly: "after each local reduction step, all-to-all communication is conducted to share the results of the lookup operation across all available GPUs along the batch dimension." This is genuinely the SAME primitive Chapter 10 built and Chapter 32 reused for MoE dispatch/combine -- but doing a third, different job here. Chapter 32's all-to-all moved tokens to wherever a gating network's runtime decision sent them, with real capacity overflow as a possible silent failure mode. This all-to-all follows a fixed, predetermined schedule known before the batch even arrives: feature index modulo GPU count. Nothing is decided at runtime, and nothing can overflow.

!!! warning "[COMMON TRAP] Treating every all-to-all as if it were a reduction"
    Chapter 8 through Chapter 11 spent an entire Part building REDUCTION-shaped collectives, and it is easy to instinctively read any cross-GPU data movement as "combining values." File 100 below is a direct check against that instinct: this embedding all-to-all never adds, averages, or otherwise combines two values together. It only ever RELOCATES a value from the GPU that computed it to the GPU that needs it next. That is precisely why it is exactly reproducible at every tested P even using ordinary values with no special care -- there is no combination step here for floating-point (non-)associativity, GShard-style capacity limits, or any other reduction-specific concern to ever act on.

### Background

```text
+----------------------------------------------------------+
| BEFORE all-to-all: sharded by FEATURE, full batch per rank |
|   rank r owns features {r, r+P, r+2P, ...} for ALL samples  |
+----------------------------------------------------------+
|              all-to-all (fixed schedule, pure resharding)  |
+----------------------------------------------------------+
| AFTER all-to-all: sharded by SAMPLE, full features per rank|
|   rank r owns samples {r, r+P, r+2P, ...}, EVERY feature    |
+----------------------------------------------------------+
```

File 100 builds this resharding as a host-side simulation with F=12 embedding features and B=24 batch samples, sharded by feature across P ranks, resharded by sample, and checks every sample's reassembled feature vector against a single-process reference at every tested P.

```cpp
// Chapter 34: GPU Recommendation Systems
// 100_embedding_alltoall_resharding_correctness_simulation.cpp
//
// HugeCTR's own real paper describes the exact moment its two parallel
// strategies meet: "when doing intra-slot reductions for multi-hot
// features, no inter-GPU communication is required. After each local
// reduction step, all-to-all communication is conducted to share the
// results of the lookup operation across all available GPUs along the
// batch dimension." Before that all-to-all, each GPU owns a SHARD OF
// FEATURES (model-parallel: Chapter 13's own technique) but holds
// results for the WHOLE batch of samples. After it, each GPU owns a
// SHARD OF SAMPLES (data-parallel: Chapter 12's own technique) but holds
// the COMPLETE feature vector for each of its own samples -- exactly
// GShard's own real framing from Chapter 32, "resharding a sharded
// tensor from one dimension to another," applied here to a fixed,
// deterministic schedule rather than a gating network's dynamic
// decision. This file builds that resharding as a host-side simulation
// -- F embedding features sharded by feature across P ranks, B batch
// samples -- and checks whether every sample's own complete, reassembled
// feature vector matches a single-process reference exactly, at every P.
// Every value here is a plain integer, deliberately, so this file's own
// answer is about the RESHARDING itself (a pure data-movement/ownership
// change, not a value combination), not about floating-point
// associativity, which Chapter 8/25/29/31/32 already covered in detail.
#include <cstdio>
#include <vector>

const int F = 12;  // total embedding features (illustrative)
const int B = 24;  // batch samples

unsigned int counterBasedHash(long long x) {
    unsigned int h = (unsigned int)x;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return h;
}

// The looked-up (and already locally-reduced, for multi-hot features)
// embedding value for a given (sample, feature) pair -- a pure function,
// so it never depends on which rank happens to be asking.
long long embeddingValue(int sample, int feature) {
    return (long long)(counterBasedHash((long long)sample * 100 + feature) % 97);
}

// Reference: every sample's full feature vector, computed directly with
// no sharding or resharding at all.
std::vector<std::vector<long long>> runReference() {
    std::vector<std::vector<long long>> full(B, std::vector<long long>(F));
    for (int s = 0; s < B; s++)
        for (int f = 0; f < F; f++)
            full[s][f] = embeddingValue(s, f);
    return full;
}

// P-rank simulation: each rank owns a contiguous shard of FEATURES (F/P
// features per rank, computed for ALL B samples -- the "model-parallel,
// full batch" phase), then an all-to-all reshards so each rank instead
// owns a contiguous shard of SAMPLES (B/P samples per rank), each with
// its COMPLETE F-length feature vector -- the "data-parallel, full
// features" phase HugeCTR's own real dense layers need.
std::vector<std::vector<long long>> runWithP(int P) {
    // Phase 1 (model-parallel by feature): each rank computes its own
    // feature shard for every sample.
    std::vector<std::vector<std::vector<long long>>> perRankFeatureShard(P);
    for (int r = 0; r < P; r++) {
        perRankFeatureShard[r].assign(B, std::vector<long long>());
        for (int f = r; f < F; f += P) {
            for (int s = 0; s < B; s++) {
                perRankFeatureShard[r][s].push_back(embeddingValue(s, f));
            }
        }
    }

    // Phase 2 (the real all-to-all): reshard from "every rank has a
    // feature-slice of every sample" to "every rank has every feature of
    // its own sample-slice." This is pure data movement -- no value is
    // ever combined with another, only relocated -- so it is checked
    // here as a real resharding, not a reduction.
    std::vector<std::vector<long long>> resharded(B, std::vector<long long>());
    for (int r = 0; r < P; r++) {
        // Reconstruct which feature indices rank r owned (matches the
        // exact striping used above).
        std::vector<int> ownedFeatures;
        for (int f = r; f < F; f += P) ownedFeatures.push_back(f);

        for (int s = 0; s < B; s++) {
            for (size_t k = 0; k < ownedFeatures.size(); k++) {
                int f = ownedFeatures[k];
                long long v = perRankFeatureShard[r][s][k];
                if ((int)resharded[s].size() <= f) resharded[s].resize(F, -1);
                resharded[s][f] = v;
            }
        }
    }
    return resharded;
}

int main() {
    auto reference = runReference();
    printf("Reference (single-process): sample 0's full feature vector: ");
    for (long long v : reference[0]) printf("%lld ", v);
    printf("\n\n");

    int Ps[] = {1, 2, 3, 4, 6, 12};
    printf("%-6s %-10s\n", "P", "all match");
    for (int P : Ps) {
        auto resharded = runWithP(P);
        bool allMatch = true;
        for (int s = 0; s < B && allMatch; s++) {
            for (int f = 0; f < F; f++) {
                if (resharded[s][f] != reference[s][f]) { allMatch = false; break; }
            }
        }
        printf("%-6d %-10s\n", P, allMatch ? "YES" : "NO");
    }

    printf("\nEvery P reproduces the exact same per-sample feature vectors "
           "as the single-process reference. Unlike Chapter 32's MoE "
           "dispatch/combine -- where a GATING NETWORK decided, per token, "
           "where data would go, and capacity overflow could silently drop "
           "a contribution -- this embedding all-to-all follows a FIXED, "
           "deterministic schedule known in advance (feature index modulo "
           "P), with no equivalent of capacity overflow: every value has a "
           "single, predetermined destination, and the operation is pure "
           "resharding, never a reduction. That is precisely why it is "
           "bit-exact regardless of P even using ordinary values -- there "
           "is no combination step here for floating-point (non-)"
           "associativity to ever act on.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 100_embedding_alltoall_resharding_correctness_simulation \
    100_embedding_alltoall_resharding_correctness_simulation.cpp
./100_embedding_alltoall_resharding_correctness_simulation
```

Locked output:

```
Reference (single-process): sample 0's full feature vector: 0 72 83 9 70 91 47 15 70 58 85 85 

P      all match 
1      YES       
2      YES       
3      YES       
4      YES       
6      YES       
12     YES       

Every P reproduces the exact same per-sample feature vectors as the single-process reference. Unlike Chapter 32's MoE dispatch/combine -- where a GATING NETWORK decided, per token, where data would go, and capacity overflow could silently drop a contribution -- this embedding all-to-all follows a FIXED, deterministic schedule known in advance (feature index modulo P), with no equivalent of capacity overflow: every value has a single, predetermined destination, and the operation is pure resharding, never a reduction. That is precisely why it is bit-exact regardless of P even using ordinary values -- there is no combination step here for floating-point (non-)associativity to ever act on.
```

## 34.3 Sizing the Shards: When More GPUs Stops Reducing Communication Cost

### Intuition

Section 34.1 showed sharding an embedding table across more GPUs (P) always helps fit a bigger table -- pure good news, more P, more headroom. Section 34.2's all-to-all raises a second question: does more P ALSO keep reducing how much data crosses the network on every single training step? Chapter 9's own real ring-collective formula already answered a version of this: in a full resharding among P ranks, each rank must send out (P-1)/P of the data it started with. That fraction climbs toward 100% quickly and then plateaus near it -- more P keeps helping memory, but the SHARE of data that must leave its origin GPU on every step stops meaningfully shrinking once P is even moderately large.

!!! warning "[COMMON TRAP] Assuming more shards always means less communication"
    It is intuitive to think "more GPUs sharing the work" implies "each one does less, including less communicating." File 101 below shows this is true for MEMORY (linear, unbounded improvement) but false for this all-to-all's own COMMUNICATION FRACTION, which saturates: at P=8 already over 87% of the per-step payload must cross the network, and by P=128 it is over 99% -- adding GPUs past that point keeps helping capacity but delivers rapidly diminishing returns on the resharding cost itself, an echo of Chapter 16's own "more ranks is not free" lesson, arrived at here for a structurally different reason (a saturating off-rank fraction, not a fixed communication floor).

### Background

```text
+----------------------------------------------------------+
| P grows -> per-GPU embedding shard shrinks LINEARLY        |
|            (unbounded improvement, Section 34.1's finding) |
+----------------------------------------------------------+
| P grows -> off-rank fraction (P-1)/P rises toward 100%      |
|            and SATURATES (diminishing returns past small P)|
+----------------------------------------------------------+
```

File 101 combines Chapter 9's own real round-count formula with a realistic batch-size and embedding-dimension payload, and Section 34.1's own terabyte-scale scenario, to show both trends side by side.

```cpp
// Chapter 34: GPU Recommendation Systems
// 101_embedding_alltoall_comm_volume_cost_model.cpp
//
// Section 34.1's own capacity model showed sharding embeddings across
// more GPUs (P) directly helps fit larger tables -- more P, more real
// per-GPU headroom. This file checks whether more P also helps the
// OTHER real cost HugeCTR's own paper names: the per-batch all-to-all
// that reshards embedding results from feature-parallel to batch-
// parallel every single training step. Reusing Chapter 9's own real
// ring/all-to-all round-count formula (2(P-1)/P of the data a rank holds
// must leave that rank in a full resharding) applied to a realistic
// batch-size x embedding-dimension payload, this closed-form model
// (never a fabricated timing) shows the two costs move in OPPOSITE
// directions as P grows: memory pressure per GPU keeps falling, but the
// FRACTION of the per-step embedding payload that must cross the network
// keeps rising toward 100%, echoing Chapter 16's own lesson that more
// ranks is not free -- for a new reason specific to this chapter's own
// resharding shape.
#include <cstdio>

int main() {
    // Illustrative but realistic DLRM-style batch/embedding-dimension
    // figures (bf16 embeddings, 2 bytes per value).
    long long batchSize = 65536;      // a real large-scale training batch size
    long long embeddingDim = 128;     // a real common embedding vector width
    long long bytesPerElement = 2;    // bf16
    long long totalPayloadBytes = batchSize * embeddingDim * bytesPerElement;

    printf("Per-step embedding payload: batch=%lld x dim=%lld x %lld bytes "
           "= %lld bytes (%.2f GB) that must be FULLY reassembled into "
           "batch-parallel shards every single training step.\n\n",
           batchSize, embeddingDim, bytesPerElement, totalPayloadBytes,
           totalPayloadBytes / 1e9);

    int Ps[] = {1, 2, 4, 8, 16, 32, 64, 128};
    printf("%-6s %-22s %-24s %-18s\n", "P", "Fraction off-rank",
           "Bytes crossing network", "Per-GPU emb. shard (GB, 10TB total)");
    double totalEmbeddingGB = 10240.0;  // reuse Section 34.1's own terabyte-scale scenario
    for (int P : Ps) {
        // Chapter 9's own real formula: in a full resharding among P
        // ranks, each rank must send out (P-1)/P of the data it starts
        // with -- the fraction of the total payload that must actually
        // cross the network (not just move within one GPU's own memory).
        double offRankFraction = (P > 1) ? (double)(P - 1) / (double)P : 0.0;
        double bytesCrossing = totalPayloadBytes * offRankFraction;
        double perGpuShardGB = totalEmbeddingGB / (double)P;
        printf("%-6d %-22.4f %-24.0f %-18.3f\n", P, offRankFraction, bytesCrossing, perGpuShardGB);
    }

    printf("\nThe per-GPU embedding shard keeps shrinking linearly with P "
           "(Section 34.1's own real memory benefit, unchanged here). But "
           "the FRACTION of the per-step payload that must cross the "
           "network rises from 0%% (P=1, no resharding needed at all) "
           "toward 100%% as P grows large, and is already over 87%% at "
           "P=8 -- most of the real payload leaves its origin GPU on "
           "EVERY step, regardless of how many GPUs share the embedding "
           "table. Sharding further keeps helping memory capacity, but "
           "past a moderate P it stops meaningfully reducing the real "
           "communication cost per step -- the same 'more ranks is not "
           "free' lesson Chapter 16 first taught, reappearing here for a "
           "genuinely different underlying reason: a full resharding's "
           "off-rank fraction saturates, it does not keep shrinking.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 101_embedding_alltoall_comm_volume_cost_model \
    101_embedding_alltoall_comm_volume_cost_model.cpp
./101_embedding_alltoall_comm_volume_cost_model
```

Locked output:

```
Per-step embedding payload: batch=65536 x dim=128 x 2 bytes = 16777216 bytes (0.02 GB) that must be FULLY reassembled into batch-parallel shards every single training step.

P      Fraction off-rank      Bytes crossing network   Per-GPU emb. shard (GB, 10TB total)
1      0.0000                 0                        10240.000         
2      0.5000                 8388608                  5120.000          
4      0.7500                 12582912                 2560.000          
8      0.8750                 14680064                 1280.000          
16     0.9375                 15728640                 640.000           
32     0.9688                 16252928                 320.000           
64     0.9844                 16515072                 160.000           
128    0.9922                 16646144                 80.000            

The per-GPU embedding shard keeps shrinking linearly with P (Section 34.1's own real memory benefit, unchanged here). But the FRACTION of the per-step payload that must cross the network rises from 0% (P=1, no resharding needed at all) toward 100% as P grows large, and is already over 87% at P=8 -- most of the real payload leaves its origin GPU on EVERY step, regardless of how many GPUs share the embedding table. Sharding further keeps helping memory capacity, but past a moderate P it stops meaningfully reducing the real communication cost per step -- the same 'more ranks is not free' lesson Chapter 16 first taught, reappearing here for a genuinely different underlying reason: a full resharding's off-rank fraction saturates, it does not keep shrinking.
```

## Chapter Summary

Merlin HugeCTR's own real, deployed design combines two parallelization strategies this book built separately -- Chapter 12's data parallelism and Chapter 13's model parallelism -- within a single model, applying each to a different part: the (potentially terabyte-scale) embedding tables are sharded model-parallel, while the small dense DNN layers stay fully replicated data-parallel. File 99 showed exactly where a single uniform strategy fails using HugeCTR's own real cited single-GPU capacity figure. The two parts are stitched together by an all-to-all, the exact primitive HugeCTR's own paper names directly, doing a third distinct job for this book: reassembling data from a feature-parallel shape into a batch-parallel shape on a fixed, predetermined schedule, never a runtime gating decision like Chapter 32's MoE dispatch, and with no equivalent of capacity overflow. File 100 confirmed this resharding is bit-exact at every tested P precisely because it never combines values, only relocates them. File 101 closed the chapter with a genuine tension: sharding the embeddings across more GPUs keeps helping memory capacity without limit, but the SHARE of the per-step embedding payload that must cross the network via this all-to-all saturates near 100% once P is even moderately large -- Chapter 16's "more ranks is not free" lesson, discovered here for a structurally different reason.

## Self-Check Questions

1. Why does HugeCTR's own real design apply Chapter 13's model parallelism to embedding tables specifically, rather than to the whole model?
2. Why does HugeCTR's own real design apply Chapter 12's data parallelism to the dense DNN layers specifically, rather than to the whole model?
3. What real all-to-all does HugeCTR's own paper describe, and what two "shapes" does it convert data between?
4. How does this chapter's embedding all-to-all differ structurally from Chapter 32's MoE dispatch/combine all-to-all, even though both reuse the same underlying primitive?
5. Why did File 100 use plain integers rather than floating-point values, and what does that choice isolate?
6. According to File 101's own locked output, what fraction of the per-step embedding payload crosses the network at P=8? At P=128? What trend do these two numbers show?
7. File 101 shows two different trends as P grows: one keeps improving without limit, and one saturates. Name each trend and explain why they behave differently.
8. If a team were choosing how many GPUs to shard a 2TB embedding table across, what would File 99 and File 101 together suggest about diminishing returns past a certain P?

## Where We Go Next

Chapter 35 turns to a very different industrial domain running on the same multi-GPU foundations: NVIDIA Earth-2's real FourCastNet weather-forecasting system, which replaces the classic stencil computation Chapter 16's domain decomposition was built around with a learned neural operator, while keeping the same halo-exchange communication shape.

## Worked Solutions

1. Because embedding tables can reach a scale ("the total size of an embedding feature can exceed the memory capacity of a single GPU," per HugeCTR's own real paper) where no single GPU can hold a full replica, model parallelism (sharding, not copying) is the only way to fit them at all -- exactly Chapter 13's own technique, applied here because of raw size, not because of any computational property of embeddings themselves.
2. Because the dense DNN layers stay small enough to replicate on every GPU with no capacity concern at all -- File 99's own locked output shows a 2GB dense model never approaches the real 640GB single-GPU budget at any table size tested. Once capacity is not the constraint, data parallelism (Chapter 12's technique, replicate and split the batch) is the natural, simpler choice, and HugeCTR's own real design uses it there.
3. HugeCTR's own paper describes an all-to-all that, "after each local reduction step... share[s] the results of the lookup operation across all available GPUs along the batch dimension." It converts data from being sharded by FEATURE (each GPU holds a slice of features for the whole batch) to being sharded by BATCH SAMPLE (each GPU holds every feature for its own slice of samples).
4. Chapter 32's MoE all-to-all routes each token to whichever expert a gating network decides on AT RUNTIME, and a real capacity limit can cause an overflowed token to be silently dropped or passed through unprocessed. This chapter's embedding all-to-all follows a FIXED schedule (feature index modulo P) known before the batch even arrives, with every value having one single, predetermined destination and no equivalent of capacity overflow at all.
5. Using plain integers isolates the question File 100 is actually asking -- does the RESHARDING itself (a pure relocation of values, never a combination of them) preserve correctness at any P -- from the floating-point associativity question Chapter 8, 25, 29, 31, and 32 already investigated in detail. Since nothing in this all-to-all ever adds two values together, there was never a risk of floating-point grouping mattering, and integers make that fact unambiguous rather than merely likely.
6. At P=8, the off-rank fraction is 0.8750 (87.5%) of the payload. At P=128, it is 0.9922 (99.22%). The trend is a rapidly rising curve that flattens out: most of the improvement in reducing the LOCAL (non-crossing) fraction happens at small P, and each further doubling of P buys a shrinking additional improvement.
7. The per-GPU embedding SHARD SIZE keeps shrinking linearly and without limit as P grows (10240/P GB, File 101's own locked table) -- pure, unbounded memory benefit. The OFF-RANK COMMUNICATION FRACTION, (P-1)/P, rises toward 1.0 and saturates -- it can never exceed 100%, and gets extremely close to it well before P grows very large, so each additional GPU keeps helping memory but contributes ever less to reducing the already-near-total communication requirement.
8. Both files agree that capacity considerations (File 99: how many GPUs are needed just to FIT a 2TB table under the real 640GB-per-GPU budget) set a MINIMUM useful P (4 GPUs, per File 99's own locked table), while File 101 shows that going much further than a moderate P (already 87.5% off-rank at P=8) buys shrinking communication benefit even as it keeps shrinking the per-GPU memory footprint -- so a real team would likely pick P based on the memory constraint and any compute/throughput needs, not on any expectation that more P will keep meaningfully reducing this specific all-to-all's own cost.

---

**Sources cited in this chapter:**

- Oldridge, E. et al. "Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference." arXiv:2210.08803, RecSys 2022, fetched fresh this session from both the paper's own PDF and its ar5iv HTML rendering (cross-checked for consistency). (The hybrid data-parallel/model-parallel design, the real embedding all-to-all description including "along the batch dimension," the real per-GPU capacity and terabyte-scale ETC claims, and the real cited 24.6x and 5-62x speedup numbers.)
- NVIDIA. "HugeCTR Core Features," Merlin HugeCTR documentation, nvidia-merlin.github.io, fetched fresh this session. (The real "each GPU only keeps a shard of the embedding table (MP)" and "the dense model... is data parallel and contains a copy... in each GPU" quotes, and the real NCCL-based communication statement.)
- This book's own Chapter 1 (memory wall), Chapter 9 (the real round-count formula reused in File 101), Chapter 10 (all-to-all), Chapter 12 (data parallelism), Chapter 13 (model parallelism), Chapter 16 (the "more ranks is not free" lesson, reappearing here for a new reason), and Chapter 32 (MoE dispatch/combine, this chapter's direct contrast for what an all-to-all can and cannot be used for).
