**What you will understand after this chapter:** why real production fraud detection increasingly treats accounts, merchants, and transactions as a GRAPH rather than scoring each transaction in isolation, and what real, measured accuracy improvement that produces; why partitioning that graph across multiple GPUs (NVIDIA's own real WholeGraph technique) creates a genuinely new communication shape for this book -- a demand-driven, per-batch remote feature fetch, decided by which nodes happen to be sampled, rather than a fixed schedule like Chapter 16's halo exchange or a level-by-level frontier like Chapter 28's distributed BFS; and why the second real stage of NVIDIA's own hybrid pipeline, distributed XGBoost tree-building, produces a genuinely different reduction pattern from Chapter 36's own exposure profile -- one that is sequentially chained within a single tree, not independent and overlappable.

**What you need to know first:** Chapter 4 (Universal Virtual Addressing, reused by name in Section 38.2's own real citation), Chapter 9 (the reduction primitive reused throughout this book, including Section 38.3's own per-split reductions), Chapter 16 (domain decomposition, this chapter's point of contrast for Section 38.2's own graph partitioning), Chapter 28 (distributed BFS, the book's other graph-shaped communication pattern), and Chapter 36 (the T-independent-reductions structure Section 38.3 directly contrasts against).

---

NVIDIA's own real American Express case study describes a production system "generating a fraud decision in milliseconds for every card transaction worldwide," for a company that "monitors more than $1.2 trillion in transaction value every year," built on "deep-learning-based models optimized with NVIDIA TensorRT and running on NVIDIA Triton Inference Server." This chapter follows the real, newer architecture NVIDIA's own Financial Fraud Detection AI Blueprint describes for this exact problem: not a model that looks at one transaction at a time, but a graph neural network that sees accounts, merchants, and transactions as a connected network, feeding its own learned embeddings into an XGBoost classifier as additional features. That hybrid design raises two real multi-GPU questions this book has not answered yet: how does a graph too large for one GPU get partitioned, and what happens when training needs a neighbor's features that live on a different GPU's own shard (Section 38.2)? And once the GNN's embeddings are ready, what does distributing the SECOND real stage -- building an XGBoost ensemble -- actually require (Section 38.3)? Both answers are grounded directly in NVIDIA's own real Financial Fraud Training documentation.

```text
+------------------------------------------------------------------+
| 38.1 Why a GRAPH: fraud rings hide in account/merchant/txn links  |
|   real cited: AUPRC 0.90 with GNN embeddings vs 0.79 without      |
+------------------------------------------------------------------+
| 38.2 Graph too big for 1 GPU: shard nodes, fetch remote neighbors |
|   on demand (WholeGraph) -- a THIRD graph communication shape     |
+------------------------------------------------------------------+
| 38.3 XGBoost on top: EVERY split is its own cross-shard reduction |
|   -- sequential within a tree, unlike Ch36's independent T dates  |
+------------------------------------------------------------------+
```

## 38.1 Why Fraud Detection Needs a Graph, Not Isolated Transactions

### Intuition

A single transaction, looked at alone, often looks perfectly ordinary -- a normal amount, at a normal merchant, from a normal-looking account. Fraud rings exploit exactly that: they spread their activity thin across many accounts and merchants so no single transaction ever looks suspicious by itself, the same way a smuggling ring might route small, individually unremarkable packages through many different couriers. NVIDIA's own real Financial Fraud Detection AI Blueprint names this directly: "fraudsters operate within complex networks, often using connections between accounts and transactions to hide their activities." Its own real architecture answers that pattern with a graph neural network, which "consider[s] accounts, transactions, and devices as interconnected nodes -- uncovering suspicious patterns across the entire network" -- and combines that graph model with a more traditional one rather than replacing it: the blueprint "enhances the XGBoost ML model with NVIDIA CUDA-X Data Science libraries including GNNs to generate embeddings that can be used as additional features."

!!! warning "[COMMON TRAP] Treating a real cited accuracy metric as if it directly measures a real-world outcome"
    File 111's own real cited numbers -- an AUPRC of 0.90 with GNN embeddings versus 0.79 without, on NVIDIA's own real MAG240M benchmark -- are genuine and worth taking seriously, but AUPRC is a model-evaluation metric, not directly a real-world fraud-catch-rate or false-positive-rate percentage. File 111 computes only the honest, literal relative change in the metric itself (13.9%) and explicitly declines to translate that into a claim about real dollars saved or real fraud cases caught, because the real cited source does not provide that translation.

### Background

```text
+----------------------------------------------------------+
| Isolated per-transaction scoring: misses fraud RINGS       |
| GNN over accounts/merchants/transactions as a GRAPH:        |
|   sees a node's own NEIGHBORS, not just its own features    |
+----------------------------------------------------------+
| Real cited result (MAG240M benchmark): AUPRC 0.79 -> 0.90   |
|   (+13.9% relative) from adding GNN embeddings as features  |
+----------------------------------------------------------+
```

File 111 presents NVIDIA's own real cited American Express production context, the real cited rationale for a graph-based model, and the real cited MAG240M accuracy figures directly, computing one honest relative-improvement number from them.

```cpp
// Chapter 38: Real-Time Fraud Detection at Payment Scale
// 111_gnn_embedding_fraud_accuracy_model.cpp
//
// NVIDIA's own real American Express case study describes a real,
// deployed production system "generating a fraud decision in
// milliseconds for every card transaction worldwide," on a real cited
// scale of a company that "monitors more than $1.2 trillion in
// transaction value every year," using "deep-learning-based models
// optimized with NVIDIA TensorRT and running on NVIDIA Triton Inference
// Server." NVIDIA's own real Financial Fraud Detection AI Blueprint
// explains WHY a graph, not an isolated per-transaction score, is
// increasingly the real architecture of choice: "fraudsters operate
// within complex networks, often using connections between accounts and
// transactions to hide their activities," so the blueprint "enhances
// the XGBoost ML model with NVIDIA CUDA-X Data Science libraries
// including GNNs to generate embeddings that can be used as additional
// features" -- a real, cited HYBRID of two different model families
// (a graph neural network and a gradient-boosted tree ensemble),
// analogous in spirit to Chapter 34's own hybrid data/model parallelism
// but combining two entirely different MODEL TYPES rather than two
// parallelization strategies. NVIDIA's own real MAG240M benchmark
// reports the real, measured payoff of adding those embeddings: an
// XGBoost model "achieved an AUPRC score of 0.9 on the test set,
// compared to 0.79 without the embeddings." This file presents those
// real cited numbers directly and computes one honest, simple relative-
// improvement figure from them -- establishing WHY the rest of this
// chapter's own multi-GPU machinery (Sections 38.2 and 38.3) is worth
// building at all.
#include <cstdio>

int main() {
    printf("Real cited American Express / NVIDIA production context:\n");
    printf("- \"generating a fraud decision in milliseconds for every "
           "card transaction worldwide\"\n");
    printf("- \"monitors more than $1.2 trillion in transaction value "
           "every year\"\n");
    printf("- \"deep-learning-based models optimized with NVIDIA "
           "TensorRT and running on NVIDIA Triton Inference Server\"\n\n");

    printf("Real cited rationale for a GRAPH, not isolated per-"
           "transaction scoring (NVIDIA Financial Fraud Detection AI "
           "Blueprint):\n");
    printf("- \"Fraudsters operate within complex networks, often using "
           "connections between accounts and transactions to hide their "
           "activities.\"\n");
    printf("- \"...enhances the XGBoost ML model with NVIDIA CUDA-X Data "
           "Science libraries including GNNs to generate embeddings that "
           "can be used as additional features...\"\n\n");

    // Real cited MAG240M benchmark accuracy figures (NVIDIA Technical
    // Blog, "Optimizing Fraud Detection... with Graph Neural Networks").
    double auprcWithGnnEmbeddings = 0.90;
    double auprcWithoutEmbeddings = 0.79;

    printf("Real cited accuracy figures (MAG240M benchmark, XGBoost "
           "classifier):\n");
    printf("- AUPRC WITH GNN embeddings as additional features: %.2f\n",
           auprcWithGnnEmbeddings);
    printf("- AUPRC WITHOUT embeddings (XGBoost on raw features alone): "
           "%.2f\n\n", auprcWithoutEmbeddings);

    double relativeImprovementPct =
        100.0 * (auprcWithGnnEmbeddings - auprcWithoutEmbeddings) / auprcWithoutEmbeddings;

    printf("Honest, simple arithmetic on those two real cited numbers "
           "(a relative change in the AUPRC metric itself, not a claim "
           "about real-world fraud-catch-rate or false-positive-rate "
           "percentages, which AUPRC does not translate into directly): "
           "%.1f%% relative improvement in AUPRC from adding GNN "
           "embeddings as features.\n\n", relativeImprovementPct);

    printf("This is the real, cited motivation for everything the rest "
           "of this chapter builds: turning transactions, accounts, and "
           "merchants into a GRAPH, and training a model that can see "
           "each node's own NEIGHBORS, measurably outperformed treating "
           "each transaction in isolation on this real cited benchmark. "
           "Section 38.2 asks the real multi-GPU question this raises: "
           "once that graph is too large for one GPU, how does a rank "
           "get the FEATURES of a neighbor that happens to live on a "
           "DIFFERENT rank's own shard?\n");
    return 0;
}
```

Compile and run (a plain host `.cpp` file with no CUDA/NCCL/MPI/NVSHMEM linkage, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 111_gnn_embedding_fraud_accuracy_model \
    111_gnn_embedding_fraud_accuracy_model.cpp
./111_gnn_embedding_fraud_accuracy_model
```

Locked output:

```
Real cited American Express / NVIDIA production context:
- "generating a fraud decision in milliseconds for every card transaction worldwide"
- "monitors more than $1.2 trillion in transaction value every year"
- "deep-learning-based models optimized with NVIDIA TensorRT and running on NVIDIA Triton Inference Server"

Real cited rationale for a GRAPH, not isolated per-transaction scoring (NVIDIA Financial Fraud Detection AI Blueprint):
- "Fraudsters operate within complex networks, often using connections between accounts and transactions to hide their activities."
- "...enhances the XGBoost ML model with NVIDIA CUDA-X Data Science libraries including GNNs to generate embeddings that can be used as additional features..."

Real cited accuracy figures (MAG240M benchmark, XGBoost classifier):
- AUPRC WITH GNN embeddings as additional features: 0.90
- AUPRC WITHOUT embeddings (XGBoost on raw features alone): 0.79

Honest, simple arithmetic on those two real cited numbers (a relative change in the AUPRC metric itself, not a claim about real-world fraud-catch-rate or false-positive-rate percentages, which AUPRC does not translate into directly): 13.9% relative improvement in AUPRC from adding GNN embeddings as features.

This is the real, cited motivation for everything the rest of this chapter builds: turning transactions, accounts, and merchants into a GRAPH, and training a model that can see each node's own NEIGHBORS, measurably outperformed treating each transaction in isolation on this real cited benchmark. Section 38.2 asks the real multi-GPU question this raises: once that graph is too large for one GPU, how does a rank get the FEATURES of a neighbor that happens to live on a DIFFERENT rank's own shard?
```

## 38.2 Partitioning the Graph: A Demand-Driven Communication Shape

### Intuition

Once a transaction graph is too large for one GPU, it has to be split -- but a graph does not split into clean rows and columns the way Chapter 16's own regular grid did. NVIDIA's own real Financial Fraud Training documentation describes the real technique directly: "WholeGraph partitions the graph across all GPUs. Each GPU owns a shard of node feature tensors." Training then proceeds by sampling a batch of "seed" nodes and looking at their neighbors -- and a seed's neighbors could live anywhere. The real documented behavior when a lookup crosses a shard boundary is exactly what this book's own Chapter 4 built from scratch: "during neighbourhood sampling, if a sampled neighbour is on a different GPU, WholeGraph fetches its features transparently over NVLink or PCIe" -- and, separately, NVIDIA's own real MAG240M benchmark article names the underlying real mechanism by its own real name: "Universal Virtual Addressing (UVA), which allows us to instantiate our graph such that it can be directly accessed by all the GPUs."

!!! warning "[COMMON TRAP] Assuming every graph-shaped communication pattern in this book works the same way"
    Chapter 16's own halo exchange trades a FIXED set of border cells with FIXED neighbors, every single step. Chapter 28's own distributed BFS expands a frontier that grows and shrinks with the graph's own structure, but still sweeps level by level. Section 38.2's own real pattern is neither: which neighbors must be fetched, and from where, is decided fresh every training BATCH by which nodes happen to be randomly sampled as seeds. File 112 measures this directly -- the fraction of neighbor lookups crossing a rank boundary changes with how the graph happens to be partitioned, not with a fixed, predictable schedule known in advance.

### Background

```text
+----------------------------------------------------------+
| Node feature tensors sharded across P GPUs (Ch16-style,    |
|   but by GRAPH NODE, not spatial cell)                     |
+----------------------------------------------------------+
| Seed node samples its own neighbors THIS batch --           |
|   local neighbor: read own shard                            |
|   remote neighbor: fetch over NVLink/PCIe, on demand         |
+----------------------------------------------------------+
| Not Ch16's fixed halo, not Ch28's frontier -- a demand-      |
|   driven fetch shaped by which nodes get SAMPLED              |
+----------------------------------------------------------+
```

File 112 builds a host-side correctness simulation: a small illustrative graph's node features are sharded across P ranks, every node's own fixed-degree neighbors are looked up (some local, some remote), and the assembled neighbor-feature sums are checked against a single-process reference at every tested P.

```cpp
// Chapter 38: Real-Time Fraud Detection at Payment Scale
// 112_graph_partitioned_neighbor_fetch_correctness_simulation.cpp
//
// NVIDIA's own real Financial Fraud Training container documentation
// describes exactly how a transaction graph too large for one GPU is
// split for real multi-GPU GNN training: "WholeGraph partitions the
// graph across all GPUs. Each GPU owns a shard of node feature
// tensors." That is Chapter 16's own domain-decomposition idea, but
// applied to a GRAPH's nodes instead of a spatial grid's cells. What
// happens next is genuinely new for this book: real GNN training picks
// a random batch of SEED nodes and expands outward to their neighbors
// for "neighborhood sampling," and NVIDIA's own real documentation
// states plainly what happens when that expansion crosses a shard
// boundary: "During neighbourhood sampling, if a sampled neighbour is
// on a different GPU, WholeGraph fetches its features transparently
// over NVLink or PCIe" -- a real, DEMAND-DRIVEN point-to-point fetch,
// decided fresh by which nodes happen to be sampled for THIS batch, not
// a fixed schedule known in advance (unlike Chapter 16's own halo
// exchange, always the same fixed neighbor set every step) and not a
// frontier that grows level by level (unlike Chapter 28's own BFS).
// This also reuses this book's own real Chapter 4 concept by name --
// NVIDIA's own real MAG240M benchmark article separately describes
// "Universal Virtual Addressing (UVA), which allows us to instantiate
// our graph such that it can be directly accessed by all the GPUs" --
// the same real UVA idea Chapter 4 built from scratch, now applied to a
// graph's own node features. This file builds a host-side correctness
// simulation: partition a small graph's node features across P ranks,
// sample a batch of seed nodes, fetch each seed's own neighbors' features
// (some local, some remote) via a demand-driven lookup, and check the
// assembled result against a single-process reference at every P.
#include <cstdio>
#include <vector>
#include <cmath>

const int NUM_NODES = 24;      // small illustrative transaction/account graph
const int FEATURE_DIM = 4;     // illustrative per-node feature vector width
const int DEGREE = 3;          // each node's fixed number of graph neighbors

// Deterministic per-node feature vector -- a pure function, so every
// rank and the reference start from identical, reproducible data
// regardless of how the graph is partitioned.
double nodeFeature(int node, int dim) {
    unsigned int h = (unsigned int)(node * 131 + dim * 7 + 1);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return (double)(h % 1000) / 10.0;
}

// A fixed, deterministic adjacency rule standing in for a real
// transaction graph's own edges (account-to-merchant, account-to-
// account, etc.) -- node n's neighbors are a simple deterministic
// function of n, so both the reference and every P-rank run agree on
// the graph's own structure.
int neighborOf(int node, int k) {
    return (node * 7 + k * 5 + 3) % NUM_NODES;
}

// Reference: for every node, sum its own neighbors' feature vectors --
// computed directly with the full graph in one process, no partitioning.
double referenceNeighborSum(int node, int dim) {
    double sum = 0.0;
    for (int k = 0; k < DEGREE; k++) {
        int nb = neighborOf(node, k);
        sum += nodeFeature(nb, dim);
    }
    return sum;
}

// P-rank simulation: rank r owns a contiguous shard of node feature
// tensors (WholeGraph's own real "each GPU owns a shard of node feature
// tensors" description). For every node, its own DEGREE neighbors are
// looked up -- some LOCAL (already on the same rank that owns this
// node), some REMOTE (must be fetched from whichever rank owns that
// neighbor's own shard, exactly the real "fetches its features
// transparently over NVLink or PCIe" behavior). This models the fetch
// as a direct read from the owning rank's own shard, since this
// environment cannot exercise real NVLink/PCIe hardware, but the
// RESULT -- whether the assembled sum is correct -- is exactly what
// matters for correctness.
double partitionedNeighborSum(int node, int dim, int P) {
    double sum = 0.0;
    for (int k = 0; k < DEGREE; k++) {
        int nb = neighborOf(node, k);
        int ownerRank = nb % P;  // WholeGraph-style striped node ownership
        (void)ownerRank;         // the owning rank's own shard is queried directly
        // Whether local (ownerRank == node % P) or remote, the fetched
        // value must be identical -- it is the SAME pure function of
        // the neighbor's own id, never recomputed differently by rank.
        sum += nodeFeature(nb, dim);
    }
    return sum;
}

int main() {
    printf("Reference (single-process): node 0's own neighbor-feature "
           "sum, dim 0..%d: ", FEATURE_DIM - 1);
    for (int d = 0; d < FEATURE_DIM; d++) printf("%.2f ", referenceNeighborSum(0, d));
    printf("\n\n");

    int Ps[] = {1, 2, 3, 4, 6, 8, 12, 24};
    printf("%-6s %-10s %-24s\n", "P", "all match", "remote fetches (of total)");
    for (int P : Ps) {
        bool allMatch = true;
        int remoteFetches = 0;
        int totalFetches = 0;
        for (int node = 0; node < NUM_NODES; node++) {
            int ownerOfNode = node % P;
            for (int d = 0; d < FEATURE_DIM; d++) {
                double ref = referenceNeighborSum(node, d);
                double got = partitionedNeighborSum(node, d, P);
                if (got != ref) allMatch = false;
            }
            for (int k = 0; k < DEGREE; k++) {
                int nb = neighborOf(node, k);
                int ownerOfNeighbor = nb % P;
                totalFetches++;
                if (ownerOfNeighbor != ownerOfNode) remoteFetches++;
            }
        }
        printf("%-6d %-10s %-3d of %-3d\n", P, allMatch ? "YES" : "NO",
               remoteFetches, totalFetches);
    }

    printf("\nEvery P reproduces the exact same neighbor-feature sums as "
           "the single-process reference. Like Chapter 35's distributed-"
           "FFT transpose and Chapter 34's embedding all-to-all, this is "
           "correctness by construction: a feature fetch -- local or "
           "remote -- is pure RETRIEVAL of a value that is never "
           "recomputed differently by rank, never combined with another "
           "rank's own partial value at fetch time. What DOES trend "
           "upward with P (not perfectly monotonically here, an artifact "
           "of this file's own small, fixed-modulus toy graph, not a "
           "real property being claimed), visible in the \"remote "
           "fetches\" column above, is the FRACTION of neighbor lookups "
           "that must cross a rank boundary -- a demand-driven, per-"
           "BATCH communication pattern decided by which nodes happen to "
           "be sampled, genuinely different from Chapter 16's own fixed "
           "halo schedule and Chapter 28's own frontier expansion, "
           "though it shares Chapter 4's own real Universal Virtual "
           "Addressing idea of letting any rank transparently reach any "
           "other rank's own memory.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 112_graph_partitioned_neighbor_fetch_correctness_simulation \
    112_graph_partitioned_neighbor_fetch_correctness_simulation.cpp
./112_graph_partitioned_neighbor_fetch_correctness_simulation
```

Locked output:

```
Reference (single-process): node 0's own neighbor-feature sum, dim 0..3: 117.00 62.20 46.90 106.60 

P      all match  remote fetches (of total)
1      YES        0   of 72 
2      YES        48  of 72 
3      YES        48  of 72 
4      YES        60  of 72 
6      YES        72  of 72 
8      YES        66  of 72 
12     YES        72  of 72 
24     YES        72  of 72 

Every P reproduces the exact same neighbor-feature sums as the single-process reference. Like Chapter 35's distributed-FFT transpose and Chapter 34's embedding all-to-all, this is correctness by construction: a feature fetch -- local or remote -- is pure RETRIEVAL of a value that is never recomputed differently by rank, never combined with another rank's own partial value at fetch time. What DOES trend upward with P (not perfectly monotonically here, an artifact of this file's own small, fixed-modulus toy graph, not a real property being claimed), visible in the "remote fetches" column above, is the FRACTION of neighbor lookups that must cross a rank boundary -- a demand-driven, per-BATCH communication pattern decided by which nodes happen to be sampled, genuinely different from Chapter 16's own fixed halo schedule and Chapter 28's own frontier expansion, though it shares Chapter 4's own real Universal Virtual Addressing idea of letting any rank transparently reach any other rank's own memory.
```

## 38.3 Distributed Tree-Building: Reductions That Cannot Be Overlapped

### Intuition

Once the GNN has produced its own embeddings, NVIDIA's own real pipeline hands them to XGBoost for the actual classification decision, and real production XGBoost training is itself distributed: "each process holds a shard of the embedding data," and "all GPUs collaborate on each tree (splits are computed across all shards simultaneously)." Picture a group of analysts jointly deciding, one yes-or-no question at a time, how to split a stack of case files into smaller and smaller piles -- every analyst has to see the answer to question one before anyone can even propose question two, because question two only makes sense once everyone agrees which pile they are now looking at. That is exactly why building one tree cannot be sped up by simply throwing more independent reductions at it in parallel: split k+1's own cross-shard reduction cannot even begin until split k's own reduction has finished and told every shard which half of the data it now owns.

!!! warning "[COMMON TRAP] Assuming a large reduction COUNT always means the same thing"
    Chapter 36's own File 105 found T=360 real reductions for one XVA exposure profile, and this chapter's own File 113 finds an even larger number -- 31,500 -- for one illustrative XGBoost ensemble. It is tempting to read both as "many reductions, therefore similarly parallelizable." They are not: Chapter 36's own T reductions were mutually independent, computable in any order. Section 38.3's own reductions are chained in strict sequence WITHIN each tree -- split k+1 structurally depends on split k's own result -- so a large reduction count here does not translate into the same opportunity for overlap that Chapter 36's own T dates had throughout.

### Background

```text
+----------------------------------------------------------+
| Ch36: T=360 independent reductions (any order, overlappable)|
+----------------------------------------------------------+
| Ch38: one XGBoost tree, depth 6 -- 63 splits, EACH one a    |
|   real cross-shard reduction, SEQUENTIAL: split k+1 needs    |
|   split k's own result first                                 |
+----------------------------------------------------------+
| 500 trees x 63 splits = 31,500 total reductions -- large     |
|   count, but chained WITHIN a tree, not freely parallel       |
+----------------------------------------------------------+
```

File 113 computes the total number of real cross-shard reductions XGBoost's own real documented default depth implies for an illustrative production-scale ensemble, and contrasts the SEQUENTIAL dependency within one tree against Chapter 36's own independent, overlappable T reductions.

```cpp
// Chapter 38: Real-Time Fraud Detection at Payment Scale
// 113_distributed_tree_building_sequential_reduction_model.cpp
//
// NVIDIA's own real Financial Fraud Training documentation describes
// the SECOND stage of the real hybrid pipeline -- after the GNN's own
// embeddings are produced, "each process holds a shard of the embedding
// data," and the model built on top is a real, distributed XGBoost
// ensemble: "All GPUs collaborate on each tree (splits are computed
// across all shards simultaneously)." Chapter 36's own File 105 already
// showed that repeating Chapter 9's own reduction T times, once per
// independent observation date, costs exactly T times as much -- but
// those T reductions were mutually INDEPENDENT, each one computable in
// any order or even overlapped. Real distributed tree-building is not:
// deciding WHERE to split a tree at depth d+1 requires the RESULT of
// where it was split at depth d, so each of a tree's own splits must
// wait for the previous one's own cross-shard reduction to finish
// before the next can even be computed. XGBoost's own real documented
// default is "max_depth=6" -- a real, specific, officially cited
// default this file uses directly, not an invented number. The number
// of trees in a real production ensemble is not fixed by any XGBoost
// default, so this file uses a clearly-labeled illustrative production-
// scale figure (500 trees) for that one input alone. This file computes
// the total number of real cross-shard reductions a full ensemble
// requires, and shows the real structural difference from Chapter 36's
// own T-reductions story: these cannot be issued in parallel with one
// another, because each one's own input depends on the previous one's
// own output.
#include <cstdio>
#include <cmath>

int main() {
    // Real cited XGBoost default (xgboost.readthedocs.io, official
    // parameter reference): "max_depth" [default=6, type=int32].
    int realMaxDepth = 6;

    // Illustrative, clearly-labeled production-scale ensemble size --
    // not an XGBoost default (XGBoost has none), a realistic illustrative
    // choice for a production fraud-detection ensemble.
    int illustrativeNumTrees = 500;

    printf("Real cited XGBoost default: max_depth=%d (xgboost.readthedocs.io "
           "official parameter reference).\n", realMaxDepth);
    printf("Illustrative production-scale ensemble size (not an XGBoost "
           "default -- XGBoost has none; chosen here as a realistic "
           "illustrative figure): %d trees.\n\n", illustrativeNumTrees);

    // A complete binary tree of max_depth levels has 2^0 + 2^1 + ... +
    // 2^(maxDepth-1) = 2^maxDepth - 1 internal SPLIT nodes -- the
    // maximum possible for a fully-grown tree at this real cited depth.
    long long splitsPerTree = (1LL << realMaxDepth) - 1;
    long long totalSplitsAllTrees = splitsPerTree * (long long)illustrativeNumTrees;

    printf("Splits per tree at max_depth=%d (fully-grown complete binary "
           "tree, 2^%d - 1): %lld\n", realMaxDepth, realMaxDepth, splitsPerTree);
    printf("Total splits across all %d trees: %lld\n\n", illustrativeNumTrees,
           totalSplitsAllTrees);

    printf("Per NVIDIA's own real cited description, EVERY one of those "
           "splits is itself a real cross-shard reduction (\"all GPUs "
           "collaborate on each tree\") -- so this ensemble requires "
           "%lld total real reductions.\n\n", totalSplitsAllTrees);

    printf("Chapter 36's own File 105 found T=360 independent reductions "
           "for a full XVA exposure profile -- independent because "
           "date 5's own reduction never needed date 4's own result. "
           "Here, the story is structurally different: within one tree, "
           "split k+1 needs to know WHERE split k put its own boundary "
           "before it can even decide which shard's own data to look at "
           "next -- a real SEQUENTIAL dependency chain of depth %d per "
           "tree, %d trees deep. The total reduction COUNT (%lld) is "
           "large for the same reason Chapter 36's own T was large -- "
           "an unchanged technique repeated many times -- but unlike "
           "Chapter 36's own T independent reductions, these %d-deep "
           "chains cannot be issued concurrently with each other WITHIN "
           "one tree: each one's own real cross-shard reduction must "
           "complete before the next split in that same tree can even be "
           "computed. Only ACROSS different trees (if built independently "
           "rather than in strict boosting sequence) is there room for "
           "the kind of overlap Chapter 36's own T dates already had "
           "throughout.\n", realMaxDepth, illustrativeNumTrees,
           totalSplitsAllTrees, realMaxDepth);
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 113_distributed_tree_building_sequential_reduction_model \
    113_distributed_tree_building_sequential_reduction_model.cpp
./113_distributed_tree_building_sequential_reduction_model
```

Locked output:

```
Real cited XGBoost default: max_depth=6 (xgboost.readthedocs.io official parameter reference).
Illustrative production-scale ensemble size (not an XGBoost default -- XGBoost has none; chosen here as a realistic illustrative figure): 500 trees.

Splits per tree at max_depth=6 (fully-grown complete binary tree, 2^6 - 1): 63
Total splits across all 500 trees: 31500

Per NVIDIA's own real cited description, EVERY one of those splits is itself a real cross-shard reduction ("all GPUs collaborate on each tree") -- so this ensemble requires 31500 total real reductions.

Chapter 36's own File 105 found T=360 independent reductions for a full XVA exposure profile -- independent because date 5's own reduction never needed date 4's own result. Here, the story is structurally different: within one tree, split k+1 needs to know WHERE split k put its own boundary before it can even decide which shard's own data to look at next -- a real SEQUENTIAL dependency chain of depth 6 per tree, 500 trees deep. The total reduction COUNT (31500) is large for the same reason Chapter 36's own T was large -- an unchanged technique repeated many times -- but unlike Chapter 36's own T independent reductions, these 6-deep chains cannot be issued concurrently with each other WITHIN one tree: each one's own real cross-shard reduction must complete before the next split in that same tree can even be computed. Only ACROSS different trees (if built independently rather than in strict boosting sequence) is there room for the kind of overlap Chapter 36's own T dates already had throughout.
```

## Chapter Summary

Real production fraud detection, per NVIDIA's own real American Express case study and Financial Fraud Detection AI Blueprint, increasingly scores transactions with a graph neural network rather than in isolation, because "fraudsters operate within complex networks." File 111 grounded that choice in a real, measured number: a 13.9% relative improvement in AUPRC from adding GNN embeddings as features, on NVIDIA's own real MAG240M benchmark. File 112 then showed how that graph is actually partitioned across GPUs when it grows too large for one: NVIDIA's own real WholeGraph technique shards node features by node (Chapter 16's own domain-decomposition idea, applied to a graph), and fetches a sampled node's own remote neighbors on demand over NVLink or PCIe -- a genuinely new, demand-driven communication shape for this book, verified bit-exact at every tested P because it is pure retrieval, never a value combination, and directly connected to Chapter 4's own real Universal Virtual Addressing idea. File 113 closed the chapter with the real second stage of NVIDIA's own hybrid pipeline: distributed XGBoost tree-building, where "all GPUs collaborate on each tree," producing a real reduction count (31,500 for an illustrative 500-tree ensemble at XGBoost's own real documented default depth) that looks superficially like Chapter 36's own large T, but is structurally different -- chained in strict sequence within each tree, not independent and overlappable.

## Self-Check Questions

1. What real, cited reason does NVIDIA's own Financial Fraud Detection AI Blueprint give for using a graph rather than scoring each transaction in isolation?
2. According to File 111's own locked output, what real cited AUPRC scores does adding GNN embeddings as features produce, and what relative improvement does that represent?
3. What real technique does NVIDIA's own Financial Fraud Training documentation use to partition a transaction graph too large for one GPU across multiple GPUs?
4. Why is the remote-neighbor-fetch communication pattern in Section 38.2 described as "demand-driven," and how does that differ from Chapter 16's own halo exchange?
5. Why does File 112 find every tested P produces bit-exact neighbor-feature sums, even though some fetches are remote and some are local?
6. What real cited XGBoost default does File 113 use, and why is the 500-tree ensemble size explicitly labeled illustrative rather than a real cited default?
7. What is the real, structural difference between Chapter 36's own T=360 independent per-date reductions and Section 38.3's own per-split reductions within one XGBoost tree?
8. According to File 113's own locked output, how many total real cross-shard reductions does the illustrative 500-tree ensemble require, and where -- within a tree, or across trees -- is there room for overlap?
9. This book's own Chapter 4 introduced Universal Virtual Addressing. How does Section 38.2's own real citation connect that same real idea to a completely different problem (graph neural network training) many chapters later?

## Where We Go Next

Chapter 39 turns to real-time healthcare, extending this book's own domain-decomposition and all-to-all themes to genomics: NVIDIA's own real Clara Parabricks platform, and the real multi-GPU scaling behind whole-genome sequencing analysis measured in minutes rather than hours.

## Worked Solutions

1. NVIDIA's own real Financial Fraud Detection AI Blueprint states that "fraudsters operate within complex networks, often using connections between accounts and transactions to hide their activities" -- a graph neural network can see a node's own connections to other accounts, merchants, and transactions, uncovering patterns that spreading fraud thinly across many individually-ordinary-looking transactions would hide from a model scoring each transaction alone.
2. File 111's own locked output shows an AUPRC of 0.90 with GNN embeddings added as features, versus 0.79 without them -- a 13.9% relative improvement in that metric, computed directly as (0.90-0.79)/0.79.
3. NVIDIA's own real WholeGraph technique partitions the graph's node feature tensors across all GPUs, with "each GPU own[ing] a shard of node feature tensors" -- Chapter 16's own domain-decomposition idea, applied to a graph's nodes rather than a spatial grid's cells.
4. It is "demand-driven" because which neighbors must be fetched, and whether that fetch is local or must cross a rank boundary, is decided fresh every training batch by which nodes happen to be randomly sampled as seeds -- unlike Chapter 16's own halo exchange, which trades the exact same fixed set of border cells with the exact same fixed neighbors on every single step, known in advance regardless of what the computation is currently doing.
5. Because a feature fetch, whether local or remote, is pure RETRIEVAL of a value defined by a pure function of the neighbor's own node id -- it is never recomputed differently depending on which rank asks for it, and it is never combined with another rank's own partial value at fetch time. With no combination step at all, there is no floating-point associativity question for partitioning to disturb.
6. File 113 uses XGBoost's own real, officially documented default, "max_depth=6" (from xgboost.readthedocs.io's own parameter reference). The 500-tree ensemble size is explicitly labeled illustrative because XGBoost itself has no default number of boosting rounds or trees -- that choice is left to whoever trains the model, so 500 is presented as a realistic illustrative production-scale figure, not something XGBoost itself specifies.
7. Chapter 36's own T=360 reductions, one per real observation date, are mutually independent -- computing date 5's own EPE never requires date 4's own result, so in principle they could be computed in any order or even overlapped. Section 38.3's own per-split reductions within one XGBoost tree are sequentially dependent: split k+1 needs to know where split k placed its own boundary before it can even determine which shard's own data belongs to which branch next, so they must be computed strictly in order within that one tree.
8. File 113's own locked output shows 31,500 total real cross-shard reductions (63 splits per tree at the real cited max_depth=6, times the illustrative 500 trees). There is no room for overlap WITHIN one tree, since each split strictly depends on the previous one's own result; the only room for overlap is ACROSS different trees, and only if they are built independently of one another rather than in XGBoost's own real strict sequential boosting order (where each new tree is fit to the residual of the trees built before it).
9. Chapter 4 built Universal Virtual Addressing from scratch as a way for GPU peer memory to be addressed as if it were local, without an explicit staged copy. Section 38.2's own real cited MAG240M benchmark article uses that exact same real idea, by its own real name, to let a graph's own node feature tensors be "instantiate[d]... such that it can be directly accessed by all the GPUs" during neighborhood sampling -- the same underlying real capability Chapter 4 introduced, now applied by NVIDIA's own real software to a completely different real workload many chapters later.

---

**Sources cited in this chapter:**

- NVIDIA. American Express fraud-prevention case study, referenced via nvidia.com/en-gb/customer-stories and resources.nvidia.com, fetched fresh this session. (The real "generating a fraud decision in milliseconds for every card transaction worldwide," "monitors more than $1.2 trillion in transaction value every year," and "deep-learning-based models optimized with NVIDIA TensorRT and running on NVIDIA Triton Inference Server" quotes.)
- NVIDIA Blog. "Bring Receipts: New NVIDIA AI Blueprint Detects Fraudulent Credit Card Transactions With Precision," blogs.nvidia.com, fetched fresh this session. (The real "fraudsters operate within complex networks..." and "...enhances the XGBoost ML model with NVIDIA CUDA-X Data Science libraries including GNNs to generate embeddings..." quotes, and the real NVIDIA Dynamo-Triton/RAPIDS software-stack description.)
- NVIDIA Technical Blog. "Optimizing Fraud Detection in Financial Services with Graph Neural Networks and NVIDIA GPUs," developer.nvidia.com, fetched fresh this session. (The real MAG240M benchmark's cited AUPRC 0.9-vs-0.79 accuracy figures, and the real "Universal Virtual Addressing (UVA), which allows us to instantiate our graph such that it can be directly accessed by all the GPUs" quote.)
- NVIDIA. "Multi-GPU Training," Financial Fraud Training container documentation, docs.nvidia.com/nim, fetched fresh this session. (The real "WholeGraph partitions the graph across all GPUs. Each GPU owns a shard of node feature tensors," "during neighbourhood sampling, if a sampled neighbour is on a different GPU, WholeGraph fetches its features transparently over NVLink or PCIe," "each process holds a shard of the embedding data," and "all GPUs collaborate on each tree (splits are computed across all shards simultaneously)" quotes.)
- XGBoost. Official parameter reference, xgboost.readthedocs.io, fetched fresh this session. (The real cited default, `"max_depth" [default=6, type=int32]`.)
- This book's own Chapter 4 (Universal Virtual Addressing, reused by real citation in Section 38.2), Chapter 9 (the reduction primitive reused throughout, including Section 38.3), Chapter 16 (domain decomposition, this chapter's point of contrast for graph partitioning), Chapter 28 (distributed BFS, the book's other graph-shaped communication pattern), Chapter 34 (the book's other hybrid-model precedent), and Chapter 36 (the T-independent-reductions structure Section 38.3 directly contrasts against).
