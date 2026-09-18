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
