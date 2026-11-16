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
// grounded in HugeCTR's own ETC documentation, which sizes memory per
// 8-GPU NODE, not per GPU: "with a 8x Tesla A100 (80 GB per GPU) setup,
// the cumulative GPU memory size ... is 640 GB." The per-GPU budget is
// therefore 80 GB (and less in practice, since activations, the dense
// model, optimizer state and workspace share the same HBM); 640 GB is
// what a whole 8-GPU node holds in aggregate.
#include <cstdio>

int main() {
    // HugeCTR's ETC documentation's own setup: 8x A100, 80 GB per GPU.
    double perGpuBudgetGB = 80.0;
    int gpusPerNode = 8;
    double perNodeAggregateGB = perGpuBudgetGB * gpusPerNode;  // 640 GB

    // The dense DNN part stays tiny by comparison -- illustrative size
    // for a realistic DLRM-style bottom+top MLP stack.
    double denseModelGB = 2.0;

    printf("Per-GPU HBM: %.0f GB (A100 80GB); one 8-GPU node in aggregate: "
           "%.0f GB -- the 640 GB figure in HugeCTR's ETC documentation is "
           "this NODE aggregate, not a single-GPU capacity.\n",
           perGpuBudgetGB, perNodeAggregateGB);
    printf("Dense model size: %.1f GB -- comfortably replicated on EVERY "
           "GPU regardless of GPU count (Chapter 12's data parallelism), "
           "since it is a small fraction of one GPU's %.0f GB.\n\n",
           denseModelGB, perGpuBudgetGB);

    struct Scenario { const char *label; double totalEmbeddingGB; };
    Scenario scenarios[] = {
        {"Small catalog",                                   50.0},
        {"One 8x A100 node's aggregate (HugeCTR ETC doc)",  640.0},
        {"Production-scale",                                2048.0},
        {"Terabyte-scale (10 TB)",                          10240.0},
    };

    printf("%-50s %-16s %-12s %-14s %-12s\n", "Scenario", "Total emb. (GB)",
           "Fits 1 GPU?", "GPUs (MP)", "8-GPU nodes");
    for (auto &s : scenarios) {
        bool fitsOne = s.totalEmbeddingGB <= perGpuBudgetGB;
        int gpusNeeded = (int)((s.totalEmbeddingGB + perGpuBudgetGB - 1.0) / perGpuBudgetGB);
        int nodesNeeded = (gpusNeeded + gpusPerNode - 1) / gpusPerNode;
        printf("%-50s %-16.1f %-12s %-14d %-12d\n", s.label, s.totalEmbeddingGB,
               fitsOne ? "YES" : "NO", gpusNeeded, nodesNeeded);
    }

    printf("\nThese GPU counts are LOWER bounds: they assume the whole %.0f GB "
           "of every GPU holds embedding rows and nothing else. Pure data "
           "parallelism (replicate the WHOLE model on every GPU, Chapter 12's "
           "technique alone) only works for the first scenario -- a 10 TB "
           "table is %.0fx one GPU's memory, and %.0fx even a whole 8-GPU "
           "node's aggregate. That is exactly why HugeCTR's own real design "
           "shards the embeddings (Chapter 13's technique) while keeping the "
           "dense layers replicated (Chapter 12's technique) -- a genuine "
           "HYBRID of two techniques this book built separately, applied to "
           "two different PARTS of the same model rather than to the whole "
           "model uniformly.\n",
           perGpuBudgetGB, 10240.0 / perGpuBudgetGB, 10240.0 / perNodeAggregateGB);
    return 0;
}
