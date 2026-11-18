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
