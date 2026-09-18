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
