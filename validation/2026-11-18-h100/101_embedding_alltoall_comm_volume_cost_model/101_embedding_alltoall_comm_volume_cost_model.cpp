// Chapter 34: GPU Recommendation Systems
// 101_embedding_alltoall_comm_volume_cost_model.cpp
//
// Section 34.1's own capacity model showed sharding embeddings across
// more GPUs (P) directly helps fit larger tables -- more P, more real
// per-GPU headroom. This file checks whether more P also helps the
// OTHER real cost HugeCTR's own paper names: the per-batch all-to-all
// that reshards embedding results from feature-parallel to batch-
// parallel every single training step. The volume rule is the standard
// one for a full all-to-all resharding: each rank keeps 1/P of the data
// it holds and must send the other (P-1)/P to the other ranks. Applied to
// a realistic batch x feature-slot x embedding-dimension payload, this
// closed-form model (never a fabricated timing) separates two numbers
// that are easy to confuse: the FRACTION of the payload that must cross
// the network, which rises toward 100% and saturates, and the BYTES EACH
// GPU must send per step, which keeps falling roughly as 1/P.
#include <cstdio>

int main() {
    // Illustrative but realistic DLRM-style figures (bf16 embeddings,
    // 2 bytes per value). 26 categorical feature slots is the Criteo
    // click-log layout used by the DLRM benchmarks.
    long long batchSize = 65536;      // a real large-scale training batch size
    long long numSlots = 26;          // categorical feature slots per sample
    long long embeddingDim = 128;     // a real common embedding vector width
    long long bytesPerElement = 2;    // bf16
    long long totalPayloadBytes = batchSize * numSlots * embeddingDim * bytesPerElement;

    printf("Per-step embedding payload: batch=%lld x slots=%lld x dim=%lld x "
           "%lld bytes = %lld bytes (%.3f GB) that must be FULLY reassembled "
           "into batch-parallel shards every single training step.\n\n",
           batchSize, numSlots, embeddingDim, bytesPerElement, totalPayloadBytes,
           totalPayloadBytes / 1e9);

    int Ps[] = {1, 2, 4, 8, 16, 32, 64, 128};
    printf("%-6s %-18s %-22s %-22s %-18s\n", "P", "Fraction off-rank",
           "Total bytes on network", "Bytes sent PER GPU", "Per-GPU emb. shard (GB, 10TB total)");
    double totalEmbeddingGB = 10240.0;  // reuse Section 34.1's terabyte-scale scenario
    for (int P : Ps) {
        // Each rank holds 1/P of the payload before the all-to-all and
        // sends (P-1)/P of what it holds to other ranks.
        double offRankFraction = (P > 1) ? (double)(P - 1) / (double)P : 0.0;
        double bytesCrossing = totalPayloadBytes * offRankFraction;
        double bytesPerGpu = bytesCrossing / (double)P;
        double perGpuShardGB = totalEmbeddingGB / (double)P;
        printf("%-6d %-18.4f %-22.0f %-22.0f %-18.3f\n", P, offRankFraction,
               bytesCrossing, bytesPerGpu, perGpuShardGB);
    }

    printf("\nTwo different quantities, two different trends. The FRACTION of "
           "the per-step payload that must cross the network rises from 0%% "
           "(P=1) toward 100%% and saturates -- already over 87%% at P=8 -- so "
           "the TOTAL bytes on the network approach the whole payload and stop "
           "growing. But the bytes EACH GPU must send keep falling roughly as "
           "1/P (exactly payload*(P-1)/P^2), because each GPU holds a smaller "
           "slice to begin with. So more GPUs DO keep shrinking each GPU's "
           "per-step all-to-all traffic; what they cannot do is shrink the "
           "aggregate traffic below (nearly) the whole payload, which matters "
           "when a shared resource -- a switch's bisection bandwidth, the "
           "inter-node fabric -- carries all of it. Real all-to-all time also "
           "has a per-message latency term that grows with P, which this "
           "volume-only model omits.\n");
    return 0;
}
