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
// answer is about the RESHARDING itself -- the pack/send/unpack index
// bookkeeping -- not about floating-point associativity.
//
// What makes this a real test rather than a tautology: the P-rank path
// never reads the global answer. Each rank computes only its own feature
// shard (features f with f % P == r -- CYCLIC striping, HugeCTR-style
// "feature index modulo P"), packs one send buffer per destination rank,
// the all-to-all delivers buffers, and each destination rank (owning a
// CONTIGUOUS block of B/P samples) must unpack every received value into
// the right (sample, feature) slot using only the schedule. A wrong
// assumption about the schedule produces wrong vectors -- which the file
// demonstrates with a second, deliberately buggy unpacker that assumes
// each source rank sent a CONTIGUOUS block of features.
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

// Feature ownership: CYCLIC, rank r owns features r, r+P, r+2P, ...
int featureOwner(int f, int P) { return f % P; }
// Sample ownership after the all-to-all: CONTIGUOUS blocks of B/P.
int sampleOwner(int s, int P) { return s / (B / P); }

// One rank's view after the all-to-all: its B/P samples, each with a full
// F-length feature vector (-1 marks a slot nobody filled).
typedef std::vector<std::vector<long long>> LocalSamples;

// P-rank simulation. Returns the global (B x F) matrix stitched together
// from every rank's LOCAL post-all-to-all samples, so it can be compared
// with the reference. contiguousBug selects the deliberately wrong unpack.
std::vector<std::vector<long long>> runWithP(int P, bool contiguousBug,
                                             long long *valuesOffRank) {
    // Phase 1 (model-parallel): rank r looks up ITS features for ALL B
    // samples, and packs them into one send buffer per destination rank d,
    // ordered (sample in d's block, then r's features in ascending order).
    // sendBuf[r][d] is the message rank r sends to rank d.
    std::vector<std::vector<std::vector<long long>>> sendBuf(
        P, std::vector<std::vector<long long>>(P));
    for (int r = 0; r < P; r++) {
        for (int s = 0; s < B; s++) {
            int d = sampleOwner(s, P);
            for (int f = r; f < F; f += P) sendBuf[r][d].push_back(embeddingValue(s, f));
        }
    }

    // Phase 2 (the all-to-all): message (r -> d) arrives at d as recvBuf[d][r].
    *valuesOffRank = 0;
    std::vector<std::vector<std::vector<long long>>> recvBuf(
        P, std::vector<std::vector<long long>>(P));
    for (int r = 0; r < P; r++)
        for (int d = 0; d < P; d++) {
            recvBuf[d][r] = sendBuf[r][d];
            if (d != r) *valuesOffRank += (long long)sendBuf[r][d].size();
        }

    // Phase 3 (unpack, data-parallel): rank d rebuilds full feature vectors
    // for its own contiguous sample block from the P received messages.
    int samplesPerRank = B / P;
    std::vector<LocalSamples> local(P, LocalSamples(samplesPerRank, std::vector<long long>(F, -1)));
    for (int d = 0; d < P; d++) {
        for (int r = 0; r < P; r++) {
            const std::vector<long long> &msg = recvBuf[d][r];
            int featuresFromR = (int)msg.size() / samplesPerRank;
            size_t pos = 0;
            for (int ls = 0; ls < samplesPerRank; ls++) {
                for (int k = 0; k < featuresFromR; k++) {
                    // Correct: the k-th feature rank r owns is r + k*P (cyclic).
                    // Bug: assume r owned a contiguous block r*(F/P) + k.
                    int f = contiguousBug ? r * (F / P) + k : r + k * P;
                    local[d][ls][f] = msg[pos++];
                }
            }
        }
    }

    // Stitch the per-rank local results into a global matrix for checking.
    std::vector<std::vector<long long>> global(B);
    for (int s = 0; s < B; s++) global[s] = local[sampleOwner(s, P)][s % samplesPerRank];
    return global;
}

int main() {
    auto reference = runReference();
    printf("Reference (single-process): sample 0's full feature vector: ");
    for (long long v : reference[0]) printf("%lld ", v);
    printf("\n\n");

    int Ps[] = {1, 2, 3, 4, 6, 12};
    printf("%-6s %-22s %-26s %-28s\n", "P", "values sent off-rank",
           "cyclic unpack: all match", "contiguous-bug unpack: wrong");
    for (int P : Ps) {
        long long offRank = 0, offRankBug = 0;
        auto good = runWithP(P, false, &offRank);
        auto bad = runWithP(P, true, &offRankBug);
        bool allMatch = true;
        int wrongSlots = 0;
        for (int s = 0; s < B; s++)
            for (int f = 0; f < F; f++) {
                if (good[s][f] != reference[s][f]) allMatch = false;
                if (bad[s][f] != reference[s][f]) wrongSlots++;
            }
        printf("%-6d %-22lld %-26s %d of %d slots\n", P, offRank,
               allMatch ? "YES" : "NO", wrongSlots, B * F);
    }

    printf("\nThe correct unpacker reproduces every sample's feature vector "
           "at every P, and the number of values that cross the network is "
           "exactly B*F*(P-1)/P. The buggy unpacker -- which assumes each "
           "rank owned a CONTIGUOUS block of features when the schedule "
           "actually stripes them CYCLICALLY (feature index modulo P) -- "
           "scrambles feature slots at every P where the two layouts differ "
           "(they coincide only at P=1 and P=F, where each rank owns either "
           "every feature or exactly one). Nothing here is ever summed, so "
           "the operation is exact; what CAN go wrong is the bookkeeping, "
           "which is why the pack and unpack sides of an all-to-all must "
           "agree on one layout.\n");
    return 0;
}
