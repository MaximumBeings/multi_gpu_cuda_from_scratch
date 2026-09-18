// Chapter 33: Disaggregated LLM Inference
// 97_phase_specific_tensor_parallelism_correctness_simulation.cpp
//
// NVIDIA's own real Dynamo documentation states the design point behind
// disaggregation directly: separate prefill and decode workers let a
// system use "a larger TP for the memory-bound decoding phase while a
// smaller TP for the computation-bound prefill phase" -- a genuinely
// different tensor-parallelism DEGREE for each phase of the SAME
// request, something Chapter 14 and Chapter 26 never needed to consider
// because they treated a request as one uniform TP-parallel computation
// throughout. This file checks the obvious correctness question that
// design raises: if the prefill phase runs under TP degree Tp and the
// decode phase runs under a DIFFERENT TP degree Td, connected only by
// the transferred KV cache (Section 33.3's own subject), does the final
// generated sequence still match a single-process (Tp=Td=1) reference
// exactly? Every value here is a plain integer specifically so this
// file's own answer is not entangled with the floating-point
// associativity question Chapter 8/25/29/31/32 already covered in
// detail -- the question this file asks is a DIFFERENT one (does
// splitting a single request's two phases onto independently-sized
// device groups preserve correctness), not a repeat of that one.
#include <cstdio>
#include <vector>

const int H = 8;             // total attention heads (illustrative)
const int PROMPT_LEN = 16;   // prompt tokens
const int NUM_DECODE_STEPS = 6;

unsigned int counterBasedHash(long long x) {
    unsigned int h = (unsigned int)x;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return h;
}

long long headWeight(int h) { return (long long)((counterBasedHash(h + 9000) % 9) + 1); }
long long promptValue(int i) { return (long long)((counterBasedHash(i + 1000) % 20) + 1); }
long long decodeWeight(int t) { return (long long)((counterBasedHash(t + 5000) % 7) + 1); }

// Prefill: for each prompt position, sum head contributions -- split
// across Tp shards of heads, each shard reduced locally, then the Tp
// partial sums combined (integer addition: exactly associative, so this
// step alone could never show a mismatch regardless of Tp).
std::vector<long long> runPrefill(int Tp) {
    std::vector<long long> kv(PROMPT_LEN, 0);
    for (int i = 0; i < PROMPT_LEN; i++) {
        long long total = 0;
        for (int shard = 0; shard < Tp; shard++) {
            long long partial = 0;
            for (int h = shard; h < H; h += Tp) partial += headWeight(h) * promptValue(i);
            total += partial;
        }
        kv[i] = total;
    }
    return kv;
}

// Decode: each step reads the FULL kv cache (the transferred prefill
// result, plus every previously-generated decode value) and produces
// one new value, split across Td shards of kv-cache POSITIONS, combined
// the same way.
std::vector<long long> runDecode(const std::vector<long long> &kv, int Td) {
    std::vector<long long> cache = kv;  // grows as decode proceeds
    std::vector<long long> generated;
    for (int t = 0; t < NUM_DECODE_STEPS; t++) {
        int cacheLen = (int)cache.size();
        long long total = 0;
        for (int shard = 0; shard < Td; shard++) {
            long long partial = 0;
            for (int i = shard; i < cacheLen; i += Td) partial += cache[i];
            total += partial;
        }
        long long value = total * decodeWeight(t);
        generated.push_back(value);
        cache.push_back(value);
    }
    return generated;
}

int main() {
    std::vector<long long> referenceKv = runPrefill(1);
    std::vector<long long> referenceDecode = runDecode(referenceKv, 1);

    printf("Reference (Tp=1, Td=1): kv[0..3]=%lld,%lld,%lld,%lld  "
           "decode tokens: ", referenceKv[0], referenceKv[1], referenceKv[2], referenceKv[3]);
    for (long long v : referenceDecode) printf("%lld ", v);
    printf("\n\n");

    int prefillTPs[] = {1, 2, 4, 8};
    int decodeTPs[] = {1, 2, 3, 6};
    printf("%-6s %-6s %-10s %-30s\n", "Tp", "Td", "match", "note");
    for (int Tp : prefillTPs) {
        for (int Td : decodeTPs) {
            std::vector<long long> kv = runPrefill(Tp);
            std::vector<long long> decoded = runDecode(kv, Td);
            bool match = (kv == referenceKv) && (decoded == referenceDecode);
            const char *note = (Tp <= Td) ? "Tp<=Td (Dynamo's own recommended shape)"
                                           : "Tp>Td (not the recommended shape, still correct)";
            printf("%-6d %-6d %-10s %-30s\n", Tp, Td, match ? "YES" : "NO", note);
        }
    }

    printf("\nEvery single (Tp, Td) combination matches the Tp=1/Td=1 reference "
           "exactly, whether or not Tp and Td are equal, and whether or not the "
           "pair follows NVIDIA Dynamo's own recommended shape (smaller TP for "
           "prefill, larger TP for decode). Correctness here does not come from "
           "the two phases sharing a device-group size -- it comes from each "
           "phase's own internal reduction being exact on its own, and the two "
           "phases being connected ONLY through the kv cache values themselves, "
           "never through shared parallelism structure. This is the real "
           "correctness argument behind disaggregation: prefill and decode are "
           "allowed to be sized independently precisely because nothing about "
           "their correctness depends on being sized the same.\n");
    return 0;
}
