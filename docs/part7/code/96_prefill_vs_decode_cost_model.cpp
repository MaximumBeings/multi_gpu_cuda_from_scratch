// Chapter 33: projection and attention costs are separate terms.
// Arithmetic illustration only: no transformer or GPU benchmark is executed.
#include <cstdio>
#include <initializer_list>
int main() {
    constexpr long long d = 8192;
    std::printf("Illustrative single-layer MHA terms; multiply-add counts as 2 FLOPs.\n");
    std::printf("One d-by-d projection: prefill=2*N*d*d; decode=2*d*d.\n");
    std::printf("Dense decode attention QK+PV=4*N*d FLOPs; BF16 K+V cache=4*N*d bytes.\n");
    std::printf("N projection_prefill_FLOPs projection_decode_FLOPs attention_decode_FLOPs KV_read_bytes\n");
    for (long long n : {128LL, 512LL, 2048LL, 8192LL}) {
        const long long prefill = 2*n*d*d, projection = 2*d*d;
        const long long attention = 4*n*d, kv = 2*n*d*2;
        std::printf("%lld %lld %lld %lld %lld\n", n, prefill, projection, attention, kv);
        if (prefill != n*projection || attention != kv ||
            (n == 128 && (attention != 4194304 || projection != 134217728))) return 1;
    }
    std::printf("Attention work grows with context. These are selected terms, not total layer FLOPs or traffic.\n");
    std::printf("For GQA/MQA, cache bytes=2*N*Hkv*Dh*element_bytes; attention work still depends on query heads.\n");
    std::printf("Weights, other projections, MLPs, softmax, batch size and reuse must enter a full roofline model.\n");
    std::printf("MODEL CHECK PASS: projection/attention separation and both K/V byte counts\n");
    return 0;
}
