// Chapter 33: Disaggregated LLM Inference
// 96_prefill_vs_decode_cost_model.cpp
//
// Chapter 26 already built multi-GPU LLM inference as a combination of
// Chapter 14's tensor parallelism and Chapter 15's pipeline parallelism,
// treating "run the model" as one uniform workload. Zhong et al.'s own
// real DistServe paper ("DistServe: Disaggregating Prefill and Decoding
// for Goodput-optimized Large Language Model Serving," arXiv:2401.09670,
// USENIX OSDI 2024) makes the case that a single LLM inference request
// is actually TWO very different workloads glued together: "the prefill
// step deals with a new sequence, often comprising many tokens, and
// processes these tokens concurrently... the prefill step tends to be
// computation-bound," while, "in contrast, the decoding phase, despite
// processing only one new token per step, incurs a similar level of I/O
// to the prefill phase, making it constrained by the GPU's memory
// bandwidth." This file builds a small, clearly-labeled ILLUSTRATIVE
// model of that same contrast (never a measured benchmark -- this
// sandbox has no real GPU to benchmark on), grounded in Chapter 1's own
// arithmetic-intensity framing, and then presents DistServe's own real
// cited numbers -- its "goodput" metric and its measured 4.48x/10.2x
// improvement -- as real data, not something this book derived.
#include <cstdio>

int main() {
    printf("Illustrative model (Chapter 1's arithmetic-intensity framing "
           "applied to one transformer layer of hidden dimension d):\n");
    printf("%-12s %-22s %-22s %-18s\n", "Prompt N", "Prefill FLOPs (~N*d^2)",
           "Decode FLOPs/step (~d^2)", "Decode bytes/step (~N*d)");
    long long d = 8192;  // illustrative hidden dimension
    long long promptLens[] = {128, 512, 2048, 8192};
    for (long long N : promptLens) {
        long long prefillFlops = 2 * N * d * d;      // scales with N, parallel over tokens
        long long decodeFlopsPerStep = 2 * d * d;    // ONE new token: independent of N
        long long decodeBytesPerStep = N * d * 2;     // must still read the whole KV cache (bf16, 2 bytes)
        printf("%-12lld %-22lld %-22lld %-18lld\n", N, prefillFlops, decodeFlopsPerStep, decodeBytesPerStep);
    }
    printf("\nDecode's own FLOPs per step never grows with the prompt -- exactly "
           "DistServe's own point that decode is 'despite processing only one new "
           "token per step.' But decode's own BYTES MOVED per step (reading the "
           "growing KV cache) grows exactly as fast as prefill's own compute does. "
           "Arithmetic intensity (FLOPs per byte) therefore FALLS as the prompt "
           "grows for decode, while prefill's stays roughly constant -- the real "
           "reason decode is memory-bandwidth-bound and prefill is compute-bound, "
           "the same distinction Chapter 1's own memory-wall discussion introduced.\n");

    printf("\nDistServe's own real cited numbers (not derived by this book):\n");
    printf("- Goodput, DistServe's own definition: \"the maximum request rate that "
           "can be served adhering to the SLO attainment goal (say, 90%%) for each "
           "GPU provisioned -- higher per-GPU goodput directly translates into "
           "lower cost per query.\"\n");
    printf("- Measured result: \"DistServe can serve 4.48x more requests or 10.2x "
           "tighter SLO, compared to state-of-the-art systems, while staying "
           "within latency constraints for >90%% of requests.\"\n");
    printf("- Real KV cache size example cited in the same paper: \"the KV cache "
           "size of a single 512-token request on OPT-66B is approximately "
           "1.13GB\" -- Section 33.3 below uses this exact real number.\n");
    return 0;
}
