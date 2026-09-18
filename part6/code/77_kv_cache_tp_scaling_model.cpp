// Chapter 26: Multi-GPU LLM Inference: Tensor and Pipeline Parallelism in
// Practice
// 77_kv_cache_tp_scaling_model.cpp
//
// Section 26.2 modeled why tensor parallelism stays bounded by a node's
// own GPU count. This section asks a different real question: within
// that node, why does adding tensor-parallel GPUs to a real inference
// SERVER help by more than the "one more GPU" arithmetic would suggest?
// vLLM's own real blog post states a genuinely surprising measured
// result: "Between TP=1 and TP=2, we are able to increase the amount of
// KV Cache blocks by 13.9x which allows us to observe 3.9x more token
// throughput -- much more than the linear 2x we would expect from
// using 2 GPUs instead of 1." This file does NOT re-derive vLLM's own
// internal accounting -- that real 13.9x/3.9x pair is cited, not
// computed by this program. What this section DOES build, as its own
// worked calculation, is a simple closed-form model of the real
// MECHANISM that can produce a super-linear KV-cache gain from tensor
// parallelism: at TP=1, one GPU's model weights (Chapter 13's own
// capacity-shard idea) eat into the SAME capacity budget the KV cache
// draws from; at TP=2, each GPU's weight shard shrinks, but its TOTAL
// capacity does not, so the memory freed for KV cache grows faster than
// linearly whenever the weights already occupy a large fraction of one
// GPU's memory. This section solves that model BACKWARD, asking what
// weight-fraction would reproduce vLLM's own cited 13.9x under this
// simplified shape -- an illustration of the mechanism's plausible
// scale, explicitly not a claim to have reconstructed vLLM's own real
// internal number.
#include <cstdio>

int main() {
    printf("--- Real, cited numbers (vLLM blog post) -- NOT computed by "
           "this program ---\n");
    printf("Between TP=1 and TP=2: KV cache blocks increase 13.9x; "
           "measured token throughput increases 3.9x (vs. a naive 2x "
           "expectation from doubling GPU count alone).\n\n");

    // --- This section's own worked model: capacity C per GPU, model
    // weight bytes W (shared across TP ranks at TP=2, full on the one
    // GPU at TP=1). KV-cache-available capacity:
    //   TP=1: C - W                     (one GPU, full weight)
    //   TP=2: 2*(C - W/2) = 2C - W      (two GPUs, each holds W/2)
    // Ratio (TP=2 KV capacity) / (TP=1 KV capacity) = (2C - W) / (C - W).
    // Solve for the weight fraction w = W/C that reproduces vLLM's own
    // cited 13.9x under this simplified shape. ---
    double target = 13.9;
    // (2 - w) / (1 - w) = target  =>  w = (target - 2) / (target - 1)
    double w = (target - 2.0) / (target - 1.0);
    printf("--- This chapter's own worked model (illustrative, NOT a "
           "reconstruction of vLLM's real internal accounting) ---\n");
    printf("Model: KV-cache-available capacity at TP=1 is (C - W); at "
           "TP=2 it is (2C - W), where C = one GPU's capacity, W = full "
           "model weight bytes.\n");
    printf("Solving (2 - w)/(1 - w) = %.1f for w = W/C: w = %.4f\n", target, w);
    printf("-> Under this simplified model, vLLM's own cited 13.9x KV-cache "
           "gain is the shape you'd expect if the model's weights alone "
           "were already consuming about %.1f%% of one GPU's memory at "
           "TP=1 -- leaving very little headroom for KV cache until a "
           "second GPU's capacity is added.\n\n", w * 100.0);

    // Show the same ratio across a small table of weight fractions, to
    // make the SHAPE of the effect visible, not just the one solved point.
    printf("%-12s %-24s\n", "W/C", "(2-w)/(1-w) KV-capacity ratio");
    double fractions[] = {0.50, 0.70, 0.80, 0.90, w, 0.95};
    for (double f : fractions) {
        double ratio = (2.0 - f) / (1.0 - f);
        printf("%-12.4f %-24.2f\n", f, ratio);
    }

    printf("\n--- The honest gap this section does NOT fill ---\n");
    printf("No published source found (vLLM docs, TensorRT-LLM docs, or an "
           "academic paper) gives a direct, apples-to-apples throughput or "
           "latency comparison of a COMBINED TP+PP configuration against "
           "TP-only or PP-only at the same total GPU count. TensorRT-LLM's "
           "own public docs describe the two qualitatively instead: TP is "
           "\"Best for: Small batch sizes, memory-constrained scenarios\"; "
           "PP is \"Best for: Large models that don't fit in single GPU "
           "memory.\" This section cites that real qualitative guidance "
           "rather than fabricating a number to fill the gap -- the same "
           "discipline this book applied in Chapter 21's own GPUDirect RDMA "
           "section and Chapter 25's own communication-cost model.\n");

    return 0;
}
