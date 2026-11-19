// Chapter 26: Multi-GPU LLM Inference: Tensor and Pipeline Parallelism in
// Practice
// 76_tp_pp_scale_cost_model.cpp
//
// Section 26.1 proved the combined TP+PP grid's ROUTING is correct at
// toy scale. This section asks the real question that shape answers:
// why 8-way tensor parallelism and 35-way pipeline parallelism,
// specifically, for a real 530-billion-parameter model? The Megatron-
// Turing NLG 530B paper (Smith et al.) states its own real config
// directly: "each 530 billion parameter model replica spans 280 NVIDIA
// A100 GPUs, with 8-way tensor-slicing within a node and 35-way
// pipeline parallelism across nodes" (8 x 35 = 280). The Megatron-LM
// paper explains WHY the split is drawn exactly at the node boundary:
// "the all-reduce communication required for tensor parallelism needs
// to go through inter-server links, which are slower than the high-
// bandwidth NVLink available within a multi-GPU server," and therefore
// "tensor model parallelism should generally be used up to degree g
// when using g-GPU servers, and then pipeline model parallelism can be
// used to scale up to larger models across servers." This section
// builds a real closed-form model of exactly WHY that specific
// boundary matters, reusing Chapter 9's own ring all-reduce cost
// formula -- and finds, honestly, that the answer is NOT primarily
// about communication VOLUME (which Chapter 9/25 already showed stays
// nearly flat regardless of N), but about ROUND COUNT, which does NOT
// stay flat, combined with the real fact that every one of those extra
// rounds would have to cross the slower inter-server link the
// Megatron-LM paper's own quote names.
#include <cstdio>
#include <cstdint>
#include <initializer_list>

int main() {
    // Real Megatron-Turing NLG 530B configuration (Smith et al.):
    const uint64_t PARAMS = 530000000000ULL; // 530 billion
    const int TP_DEGREE = 8;   // real: within a node
    const int PP_DEGREE = 35;  // real: across nodes
    const int TOTAL_GPUS = TP_DEGREE * PP_DEGREE; // 280, matches the paper

    printf("Megatron-Turing NLG 530B (Smith et al.): %llu params, "
           "TP=%d (within a node), PP=%d (across nodes), TOTAL=%d GPUs "
           "per model replica\n\n",
           (unsigned long long)PARAMS, TP_DEGREE, PP_DEGREE, TOTAL_GPUS);

    // --- Capacity: this chapter's own worked calculation, reusing
    // Chapter 13's own per-shard memory model. fp16 (2 bytes/param) is
    // this section's own stated assumption -- a common LLM-serving
    // precision, not itself a number quoted from the paper. ---
    const double BYTES_PER_PARAM_FP16 = 2.0;
    double totalBytes = (double)PARAMS * BYTES_PER_PARAM_FP16;
    double totalGiB = totalBytes / (1024.0 * 1024.0 * 1024.0);
    double perGpuGiB = totalGiB / TOTAL_GPUS;
    printf("Total fp16 parameter bytes: %.2f GiB. Sharded evenly across "
           "all %d GPUs (TP x PP combined): %.3f GiB/GPU -- this chapter's "
           "own worked calculation, not a number from the paper.\n\n",
           totalGiB, TOTAL_GPUS, perGpuGiB);

    // --- Communication: Chapter 9's own real ring all-reduce formula,
    // applied to TWO cases -- the REAL TP=8 config, and a HYPOTHETICAL
    // TP=280 (using tensor parallelism for the ENTIRE replica, with no
    // pipeline parallelism at all). K is left symbolic (an arbitrary
    // activation buffer size) since this section compares RATIOS, not
    // absolute bytes -- no fabricated timing, matching this book's own
    // established practice (Ch21/Ch25). ---
    printf("--- Ch9's own ring all-reduce formula, applied to TWO cases ---\n");
    for (int N : {TP_DEGREE, TOTAL_GPUS}) {
        int rounds = 2 * (N - 1);
        double volumeRatioToConstant = 2.0 * (double)(N - 1) / (double)N; // -> 2 as N grows
        printf("N=%-4d rounds=2(N-1)=%-4d   per-rank volume factor 2(N-1)/N = %.4f "
               "(approaches the constant 2 either way)\n",
               N, rounds, volumeRatioToConstant);
    }

    double volN8 = 2.0 * (TP_DEGREE - 1) / (double)TP_DEGREE;
    double volN280 = 2.0 * (TOTAL_GPUS - 1) / (double)TOTAL_GPUS;
    int roundsN8 = 2 * (TP_DEGREE - 1);
    int roundsN280 = 2 * (TOTAL_GPUS - 1);
    printf("\nCommunication VOLUME ratio, TP=280 vs TP=8: %.4fx "
           "(nearly flat -- Chapter 9/25's own finding still holds: volume "
           "barely grows with N).\n", volN280 / volN8);
    printf("Communication ROUND-COUNT ratio, TP=280 vs TP=8: %dx / %dx = %.1fx "
           "(this does NOT stay flat -- it scales directly with N-1).\n",
           roundsN280, roundsN8, (double)roundsN280 / (double)roundsN8);

    printf("\nAnd critically: EVERY one of TP=8's %d rounds stays inside "
           "one node (real NVLink, per the Megatron-LM paper's own "
           "quote), while ALL %d of TP=280's rounds would have to cross "
           "the slower inter-server link the SAME paper names as the "
           "real reason to keep tensor parallelism inside a node.\n",
           roundsN8, roundsN280);

    return 0;
}
