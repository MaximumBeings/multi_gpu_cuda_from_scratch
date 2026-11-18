// Appendix F: From PyTorch Distributed and DeepSpeed to C++: A Rosetta Stone
// 137_zero_stage_memory_model.cpp
//
// Chapter 13's own File 36 re-derived the real 16-bytes-per-parameter
// mixed-precision Adam breakdown (Rajbhandari et al., ZeRO, SC'20; 2
// bytes fp16 param + 2 bytes fp16 grad + 4 bytes fp32 param + 4 bytes
// fp32 momentum + 4 bytes fp32 variance) and asked how many DEVICES a
// model's training state must be split across to exist at all. That
// question is Chapter 13's own model-PARALLEL question -- splitting the
// model's actual layers, computed by DIFFERENT devices. DeepSpeed's
// ZeRO asks a genuinely different question: across N DATA-parallel
// REPLICAS (Chapter 12's own kind of replica, where every device still
// computes the SAME full model on different data), how much of each
// replica's OWN copy of the training state can be sharded away, since
// every replica needs the state only transiently, not permanently. The
// two mechanisms are not the same thing wearing two names -- this file
// keeps them honestly separate, and derives ZeRO's own real published
// stage 1/2/3 formulas directly from Chapter 13's own 16-byte
// breakdown, replacing none of Chapter 13's numbers.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 137_zero_stage_memory_model.cpp -o 137_zero_stage_memory_model
// Run:     ./137_zero_stage_memory_model
#include <cstdio>
#include <cmath>

int main() {
    printf("=== Section F.3: DeepSpeed's ZeRO stages, derived from Chapter 13's own\n");
    printf("    16-bytes-per-parameter breakdown, not a new formula ===\n\n");

    printf("Chapter 13's own real per-parameter breakdown (mixed-precision Adam):\n");
    printf("  2 bytes fp16 parameter   + 2 bytes fp16 gradient   = 4 bytes  (unsharded by Pos/Pos+g)\n");
    printf("  4 bytes fp32 parameter + 4 bytes momentum + 4 bytes variance = 12 bytes  (DeepSpeed's own \"optimizer states\")\n");
    printf("  total: 16 bytes/parameter, exactly Chapter 13's own already-locked figure\n\n");

    printf("DeepSpeed's own real documented stages (deepspeed.readthedocs.io, quoted):\n");
    printf("  stage 1 (Pos):     \"optimizer states...are partitioned across the\n");
    printf("                      processes, so that each process updates only its\n");
    printf("                      partition\"\n");
    printf("  stage 2 (Pos+g):   \"the reduced 16-bit gradients...are also partitioned\n");
    printf("                      such that each process retains only the gradients\n");
    printf("                      corresponding to its portion of the optimizer states\"\n");
    printf("  stage 3 (Pos+g+p): \"the 16-bit model parameters are partitioned across\n");
    printf("                      the processes. ZeRO-3 will automatically collect and\n");
    printf("                      partition them during the forward and backward\n");
    printf("                      passes\"\n");
    printf("  config: DeepSpeedZeroConfig's own \"stage\" field, 1/2/3, inside the\n");
    printf("  zero_optimization dict -- a real, documented config key, not renamed here.\n\n");

    // Per-GPU bytes/parameter formula, derived (not looked up) from
    // Chapter 13's own 16-byte breakdown, sharded N ways:
    //   stage 1: 4 unsharded (fp16 param+grad) + 12/N sharded (optimizer states)
    //   stage 2: 2 unsharded (fp16 param only)  + 14/N sharded (optimizer states + fp16 grad)
    //   stage 3: 16/N -- everything sharded
    auto stage1 = [](double N) { return 4.0 + 12.0 / N; };
    auto stage2 = [](double N) { return 2.0 + 14.0 / N; };
    auto stage3 = [](double N) { return 16.0 / N; };

    // Self-consistency check against the ZeRO paper's own published
    // worked example (7.5B-parameter model, Nd=64): Pos=31.4GB,
    // Pos+g=16.6GB, Pos+g+p=1.9GB. If Chapter 13's own 16-byte
    // breakdown, sharded with this file's own formula, doesn't
    // reproduce those three published numbers, one of the two has a bug.
    {
        double psi = 7.5e9, N = 64.0;
        double s1GB = stage1(N) * psi / 1e9;
        double s2GB = stage2(N) * psi / 1e9;
        double s3GB = stage3(N) * psi / 1e9;
        printf("self-consistency check against the ZeRO paper's own published 7.5B-param,\n");
        printf("Nd=64 worked example (31.4 / 16.6 / 1.9 GB):\n");
        printf("  Pos     = %.1f GB\n", s1GB);
        printf("  Pos+g   = %.1f GB\n", s2GB);
        printf("  Pos+g+p = %.1f GB\n", s3GB);
        bool matches = (std::fabs(s1GB - 31.4) < 0.1) && (std::fabs(s2GB - 16.6) < 0.1) && (std::fabs(s3GB - 1.9) < 0.1);
        printf("  matches published figures: %s\n\n", matches ? "PASS" : "FAIL");
        if (!matches) return 1;
    }

    // Now apply the same, already-validated formula to Chapter 12's own
    // REPLICAS=8 data-parallel group of GPT-3 (Chapter 13's own 175B
    // model, Chapter 1's own already-locked 2800 GB unsharded figure).
    double psiGPT3 = 175.0e9;
    double unshardedGB = psiGPT3 * 16.0 / 1e9;
    printf("applying the SAME validated formula to Chapter 13's own GPT-3 figure,\n");
    printf("sharded across Chapter 12's own REPLICAS=8 data-parallel group:\n\n");
    printf("%-24s %14s\n", "configuration", "GB/replica");
    printf("%-24s %14.1f\n", "no ZeRO (Ch13 baseline)", unshardedGB);
    printf("%-24s %14.1f\n", "ZeRO stage 1 (Pos)", stage1(8.0) * psiGPT3 / 1e9);
    printf("%-24s %14.1f\n", "ZeRO stage 2 (Pos+g)", stage2(8.0) * psiGPT3 / 1e9);
    printf("%-24s %14.1f\n", "ZeRO stage 3 (Pos+g+p)", stage3(8.0) * psiGPT3 / 1e9);

    printf("\nnote what ZeRO does NOT change: Chapter 13's own model-parallel shard\n");
    printf("count (35 H100 SXM GPUs, from its own File 36) answers a different question\n");
    printf("-- how many devices must COOPERATE to hold one copy at all -- and ZeRO stage\n");
    printf("3's own %.1f GB/replica figure above is still larger than one H100 SXM's 80\n", stage3(8.0) * psiGPT3 / 1e9);
    printf("GB, meaning GPT-3 at this size needs BOTH: Chapter 13's model parallelism to\n");
    printf("split the model itself, and ZeRO-style sharding within each data-parallel\n");
    printf("replica group -- exactly the combination real large-model training uses, and\n");
    printf("exactly why DeepSpeed documents ZeRO stage 3 alongside, not instead of,\n");
    printf("model-parallel/tensor-parallel splitting.\n");

    return 0;
}
