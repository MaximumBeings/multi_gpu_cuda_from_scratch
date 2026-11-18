// Chapter 1: Why One GPU Is Not Enough
// 03_compute_wall.cpp
//
// Plain host C++. Computes a deterministic FLOP count and divides by a
// cited real peak-FLOPS figure -- never a fabricated wall-clock number,
// per this book's own stated discipline (see getting-started.md).
#include <cstdio>

int main() {
    // Kaplan et al., "Scaling Laws for Neural Language Models" (2020):
    // C ~= 6*N*D, where N is (non-embedding) parameter count and D is
    // dataset size in tokens. We use GPT-3's reported total parameter
    // count (175B, Brown et al. 2020, Table 2.1) as an approximation of
    // N, and the paper's own stated token budget, D = 300e9 tokens
    // ("All models were trained for a total of 300 billion tokens").
    const double N = 175.0e9;
    const double D = 300.0e9;
    const double totalFLOPs = 6.0 * N * D;

    // NVIDIA H100 SXM5: 989 TFLOP/s peak dense BF16/FP16 Tensor Core
    // throughput (no sparsity). Real vendor-cited figure.
    const double H100_PEAK_FLOPS = 989.0e12;
    const double SECONDS_PER_DAY = 86400.0;

    const double idealFlopsPerDay = H100_PEAK_FLOPS * SECONDS_PER_DAY;
    const double idealGpuDays = totalFLOPs / idealFlopsPerDay;

    // Real training never sustains 100% of peak. We label this an
    // illustrative assumption, not a measurement: many public reports of
    // well-optimized large-model training put achieved throughput in the
    // 30-50% range of peak (Model FLOPs Utilization, or MFU); we use 50%
    // here as a round, explicitly-labeled upper-end illustrative figure.
    const double assumedMFU = 0.50;
    const double realisticGpuDays = idealGpuDays / assumedMFU;

    printf("Total training compute (C = 6*N*D): %.3e FLOPs\n", totalFLOPs);
    printf("N (parameters, approx.):            %.3e\n", N);
    printf("D (tokens):                         %.3e\n", D);
    printf("H100 SXM peak dense BF16/FP16:       %.3e FLOP/s\n",
           H100_PEAK_FLOPS);
    printf("\n");
    printf("GPU-days on a single H100 at 100%% of peak (ideal, "
           "unreachable in practice): %.1f\n", idealGpuDays);
    printf("GPU-days on a single H100 at an assumed %.0f%% of peak "
           "(illustrative MFU, not measured):   %.1f\n",
           assumedMFU * 100.0, realisticGpuDays);
    printf("Same total compute spread across 1024 such GPUs at the "
           "assumed %.0f%% MFU: %.2f days\n",
           assumedMFU * 100.0, realisticGpuDays / 1024.0);

    return 0;
}
