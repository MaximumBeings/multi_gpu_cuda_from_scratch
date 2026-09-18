// Chapter 24: Multi-GPU Dense Matrix Multiplication at Scale
// 69_gemm_memory_wall_model.cpp
//
// A real, closed-form model (no fabricated timings, matching this book's
// own established practice since Ch2/Ch5/Ch16) of when a dense GEMM's
// three matrices genuinely stop fitting on one real GPU. Reuses Chapter
// 1's own real H100 SXM figure (80 GB HBM3) as the single-device budget.
// A real NVIDIA forum thread on cuBLASXt states plainly what problem
// this section is building toward: cuBLASXt's "main point" is to "work
// around GPU memory capacity limits" -- i.e. out-of-core / multi-GPU
// GEMM exists FIRST because of capacity, not (only) because of speed.
#include <cstdio>
#include <cstdint>
#include <cmath>

struct GemmShape {
    const char *label;
    uint64_t m, k, n;
};

int main() {
    const double H100_SXM_HBM_BYTES = 80.0 * 1024.0 * 1024.0 * 1024.0; // Ch1's own real figure
    const double BYTES_PER_FP32 = 4.0;

    // Real GPT-3 175B dimensions (Brown et al. Table 2.1, already cited
    // in Ch1/Ch13/Ch14): d_model=12288. A single FFN weight matrix is
    // d_model x 4*d_model. This section asks a genuinely different
    // question from Ch13's per-layer question: not "does ONE weight
    // matrix fit," but "do the three GEMM operands -- A, B, and the
    // output C -- fit AT ONCE, at a batch size and sequence length a
    // real training run actually uses."
    GemmShape shapes[] = {
        {"GPT-3 FFN weight alone (d_model x 4d_model)", 12288, 12288, 49152},
        {"GPT-3 FFN forward, batch=512, seq=2048 (M=batch*seq)",
         512ull * 2048ull, 12288, 49152},
        {"A hypothetical 10x-larger FFN forward at the same batch/seq",
         512ull * 2048ull, 122880, 491520},
    };

    printf("Single-device budget (Ch1's own real H100 SXM figure): %.1f GiB\n\n",
           H100_SXM_HBM_BYTES / (1024.0 * 1024.0 * 1024.0));

    for (auto &s : shapes) {
        double bytesA = (double)s.m * (double)s.k * BYTES_PER_FP32;
        double bytesB = (double)s.k * (double)s.n * BYTES_PER_FP32;
        double bytesC = (double)s.m * (double)s.n * BYTES_PER_FP32;
        double totalGiB = (bytesA + bytesB + bytesC) / (1024.0 * 1024.0 * 1024.0);
        bool fits = (bytesA + bytesB + bytesC) <= H100_SXM_HBM_BYTES;
        printf("%s\n", s.label);
        printf("  M=%llu K=%llu N=%llu (fp32)\n",
               (unsigned long long)s.m, (unsigned long long)s.k, (unsigned long long)s.n);
        printf("  A+B+C = %.2f GiB -> %s on one H100 SXM\n\n",
               totalGiB, fits ? "FITS" : "DOES NOT FIT");
    }

    // A real, closed-form minimum-device-count model: if the three
    // operands don't fit on one device, how many devices, under a
    // cuBLASXt-style even block split of A/B/C across devices, are
    // needed at minimum? This does NOT model communication cost (that
    // is Section 24.2's own job) -- only capacity.
    printf("--- Minimum device count under even block distribution (capacity only) ---\n");
    for (auto &s : shapes) {
        double bytesA = (double)s.m * (double)s.k * BYTES_PER_FP32;
        double bytesB = (double)s.k * (double)s.n * BYTES_PER_FP32;
        double bytesC = (double)s.m * (double)s.n * BYTES_PER_FP32;
        double totalBytes = bytesA + bytesB + bytesC;
        int minDevices = (int)std::ceil(totalBytes / H100_SXM_HBM_BYTES);
        if (minDevices < 1) minDevices = 1;
        printf("%s -> minimum %d device(s)\n", s.label, minDevices);
    }

    return 0;
}
