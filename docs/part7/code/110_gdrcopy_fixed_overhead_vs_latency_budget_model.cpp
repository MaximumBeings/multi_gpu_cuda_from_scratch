// Chapter 37: GPUDirect RDMA in Capital Markets
// 110_gdrcopy_fixed_overhead_vs_latency_budget_model.cpp
//
// Every earlier chapter's own cost model (Chapter 5, 9, 16, 27, 28, 34,
// 35) found that fixed per-call overhead becomes NEGLIGIBLE once enough
// data moves -- communication VOLUME was always what eventually
// dominated. Real single-digit-microsecond capital-markets inference
// inverts that lesson completely. NVIDIA's own real "Achieving Single-
// Digit Microsecond Latency Inference for Capital Markets" blog reports
// real cited P99 latencies this small: "4.61 microseconds p99" on a
// real GH200 (four LSTM_A instances) and a real cited "4.3" microsecond
// P99 figure on RTX PRO 6000 Blackwell for a small model -- and
// identifies the real bottleneck directly: "the overhead of CPU-GPU
// synchronization and reading the input vector from host memory... is
// the dominant contributor to latency for the small model." NVIDIA's
// own real GDRCopy library exists for exactly this problem -- its own
// real README states "while GPUDirect RDMA is meant for direct access
// to GPU memory from third-party devices, it is possible to use these
// same APIs to create perfectly valid CPU mappings of the GPU memory.
// The advantage of a CPU driven copy is the very small overhead
// involved," and cites real, specific numbers: an ordinary "cudaMemcpy
// can incur in a 6-7us overhead," against GDRCopy's own real cited
// ping-pong round-trip latency of "1.08762 us" (on a real A40 GPU).
// This file places those real, separately-cited numbers side by side
// against the real cited total latency budgets above, to show something
// no earlier chapter's cost model needed to show: a FIXED per-call
// overhead alone -- not any payload size -- can exceed an entire real
// production latency budget.
#include <cstdio>

int main() {
    // Real cited GDRCopy README numbers.
    double cudaMemcpyOverheadLowUs = 6.0;
    double cudaMemcpyOverheadHighUs = 7.0;
    double gdrCopyRoundTripUs = 1.08762;  // real cited A40 ping-pong result

    // Real cited capital-markets total inference latency budgets.
    double gh200TotalP99Us = 4.61;        // real cited GH200, LSTM_A x4
    double rtxPro6000TotalP99Us = 4.3;    // real cited RTX PRO 6000, small model

    printf("Real cited fixed per-call overhead (NVIDIA/gdrcopy README):\n");
    printf("- Ordinary cudaMemcpy round trip: %.1f-%.1f microseconds\n",
           cudaMemcpyOverheadLowUs, cudaMemcpyOverheadHighUs);
    printf("- GDRCopy (CPU-mapped direct read via GPUDirect RDMA): %.5f "
           "microseconds (real cited A40 ping-pong result)\n\n", gdrCopyRoundTripUs);

    printf("Real cited TOTAL P99 inference latency budgets (NVIDIA capital-"
           "markets blog):\n");
    printf("- GH200, four LSTM_A instances: %.2f microseconds p99\n", gh200TotalP99Us);
    printf("- RTX PRO 6000 Blackwell, small model: %.2f microseconds p99\n\n",
           rtxPro6000TotalP99Us);

    printf("%-46s %-18s %-18s\n", "Comparison", "vs GH200 budget", "vs RTX PRO 6000 budget");
    double memcpyFracGh200 = 100.0 * cudaMemcpyOverheadLowUs / gh200TotalP99Us;
    double memcpyFracRtx = 100.0 * cudaMemcpyOverheadLowUs / rtxPro6000TotalP99Us;
    double gdrFracGh200 = 100.0 * gdrCopyRoundTripUs / gh200TotalP99Us;
    double gdrFracRtx = 100.0 * gdrCopyRoundTripUs / rtxPro6000TotalP99Us;
    printf("%-46s %-17.1f%% %-17.1f%%\n", "cudaMemcpy overhead alone (low end, 6us)",
           memcpyFracGh200, memcpyFracRtx);
    printf("%-46s %-17.1f%% %-17.1f%%\n", "GDRCopy round trip alone",
           gdrFracGh200, gdrFracRtx);

    printf("\nA plain cudaMemcpy round trip's own real cited FIXED "
           "overhead (%.1f-%.1fus) is, by itself, %.0f%%-%.0f%% of these "
           "real total P99 latency budgets -- at the low end of NVIDIA's "
           "own cited range, it alone would consume MORE than the "
           "entire real RTX PRO 6000 budget, before a single byte of the "
           "actual input vector or model computation is counted. GDRCopy's "
           "own real cited round trip (%.5fus) is small enough, by "
           "contrast, to leave the vast majority of either real budget "
           "for the model's own computation. This is the opposite lesson "
           "from every earlier chapter's own cost model: there, a fixed "
           "per-call cost eventually became negligible as payload grew; "
           "here, the payload (one small input vector) was always tiny, "
           "so it is the FIXED overhead itself, not any volume, that "
           "threatens to consume the entire real latency budget -- "
           "exactly the real problem GDRCopy's own GPUDirect-RDMA-based "
           "CPU mapping exists to solve.\n",
           cudaMemcpyOverheadLowUs, cudaMemcpyOverheadHighUs,
           memcpyFracRtx < memcpyFracGh200 ? memcpyFracRtx : memcpyFracGh200,
           memcpyFracRtx > memcpyFracGh200 ? memcpyFracRtx : memcpyFracGh200,
           gdrCopyRoundTripUs);
    return 0;
}
