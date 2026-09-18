// Chapter 21: GPUDirect RDMA: Bypassing the Host Entirely
// 62_staged_vs_rdma_cost_model.cpp
//
// Plain host C++, no device/MPI/NCCL needed -- this section's job is
// to quantify, using cited real numbers (not new measurements this
// book has no hardware to take), the gap between Section 20.2's own
// real staged path (cudaMemcpy D2H -> network -> cudaMemcpy H2D) and
// GPUDirect RDMA's direct path. Every constant below is cited to
// NVIDIA's own "Benchmarking GPUDirect RDMA on Modern Server
// Platforms" post; every combination of them is this book's own
// arithmetic, not a reproduction of an unpublished internal model --
// Part C below is explicit about that boundary.
#include <cstdio>

int main() {
    // ---- Part A: small-message latency, cited fixed numbers ----
    // "GPUDirect RDMA provides a latency consistently below 2us" --
    // measured 1.7us (GPU-to-Host) and 1.9us (Host-to-GPU direction).
    // The staged path's own cited breakdown: cudaMemcpy/cudaMemcpyAsync
    // "can easily take 8us and 9us respectively," plus InfiniBand's own
    // "1.3us" host-to-host latency -- the post's own rounded total for
    // the full GPU-to-GPU staged round trip is "approximately 17us."
    const double RDMA_LATENCY_LOW_US = 1.7;
    const double RDMA_LATENCY_HIGH_US = 1.9;
    const double STAGED_LATENCY_US = 17.0;

    double rdmaAvg = (RDMA_LATENCY_LOW_US + RDMA_LATENCY_HIGH_US) / 2.0;
    double speedup = STAGED_LATENCY_US / rdmaAvg;
    printf("Part A -- small-message fixed latency (cited, not measured here):\n");
    printf("  GPUDirect RDMA: %.1f-%.1fus (avg %.2fus)\n",
           RDMA_LATENCY_LOW_US, RDMA_LATENCY_HIGH_US, rdmaAvg);
    printf("  Staged (D2H + IB host-to-host + H2D): ~%.1fus\n", STAGED_LATENCY_US);
    printf("  Ratio: %.2fx -- matches the source's own cited "
           "\"approximately 9x faster\" claim (computed here from its own "
           "two numbers, not independently re-measured).\n\n", speedup);

    // ---- Part B: large-message bandwidth, topology-dependent ----
    // Best case, both devices under the SAME PCIe switch (the
    // GPUDirect RDMA design guide's own top-tier topology): 11.6 GB/s
    // Host-to-GPU with a dual-rail setup. Worst real case in the SAME
    // post: crossing the inter-socket QPI link collapses write
    // bandwidth to "250 MB/s when the IB adapter pushes data to a GPU
    // on a different socket." Same two endpoints NCCL's own
    // NCCL_NET_GDR_LEVEL knob names directly -- PIX (same PCIe switch)
    // down through SYS (cross-NUMA-node, over the SMP interconnect) --
    // reusing Chapter 2's own topology-matters framing, now for a
    // GPU-NIC pair instead of a GPU-GPU pair.
    const double BEST_CASE_GBPS = 11.6;   // same PCIe switch (PIX-class)
    const double WORST_CASE_GBPS = 0.25;  // 250 MB/s, crossing QPI (SYS-class)
    double degradation = BEST_CASE_GBPS / WORST_CASE_GBPS;
    printf("Part B -- large-message bandwidth, same real post, different "
           "PCIe topology:\n");
    printf("  Same PCIe switch (NCCL_NET_GDR_LEVEL=PIX-class): %.1f GB/s\n",
           BEST_CASE_GBPS);
    printf("  Crossing inter-socket QPI (NCCL_NET_GDR_LEVEL=SYS-class): "
           "%.2f GB/s\n", WORST_CASE_GBPS);
    printf("  Degradation: %.1fx -- \"GPUDirect RDMA capable\" (Section "
           "21.2's own boolean attribute) is NECESSARY but not "
           "SUFFICIENT: the same capable GPU, paired with the same "
           "capable NIC, is %.0fx slower if the PCIe topology between "
           "them is wrong, with no code change at all.\n\n", degradation, degradation);

    // ---- Part C: the real crossover, cited as a finding, not re-derived ----
    printf("Part C -- the source's own real crossover finding:\n");
    printf("  The same post states: \"GPUDirect RDMA is faster than the "
           "staging approach for message sizes up to 400-500KB (on Ivy "
           "Bridge Xeon)\" -- a real, cited finding this book reports "
           "rather than re-derives: NVIDIA's post names the crossover "
           "but does not publish the underlying model that produces it, "
           "and Parts A and B above already show why a simple two-term "
           "(fixed latency + bandwidth) model built from this post's own "
           "OTHER numbers should not be expected to reproduce it exactly "
           "-- Part A's numbers alone would put GPUDirect RDMA ahead at "
           "every size, since it has both the lower fixed latency AND "
           "(Part B, best case) the higher bandwidth. The real crossover "
           "almost certainly depends on GPUDirect RDMA's own bandwidth "
           "ceiling being LOWER than the staged path's at large sizes on "
           "specifically the Ivy Bridge platform tested -- a real, "
           "platform-specific effect this book has no cited numbers for, "
           "so it is reported as a limit of what is known here, not "
           "quietly modeled around.\n");

    return 0;
}
