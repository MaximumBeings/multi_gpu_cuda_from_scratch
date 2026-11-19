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
    // "GPUDirect RDMA provides a latency consistently below 2us." The
    // post's two figures are NOT two ends of one range: 1.7us is its
    // Host-to-GPU figure, and 1.9us is the GPU-to-GPU figure. The
    // staged path's own cited breakdown: cudaMemcpy/cudaMemcpyAsync
    // "can easily take 8us and 9us respectively," plus InfiniBand's own
    // "1.3us" host-to-host latency -- the post's own rounded total for
    // the full GPU-to-GPU staged round trip is "approximately 17us."
    // A like-for-like comparison therefore pairs the GPU-to-GPU staged
    // total with the GPU-to-GPU RDMA figure -- not an average across
    // directions.
    const double RDMA_HOST_TO_GPU_US = 1.7; // not used in the ratio
    const double RDMA_GPU_TO_GPU_US = 1.9;
    const double STAGED_GPU_TO_GPU_US = 17.0;

    double speedup = STAGED_GPU_TO_GPU_US / RDMA_GPU_TO_GPU_US;
    printf("Part A -- small-message fixed latency (cited, not measured here):\n");
    printf("  GPUDirect RDMA, GPU-to-GPU:  %.1fus  (Host-to-GPU: %.1fus, "
           "a different path, not averaged in)\n",
           RDMA_GPU_TO_GPU_US, RDMA_HOST_TO_GPU_US);
    printf("  Staged GPU-to-GPU (D2H + IB host-to-host + H2D): ~%.1fus\n",
           STAGED_GPU_TO_GPU_US);
    printf("  Ratio: %.1fx -- this book's own arithmetic on the post's two "
           "GPU-to-GPU numbers; the post itself states no single "
           "speedup factor.\n\n", speedup);

    // ---- Part B: large-message bandwidth, topology-dependent ----
    // The post's headline 11.6 GB/s Host-to-GPU figure is a DUAL-rail
    // (two NIC) result; its 250 MB/s figure -- "when the IB adapter
    // pushes data to a GPU on a different socket," crossing QPI -- is a
    // SINGLE-rail result on the Ivy Bridge Xeon platform. Dividing one
    // by the other (46x) mixes NIC counts AND platforms. Matching at
    // least the rail count, the post's single-rail GPUDirect RDMA figure
    // is about 6.1 GB/s. Same two endpoints NCCL's own
    // NCCL_NET_GDR_LEVEL knob names directly -- PIX (same PCIe switch)
    // down through SYS (cross-NUMA-node, over the SMP interconnect) --
    // reusing Chapter 2's own topology-matters framing, now for a
    // GPU-NIC pair instead of a GPU-GPU pair.
    const double DUAL_RAIL_GBPS = 11.6;   // dual-rail, NOT comparable to the QPI case
    const double SINGLE_RAIL_GBPS = 6.1;  // single-rail GPUDirect RDMA, good PCIe path
    const double QPI_SINGLE_RAIL_GBPS = 0.25; // 250 MB/s, crossing QPI, Ivy Bridge (SYS-class)
    double degradation = SINGLE_RAIL_GBPS / QPI_SINGLE_RAIL_GBPS;
    printf("Part B -- large-message bandwidth, same real post, single-rail "
           "vs single-rail:\n");
    printf("  Single rail, GPU and NIC on a good PCIe path: "
           "~%.1f GB/s\n", SINGLE_RAIL_GBPS);
    printf("  Single rail, crossing inter-socket QPI on Ivy Bridge "
           "(SYS-class): %.2f GB/s\n", QPI_SINGLE_RAIL_GBPS);
    printf("  Degradation: ~%.0fx (the post's %.1f GB/s headline is "
           "dual-rail, so %.1f / %.2f would double-count the second NIC). "
           "Even this ~%.0fx is not one controlled experiment -- the two "
           "figures need not come from the same server platform -- but "
           "the order of magnitude is the point: \"GPUDirect RDMA "
           "capable\" (Section 21.2's own boolean attribute) is NECESSARY "
           "but not SUFFICIENT; a bad PCIe path between a capable GPU and "
           "a capable NIC can cost more than an order of magnitude of "
           "bandwidth with no code change at all.\n\n",
           degradation, DUAL_RAIL_GBPS, DUAL_RAIL_GBPS, QPI_SINGLE_RAIL_GBPS,
           degradation);

    // ---- Part C: the real crossover, cited as a finding, not re-derived ----
    printf("Part C -- the source's own real crossover finding:\n");
    printf("  The same post states: \"GPUDirect RDMA is faster than the "
           "staging approach for message sizes up to 400-500KB (on Ivy "
           "Bridge Xeon)\" -- a real, cited finding this book reports "
           "rather than re-derives. A simple two-term (fixed latency + "
           "bandwidth) model built from Part A's latency and Part B's "
           "good-topology bandwidth would put GPUDirect RDMA ahead at "
           "every size, so it cannot reproduce the crossover; the post "
           "ties the crossover to the platform it measured (Ivy Bridge "
           "Xeon, whose cross-socket QPI path is also Part B's 250 MB/s "
           "case), and this book does not rebuild that platform-specific "
           "bandwidth model here.\n");

    return 0;
}
