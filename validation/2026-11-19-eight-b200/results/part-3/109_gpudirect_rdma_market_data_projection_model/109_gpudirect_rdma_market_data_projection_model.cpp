// Chapter 37: GPUDirect RDMA in Capital Markets
// 109_gpudirect_rdma_market_data_projection_model.cpp
//
// Section 37.1 established a real, CURRENTLY-DEPLOYED baseline: kernel-
// bypass NIC processing delivers market data to HOST memory at a real
// cited sub-60-microsecond median latency. The obvious next question is
// whether Chapter 21's own real GPUDirect RDMA technique extends that
// pipeline the rest of the way into GPU memory for this specific
// workload TODAY. NVIDIA's own real blog post on Rivermax and NEIO
// FastSocket for financial services answers this honestly, and the
// honest answer matters: it describes GPUDirect for market data as a
// FUTURE direction, not a currently-deployed one -- "GPUDirect
// technology is poised to improve the performance of trading systems by
// enabling direct memory access between NICs and GPUs, bypassing the
// CPU to reduce latency," and "with Rivermax and GPUDirect powering
// zero-copy access, market data is streamed directly from high-speed
// NICs into GPU memory, eliminating PCIe bottlenecks" -- both quotes
// appear under that post's own "What's next" heading, not as a deployed
// result. This file does NOT claim
// NVIDIA has published a combined market-data-plus-GPUDirect-RDMA
// latency number, because no such real, specific number was found. It
// instead builds an explicitly-labeled ILLUSTRATIVE projection: applying
// Chapter 21's own real cited GPUDirect RDMA speedup (9.44x, computed
// from NVIDIA's own real "Benchmarking GPUDirect RDMA on Modern Server
// Platforms" post) to Section 37.1's own real cited kernel-bypass
// baseline (sub-60-microsecond median), to show what removing the
// remaining host-memory hop could plausibly be worth IF this real
// Chapter 21 technique were applied to this real Section 37.1 workload
// -- this book's own synthesis of two independently real, separately-
// cited numbers, not a claim NVIDIA has published this specific result.
#include <cstdio>

int main() {
    printf("Real quotes from NVIDIA's own Rivermax/NEIO FastSocket blog "
           "post, under its own \"What's next\" heading (not a deployed-"
           "today claim):\n");
    printf("- \"GPUDirect technology is poised to improve the performance "
           "of trading systems by enabling direct memory access between "
           "NICs and GPUs, bypassing the CPU to reduce latency.\"\n");
    printf("- \"With Rivermax and GPUDirect powering zero-copy access, "
           "market data is streamed directly from high-speed NICs into "
           "GPU memory, eliminating PCIe bottlenecks.\"\n\n");

    printf("Real, separately-cited numbers this file combines "
           "illustratively (never as a single NVIDIA-published result):\n");
    double realKernelBypassLatencyUs = 60.0;   // Section 37.1's own real Databento figure
    double realGpuDirectRdmaSpeedup = 9.44;    // Chapter 21's own real cited ratio

    printf("- Section 37.1's own real cited kernel-bypass-to-host median "
           "latency: below %.0f microseconds (Databento/NVIDIA case "
           "study).\n", realKernelBypassLatencyUs);
    printf("- Chapter 21's own real cited GPUDirect RDMA vs. staged-"
           "transfer speedup: %.2fx (computed in Chapter 21 from "
           "NVIDIA's own \"Benchmarking GPUDirect RDMA on Modern Server "
           "Platforms\" post, matching that post's own cited "
           "\"approximately 9x faster\" claim).\n\n", realGpuDirectRdmaSpeedup);

    double illustrativeProjectedLatencyUs = realKernelBypassLatencyUs / realGpuDirectRdmaSpeedup;

    printf("THIS BOOK'S OWN ILLUSTRATIVE PROJECTION (not an NVIDIA-"
           "published number): if Chapter 21's own real %.2fx GPUDirect "
           "RDMA speedup applied directly to Section 37.1's own real "
           "%.0fus kernel-bypass baseline, removing its remaining host-"
           "memory hop, the projected latency would be approximately "
           "%.2fus.\n\n", realGpuDirectRdmaSpeedup, realKernelBypassLatencyUs,
           illustrativeProjectedLatencyUs);

    printf("Why this is labeled a projection and not a finding: Chapter "
           "21's own %.2fx figure was measured on a synthetic small-"
           "message ping-pong benchmark on specific Ivy Bridge-era "
           "hardware, not on Databento's own real market-data pipeline or "
           "its own real NIC/GPU topology. Combining the two real numbers "
           "this way assumes the same relative speedup would carry over "
           "to a different, real, specific workload -- an assumption this "
           "book has no cited measurement to confirm. What IS real and "
           "directly cited is the DIRECTION: NVIDIA's own documentation "
           "places NIC-to-GPU GPUDirect RDMA for market data under \"What's "
           "next,\" not among its own already-deployed results, unlike "
           "Chapter 21's own GPU-to-GPU and storage-to-GPU GPUDirect RDMA "
           "use cases, which that chapter cited as already real and "
           "measured.\n", realGpuDirectRdmaSpeedup);
    return 0;
}
