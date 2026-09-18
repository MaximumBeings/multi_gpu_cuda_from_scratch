// Chapter 37: GPUDirect RDMA in Capital Markets
// 108_kernel_bypass_market_data_baseline_model.cpp
//
// Chapter 21 built real GPUDirect RDMA between a NIC and a GPU, but its
// own hybrid communicator already assumed data had somehow reached a
// network interface in the first place. Real production market-data
// systems answer that "somehow" with a specific, currently-DEPLOYED
// technique that has nothing to do with GPUs yet: kernel-bypass NIC
// processing. NVIDIA's own real Databento case study describes exactly
// this, at real production scale: Databento's real platform "exceeds 80
// Gbps and handles over 200 billion market updates each day," with "US
// options data alone surpassing standard 40G network links," using
// "kernel-bypass paths on NVIDIA NICs" and "DPDK acceleration" to reach
// a real cited "median network latency [that] dropped below 60
// microseconds," with "ConnectX hardware timestamping" for sequencing.
// This file presents those real cited numbers directly and does simple,
// clearly-labeled arithmetic on them (average updates/second, average
// bytes/update at 80 Gbps) -- establishing the real, deployed starting
// point this chapter's own GPUDirect RDMA discussion (Sections 37.2 and
// 37.3) builds on: market data reaching HOST memory via kernel bypass,
// not yet GPU memory.
#include <cstdio>

int main() {
    // Real cited Databento production figures.
    double realGbps = 80.0;
    double realUpdatesPerDay = 200.0e9;
    double realMedianLatencyUs = 60.0;  // "dropped below 60 microseconds"

    printf("Real cited Databento production figures (NVIDIA case study):\n");
    printf("- Sustained throughput: over %.0f Gbps (\"exceeds 80 Gbps\")\n", realGbps);
    printf("- Daily volume: over %.0e market updates/day (\"over 200 billion "
           "market updates each day\")\n", realUpdatesPerDay);
    printf("- Real cited median network latency (kernel-bypass + DPDK on "
           "NVIDIA NICs): below %.0f microseconds\n\n", realMedianLatencyUs);

    // Simple, clearly-labeled arithmetic on the real cited figures above
    // -- not new measurements, just unit conversions.
    double avgUpdatesPerSecond = realUpdatesPerDay / 86400.0;
    double bytesPerSecondAt80Gbps = realGbps * 1.0e9 / 8.0;
    double avgBytesPerUpdate = bytesPerSecondAt80Gbps / avgUpdatesPerSecond;

    printf("Derived (plain arithmetic on the real cited figures above):\n");
    printf("Average updates/second across a full day: %.2e\n", avgUpdatesPerSecond);
    printf("Bytes/second sustained at the real cited 80 Gbps figure: %.2e\n",
           bytesPerSecondAt80Gbps);
    printf("Implied average bytes/update if 80 Gbps were sustained at "
           "this DAILY-AVERAGE update rate: %.1f bytes\n\n", avgBytesPerUpdate);

    printf("A honest flag on that last number: real market-data messages "
           "are typically tens to a few hundred bytes, not %.0f bytes -- "
           "so this arithmetic result is a mismatch, not a real average "
           "message size. It shows that the real cited \"80 Gbps\" figure "
           "describes sustained PEAK capacity, while the real cited "
           "\"200 billion updates/day\" figure is a DAILY AVERAGE across "
           "quiet and busy periods alike; dividing one by the other "
           "conflates two different real numbers that were never meant "
           "to describe the same moment. Reporting this mismatch honestly, "
           "rather than quietly presenting %.0f bytes as a real message "
           "size, matters more here than the arithmetic itself.\n\n",
           avgBytesPerUpdate, avgBytesPerUpdate);

    printf("This is the real, CURRENTLY-DEPLOYED starting point for "
           "everything this chapter builds on: kernel-bypass NIC "
           "processing (DPDK, hardware timestamping) gets market data "
           "from the wire into HOST memory at real sub-%.0f-microsecond "
           "median latency, at real production scale -- but it stops at "
           "host memory. Nothing in this real, cited pipeline yet "
           "involves a GPU at all. Section 37.2 asks the honest question "
           "this chapter exists to answer: does Chapter 21's own "
           "GPUDirect RDMA extend this real pipeline the rest of the way, "
           "directly into GPU memory, for this specific workload today?\n",
           realMedianLatencyUs);
    return 0;
}
