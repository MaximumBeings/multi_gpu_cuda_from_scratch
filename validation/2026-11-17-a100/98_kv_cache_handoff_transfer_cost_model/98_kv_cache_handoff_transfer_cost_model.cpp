// Chapter 33: Disaggregated LLM Inference
// 98_kv_cache_handoff_transfer_cost_model.cpp
//
// Every collective this book built since Chapter 8 moved data among ALL
// members of a group (a broadcast to everyone, a reduce from everyone,
// an all-to-all among everyone). The KV cache handoff between a prefill
// instance and a decode instance is a genuinely different shape: a
// targeted, one-time, point-to-point bulk transfer from one specific
// device (or TP-group) to one other specific device (or TP-group) --
// the closest ancestor in this book is Chapter 5's explicit
// cudaMemcpyPeer() and Chapter 21's GPUDirect RDMA, not any collective.
// Zhong et al.'s own real DistServe paper is candid that this transfer
// is not free -- "transferring KV caches from prefill to decoding
// instances incurs notable overheads" -- and gives a real, concrete
// example: "the KV cache size of a single 512-token request on OPT-66B
// is approximately 1.13GB." This file takes that exact real number and
// combines it with DistServe's own real cited interconnect bandwidths
// -- "Infiniband (e.g., 800 Gbps)" for cross-node transfer and
// "intra-node NVLINK, where the peak bandwidth between A100 GPUs is 600
// GB/s" -- to compute the real transfer TIME implied by those real
// numbers, the same arithmetic DistServe's own paper uses to conclude
// the overhead is negligible. No timing here is measured (this sandbox
// has no real GPU or RDMA fabric); every input is a real cited number,
// and the only computation performed is bytes divided by bandwidth.
#include <cstdio>

int main() {
    double kvCacheGB = 1.13;  // DistServe's own real cited OPT-66B/512-token example

    printf("DistServe's own real cited KV cache example: a single 512-token "
           "OPT-66B request's KV cache is approximately %.2f GB.\n\n", kvCacheGB);

    struct Link { const char *name; double gbPerSec; const char *source; };
    Link links[] = {
        {"Intra-node NVLink (A100)", 600.0, "DistServe paper: \"peak bandwidth between A100 GPUs is 600 GB/s\""},
        {"Cross-node InfiniBand",    100.0, "DistServe paper: \"Infiniband (e.g., 800 Gbps)\" = 800/8 GB/s"},
    };

    printf("%-28s %-12s %-16s\n", "Link", "GB/s", "Transfer time (ms)");
    for (auto &l : links) {
        double seconds = kvCacheGB / l.gbPerSec;
        double ms = seconds * 1000.0;
        printf("%-28s %-12.1f %-16.3f\n", l.name, l.gbPerSec, ms);
    }

    printf("\nAt real NVLink bandwidth, moving this real 1.13GB KV cache takes "
           "about %.2f milliseconds; even over real cross-node InfiniBand, about "
           "%.2f milliseconds. Both numbers are small enough next to a real LLM "
           "serving request's own end-to-end latency (typically hundreds of "
           "milliseconds to multiple seconds for a multi-hundred-token response) "
           "that DistServe's own paper calls the transmission overhead "
           "\"negligible\" -- this is the exact arithmetic behind that real "
           "conclusion, not a new estimate this book is introducing.\n",
           (kvCacheGB / 600.0) * 1000.0, (kvCacheGB / 100.0) * 1000.0);

    printf("\nNVIDIA's own real Dynamo documentation adds one further real design "
           "point this arithmetic alone does not show: the transfer does not even "
           "need to be on the critical path at all -- \"the KV transfer is "
           "non-blocking, allowing GPU forward passes to continue serving other "
           "requests during the transfer.\" A cost small enough to call negligible "
           "AND overlapped with other useful work is the real, two-part reason "
           "production disaggregated serving systems treat this handoff as a "
           "solved problem rather than their main bottleneck.\n");
    return 0;
}
