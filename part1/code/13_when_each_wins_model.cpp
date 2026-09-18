// Chapter 5: Explicit Transfers -- cudaMemcpyPeer, Staged Host Transfers,
// and When Each Wins
// 13_when_each_wins_model.cpp
//
// Plain host C++ cost model, reusing this book's own already-cited
// bandwidth figures (Chapter 2, docs/part0/02-multi-gpu-hardware-topology.md):
// PCIe 4.0 x16 per-direction bandwidth (31.5 GB/s) and a direct, wired
// NVLink connection in a partial-mesh topology (50 GB/s). This is a
// closed-form arithmetic model of transfer TIME, not a measurement of
// real hardware -- it answers "which route wins, and by how much" using
// numbers this book has already sourced and cited, not new fabricated
// timings.
#include <cstdio>

int main() {
    const double PCIE_GBPS = 31.5;   // direct PCIe 4.0 x16, per direction
    const double NVLINK_GBPS = 50.0; // direct wired NVLink pair, partial mesh

    const double transferSizeGB = 1.0;

    // Peer-direct route: one hop, over whichever direct link exists.
    // If the pair is NVLink-wired, that's the NVLink figure; if the
    // pair's only direct link is PCIe, that's the PCIe figure.
    double peerTimeNVLink = transferSizeGB / NVLINK_GBPS;
    double peerTimePCIe   = transferSizeGB / PCIE_GBPS;

    // Staged route: two hops through the host, device->host and
    // host->device, each paying the PCIe rate -- this is exactly what
    // Section 5.2's explicit two-step transfer measured the call
    // pattern for, and what cudaMemcpyPeer itself falls back to
    // whenever peer access has not been enabled for that pair.
    double stagedTime = 2.0 * (transferSizeGB / PCIE_GBPS);

    printf("Transfer size: %.1f GB\n\n", transferSizeGB);
    printf("Peer-direct route, NVLink-wired pair:   %.6f s (%.1f GB/s, one hop)\n",
           peerTimeNVLink, NVLINK_GBPS);
    printf("Peer-direct route, PCIe-only direct pair: %.6f s (%.1f GB/s, one hop)\n",
           peerTimePCIe, PCIE_GBPS);
    printf("Staged host route (any pair):            %.6f s (%.1f GB/s, two hops)\n\n",
           stagedTime, PCIE_GBPS);

    double speedupNVLinkVsStaged = stagedTime / peerTimeNVLink;
    double speedupPCIeDirectVsStaged = stagedTime / peerTimePCIe;

    printf("Peer-direct over NVLink is %.4fx faster than staging, for a wired pair.\n",
           speedupNVLinkVsStaged);
    printf("Peer-direct over a direct PCIe link is %.4fx faster than staging,\n"
           "for a pair with a direct PCIe route but no NVLink wire.\n",
           speedupPCIeDirectVsStaged);
    printf("\nWhen the pair has NO direct link at all (per Chapter 2's partial-mesh\n"
           "topology), peer access cannot be enabled between them in the first\n"
           "place (Chapter 4, Section 4.2) -- staging is then not merely faster\n"
           "or slower, it is the ONLY route that exists.\n");

    return 0;
}
