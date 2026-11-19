// Chapter 33: KV-cache handoff lower-bound model.
// DistServe cites 1.13 GB for an OPT-66B, 512-token request.
// A100's 600 GB/s NVLink specification sums both directions; use at
// most 300 GB/s for one sender's one-way transfer. These are ideal
// bytes/bandwidth bounds, not measured latency or a serving guarantee.
#include <cstdio>
#include <cmath>
int main() {
    const double cacheGB = 1.13;
    const double nvlinkOneWayGBps = 600.0 / 2.0;
    const double infinibandGBps = 800.0 / 8.0;
    const double nvlinkMs = 1000.0 * cacheGB / nvlinkOneWayGBps;
    const double infinibandMs = 1000.0 * cacheGB / infinibandGBps;
    std::printf("Ideal one-way transfer bounds for %.2f decimal GB:\n", cacheGB);
    std::printf("A100 NVLink: %.0f GB/s one-way ceiling -> %.3f ms\n", nvlinkOneWayGBps, nvlinkMs);
    std::printf("800 Gbit/s InfiniBand: %.0f GB/s line rate -> %.3f ms\n", infinibandGBps, infinibandMs);
    std::printf("Protocol overhead, topology, tensor-parallel sharding and contention change achieved latency.\n");
    std::printf("Overlap can hide work for other requests; this request still depends on its KV arrival.\n");
    if (std::fabs(nvlinkMs - 3.766666666666667) > 1e-12 ||
        std::fabs(infinibandMs - 11.3) > 1e-12) return 1;
    std::printf("MODEL CHECK PASS: directional bandwidth arithmetic\n");
    return 0;
}
