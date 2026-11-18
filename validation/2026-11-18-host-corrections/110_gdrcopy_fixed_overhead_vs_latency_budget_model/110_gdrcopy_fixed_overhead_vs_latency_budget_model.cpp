// Chapter 37: compare published magnitudes without equating experiments.
// NVIDIA/gdrcopy README: cudaMemcpy can incur 6-7 us overhead;
// its A40 CPU/GPU ping-pong example reports 1.08762 us round-trip.
// The NVIDIA capital-markets article reports separate GH200 and
// RTX PRO 6000 inference experiments. This program benchmarks neither.
#include <cstdio>
#include <cmath>
int main() {
    const double memcpyLowUs = 6.0, memcpyHighUs = 7.0;
    const double a40PingPongUs = 1.08762;
    const double gh200P99Us = 4.61, rtxP99Us = 4.3;
    std::printf("Published figures from distinct experiments (not a controlled comparison):\n");
    std::printf("cudaMemcpy per-call overhead illustration: %.1f-%.1f us; not a round-trip measurement\n", memcpyLowUs, memcpyHighUs);
    std::printf("A40 GDRCopy CPU/GPU ping-pong: %.5f us round-trip; not a CPU read-latency figure\n", a40PingPongUs);
    std::printf("GH200 four LSTM_A instances: %.2f us p99; RTX PRO 6000 small model: %.2f us p99\n", gh200P99Us, rtxP99Us);
    const double ratios[] = {100.0 * memcpyLowUs / gh200P99Us,
                             100.0 * memcpyLowUs / rtxP99Us,
                             100.0 * a40PingPongUs / gh200P99Us,
                             100.0 * a40PingPongUs / rtxP99Us};
    std::printf("Arithmetic ratios to those separate budgets: memcpy %.1f%%/%.1f%%; ping-pong %.1f%%/%.1f%%\n", ratios[0], ratios[1], ratios[2], ratios[3]);
    std::printf("Ratios do not predict a replacement's latency, throughput or tail percentile.\n");
    std::printf("CPU writes through mapped GPU memory and CPU reads have different costs; GDRCopy warns device-to-host reads can be slow.\n");
    if (std::fabs(ratios[0] - 130.1518438177874) > 1e-10) return 1;
    std::printf("MODEL CHECK PASS: distinct-operation comparison\n");
    return 0;
}
