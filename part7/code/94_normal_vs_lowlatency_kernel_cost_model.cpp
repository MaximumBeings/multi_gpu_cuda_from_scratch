// Chapter 32: Mixture-of-Experts at Scale
// 94_normal_vs_lowlatency_kernel_cost_model.cpp
//
// DeepSeek-AI's own real DeepEP library ships TWO separate kernel
// families for the exact same operation (MoE dispatch and combine), not
// one "best" kernel -- and its own real documented numbers (V1, at
// docs/legacy.md in the deepseek-ai/DeepEP GitHub repository, freshly
// re-verified this session) explain why. "Normal kernels" are described
// as "high-throughput and low-latency all-to-all GPU kernels" tuned for
// TRAINING-sized batches; "low-latency kernels" are "a set of low-
// latency kernels with pure RDMA to minimize delays," built specifically
// "for inference decoding" where batches are tiny (as few as 8 tokens
// per expert) and every microsecond of end-to-end latency is user-
// visible. This file prints DeepEP's own two real cited tables side by
// side -- no timing here is measured or fabricated by this book; every
// number below is copied verbatim from DeepEP's own published benchmark
// data -- and computes simple, honest arithmetic ON those real numbers
// (a per-token latency, and a bandwidth-degradation ratio) to make the
// actual tradeoff visible.
#include <cstdio>

struct NormalRow { const char *type; int ep; double dispatchGBps; double combineGBps; };
struct LowLatRow { int ep; double dispatchUs; double dispatchGBps; double combineUs; double combineGBps; };

int main() {
    // DeepEP V1's own real "Normal kernels" table (H800, DeepSeek-V3/R1
    // settings), from docs/legacy.md.
    NormalRow normal[] = {
        {"Intranode", 8,  153.0, 158.0},
        {"Internode", 16, 43.0,  43.0},
        {"Internode", 32, 58.0,  57.0},
        {"Internode", 64, 51.0,  50.0},
    };
    printf("DeepEP V1 -- Normal kernels (real cited data, docs/legacy.md):\n");
    printf("%-10s %-6s %-20s %-20s\n", "Type", "EP", "Dispatch BW (GB/s)", "Combine BW (GB/s)");
    for (auto &r : normal) {
        printf("%-10s %-6d %-20.1f %-20.1f\n", r.type, r.ep, r.dispatchGBps, r.combineGBps);
    }

    // DeepEP V1's own real "Low-latency kernels" table (pure RDMA, 128
    // tokens per batch), same source.
    LowLatRow lowlat[] = {
        {8,   77.0,  98.0,  114.0, 127.0},
        {16,  118.0, 63.0,  195.0, 74.0},
        {32,  155.0, 48.0,  273.0, 53.0},
        {64,  173.0, 43.0,  314.0, 46.0},
        {128, 192.0, 39.0,  369.0, 39.0},
        {256, 194.0, 39.0,  360.0, 40.0},
    };
    printf("\nDeepEP V1 -- Low-latency kernels (real cited data, 128 tokens/batch):\n");
    printf("%-6s %-14s %-16s %-14s %-16s %-16s\n", "EP", "Dispatch us",
           "Dispatch GB/s", "Combine us", "Combine GB/s", "us/token (dispatch)");
    for (auto &r : lowlat) {
        double perTokenUs = r.dispatchUs / 128.0;
        printf("%-6d %-14.1f %-16.1f %-14.1f %-16.1f %-16.3f\n",
               r.ep, r.dispatchUs, r.dispatchGBps, r.combineUs, r.combineGBps, perTokenUs);
    }

    double bwAt8 = lowlat[0].dispatchGBps;
    double bwAt256 = lowlat[5].dispatchGBps;
    printf("\nLow-latency dispatch bandwidth falls from %.0f GB/s at EP=8 to "
           "%.0f GB/s at EP=256 -- a %.2fx drop -- as latency simultaneously "
           "RISES from %.0f us to %.0f us, a %.2fx increase. Both real "
           "numbers move in the same direction as EP grows: more experts to "
           "reach means more small, separate RDMA operations per token, not "
           "one bigger one.\n",
           bwAt8, bwAt256, bwAt8 / bwAt256,
           lowlat[0].dispatchUs, lowlat[5].dispatchUs, lowlat[5].dispatchUs / lowlat[0].dispatchUs);

    printf("\nNotice what DeepEP's own real documentation does NOT publish: "
           "no latency number at all for the normal kernels, at any EP "
           "degree. That omission is itself informative -- normal kernels "
           "are benchmarked and tuned purely as a throughput number because "
           "they are meant to move whole TRAINING batches (thousands of "
           "tokens) where per-call fixed overhead is amortized away; low-"
           "latency kernels report latency because at INFERENCE-DECODE "
           "batch sizes (the 128-token, 8-to-256-expert range above) that "
           "fixed overhead is exactly what a user waiting on a response "
           "feels. This mirrors Chapter 18's own finding that no single "
           "load-balancing strategy dominates across every workload shape: "
           "here, no single ALL-TO-ALL KERNEL dominates across every batch "
           "size either, which is exactly why DeepEP ships both.\n");
    return 0;
}
