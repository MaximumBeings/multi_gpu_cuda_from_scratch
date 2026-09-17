// Chapter 1: Why One GPU Is Not Enough
// 02_memory_wall.cpp
//
// Plain host C++. No CUDA, no simulation, nothing to fake: this is ordinary
// arithmetic on numbers taken from cited real sources (see the chapter
// text for citations), computed exactly and reproducibly.
#include <cstdio>
#include <cstdint>
#include <cmath>

struct Model {
    const char* name;
    double params; // parameter count
};

int main() {
    const Model models[] = {
        {"7B-class model",   7.0e9},
        {"GPT-3 (175B)",   175.0e9},
    };

    // Real GPU memory capacities, decimal GB as the vendor datasheets state
    // them (NVIDIA H100 datasheet / product page).
    const double H100_SXM_GB = 80.0;
    const double H100_NVL_GB = 94.0;
    const double BYTES_PER_GB = 1.0e9;

    printf("%-16s %10s %16s %16s %10s %10s\n",
           "Model", "Params", "Inference(fp16)", "Training(Adam,mp)",
           "H100 SXM x", "H100 NVL x");

    for (const auto& m : models) {
        // Inference footprint: weights only, fp16/bf16 -> 2 bytes/param.
        double inferenceBytes = m.params * 2.0;

        // Training footprint, mixed-precision Adam, per the ZeRO paper
        // (Rajbhandari et al., SC'20): 2 (fp16 params) + 2 (fp16 grads)
        // + 4 (fp32 params) + 4 (fp32 momentum) + 4 (fp32 variance)
        // = 16 bytes per parameter. This excludes activation memory,
        // which is workload- (batch size, sequence length) dependent
        // and addressed separately in the chapter text.
        double trainingBytes = m.params * 16.0;

        double inferenceGB = inferenceBytes / BYTES_PER_GB;
        double trainingGB  = trainingBytes  / BYTES_PER_GB;

        double sxmNeeded = std::ceil(trainingGB / H100_SXM_GB);
        double nvlNeeded = std::ceil(trainingGB / H100_NVL_GB);

        printf("%-16s %8.1fB %13.1f GB %13.1f GB %10.0f %10.0f\n",
               m.name, m.params / 1e9, inferenceGB, trainingGB,
               sxmNeeded, nvlNeeded);
    }

    printf("\nMinimum H100 SXM (80 GB) GPUs to hold GPT-3's training state "
           "(params+grads+Adam state only, no activations): %.0f\n",
           std::ceil((175.0e9 * 16.0 / BYTES_PER_GB) / H100_SXM_GB));

    return 0;
}
