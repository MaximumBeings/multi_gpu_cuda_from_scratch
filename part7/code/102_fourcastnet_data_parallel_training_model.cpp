// Chapter 35: AI Weather Forecasting at Scale
// 102_fourcastnet_data_parallel_training_model.cpp
//
// NVIDIA and collaborators' own real FourCastNet paper ("FourCastNet: A
// Global Data-driven High-resolution Weather Model using Adaptive
// Fourier Neural Operators," Pathak et al., arXiv:2202.11214, 2022)
// reports a real, specific multi-GPU training shape: "the end to end
// training takes about 16 hours wall-clock time on a cluster of 64
// Nvidia A100 GPUs." At the real global grid size the paper cites --
// "a resolution of 0.25 degree which corresponds to... a global grid
// size of 720x1440 pixels" -- that grid comfortably fits in a single
// A100's own memory, so this real 64-GPU training job is, structurally,
// Chapter 12's own data parallelism (whole-model replicas, batch split
// across GPUs, gradients synchronized by an all-reduce), NOT Chapter
// 16's spatial domain decomposition -- a genuine surprise for a "weather
// grid" model, and the reason this section exists before Section 35.2
// gets to the real technique (distributed FFT) that DOES eventually
// require spatial splitting. This file builds a simple, clearly-labeled
// illustrative scaling model (not a fabricated timing) around the real
// 16-hour/64-GPU anchor point, and prints FourCastNet's own real cited
// speedup and energy numbers directly.
#include <cstdio>

int main() {
    double realHoursAt64GPUs = 16.0;
    int realGpuCount = 64;

    printf("Real cited anchor point (Pathak et al. 2022): %.0f hours "
           "wall-clock on %d real A100 GPUs.\n\n", realHoursAt64GPUs, realGpuCount);

    printf("Illustrative near-linear data-parallel scaling model (Chapter "
           "12's own technique: whole-model replicas, gradient all-reduce "
           "each step) anchored to that real point -- NOT a claim about "
           "measured scaling efficiency, which this sandbox cannot "
           "measure:\n");
    printf("%-12s %-20s\n", "GPU count", "Illustrative hours");
    int gpuCounts[] = {8, 16, 32, 64, 128, 256};
    for (int g : gpuCounts) {
        double hours = realHoursAt64GPUs * (double)realGpuCount / (double)g;
        printf("%-12d %-20.2f\n", g, hours);
    }

    printf("\nFourCastNet's own real cited results (not derived by this "
           "book):\n");
    printf("- \"FourCastNet generates a week-long forecast in less than 2 "
           "seconds, orders of magnitude faster than IFS.\"\n");
    printf("- \"FourCastNet is about 45,000 times faster than traditional "
           "NWP models on a node-hour basis.\"\n");
    printf("- \"Once trained, however, FourCastNet uses about 12,000 times "
           "less energy to generate a forecast than the IFS model.\"\n");
    printf("- Real global grid: \"a resolution of 0.25 degree... a global "
           "grid size of 720x1440 pixels\" -- comfortably fits one real "
           "A100's memory, which is exactly why the real 64-GPU/16-hour "
           "training run above is ordinary data parallelism, not spatial "
           "domain decomposition.\n");
    return 0;
}
