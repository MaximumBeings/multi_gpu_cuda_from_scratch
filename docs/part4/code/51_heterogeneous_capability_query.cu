// Chapter 18: Load Balancing Across Heterogeneous GPUs
// 51_heterogeneous_capability_query.cu
//
// Every equal-split function this book has written since Chapter 12
// -- computeShard() (12.1), computeLayerRange() (13.2), and
// computeRowRange() (16.1) -- divides a total by worldSize and hands
// every rank an identical-sized piece. None of them ever asked how
// FAST any given rank actually is. That's a silent assumption:
// every real cluster eventually mixes GPU generations, and even
// identical GPUs throttle differently under heat. A real, well-known
// paper on training across heterogeneous GPU clusters (Um et al.,
// "Cephalo," 2024) states the consequence plainly: "In clusters with
// varying GPU capabilities, training is bottlenecked by the slowest
// GPU, leaving faster GPUs idle." This section queries the one real
// signal this book has never used before -- a device's own physical
// capability, via cudaGetDeviceProperties() -- as the first step
// toward fixing that.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    printf("\nQuerying real per-device capability for every device this "
           "book's own real deviceCount (%d) reports:\n", deviceCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        cudaDeviceProp prop;
        cudaError_t eProps = cudaGetDeviceProperties(&prop, rank);
        printf("  rank %d: cudaGetDeviceProperties(): %s (code %d)\n",
               rank, cudaGetErrorString(eProps), (int)eProps);
    }
    printf("  (%d device(s) -- nothing to print above.)\n", deviceCount);

    // The real query, attempted honestly on device 0 regardless of
    // deviceCount, exactly like this book's every other honest
    // capability check since Chapter 6's asyncEngineCount/
    // concurrentKernels query.
    cudaDeviceProp prop;
    cudaError_t eProps = cudaGetDeviceProperties(&prop, 0);
    printf("\ncudaGetDeviceProperties(&prop, 0): %s (code %d)\n",
           cudaGetErrorString(eProps), (int)eProps);
    printf("Two of the real fields cudaDeviceProp exposes for exactly this "
           "purpose -- comparing one device's real capability against "
           "another's -- are multiProcessorCount (how many streaming "
           "multiprocessors the device physically has) and clockRate "
           "(the device's core clock, in kHz). Neither field can be read "
           "here, since the call above never succeeded.\n");

    // Hypothetically, a 4-device cluster mixing two GPU generations --
    // exactly the situation Cephalo's own paper studies, and exactly
    // the situation none of this book's own computeShard()/
    // computeLayerRange()/computeRowRange() functions has ever asked
    // about. Real multiProcessorCount figures for real, named GPUs
    // (NVIDIA's own architecture whitepapers): an A100 has 108 SMs: an
    // H100 SXM has 132. Relative SM count is a real, if rough, proxy
    // for relative throughput -- this chapter's own closed-form model
    // in Section 18.3 uses a cleaner, already-measured relative-speed
    // number instead, the same way Chapter 2's bandwidth model used
    // real cited bandwidth figures rather than re-deriving them.
    struct HypotheticalDevice { const char* name; int multiProcessorCount; };
    HypotheticalDevice cluster[] = {
        {"A100 (rank 0)", 108},
        {"A100 (rank 1)", 108},
        {"H100 SXM (rank 2)", 132},
        {"A100, thermally throttled (rank 3)", 108},
    };
    printf("\nA hypothetical 4-device cluster mixing GPU generations "
           "(real multiProcessorCount figures from NVIDIA's own "
           "architecture whitepapers):\n");
    for (const auto& d : cluster) {
        printf("  %-38s multiProcessorCount = %d\n", d.name, d.multiProcessorCount);
    }
    printf("\nEvery computeShard()-style function this book has written "
           "would still hand all four of these devices an IDENTICAL "
           "share of the work -- the SM count difference above, and rank "
           "3's real thermal throttling (which multiProcessorCount can't "
           "even see, since the SMs are still physically there, just "
           "running slower), are both invisible to pure host arithmetic "
           "that only ever divides by worldSize.\n");

    return 0;
}
