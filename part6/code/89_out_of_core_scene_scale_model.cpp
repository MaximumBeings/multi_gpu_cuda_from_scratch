// Chapter 30: Multi-GPU Ray Tracing and Rendering
// 89_out_of_core_scene_scale_model.cpp
//
// Sections 30.1-30.2 showed rendering's clean case: zero communication
// during computation, a fixed one-time final-image cost, and a real but
// LOCAL complication (per-tile load imbalance, fixable by reassigning
// which rows go to which rank -- itself only possible because rows have
// no neighbor dependency, unlike Chapter 16/29's contiguous halo-exchange
// strips). This section presents the real complication that reintroduces
// genuine cross-device communication DURING rendering: a real, cited
// SIGGRAPH 2022 paper -- Fouladi, Shacklett, Poms, Arora, Ozdemir,
// Raghavan, Hanrahan, Fatahalian & Winstein, "R2E2: Low-Latency Path
// Tracing of Terabyte-Scale Scenes using Thousands of Cloud CPUs," ACM
// Transactions on Graphics 41(4), Article 76, 2022 -- describes real
// production scenes "with up to a terabyte of geometry and texture data
// (where as little as 1/250th of the scene can fit on any one node)."
// Once a SINGLE frame's own scene data cannot fit on one device, this
// chapter's own Section 30.1 no-communication assumption breaks: a ray
// traveling through the scene can need geometry another node owns,
// forcing genuine cross-node data movement DURING rendering -- rejoining
// the rest of this book's own distributed-systems concerns (echoing
// Chapter 1's own original memory-wall motivation for this entire book).
#include <cstdio>

int main() {
    printf("--- R2E2's own real scale problem ---\n");
    printf("Real cited fact: \"scenes with up to a terabyte of geometry and "
           "texture data (where as little as 1/250th of the scene can fit "
           "on any one node)\" -- Fouladi et al., SIGGRAPH 2022.\n\n");

    printf("%-24s %-20s %-24s\n", "total scene size", "per-node capacity",
           "nodes needed (this chapter's own model)");
    // Real cited fraction: 1/250th per node. Applied here to a few
    // illustrative total-scene sizes (the 1 TB figure itself is real and
    // cited; the smaller sizes are this chapter's own illustrative scaling
    // points, explicitly NOT claimed as R2E2's own measured configurations).
    struct Scene { const char *label; double totalGB; };
    Scene scenes[] = {
        {"100 GB (illustrative)", 100.0},
        {"500 GB (illustrative)", 500.0},
        {"1024 GB = 1 TB (R2E2 cited)", 1024.0}
    };
    const double perNodeFraction = 1.0 / 250.0; // R2E2's own real cited fraction
    for (auto &s : scenes) {
        double perNodeGB = s.totalGB * perNodeFraction;
        int nodesNeeded = 250; // by construction: 1/250th per node -> 250 nodes
        printf("%-24s %-20.2f %-24d\n", s.label, perNodeGB, nodesNeeded);
    }
    printf("\nAt R2E2's own real terabyte scale, roughly 250 nodes are needed "
           "just to HOLD the scene data -- this is now a genuine data "
           "DECOMPOSITION problem, structurally the same kind of problem "
           "Chapter 24 solved for a matrix too large for one device's "
           "memory, not the zero-communication case Section 30.1 built.\n\n");

    printf("--- Why this reintroduces communication DURING rendering ---\n");
    printf("Section 30.1's own simulation worked because every pixel's ray "
           "test needed ONLY the fixed scene description and that pixel's "
           "own coordinates -- and the whole scene fit in that simulation's "
           "own memory. Once the scene itself is partitioned across nodes "
           "(because no single node can hold it), a ray that would need to "
           "test against geometry owned by ANOTHER node must either fetch "
           "that geometry or forward the ray's own state to the node that "
           "owns it -- a real per-ray communication requirement Section "
           "30.1's clean model never had to pay.\n\n");

    printf("--- Real payoff, cited ---\n");
    printf("R2E2's own real reported results against a single-machine "
           "in-memory baseline (pbrt-treelet): \"R2E2 is 3.5-7.8x faster "
           "than pbrt-treelet's single-machine in-memory path tracer,\" "
           "with scene loading itself \"4.9x (Moana-XL) and 2.9x (Terrace) "
           "faster.\" The real headline finding: \"R2E2 completes the "
           "entire path tracing job before pbrt-treelet begins tracing a "
           "single ray\" for most of their tested configurations -- because "
           "the single-machine baseline cannot even START until the whole "
           "scene has been paged in from disk, while R2E2's own distributed "
           "approach never needs the whole scene on any one node at all.\n");

    return 0;
}
