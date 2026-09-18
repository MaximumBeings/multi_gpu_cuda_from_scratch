// Chapter 27: N-Body Simulation at Scale
// 79_nbody_vs_halo_communication_model.cpp
//
// Section 27.1 proved a distributed all-pairs N-body force calculation
// needs an ALL-GATHER, not a halo exchange, to get the right answer.
// This section quantifies exactly how much worse that makes N-body's
// own communication SHAPE, reusing Chapter 9/10's own real ring all-
// gather formula: each of P ranks ends a ring all-gather having
// received (P-1)/P of the total K bytes being gathered. Chapter 16's
// own halo-exchange volume, by contrast, depended only on a fixed
// BOUNDARY size between adjacent shards -- never on the total domain
// size -- which is exactly why adding more shards made Chapter 16's
// own per-shard communication SHRINK. This section's own worked
// calculation, applied to a real body count from a real cited source
// (Burtscher & Pingali's own "5,000,000 bodies" CUDA Barnes-Hut
// benchmark), shows the opposite: N-body's own per-GPU all-gather
// volume stays close to the ENTIRE dataset's size, almost regardless
// of how many GPUs are added, because every GPU still needs every
// body's data every step.
#include <cstdio>
#include <cstdint>

int main() {
    // Burtscher & Pingali's own real cited body count, reused here as
    // a real-scale example (not re-derived by this program).
    const uint64_t N_BODIES = 5000000ULL;
    // This section's own stated assumption -- 3D position (3 doubles)
    // + mass (1 double) = 32 bytes/body -- a common representation,
    // not itself a number quoted from a paper (Chapter 26's own
    // fp16-assumption practice, reused here).
    const double BYTES_PER_BODY = 32.0;
    double totalBytes = (double)N_BODIES * BYTES_PER_BODY;
    double totalMiB = totalBytes / (1024.0 * 1024.0);

    printf("N=%llu bodies (Burtscher & Pingali's own real cited scale), "
           "%.0f bytes/body (this chapter's own stated assumption) -> "
           "total dataset = %.2f MiB\n\n",
           (unsigned long long)N_BODIES, BYTES_PER_BODY, totalMiB);

    printf("%-6s %-28s %-28s\n", "P", "N-body all-gather MiB/GPU",
           "Ch16 halo-exchange MiB/GPU");
    printf("--------------------------------------------------------------"
           "----\n");

    // Chapter 16's own real halo-exchange model used a FIXED boundary
    // size, independent of total domain size -- represented here by a
    // fixed per-shard boundary of BOUNDARY_MIB, unaffected by P,
    // reused only for shape contrast (not Ch16's own literal numbers).
    const double BOUNDARY_MIB = 0.5; // a fixed boundary exchange size

    const int Ps[] = {2, 4, 8, 16, 32, 64};
    for (int P : Ps) {
        // Chapter 9/10's own real ring all-gather formula: each rank
        // receives (P-1)/P of the total K bytes being gathered.
        double perGpuAllGatherMiB = (double)(P - 1) / (double)P * totalMiB;
        printf("%-6d %-28.2f %-28.2f\n", P, perGpuAllGatherMiB, BOUNDARY_MIB);
    }

    printf("\nAs P grows, N-body's own per-GPU all-gather volume "
           "APPROACHES the FULL dataset size (%.2f MiB) and never drops "
           "below it by much -- every GPU still needs every body's data, "
           "every step. Chapter 16's own halo-exchange volume stayed "
           "FLAT at a fixed boundary size regardless of P, because a "
           "stencil update only ever reads immediately adjacent cells. "
           "Adding more GPUs make Chapter 16's own per-GPU communication "
           "SHRINK relative to the growing problem; it does nothing "
           "comparable for this chapter's own all-pairs N-body case.\n",
           totalMiB);

    return 0;
}
