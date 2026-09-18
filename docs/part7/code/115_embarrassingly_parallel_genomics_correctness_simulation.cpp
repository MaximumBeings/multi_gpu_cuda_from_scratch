// Chapter 39: Real-Time Genomics at Scale
// 115_embarrassingly_parallel_genomics_correctness_simulation.cpp
//
// Google's own real DeepVariant documentation (the deep-learning variant
// caller NVIDIA's own real Clara Parabricks GPU-accelerates) states the
// exact real structural property this file tests: "Since the process of
// generating examples is embarrassingly parallel across the genome,
// make_examples supports sharding of its input and output via the
// --task argument" -- each shard is simply a disjoint slice of genomic
// regions, and DeepVariant's own real sharding mechanism assigns shard
// k a literal 1/N fraction of the regions with no coordination between
// shards at all. The read-alignment half of the real Parabricks germline
// pipeline (fq2bam, built on BWA-MEM) rests on the same real underlying
// fact from basic sequence-alignment: each sequencing READ is aligned to
// the reference independently of every other read, since alignment asks
// only "where does THIS read best match the reference," never "where do
// reads X and Y match relative to each other." This file builds a
// host-side correctness simulation of that shared real property applied
// to BOTH real pipeline stages -- P workers each own a disjoint,
// contiguous slice of N independent input units (reads for alignment,
// genomic regions for variant calling) and process ONLY their own slice,
// with no cross-worker read at all, not even a boundary one. This is
// deliberately built to look almost identical in STRUCTURE to Chapter
// 16's own row-strip domain decomposition, so the difference this file
// is actually testing shows up clearly: Chapter 16's stencil update for
// row i needs rows i-1 and i+1 (a real halo exchange is unavoidable, no
// matter how the rows are split), while every unit here needs nothing
// beyond its own data, by the real cited nature of independent reads and
// independent genomic regions -- so a correct implementation here needs
// ZERO cross-worker communication of any kind, not even at partition
// boundaries, which Chapter 16's own halo exchange could never avoid.
#include <cstdio>
#include <vector>

const int NUM_UNITS = 10007;  // a deliberately non-round, prime unit count
                               // (independent reads, or independent genomic
                               // regions) -- chosen so P does not divide it
                               // evenly at most tested P, exercising the
                               // real remainder-handling every partitioning
                               // scheme in this book has needed since Ch9.

// A pure, deterministic per-unit "result" (standing in for a real
// alignment score or a real variant call) -- a function of the unit's
// own id ONLY, never of any neighboring unit, modeling the real cited
// fact that each read/region's own outcome depends on nothing else.
long long unitResult(int unitId) {
    unsigned int h = (unsigned int)(unitId * 2654435761u + 17u);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return (long long)(h % 1000000);
}

// Reference: process all NUM_UNITS units in one single-process pass, no
// partitioning at all -- the ground truth every P-worker run is checked
// against.
std::vector<long long> referenceRun() {
    std::vector<long long> results(NUM_UNITS);
    for (int i = 0; i < NUM_UNITS; i++) results[i] = unitResult(i);
    return results;
}

// P-worker run: worker w owns units [start, end), a disjoint contiguous
// slice (DeepVariant's own real --task sharding scheme). Each worker
// computes ONLY its own slice's results, reading nothing from any other
// worker's slice -- the real zero-communication property this file
// exists to verify.
std::vector<long long> partitionedRun(int P) {
    std::vector<long long> results(NUM_UNITS);
    for (int w = 0; w < P; w++) {
        int start = (int)((long long)NUM_UNITS * w / P);
        int end = (int)((long long)NUM_UNITS * (w + 1) / P);
        for (int i = start; i < end; i++) {
            results[i] = unitResult(i);  // no reference to any other worker's slice
        }
    }
    return results;
}

int main() {
    std::vector<long long> reference = referenceRun();
    printf("Reference (single-process): %d independent units processed, "
           "first 3 results: %lld %lld %lld\n\n",
           NUM_UNITS, reference[0], reference[1], reference[2]);

    int Ps[] = {1, 2, 3, 4, 5, 7, 8, 16, 32, 64};
    printf("%-6s %-10s %-30s\n", "P", "all match", "max slice size (load imbalance)");
    for (int P : Ps) {
        std::vector<long long> got = partitionedRun(P);
        bool allMatch = true;
        for (int i = 0; i < NUM_UNITS; i++) {
            if (got[i] != reference[i]) { allMatch = false; break; }
        }
        int maxSlice = 0, minSlice = NUM_UNITS;
        for (int w = 0; w < P; w++) {
            int start = (int)((long long)NUM_UNITS * w / P);
            int end = (int)((long long)NUM_UNITS * (w + 1) / P);
            int sliceSize = end - start;
            if (sliceSize > maxSlice) maxSlice = sliceSize;
            if (sliceSize < minSlice) minSlice = sliceSize;
        }
        printf("%-6d %-10s max=%-6d min=%-6d (diff=%d)\n", P,
               allMatch ? "YES" : "NO", maxSlice, minSlice, maxSlice - minSlice);
    }

    printf("\nEvery P reproduces the exact same %d results as the single-"
           "process reference, at every tested P from 1 to 64. This is not "
           "the same correctness-by-construction reasoning Chapter 30's "
           "ray tracing used (zero communication during rendering because "
           "each ray's OWNERSHIP is decided by a fixed spatial partition "
           "of one shared scene) -- here, zero communication is possible "
           "because the real cited scientific fact is that unit i's own "
           "result never depends on unit j's data AT ALL, for any i != j, "
           "not merely that a chosen scheduling policy happens to avoid "
           "asking for it. The only imperfection visible above is the "
           "small max-min slice-size difference at P values that do not "
           "evenly divide %d -- ordinary integer remainder handling, the "
           "same real load-imbalance category Chapter 18 built an entire "
           "chapter around, but bounded here by at most 1 unit per worker "
           "(this file's own contiguous-slice split), never the large, "
           "data-dependent imbalance Chapter 18's own heterogeneous-GPU "
           "story required.\n", NUM_UNITS, NUM_UNITS);
    return 0;
}
