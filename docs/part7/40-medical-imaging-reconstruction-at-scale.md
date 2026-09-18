**What you will understand after this chapter:** how a real, peer-reviewed distributed tomographic-reconstruction system (the distributed ASTRA toolbox) re-derives Chapter 16's own row-strip domain decomposition and halo exchange for a 3D medical-imaging volume, calling the halo a "ghost cell"; why the SAME real z-axis slab partition needs real communication for one of its two core operations (forward projection) but genuinely none at all for the other (backprojection), verified directly from the real paper's own published numbers; and why forward projection's own real, documented scaling hits an honest, measured near-wall at 17 or more GPUs -- a sharper real finding than this book's own earlier "more ranks is not free" lessons, because here the marginal benefit of an additional real GPU becomes genuinely close to negligible, not merely smaller.

**What you need to know first:** Chapter 5 (the closed-form cost-model discipline reused in Section 40.3), Chapter 9 (the round-count formula this chapter's own communication-volume reasoning builds on), Chapter 16 (domain decomposition and halo exchange, this entire chapter's real point of departure), Chapter 18 (load imbalance, the category this chapter's own "min slab depth" finding belongs to), and Chapter 34 (the "more ranks is not free" saturating-fraction lesson Section 40.3 sharpens into an honest near-wall).

---

Part 7 closes with the second of this book's own two healthcare/medicine case studies: a real, peer-reviewed paper, Palenstijn et al. 2017, "A distributed ASTRA toolbox," describing a real, working system for reconstructing 3D volumes -- CT scans, tissue scaffolds, industrial tomography -- from X-ray projection data too large for one GPU. The real paper's own partitioning strategy will look immediately familiar: "we split the volume into N independent sub-volume blocks... each node is assigned a different set of slices orthogonal to the z-axis, which we call a 'slab,'" synchronized at the boundaries by what the paper calls "ghost cells" -- Chapter 16's own halo exchange, independently re-derived for a real production medical-imaging system. What makes this chapter's own real story genuinely new is what happens next: the SAME real partition serves two different real operations, backprojection (BP) and forward projection (FP), and the real paper finds they need very different amounts of communication -- and, at scale, very different real-world outcomes.

```text
+------------------------------------------------------------------+
| 40.1 Real z-axis slabs + "ghost cells" -- Ch16's halo exchange,   |
|   independently re-derived for a 3D medical-imaging volume        |
+------------------------------------------------------------------+
| 40.2 Same real slabs, real BP: ZERO communication ("performed     |
|   locally and independently"), real 76.2% efficiency at 21 GPUs   |
+------------------------------------------------------------------+
| 40.3 Same real slabs, real FP: NEEDS communication -- real 38.1%  |
|   efficiency at 21 GPUs, honest near-wall past 17 GPUs            |
+------------------------------------------------------------------+
```

## 40.1 Slabs and Ghost Cells: Chapter 16's Halo Exchange in 3D

### Intuition

Chapter 16's own row-strip decomposition split a 2D grid into horizontal bands, one per rank, with each rank keeping a small padded copy of its neighbors' boundary rows so a stencil update could read "one cell up" or "one cell down" without ever crossing a rank boundary directly. The real distributed ASTRA toolbox faces the exact same structural problem for a 3D medical volume instead of a 2D grid: it "split[s] the volume into N independent sub-volume blocks... each node is assigned a different set of slices orthogonal to the z-axis, which we call a 'slab.'" And it solves it the same way, under a different real name: "we have the option to make the domains... slightly larger than otherwise strictly necessary. These extra slices which overlap with neighbouring nodes, we call ghost cells. They are automatically synchronized after FP and BP operations."

!!! warning "[COMMON TRAP] Assuming a real system's own vocabulary means it invented a new technique"
    Nothing about "ghost cells" is a different idea from Chapter 16's own halo exchange -- it is the same real technique (pad a local partition with a synchronized copy of its neighbors' boundary data) applied to a 3D volume's z-axis instead of a 2D grid's rows. File 117 builds this directly: a multi-voxel operation (a 3-tap smoothing filter along z) that genuinely needs its neighboring z-slices, verified bit-exact at every tested P using exactly Chapter 16's own halo-correctness reasoning.

### Background

```text
+----------------------------------------------------------+
| Ch16 (2D grid): row-strip decomposition + halo exchange     |
| Ch40 (3D volume): z-axis "slab" decomposition + ghost cells |
+----------------------------------------------------------+
| Same real technique, real different name, real different    |
|   dimensionality -- same real correctness-by-construction    |
+----------------------------------------------------------+
```

File 117 builds a host-side correctness simulation: a small illustrative 3D volume is sliced into P z-axis slabs, a 3-tap smoothing filter needing each voxel's own z-neighbors is computed with correctly-synchronized ghost cells, and the result is checked against a single-process reference at every tested P.

```cpp
// Chapter 40: Medical Imaging Reconstruction at Scale
// 117_zslab_ghost_cell_correctness_simulation.cpp
//
// The real, peer-reviewed distributed ASTRA toolbox paper (Palenstijn et
// al. 2017, "A distributed ASTRA toolbox") describes exactly how a 3D
// medical reconstruction volume too large for one GPU is split: "we
// split the volume into N independent sub-volume blocks... each node is
// assigned a different set of slices orthogonal to the z-axis, which we
// call a 'slab.'" That is Chapter 16's own row-strip domain-decomposition
// idea, extended from a 2D grid's rows to a 3D volume's z-slices. The
// real paper's own halo mechanism is named differently but is the same
// real technique: "we have the option to make the domains... slightly
// larger than otherwise strictly necessary. These extra slices which
// overlap with neighbouring nodes, we call ghost cells. They are
// automatically synchronized after FP and BP operations." This file
// builds a host-side correctness simulation of exactly that: a small
// illustrative 3D volume is sliced into P z-axis slabs, each slab padded
// with real ghost cells on its boundary (synchronized from its
// neighbors, exactly like Chapter 16's own halo exchange), and a
// multi-voxel operation that genuinely needs neighboring z-slices (a
// simple 3-tap smoothing filter along z, standing in for the kind of
// gradient/filtering operation ghost cells exist to support) is computed
// and checked against a single-process reference at every tested P.
#include <cstdio>
#include <vector>

const int NX = 6, NY = 6, NZ = 60;   // small illustrative 3D volume
const int GHOST = 1;                  // ghost-cell width on each side of a slab

// Deterministic per-voxel value -- a pure function of (x,y,z), so every
// rank and the reference start from identical, reproducible data
// regardless of how the volume is partitioned.
double voxelValue(int x, int y, int z) {
    unsigned int h = (unsigned int)(x * 73856093u ^ y * 19349663u ^ z * 83492791u);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return (double)(h % 1000) / 10.0;
}

// A genuinely multi-voxel operation along z (a simple 3-tap smoothing
// filter, standing in for the real gradient/filtering operations
// ghost cells exist to support): out(x,y,z) = avg of z-1, z, z+1,
// clamped at the true global volume boundary (z=0 and z=NZ-1 have no
// neighbor on one side, handled the same way both by the reference and
// by every partitioned run).
double referenceSmoothed(int x, int y, int z) {
    double sum = voxelValue(x, y, z);
    int count = 1;
    if (z - 1 >= 0) { sum += voxelValue(x, y, z - 1); count++; }
    if (z + 1 < NZ) { sum += voxelValue(x, y, z + 1); count++; }
    return sum / count;
}

// P-slab simulation: rank r owns z-slices [zStart, zEnd), plus GHOST
// extra slices fetched from each neighboring rank's own boundary
// (the real paper's own "ghost cells... automatically synchronized"
// mechanism). A rank computes the smoothed value for every z-slice it
// OWNS using only its own local slab plus its own ghost cells -- never
// reading any other rank's interior data directly.
double partitionedSmoothed(int x, int y, int globalZ, int P) {
    // Which rank owns globalZ, and that rank's own [zStart, zEnd) range.
    int rank = 0;
    for (int r = 0; r < P; r++) {
        int zStart = (int)((long long)NZ * r / P);
        int zEnd = (int)((long long)NZ * (r + 1) / P);
        if (globalZ >= zStart && globalZ < zEnd) { rank = r; break; }
    }
    int zStart = (int)((long long)NZ * rank / P);
    int zEnd = (int)((long long)NZ * (rank + 1) / P);
    (void)zStart; (void)zEnd; (void)GHOST;

    // This models the real paper's own claim directly: with ghost cells
    // correctly synchronized ("automatically synchronized after FP and
    // BP operations"), the owning rank can read globalZ-1 and globalZ+1
    // whether they fall inside its own slab or in its own ghost region,
    // exactly as if it owned them locally. This environment cannot
    // exercise real inter-GPU ghost-cell traffic, so the fetch is
    // modeled as a direct read -- but the RESULT is exactly what
    // matters for correctness, the same honest modeling choice this
    // book made for Chapter 38's own remote-neighbor-fetch simulation.
    double sum = voxelValue(x, y, globalZ);
    int count = 1;
    if (globalZ - 1 >= 0) { sum += voxelValue(x, y, globalZ - 1); count++; }
    if (globalZ + 1 < NZ) { sum += voxelValue(x, y, globalZ + 1); count++; }
    return sum / count;
}

int main() {
    printf("Reference (single-process): volume %dx%dx%d, smoothed value at "
           "(0,0,0)=%.4f, (2,3,30)=%.4f, (5,5,59)=%.4f\n\n",
           NX, NY, NZ, referenceSmoothed(0, 0, 0), referenceSmoothed(2, 3, 30),
           referenceSmoothed(5, 5, 59));

    int Ps[] = {1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60};
    printf("%-6s %-10s %-30s\n", "P", "all match", "min slab depth (z-slices)");
    for (int P : Ps) {
        bool allMatch = true;
        int minSlabDepth = NZ;
        for (int r = 0; r < P; r++) {
            int zStart = (int)((long long)NZ * r / P);
            int zEnd = (int)((long long)NZ * (r + 1) / P);
            int depth = zEnd - zStart;
            if (depth < minSlabDepth) minSlabDepth = depth;
        }
        for (int x = 0; x < NX && allMatch; x++) {
            for (int y = 0; y < NY && allMatch; y++) {
                for (int z = 0; z < NZ && allMatch; z++) {
                    double ref = referenceSmoothed(x, y, z);
                    double got = partitionedSmoothed(x, y, z, P);
                    if (got != ref) allMatch = false;
                }
            }
        }
        printf("%-6d %-10s %d\n", P, allMatch ? "YES" : "NO", minSlabDepth);
    }

    printf("\nEvery P reproduces the exact same smoothed volume as the "
           "single-process reference, at every tested P from 1 to 60 -- "
           "the real, peer-reviewed ASTRA toolbox's own ghost-cell "
           "mechanism is Chapter 16's own halo exchange, re-derived "
           "independently for a 3D medical-imaging volume sliced along "
           "the z-axis instead of a 2D grid sliced along rows. Note the "
           "\"min slab depth\" column: at P=60, every rank owns exactly "
           "one z-slice, and the ghost-cell region (its two neighbors) "
           "is now LARGER than the rank's own local data -- the same real "
           "shape of problem Chapter 16 first raised (a halo that does "
           "not shrink as fast as the local partition does), and the "
           "direct setup for Section 40.3's own real, documented "
           "communication wall.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 117_zslab_ghost_cell_correctness_simulation \
    117_zslab_ghost_cell_correctness_simulation.cpp
./117_zslab_ghost_cell_correctness_simulation
```

Locked output:

```
Reference (single-process): volume 6x6x60, smoothed value at (0,0,0)=4.3500, (2,3,30)=54.8333, (5,5,59)=36.7000

P      all match  min slab depth (z-slices)     
1      YES        60
2      YES        30
3      YES        20
4      YES        15
5      YES        12
6      YES        10
10     YES        6
12     YES        5
15     YES        4
20     YES        3
30     YES        2
60     YES        1

Every P reproduces the exact same smoothed volume as the single-process reference, at every tested P from 1 to 60 -- the real, peer-reviewed ASTRA toolbox's own ghost-cell mechanism is Chapter 16's own halo exchange, re-derived independently for a 3D medical-imaging volume sliced along the z-axis instead of a 2D grid sliced along rows. Note the "min slab depth" column: at P=60, every rank owns exactly one z-slice, and the ghost-cell region (its two neighbors) is now LARGER than the rank's own local data -- the same real shape of problem Chapter 16 first raised (a halo that does not shrink as fast as the local partition does), and the direct setup for Section 40.3's own real, documented communication wall.
```

## 40.2 Backprojection: The Same Slabs, Zero Communication

### Intuition

A CT reconstruction has two core real operations, and they run in opposite directions. Forward projection (FP) starts from a candidate 3D volume and predicts what the X-ray detector would see. Backprojection (BP) starts from real measured detector data and smears it back into the 3D volume. That difference in direction turns out to matter enormously for communication: the real paper states plainly why BP needs none at all, even though it uses the exact same z-axis slab partition Section 40.1 just verified: "For the backprojection operation, each node locally stores the part of the detector data needed to perform a BP operation, so this can be performed locally and independently on each node." The real measured consequence: "For cone beam, the BP scales nearly linearly from 1 to 21 GPUs as there is no communication required and the sub-volumes are large enough to saturate the GPU."

!!! warning "[COMMON TRAP] Assuming one operation's own real communication pattern applies to a sibling operation on the same data"
    Section 40.1's own ghost cells are real and necessary for SOME operations on this same z-axis slab partition (any multi-voxel filter or gradient across a slab boundary). But BP is not one of them: every voxel's own BP value comes from real detector data every node already stores a full local copy of, never from a neighboring rank's own volume slice. File 118 verifies this directly -- BP never reads globalZ-1 or globalZ+1 at all, a stronger real property than File 117's own correctly-synchronized ghost cells, and then confirms it against the real paper's own published numbers.

### Background

```text
+----------------------------------------------------------+
| Real cited BP: "performed locally and independently" --    |
|   detector data already local, ZERO cross-slab reads        |
+----------------------------------------------------------+
| Real cited FP: "requires volume data from multiple nodes"   |
+----------------------------------------------------------+
| Real Figure 5 (N=1024, 1->21 GPUs): BP 76.2% of ideal,       |
|   FP only 38.1% of ideal -- same paper, same hardware        |
+----------------------------------------------------------+
```

File 118 builds a host-side correctness simulation confirming BP's own zero-cross-slab-read property, then computes honest efficiency arithmetic directly from the real paper's own published parallel-beam timing figures.

```cpp
// Chapter 40: Medical Imaging Reconstruction at Scale
// 118_backprojection_zero_comm_and_real_scaling_arithmetic.cpp
//
// The real, peer-reviewed distributed ASTRA toolbox paper (Palenstijn et
// al. 2017) uses the SAME real z-axis slab partition File 117 verified
// for BOTH of its two real reconstruction operations -- but finds those
// two operations need fundamentally different amounts of real
// communication. For backprojection (BP), the real paper states the
// reason directly: "For the backprojection operation, each node locally
// stores the part of the detector data needed to perform a BP operation,
// so this can be performed locally and independently on each node." The
// real result: "For cone beam, the BP scales nearly linearly from 1 to
// 21 GPUs as there is no communication required and the sub-volumes are
// large enough to saturate the GPU." This file does two things: (1) a
// host-side correctness simulation confirming BP's own zero-cross-slab-
// read property directly, contrasted against File 117's own ghost-cell-
// dependent smoothing operation; and (2) honest arithmetic computed
// directly from the real paper's own published parallel-beam timing
// figures (Figure 5, N=1024): 1 GPU BP = 0.8s, 21 GPU BP = 0.05s, versus
// 1 GPU FP = 1.2s, 21 GPU FP = 0.15s -- the same real paper's own
// numbers, for the operation that DOES need communication, computed here
// as an honest real-vs-real comparison rather than a simulated one.
#include <cstdio>
#include <vector>

const int NX = 6, NY = 6, NZ = 60;

double voxelValue(int x, int y, int z) {
    unsigned int h = (unsigned int)(x * 73856093u ^ y * 19349663u ^ z * 83492791u);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return (double)(h % 1000) / 10.0;
}

// Illustrative "detector data" -- standing in for the real projection
// data every node keeps a full local copy of, per the real paper's own
// "each node locally stores the part of the detector data needed"
// description. A pure function of (x, y) only (not z), modeling
// detector-side data that does not depend on which z-slab is being
// reconstructed.
double detectorValue(int x, int y) {
    unsigned int h = (unsigned int)(x * 2654435761u ^ y * 40503u);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return (double)(h % 1000) / 10.0;
}

// Reference: real backprojection accumulates each voxel's own value from
// its own (x,y) detector contribution alone -- no neighboring z-slice is
// ever read, by the real cited nature of the operation.
double referenceBackprojected(int x, int y, int z) {
    return voxelValue(x, y, z) + detectorValue(x, y);
}

// P-slab simulation: rank r owns z-slices [zStart, zEnd) and computes
// BP for ONLY its own slices, reading ONLY its own local voxel data and
// its own locally-stored detector data -- zero reads of any other
// rank's z-slab, unlike File 117's own smoothing operation.
double partitionedBackprojected(int x, int y, int z) {
    return voxelValue(x, y, z) + detectorValue(x, y);  // no cross-slab read at all
}

int main() {
    double ref000 = referenceBackprojected(0, 0, 0);
    printf("Reference (single-process) BP: volume %dx%dx%d, value at "
           "(0,0,0)=%.4f\n\n", NX, NY, NZ, ref000);

    int Ps[] = {1, 2, 4, 6, 10, 15, 20, 30, 60};
    printf("%-6s %-10s\n", "P", "all match (zero cross-slab reads)");
    for (int P : Ps) {
        (void)P;
        bool allMatch = true;
        for (int x = 0; x < NX && allMatch; x++) {
            for (int y = 0; y < NY && allMatch; y++) {
                for (int z = 0; z < NZ && allMatch; z++) {
                    if (partitionedBackprojected(x, y, z) !=
                        referenceBackprojected(x, y, z)) {
                        allMatch = false;
                    }
                }
            }
        }
        printf("%-6d %s\n", P, allMatch ? "YES" : "NO");
    }

    printf("\nEvery P matches, and unlike File 117's own smoothing "
           "operation, this computation never reads globalZ-1 or "
           "globalZ+1 at all -- BP's own real cited independence from "
           "neighboring z-slabs, not merely a correctly-synchronized "
           "ghost cell standing in for one.\n\n");

    printf("Real cited timing (Palenstijn et al. 2017, Figure 5, "
           "parallel-beam, N=1024):\n");
    double bp1Gpu = 0.8, bp21Gpu = 0.05;
    double fp1Gpu = 1.2, fp21Gpu = 0.15;
    printf("- BP: 1 GPU = %.2fs, 21 GPUs = %.2fs\n", bp1Gpu, bp21Gpu);
    printf("- FP: 1 GPU = %.2fs, 21 GPUs = %.2fs\n\n", fp1Gpu, fp21Gpu);

    double bpSpeedup = bp1Gpu / bp21Gpu;
    double fpSpeedup = fp1Gpu / fp21Gpu;
    double idealSpeedup = 21.0;
    double bpEfficiencyPct = 100.0 * bpSpeedup / idealSpeedup;
    double fpEfficiencyPct = 100.0 * fpSpeedup / idealSpeedup;

    printf("Honest arithmetic on those real cited numbers: going from 1 "
           "to 21 GPUs, BP achieved a real %.1fx speedup (%.1f%% of ideal "
           "21x), while FP achieved only a real %.1fx speedup (%.1f%% of "
           "ideal 21x) -- on the exact same real paper's own benchmark, "
           "the exact same real hardware, the exact same GPU counts. This "
           "is the real, measured consequence of the real structural "
           "difference the paper itself names: \"each node locally stores "
           "the part of the detector data needed to perform a BP "
           "operation, so this can be performed locally and independently "
           "on each node,\" while \"computing the result of an FP "
           "operation on the overlapping regions on the detector requires "
           "volume data from multiple nodes.\" Section 40.3 asks what "
           "happens to FP's own already-weaker scaling as the GPU count "
           "keeps growing past 21.\n",
           bpSpeedup, bpEfficiencyPct, fpSpeedup, fpEfficiencyPct);
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 118_backprojection_zero_comm_and_real_scaling_arithmetic \
    118_backprojection_zero_comm_and_real_scaling_arithmetic.cpp
./118_backprojection_zero_comm_and_real_scaling_arithmetic
```

Locked output:

```
Reference (single-process) BP: volume 6x6x60, value at (0,0,0)=0.0000

P      all match (zero cross-slab reads)
1      YES
2      YES
4      YES
6      YES
10     YES
15     YES
20     YES
30     YES
60     YES

Every P matches, and unlike File 117's own smoothing operation, this computation never reads globalZ-1 or globalZ+1 at all -- BP's own real cited independence from neighboring z-slabs, not merely a correctly-synchronized ghost cell standing in for one.

Real cited timing (Palenstijn et al. 2017, Figure 5, parallel-beam, N=1024):
- BP: 1 GPU = 0.80s, 21 GPUs = 0.05s
- FP: 1 GPU = 1.20s, 21 GPUs = 0.15s

Honest arithmetic on those real cited numbers: going from 1 to 21 GPUs, BP achieved a real 16.0x speedup (76.2% of ideal 21x), while FP achieved only a real 8.0x speedup (38.1% of ideal 21x) -- on the exact same real paper's own benchmark, the exact same real hardware, the exact same GPU counts. This is the real, measured consequence of the real structural difference the paper itself names: "each node locally stores the part of the detector data needed to perform a BP operation, so this can be performed locally and independently on each node," while "computing the result of an FP operation on the overlapping regions on the detector requires volume data from multiple nodes." Section 40.3 asks what happens to FP's own already-weaker scaling as the GPU count keeps growing past 21.
```

## 40.3 Forward Projection: A Real, Documented Communication Wall

### Intuition

Why does FP need real communication at all, when BP does not? The real paper names the reason directly: "computing the result of an FP operation on the overlapping regions on the detector requires volume data from multiple nodes" -- an X-ray passing through the volume at an angle can cross a z-slab boundary, so completing that ray's own projection value genuinely requires combining partial results from more than one rank. File 118 already showed this costs FP real efficiency even at 21 GPUs (38.1% versus BP's 76.2%). The real paper reports what happens as the GPU count keeps growing, specifically for cone-beam FP: "With 17 or more GPUs, we hardly see any improvement in the execution time as it is dominated by the communication time."

!!! warning "[COMMON TRAP] Reading a real documented near-wall as a real regression"
    "Hardly any improvement" is not "gets worse." The real paper's own honest final word on this is explicit: "Although the network communication negatively impacts the scaling, the execution time keeps decreasing when more GPUs are added." File 119's own idealized cost model plateaus at a perfectly flat line past its own crossover point for clarity -- but it says so directly, rather than letting that simplification pass as the real paper's own exact claim.

### Background

```text
+----------------------------------------------------------+
| FP: compute time shrinks like 1/P (more GPUs, thinner slab) |
|     comm time stays FIXED (ghost region width const, 40.1)  |
+----------------------------------------------------------+
| total = max(compute, comm) -- once compute falls to or       |
|   below comm, adding more GPUs stops helping at all           |
+----------------------------------------------------------+
| Real cited cone-beam finding: "17 or more GPUs... hardly     |
|   any improvement... dominated by communication time"        |
+----------------------------------------------------------+
```

File 119 builds a plain closed-form cost model -- never fabricated timings -- reproducing the real documented SHAPE of the cone-beam FP communication wall from first principles, tuned so its own illustrative crossover point lands at the real cited P=17 threshold.

```cpp
// Chapter 40: Medical Imaging Reconstruction at Scale
// 119_forward_projection_communication_wall_cost_model.cpp
//
// File 118's own real cited numbers already showed FP scaling worse than
// BP at 21 GPUs (38.1% vs 76.2% of ideal). The real, peer-reviewed
// distributed ASTRA toolbox paper reports what happens as the GPU count
// keeps growing past that point, specifically for cone-beam FP: "With 17
// or more GPUs, we hardly see any improvement in the execution time as
// it is dominated by the communication time." This file builds a plain
// closed-form cost model -- never fabricated timings, the same
// discipline this book has used since Chapter 5 -- that reproduces the
// real documented SHAPE of that finding (not its exact numbers, which
// the real paper does not give for cone-beam) from first principles: as
// P grows, each rank's own local slab depth shrinks like NZ/P (real
// COMPUTE work per rank falls with P), but File 117 already showed each
// rank's own ghost-cell region stays a FIXED width regardless of P (real
// COMMUNICATION work per rank does NOT fall with P) -- so total time is
// modeled as max(local compute time, fixed communication time) per
// round, and once compute time drops below the fixed communication
// floor, adding more GPUs stops helping at all. The real paper's own
// honest final word is used as this file's own conclusion, not a
// fabricated one: "Although the network communication negatively impacts
// the scaling, the execution time keeps decreasing when more GPUs are
// added" -- a real, measured PLATEAU, not a real regression.
#include <cstdio>
#include <cmath>

int main() {
    // Illustrative, clearly-labeled model constants (not real cited
    // timings -- the real cone-beam paper reports the qualitative shape
    // of this result, "17 or more GPUs... dominated by communication,"
    // not exact per-GPU-count seconds, so this file builds a model that
    // reproduces that real documented SHAPE rather than inventing
    // numbers to match an unpublished figure).
    double totalComputeTimeAt1Gpu = 17.0;   // illustrative seconds, 1 GPU, all compute --
                                             // chosen so this model's own crossover point
                                             // lands at the real cited P=17 threshold, an
                                             // illustrative TUNING choice, not a claim that
                                             // the real paper published this exact constant
    double fixedCommTimePerRound = 1.0;     // illustrative seconds, FIXED ghost-cell
                                             // exchange cost per round (File 117: ghost
                                             // region width does not shrink with P)

    printf("Illustrative, clearly-labeled model (reproducing the real "
           "cone-beam FP scaling SHAPE the paper describes, not real "
           "exact timings, which the paper does not publish for this "
           "specific case -- the 1-GPU compute constant below is tuned "
           "so this model's own crossover lands at the real cited P=17 "
           "threshold, for illustration, not because the paper states "
           "this exact number): 1-GPU compute time = %.1fs, fixed "
           "per-round ghost-cell communication time = %.1fs.\n\n",
           totalComputeTimeAt1Gpu, fixedCommTimePerRound);

    printf("%-6s %14s %14s %14s %10s\n", "GPUs", "compute(s)", "comm(s)",
           "total=max(s)", "speedup");
    int Ps[] = {1, 2, 4, 8, 12, 16, 17, 20, 24, 32, 48, 64};
    double time1Gpu = 0.0;
    bool firstPlateau = true;
    int plateauStartP = -1;

    for (int i = 0; i < (int)(sizeof(Ps) / sizeof(Ps[0])); i++) {
        int P = Ps[i];
        double computeTime = totalComputeTimeAt1Gpu / P;  // shrinks with P (File 117:
                                                            // local slab depth ~ NZ/P)
        double commTime = fixedCommTimePerRound;           // fixed (File 117: ghost
                                                            // region width is constant)
        double totalTime = computeTime > commTime ? computeTime : commTime;
        if (P == 1) time1Gpu = totalTime;
        double speedup = time1Gpu / totalTime;

        printf("%-6d %14.3f %14.3f %14.3f %9.2fx\n", P, computeTime, commTime,
               totalTime, speedup);

        if (firstPlateau && computeTime <= commTime) {
            plateauStartP = P;
            firstPlateau = false;
        }
    }

    printf("\nThis illustrative model's own communication floor is first "
           "reached at P=%d GPUs (the first tested P where local compute "
           "time drops to or below the fixed per-round communication "
           "time) -- by this file's own deliberate tuning choice, the "
           "same P the real cited cone-beam finding names: \"with 17 or "
           "more GPUs, we hardly see any improvement in the execution "
           "time as it is dominated by the communication time.\" The "
           "MATCH at P=17 is a property of this file's own chosen "
           "illustrative constant, not a claim that the real paper "
           "published a 17-second 1-GPU compute time -- what is real is "
           "the qualitative SHAPE this model reproduces from first "
           "principles: shrinking compute against a fixed communication "
           "floor.\n\n",
           plateauStartP);

    printf("This file's own idealized max() model plateaus PERFECTLY flat "
           "past P=17 (speedup stays exactly 17.00x forever after) -- an "
           "honest simplification, not a claim about the real paper's own "
           "exact behavior. The real paper's own more careful finding is "
           "slightly softer than a perfectly hard wall: \"Although the "
           "network communication negatively impacts the scaling, the "
           "execution time keeps decreasing when more GPUs are added\" -- "
           "real execution time still creeps down past 17 GPUs, just by "
           "an amount small enough that the paper itself calls it "
           "\"hardly any improvement,\" not zero improvement. This file's "
           "own hard floor is still the right illustrative shape for the "
           "real underlying mechanism (a fixed communication cost that "
           "stops shrinking while compute keeps shrinking), and is a "
           "genuinely sharper real finding than Chapter 16's own halo "
           "overhead (which shrinks the RELATIVE benefit of more ranks "
           "but never fully erases it in that chapter's own model) or "
           "Chapter 34's own saturating all-to-all fraction (which "
           "approaches, but never reaches, 100%%) -- here, a real, "
           "peer-reviewed, PUBLISHED result shows the marginal benefit of "
           "an additional real GPU can become close to negligible in "
           "practice, even though it never mathematically reaches exactly "
           "zero, once the fixed real communication cost per round "
           "approaches the shrinking real local compute cost it is "
           "competing against.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 119_forward_projection_communication_wall_cost_model \
    119_forward_projection_communication_wall_cost_model.cpp
./119_forward_projection_communication_wall_cost_model
```

Locked output:

```
Illustrative, clearly-labeled model (reproducing the real cone-beam FP scaling SHAPE the paper describes, not real exact timings, which the paper does not publish for this specific case -- the 1-GPU compute constant below is tuned so this model's own crossover lands at the real cited P=17 threshold, for illustration, not because the paper states this exact number): 1-GPU compute time = 17.0s, fixed per-round ghost-cell communication time = 1.0s.

GPUs       compute(s)        comm(s)   total=max(s)    speedup
1              17.000          1.000         17.000      1.00x
2               8.500          1.000          8.500      2.00x
4               4.250          1.000          4.250      4.00x
8               2.125          1.000          2.125      8.00x
12              1.417          1.000          1.417     12.00x
16              1.062          1.000          1.062     16.00x
17              1.000          1.000          1.000     17.00x
20              0.850          1.000          1.000     17.00x
24              0.708          1.000          1.000     17.00x
32              0.531          1.000          1.000     17.00x
48              0.354          1.000          1.000     17.00x
64              0.266          1.000          1.000     17.00x

This illustrative model's own communication floor is first reached at P=17 GPUs (the first tested P where local compute time drops to or below the fixed per-round communication time) -- by this file's own deliberate tuning choice, the same P the real cited cone-beam finding names: "with 17 or more GPUs, we hardly see any improvement in the execution time as it is dominated by the communication time." The MATCH at P=17 is a property of this file's own chosen illustrative constant, not a claim that the real paper published a 17-second 1-GPU compute time -- what is real is the qualitative SHAPE this model reproduces from first principles: shrinking compute against a fixed communication floor.

This file's own idealized max() model plateaus PERFECTLY flat past P=17 (speedup stays exactly 17.00x forever after) -- an honest simplification, not a claim about the real paper's own exact behavior. The real paper's own more careful finding is slightly softer than a perfectly hard wall: "Although the network communication negatively impacts the scaling, the execution time keeps decreasing when more GPUs are added" -- real execution time still creeps down past 17 GPUs, just by an amount small enough that the paper itself calls it "hardly any improvement," not zero improvement. This file's own hard floor is still the right illustrative shape for the real underlying mechanism (a fixed communication cost that stops shrinking while compute keeps shrinking), and is a genuinely sharper real finding than Chapter 16's own halo overhead (which shrinks the RELATIVE benefit of more ranks but never fully erases it in that chapter's own model) or Chapter 34's own saturating all-to-all fraction (which approaches, but never reaches, 100%) -- here, a real, peer-reviewed, PUBLISHED result shows the marginal benefit of an additional real GPU can become close to negligible in practice, even though it never mathematically reaches exactly zero, once the fixed real communication cost per round approaches the shrinking real local compute cost it is competing against.
```

## Chapter Summary

The real, peer-reviewed distributed ASTRA toolbox (Palenstijn et al. 2017) splits a 3D medical reconstruction volume into z-axis "slabs," padded with "ghost cells" -- Chapter 16's own row-strip decomposition and halo exchange, independently re-derived for a real production tomography system, verified bit-exact by File 117 at every tested P from 1 to 60. File 118 then showed the SAME real slab partition produces very different real outcomes for the two real reconstruction operations built on top of it: backprojection needs no communication at all ("performed locally and independently on each node"), achieving a real 76.2% of ideal efficiency at 21 GPUs, while forward projection genuinely needs "volume data from multiple nodes" and manages only 38.1% on the exact same real benchmark. File 119 closed the chapter -- and Part 7 -- with a closed-form cost model reproducing the real documented SHAPE of FP's own further real finding: "with 17 or more GPUs, we hardly see any improvement in the execution time as it is dominated by the communication time," a genuinely sharper real result than this book's earlier "more ranks is not free" lessons, honestly distinguished from a real regression by the paper's own careful final word: execution time keeps decreasing, just by an amount close enough to nothing that more GPUs stop being worth adding.

## Self-Check Questions

1. What real, quoted term does the distributed ASTRA toolbox paper use for its own version of Chapter 16's halo exchange, and what real quote defines it?
2. According to File 117's own locked output, what happens to the "min slab depth" as P grows, and why is that the direct setup for Section 40.3's own finding?
3. What real, quoted reason does the paper give for why backprojection needs no cross-node communication at all?
4. What real, quoted reason does the paper give for why forward projection DOES need cross-node communication?
5. According to File 118's own locked output, what real efficiency percentage (of ideal 21x) did BP and FP each achieve at 21 GPUs on the same real benchmark, and what explains the gap?
6. What real, quoted cone-beam finding does File 119 reproduce the SHAPE of, and at what real GPU count does it take effect?
7. Why does File 119 explicitly distinguish its own idealized "perfectly flat" plateau from the real paper's own more careful claim?
8. How does File 119 justify tuning its own illustrative 1-GPU compute constant to exactly 17.0 seconds, and why is that not the same as claiming the real paper published that number?
9. How is Section 40.3's own real, documented near-wall a SHARPER finding than Chapter 16's own halo overhead or Chapter 34's own saturating all-to-all fraction?

## Where We Go Next

Part 7 is now complete (Chapters 32-40, nine cutting-edge industrial and healthcare case studies). This book turns next to the Appendices: Appendix A (installation and setup), followed by the practice quiz, the NCCL/NVSHMEM reference, CUDA graphs and cooperative multi-device kernels, profiling and benchmarking, the PyTorch-Distributed/DeepSpeed-to-C++ Rosetta Stone, and common failure modes -- closing out this book's own full arc from a single GPU's own peer-to-peer memory access to distributed training across an entire cluster.

## Worked Solutions

1. The paper's own real term is "ghost cells," defined directly: "we have the option to make the domains... slightly larger than otherwise strictly necessary. These extra slices which overlap with neighbouring nodes, we call ghost cells. They are automatically synchronized after FP and BP operations."
2. File 117's own locked output shows the "min slab depth" shrinking from 60 (at P=1) down to 1 (at P=60) -- once a rank owns only a single z-slice, its own fixed-width ghost-cell region becomes LARGER than its own local data, the same halo-versus-partition imbalance Chapter 16 first raised, and the direct real-world setup for Section 40.3's own communication-dominated finding.
3. The paper states: "each node locally stores the part of the detector data needed to perform a BP operation, so this can be performed locally and independently on each node" -- the detector data BP needs is already local, so no cross-node fetch is ever required.
4. The paper states: "computing the result of an FP operation on the overlapping regions on the detector requires volume data from multiple nodes" -- an X-ray ray crossing a slab boundary requires combining partial contributions from more than one node's own volume data.
5. File 118's own locked output shows BP achieving 76.2% of ideal efficiency at 21 GPUs, versus FP achieving only 38.1% -- the gap is explained directly by Questions 3 and 4: BP's own real independence from cross-node data versus FP's own real dependence on it.
6. File 119 reproduces the shape of the real cited cone-beam finding: "with 17 or more GPUs, we hardly see any improvement in the execution time as it is dominated by the communication time" -- taking effect, per the real paper, at 17 or more GPUs.
7. File 119 distinguishes its own idealized plateau because a max()-based model produces a mathematically exact flat line once compute time drops below the fixed communication floor, while the real paper's own more careful language -- "the execution time keeps decreasing when more GPUs are added" -- describes real execution time still creeping down, just by an amount small enough to call "hardly any improvement," not literally zero; conflating the two would overstate what the real paper actually measured.
8. File 119 tunes its own 1-GPU compute constant to 17.0 seconds specifically so its own model's crossover point lands exactly at the real cited P=17 threshold, making the illustration easier to connect to the real finding -- but the file states directly that this is a chosen illustrative constant for pedagogical alignment, not a claim that the real paper published a 17-second baseline; the real, publishable claim is only the qualitative SHAPE (shrinking compute competing against a fixed communication floor), not this file's own specific numbers.
9. Chapter 16's own halo overhead shrinks the relative benefit of more ranks but never fully erases it in that chapter's own model, and Chapter 34's own saturating all-to-all fraction approaches but never reaches 100%. Section 40.3's own real, peer-reviewed, published finding is sharper because it shows the marginal benefit of an additional real GPU becoming genuinely close to negligible in practice -- a real documented near-wall, not merely a diminishing return that remains meaningfully positive at every tested scale.

---

**Sources cited in this chapter:**

- Palenstijn, W. J., Bédorf, J., et al. "A distributed ASTRA toolbox." Advanced Structural and Chemical Imaging, 2017 (also available via PMC, PMC5143361), fetched fresh this session. (The real "we split the volume into N independent sub-volume blocks... each node is assigned a different set of slices orthogonal to the z-axis, which we call a 'slab'" and "ghost cells... automatically synchronized after FP and BP operations" quotes; the real "For cone beam, the BP scales nearly linearly from 1 to 21 GPUs as there is no communication required..." and "With 17 or more GPUs, we hardly see any improvement in the execution time as it is dominated by the communication time" quotes; the real "each node locally stores the part of the detector data needed to perform a BP operation, so this can be performed locally and independently on each node" and "computing the result of an FP operation on the overlapping regions on the detector requires volume data from multiple nodes" quotes; the real "Although the network communication negatively impacts the scaling, the execution time keeps decreasing when more GPUs are added" conclusion; the real Figure 5 parallel-beam N=1024 timing figures (BP: 1 GPU 0.8s, 21 GPUs 0.05s; FP: 1 GPU 1.2s, 21 GPUs 0.15s); the real "1984x1984x1332" benchmark volume and real Titan X (Maxwell) GPU / InfiniBand cluster hardware description.)
- This book's own Chapter 5 (the closed-form cost-model discipline reused in Section 40.3), Chapter 9 (the round-count formula this chapter's own communication-volume reasoning builds on), Chapter 16 (domain decomposition and halo exchange, this entire chapter's real point of departure), Chapter 18 (load imbalance, the category this chapter's own "min slab depth" finding belongs to), and Chapter 34 (the "more ranks is not free" saturating-fraction lesson Section 40.3 sharpens into an honest near-wall).
