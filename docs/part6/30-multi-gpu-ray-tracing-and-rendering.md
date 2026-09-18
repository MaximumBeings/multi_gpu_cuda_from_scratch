**What you will understand after reading this chapter:** why rendering an image across multiple GPUs is, in its clean case, a fundamentally different distributed-systems problem from every earlier case study in this book -- one that needs no communication at all during computation -- and exactly two real reasons that clean case breaks down in production: uneven per-tile cost, and scenes too large for one device's memory.

**What you need to know first:** Chapter 16's row-strip domain decomposition (as the contrasting case); Chapter 18's real weighted load-balancing technique; Chapter 24's real capacity/decomposition reasoning for data too large for one device.

---

Every case study since Chapter 24 needed some form of communication *during* computation: Chapter 24's SUMMA needed a round of broadcasts per grid step, Chapter 25's ring all-reduce needed a communication step per bucket, Chapter 26's tensor and pipeline parallelism needed collective rounds at fixed points in the forward pass, Chapter 27's N-body simulation needed a mandatory all-gather every step, Chapter 28's distributed BFS needed a frontier exchange whose partners changed every level, and Chapter 29's distributed Jacobi solver needed a halo exchange every iteration plus a periodic global reduction. Rendering an image, in its simplest and most common form, needs none of that. Real render farms exploit exactly this: Wikipedia's own summary of how they work states plainly that "frames and sometimes tiles can be calculated independently of the others, with the main communication between processors being the upload of the initial source material, such as models and textures, and the download of the finished images." This chapter builds that clean case first, verifies it, and then shows the two real ways production rendering systems still end up needing real communication after all.

```text
Every prior Part 6 chapter:              This chapter's clean case:

  rank 0 ---per-step comm--- rank 1        rank 0   (no comm at all)   rank 1
     |                          |             |                          |
  compute                    compute        compute                   compute
     |                          |             |                          |
  (repeat, WITH communication   |          (repeat, WITH NO communication |
   at every step)                                at any step)
                                               |                          |
                                               +---- final gather --------+
                                                    (ONE time, at the end)
```

## 30.1 The Clean Case: Zero Communication During Rendering

### Intuition

Picture an image as a grid of pixels, and imagine handing each GPU a horizontal strip of rows to render -- visually similar to Chapter 16's row-strip picture. The difference is what each pixel's computation actually needs: a stencil update in Chapter 16 needed its *neighbors'* values, so a rank near a strip boundary needed data from the *next* rank. A rendered pixel's color, in the simplest case, depends only on the fixed scene description (where the objects are) and that pixel's own position -- nothing about a neighboring pixel's color, and nothing any other rank is doing. This is the classic "sort-last" shape from parallel-graphics literature: Molnar, Cox, Ellsworth & Fuchs's own real taxonomy paper ("A Sorting Classification of Parallel Rendering," *IEEE Computer Graphics and Applications*, 1994) describes it as deferring any sorting/assembly "until the end of the rendering pipeline -- after primitives have been rasterized into pixels, samples, or pixel fragments." Nothing needs to be exchanged *before* that final assembly step.

### Background

This section's own simulation builds the simplest possible version of this idea: a fixed, deterministic two-object 2D "scene" (two circles standing in for spheres, with a nearest-hit-wins rule matching a real ray tracer's depth test), an integer-only per-pixel hit test with no floating point at all, split into row-tiles across P ranks. Each rank renders only its own rows, touching nothing but the fixed scene description and its own pixel coordinates -- it never reads another rank's tile, and there is no reduction step of any kind, so unlike every earlier chapter's collective, there is not even an associativity question to ask about the result.

```cpp
// Chapter 30: Multi-GPU Ray Tracing and Rendering
// 87_independent_tile_rendering_correctness_simulation.cpp
//
// Every case study since Chapter 24 needed SOME communication DURING
// computation: Chapter 16/29's halo exchange every step, Chapter 9/25's
// ring all-reduce, Chapter 26's tensor/pipeline-parallel rounds, Chapter
// 27's mandatory all-gather, Chapter 28's graph-dependent frontier
// exchange, Chapter 29's residual Allreduce. Rendering an image is
// structurally different in the simple case: Wikipedia's own summary of
// how real render farms work states it plainly -- "frames and sometimes
// tiles can be calculated independently of the others, with the main
// communication between processors being the upload of the initial
// source material, such as models and textures, and the download of the
// finished images." Molnar, Cox, Ellsworth & Fuchs's own real classic
// taxonomy ("A Sorting Classification of Parallel Rendering," IEEE
// Computer Graphics and Applications, 1994) calls this shape "sort-last":
// sorting (assembling results into the final image) happens only "after
// primitives have been rasterized into pixels, samples, or pixel
// fragments." This file builds the simplest possible version of that
// idea: a tiny fixed 2D "scene" (two circles standing in for spheres,
// nearest-hit-wins, matching real ray tracing's depth test), a
// deterministic integer-only per-pixel hit test with NO floating point
// and NO shared mutable state, split into row-tiles across P ranks. Each
// rank computes its OWN tile using ONLY the fixed scene description and
// its own pixel coordinates -- it never reads another rank's tile, and
// there is no reduction step at all (unlike every earlier chapter's
// collective), so there is not even an associativity question to ask.
// The assembled image is checked against a single-process reference,
// exactly, across several values of P.
#include <cstdio>
#include <vector>

const int WIDTH = 16;
const int HEIGHT = 12;

struct Circle { int cx, cy, r2; int color; }; // r2 = radius squared

// A fixed, deterministic 2-object "scene." Circle 0 is drawn in front of
// (i.e., wins ties/overlaps against) circle 1 when both are hit, exactly
// like a real ray tracer's nearest-hit rule -- here, "nearest" is modeled
// as "smaller squared distance to the object's own center," a simplified
// stand-in for real depth comparison.
std::vector<Circle> makeScene() {
    return {
        {5, 5, 9, 1},   // color 1: circle at (5,5), radius^2 = 9 (r=3)
        {11, 7, 16, 2}  // color 2: circle at (11,7), radius^2 = 16 (r=4)
    };
}

// Pure per-pixel function: given ONLY the fixed scene and this pixel's
// own (x,y), returns the pixel's color. No other pixel's data, and no
// other rank's data, is ever touched.
int shadePixel(const std::vector<Circle> &scene, int x, int y) {
    int bestColor = 0;      // 0 = background
    int bestDistSq = -1;    // -1 = "no hit yet"
    for (const Circle &c : scene) {
        int dx = x - c.cx, dy = y - c.cy;
        int distSq = dx * dx + dy * dy;
        if (distSq <= c.r2) {
            if (bestDistSq == -1 || distSq < bestDistSq) {
                bestDistSq = distSq;
                bestColor = c.color;
            }
        }
    }
    return bestColor;
}

// --- Reference: single-process render of the FULL image. ---
std::vector<int> referenceRender() {
    auto scene = makeScene();
    std::vector<int> image(WIDTH * HEIGHT);
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++)
            image[y * WIDTH + x] = shadePixel(scene, x, y);
    return image;
}

// --- Distributed: P ranks, each rendering its own row-tile, with ZERO
// communication during rendering -- no halo, no reduction, nothing. ---
std::vector<int> distributedRender(int P) {
    auto scene = makeScene();
    int rowsPerRank = HEIGHT / P;
    std::vector<int> assembled(WIDTH * HEIGHT, -999); // sentinel: "not yet written"

    for (int r = 0; r < P; r++) {
        int yStart = r * rowsPerRank;
        int yEnd = yStart + rowsPerRank;
        // This rank touches ONLY its own rows, using ONLY the fixed scene
        // and its own pixel coordinates -- no other rank's tile is read.
        for (int y = yStart; y < yEnd; y++)
            for (int x = 0; x < WIDTH; x++)
                assembled[y * WIDTH + x] = shadePixel(scene, x, y);
    }
    return assembled;
}

void printImage(const std::vector<int> &image) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int c = image[y * WIDTH + x];
            putchar(c == 0 ? '.' : (c == 1 ? '1' : '2'));
        }
        putchar('\n');
    }
}

int main() {
    std::vector<int> ref = referenceRender();

    printf("Reference (single-process) %dx%d render, 2-circle scene "
           "('.'=background, '1'/'2'=circle color, nearest hit wins):\n\n", WIDTH, HEIGHT);
    printImage(ref);
    printf("\n");

    int Ps[] = {1, 2, 3, 4, 6, 12};
    printf("%-6s %-22s %-30s\n", "P", "rows/rank", "assembled image EXACT match?");
    for (int P : Ps) {
        std::vector<int> dist = distributedRender(P);
        bool exact = (dist == ref);
        printf("%-6d %-22d %-30s\n", P, HEIGHT / P, exact ? "YES" : "NO");
    }

    return 0;
}
```

Compile and run:

```
g++ -O2 87_independent_tile_rendering_correctness_simulation.cpp -o 87_independent_tile_rendering_correctness_simulation
./87_independent_tile_rendering_correctness_simulation
```

```text
Reference (single-process) 16x12 render, 2-circle scene ('.'=background, '1'/'2'=circle color, nearest hit wins):

................
................
.....1..........
...11111...2....
...11111.22222..
..1111111222222.
...111112222222.
...1111122222222
.....1..2222222.
........2222222.
.........22222..
...........2....

P      rows/rank              assembled image EXACT match?  
1      12                     YES                           
2      6                      YES                           
3      4                      YES                           
4      3                      YES                           
6      2                      YES                           
12     1                      YES                           
```

The image matches bit-for-bit at every tested P, including P = 12 (one row per rank). Notice what this proves, and what it does *not* need to prove: unlike Chapter 16's stencil or Chapter 29's Jacobi update, there is no argument needed about operand order or reduction grouping, because there is no shared computation at all -- each pixel's color is a pure function of the fixed scene and that pixel's own coordinates. This also means the row-tiles do not need to be contiguous strips the way Chapter 16 and Chapter 29's halo exchange required: because no rank ever needs a neighboring rank's pixel, rows could just as easily be handed out round-robin or in any other pattern, with no correctness consequence at all. Section 30.2 shows why that freedom matters once tiles are not equally expensive to render.

```text
Chapter 16 / Chapter 29's tiles:            This chapter's tiles:

  MUST be contiguous strips                  can be ANY assignment of rows
  (rank N needs rank N+1's                   (no rank ever needs another
  boundary row)                              rank's row -- no dependency
                                              of any kind between tiles)
```

!!! warning "[COMMON TRAP] Assuming rendering is always communication-free"
    Section 30.1's own model is deliberately the *simplest possible* rendering case: a fixed scene fully known to every rank in advance, with no cross-pixel dependency whatsoever. Real production renderers routinely violate one or both of those assumptions -- Section 30.2 shows real per-tile cost varying enough to matter, and Section 30.3 shows a real, cited case where the scene itself does not fit on one device. "Rendering has no communication" is true only as long as the working set fits on every device and no pixel's color depends on another pixel's result; neither assumption is guaranteed at production scale.

## 30.2 A Real Complication: Uneven Per-Tile Cost

### Intuition

Section 30.1's row-tiles all took the same amount of work to render, because every pixel's hit test costs the same fixed number of operations. Real scenes are not like that: a tile that lands on empty background costs far less than a tile packed with reflective surfaces, overlapping objects, or deep ray bounces. If every rank gets an equal *number* of rows but the *cost* of those rows varies wildly, the render's total time is bounded by whichever rank got the expensive rows -- the other ranks finish early and simply wait.

### Background

NVIDIA's own real Omniverse RTX renderer documentation confirms this is a genuine production concern, not a hypothetical one: it states that the renderer "automatically balances the amount of total path tracing work to be performed by each GPU in a multi-GPU configuration" -- meaning real per-GPU tiles are *not* simply equal-sized slices. The same documentation also reveals a second real detail Section 30.1's clean model did not have: real tile splitting uses "a large tile per GPU with a small overlap region between them," because filtering and denoising at a tile's edge genuinely needs a little of the neighboring tile's data -- a small, real exception to the zero-communication claim, though nothing like Chapter 16's full halo exchange. This section models the load-imbalance problem using Chapter 18's own real weighted-sharding technique -- originally built there for uneven *device* capability, reused here for uneven *scene* content.

```cpp
// Chapter 30: Multi-GPU Ray Tracing and Rendering
// 88_rendering_comm_and_load_balance_model.cpp
//
// Section 30.1's own simulation showed rendering's clean case needs ZERO
// communication during computation. This section quantifies exactly how
// different that is from every earlier Part 6 chapter, using a closed-form
// model (never a fabricated timing, matching this book's own standing
// practice), and then shows the real complication production renderers
// actually hit: NVIDIA's own real Omniverse RTX renderer documentation
// states it "splits the rendering of the image into a large tile per GPU
// with a small overlap region between them" (the overlap is real -- tile
// edges need neighboring pixel data for filtering/denoising, a small
// real exception to Section 30.1's zero-communication claim) and that it
// "automatically balances the amount of total path tracing work to be
// performed by each GPU in a multi-GPU configuration" -- meaning real
// per-GPU tiles are NOT always equal-sized, because real scenes are not
// uniformly expensive to render across every pixel. This section models
// that with Chapter 18's own real weighted-sharding technique, applied
// here to rendering cost instead of device capability.
#include <cstdio>
#include <cmath>
#include <cstring>

int main() {
    printf("--- Total communication volume vs. earlier Part 6 chapters ---\n");
    printf("Every case study since Ch24 pays a communication cost that "
           "GROWS with the number of computational steps taken (SUMMA "
           "rounds, ring all-reduce steps, TP/PP rounds, all-gather steps, "
           "BFS levels, Jacobi iterations). This chapter's own clean case "
           "pays a FIXED cost -- the final image size -- exactly ONCE, "
           "regardless of how much internal computation (ray bounces, "
           "samples per pixel) produced that image.\n\n");
    printf("%-10s %-30s %-30s\n", "steps", "earlier chapters: cost grows",
           "this chapter: cost stays fixed");
    const int stepCounts[] = {1, 10, 100, 1000};
    const long long perStepCost = 16; // illustrative unit, e.g. Ch29's own 16-double halo cost
    const long long finalImageCost = 16 * 12; // this chapter's own WIDTH*HEIGHT from Section 30.1
    for (int steps : stepCounts) {
        long long earlierTotal = (long long)steps * perStepCost;
        printf("%-10d %-30lld %-30lld\n", steps, earlierTotal, finalImageCost);
    }
    printf("\nAt 1000 steps, an earlier chapter's own per-step cost model "
           "would have paid out %lld total units of communication; this "
           "chapter's own model pays the SAME %lld units whether the "
           "render took 1 internal step or 1000 -- because nothing about "
           "internal ray computation touches another rank's data.\n\n",
           1000LL * perStepCost, finalImageCost);

    printf("--- The real complication: uneven per-tile rendering cost ---\n");
    printf("NVIDIA's own real Omniverse RTX renderer documentation: "
           "\"automatically balances the amount of total path tracing "
           "work to be performed by each GPU.\" This illustrative model "
           "(NOT measured data) assigns each of this chapter's own 12 "
           "image rows a rendering-cost WEIGHT reflecting how many "
           "ray-object tests that row's pixels actually need (rows near "
           "the two circles from Section 30.1 cost more than empty "
           "background rows):\n\n");

    // Illustrative per-row cost weight (NOT measured): rows overlapping
    // the two circles from Section 30.1's scene cost more.
    int rowWeight[12] = {1, 1, 3, 5, 6, 7, 6, 7, 4, 3, 2, 1};
    int totalWeight = 0;
    for (int w : rowWeight) totalWeight += w;

    printf("%-6s %-10s\n", "row", "cost weight");
    for (int i = 0; i < 12; i++) printf("%-6d %-10d\n", i, rowWeight[i]);
    printf("Total weight: %d\n\n", totalWeight);

    printf("Naive EQUAL row split across P=4 ranks (3 rows each):\n");
    printf("%-8s %-14s %-10s\n", "rank", "rows", "total weight");
    int maxNaive = 0;
    for (int r = 0; r < 4; r++) {
        int w = rowWeight[r*3] + rowWeight[r*3+1] + rowWeight[r*3+2];
        if (w > maxNaive) maxNaive = w;
        printf("%-8d %d-%-12d %-10d\n", r, r*3, r*3+2, w);
    }
    printf("Slowest rank's weight: %d out of total %d -- the render's total "
           "time is bounded by this SLOWEST rank, exactly Chapter 18's own "
           "real load-imbalance finding, here caused by SCENE CONTENT "
           "rather than device capability.\n\n", maxNaive, totalWeight);

    printf("Chapter 18's own real weighted-sharding technique applied to "
           "rows instead of devices -- greedily assign rows to whichever "
           "rank currently has the SMALLEST accumulated weight:\n");
    int rankWeight[4] = {0, 0, 0, 0};
    int rankRows[4][12]; int rankRowCount[4] = {0,0,0,0};
    for (int row = 0; row < 12; row++) {
        int best = 0;
        for (int r = 1; r < 4; r++) if (rankWeight[r] < rankWeight[best]) best = r;
        rankWeight[best] += rowWeight[row];
        rankRows[best][rankRowCount[best]++] = row;
    }
    printf("%-8s %-24s %-10s\n", "rank", "rows assigned", "total weight");
    int maxBalanced = 0;
    for (int r = 0; r < 4; r++) {
        if (rankWeight[r] > maxBalanced) maxBalanced = rankWeight[r];
        printf("%-8d ", r);
        char buf[64] = ""; char tmp[8];
        for (int i = 0; i < rankRowCount[r]; i++) {
            snprintf(tmp, sizeof(tmp), "%d%s", rankRows[r][i], i+1<rankRowCount[r] ? "," : "");
            strcat(buf, tmp);
        }
        printf("%-24s %-10d\n", buf, rankWeight[r]);
    }
    printf("\nSlowest rank's weight after rebalancing: %d (down from %d) -- "
           "a %.2fx reduction in the slowest rank's load, achieved with "
           "UNEQUAL row counts per rank but more balanced TOTAL weight, "
           "exactly Chapter 18's own real point that equal SHARE COUNT "
           "and equal WORK are different things.\n",
           maxBalanced, maxNaive, (double)maxNaive / (double)maxBalanced);

    return 0;
}
```

Compile and run:

```
g++ -O2 88_rendering_comm_and_load_balance_model.cpp -o 88_rendering_comm_and_load_balance_model
./88_rendering_comm_and_load_balance_model
```

```text
--- Total communication volume vs. earlier Part 6 chapters ---
Every case study since Ch24 pays a communication cost that GROWS with the number of computational steps taken (SUMMA rounds, ring all-reduce steps, TP/PP rounds, all-gather steps, BFS levels, Jacobi iterations). This chapter's own clean case pays a FIXED cost -- the final image size -- exactly ONCE, regardless of how much internal computation (ray bounces, samples per pixel) produced that image.

steps      earlier chapters: cost grows   this chapter: cost stays fixed
1          16                             192                           
10         160                            192                           
100        1600                           192                           
1000       16000                          192                           

At 1000 steps, an earlier chapter's own per-step cost model would have paid out 16000 total units of communication; this chapter's own model pays the SAME 192 units whether the render took 1 internal step or 1000 -- because nothing about internal ray computation touches another rank's data.

--- The real complication: uneven per-tile rendering cost ---
NVIDIA's own real Omniverse RTX renderer documentation: "automatically balances the amount of total path tracing work to be performed by each GPU." This illustrative model (NOT measured data) assigns each of this chapter's own 12 image rows a rendering-cost WEIGHT reflecting how many ray-object tests that row's pixels actually need (rows near the two circles from Section 30.1 cost more than empty background rows):

row    cost weight
0      1         
1      1         
2      3         
3      5         
4      6         
5      7         
6      6         
7      7         
8      4         
9      3         
10     2         
11     1         
Total weight: 46

Naive EQUAL row split across P=4 ranks (3 rows each):
rank     rows           total weight
0        0-2            5         
1        3-5            18        
2        6-8            17        
3        9-11           6         
Slowest rank's weight: 18 out of total 46 -- the render's total time is bounded by this SLOWEST rank, exactly Chapter 18's own real load-imbalance finding, here caused by SCENE CONTENT rather than device capability.

Chapter 18's own real weighted-sharding technique applied to rows instead of devices -- greedily assign rows to whichever rank currently has the SMALLEST accumulated weight:
rank     rows assigned            total weight
0        0,4,8,11                 12        
1        1,5,9                    11        
2        2,6,10                   11        
3        3,7                      12        

Slowest rank's weight after rebalancing: 12 (down from 18) -- a 1.50x reduction in the slowest rank's load, achieved with UNEQUAL row counts per rank but more balanced TOTAL weight, exactly Chapter 18's own real point that equal SHARE COUNT and equal WORK are different things.
```

Notice the rebalanced assignment: rank 0 gets rows 0, 4, 8, and 11 -- not a contiguous strip. Chapter 18's own weighted-sharding technique could only reach this more balanced 1.50x-better assignment because Section 30.1 already established that rendering tiles have no neighbor dependency; a contiguous-strip requirement, the way Chapter 16 and Chapter 29 needed, would have ruled this exact assignment out.

```text
Naive equal split (contiguous, unbalanced):    Rebalanced (non-contiguous, balanced):

  rank 0: rows 0-2   (light)                     rank 0: rows 0,4,8,11  (mixed)
  rank 1: rows 3-5   (HEAVY, slowest)              rank 1: rows 1,5,9     (mixed)
  rank 2: rows 6-8   (heavy)                       rank 2: rows 2,6,10    (mixed)
  rank 3: rows 9-11  (light)                       rank 3: rows 3,7       (mixed)
                                                    all four ranks finish
                                                    closer together
```

!!! warning "[COMMON TRAP] Treating NVIDIA's real tile overlap as a contradiction of the zero-communication claim"
    NVIDIA's own real "small overlap region between them" is a genuinely real detail, but it is not the same kind of communication Chapter 16's halo exchange needed. Chapter 16's halo exchange feeds directly into the *numerical result* of the next iteration's stencil update -- omitting it changes the computed values. A rendering tile's overlap exists to make a *filtering/denoising* step look correct at the seam, a post-processing concern layered on top of an already-complete per-pixel computation. Do not conflate a correctness-critical exchange with a cosmetic one just because both involve neighboring tiles.

## 30.3 The Real Limit: When the Scene Itself Does Not Fit

### Intuition

Section 30.1's clean case assumed every rank already has the entire scene description available locally -- a reasonable assumption for a two-circle toy scene, but not for a real production scene with enormous amounts of geometry and texture data. Once a single frame's own scene data is too large for one device's memory, the "no communication needed" argument collapses at its foundation: a ray traveling through the scene might need to test against geometry that simply is not there.

### Background

A real, cited SIGGRAPH 2022 paper makes this concrete at genuine production scale: Fouladi, Shacklett, Poms, Arora, Ozdemir, Raghavan, Hanrahan, Fatahalian & Winstein's "R2E2: Low-Latency Path Tracing of Terabyte-Scale Scenes using Thousands of Cloud CPUs" (*ACM Transactions on Graphics*, 41(4), Article 76, 2022) describes real scenes "with up to a terabyte of geometry and texture data (where as little as 1/250th of the scene can fit on any one node)." At that scale, the scene itself has to be partitioned across nodes the same way Chapter 24 partitioned a matrix too large for one device -- and a ray that needs geometry owned by a different node must trigger real communication mid-render, something Section 30.1's clean model never needed to do.

```cpp
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
// Raghavan, Hanrahan, Fatahalian & Winston, "R2E2: Low-Latency Path
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
```

Compile and run:

```
g++ -O2 89_out_of_core_scene_scale_model.cpp -o 89_out_of_core_scene_scale_model
./89_out_of_core_scene_scale_model
```

```text
--- R2E2's own real scale problem ---
Real cited fact: "scenes with up to a terabyte of geometry and texture data (where as little as 1/250th of the scene can fit on any one node)" -- Fouladi et al., SIGGRAPH 2022.

total scene size         per-node capacity    nodes needed (this chapter's own model)
100 GB (illustrative)    0.40                 250                     
500 GB (illustrative)    2.00                 250                     
1024 GB = 1 TB (R2E2 cited) 4.10                 250                     

At R2E2's own real terabyte scale, roughly 250 nodes are needed just to HOLD the scene data -- this is now a genuine data DECOMPOSITION problem, structurally the same kind of problem Chapter 24 solved for a matrix too large for one device's memory, not the zero-communication case Section 30.1 built.

--- Why this reintroduces communication DURING rendering ---
Section 30.1's own simulation worked because every pixel's ray test needed ONLY the fixed scene description and that pixel's own coordinates -- and the whole scene fit in that simulation's own memory. Once the scene itself is partitioned across nodes (because no single node can hold it), a ray that would need to test against geometry owned by ANOTHER node must either fetch that geometry or forward the ray's own state to the node that owns it -- a real per-ray communication requirement Section 30.1's clean model never had to pay.

--- Real payoff, cited ---
R2E2's own real reported results against a single-machine in-memory baseline (pbrt-treelet): "R2E2 is 3.5-7.8x faster than pbrt-treelet's single-machine in-memory path tracer," with scene loading itself "4.9x (Moana-XL) and 2.9x (Terrace) faster." The real headline finding: "R2E2 completes the entire path tracing job before pbrt-treelet begins tracing a single ray" for most of their tested configurations -- because the single-machine baseline cannot even START until the whole scene has been paged in from disk, while R2E2's own distributed approach never needs the whole scene on any one node at all.
```

```text
Section 30.1's assumption:                  R2E2's own real production reality:

  whole scene fits in                        whole scene does NOT fit
  every rank's own memory                     on any single node
  --------------------------->                --------------------------->
  zero communication needed                   rays must fetch/forward
                                               across node boundaries
```

!!! warning "[COMMON TRAP] Assuming a bigger scene just means more tiles"
    It is tempting to assume that scaling up a scene only ever means giving each rank a smaller *tile of the image* -- more rows, more ranks, same clean model. R2E2's own real motivating problem shows this is only true while the *scene data itself* still fits per node. Once the scene's geometry and textures alone exceed a node's memory, the bottleneck moves from "how do we split the image" to "how do we split the scene" -- and splitting the scene, unlike splitting the image, reintroduces exactly the kind of cross-device data dependency this chapter's clean case was built specifically to avoid.

## Chapter Summary

Rendering, in its simplest and most common form, breaks the pattern every other Part 6 case study followed: Section 30.1's own simulation showed that when every pixel's color depends only on a fixed scene description and that pixel's own coordinates, a distributed render needs zero communication during computation, matches a single-process reference exactly at every tested partition, and does not even raise the reduction-order question earlier chapters' collectives did, because there is no reduction at all. Section 30.2 quantified how differently this chapter's total communication cost scales compared to every earlier chapter's per-step cost, then showed a real, cited complication -- NVIDIA's own real Omniverse RTX renderer both introduces a small real tile-overlap exception and automatically load-balances uneven per-tile cost -- and modeled that rebalancing with Chapter 18's own real weighted-sharding technique, made possible here specifically because rendering tiles, unlike Chapter 16 and Chapter 29's contiguous halo-exchange strips, have no neighbor dependency at all. Section 30.3 closed with the real limit: Fouladi et al.'s real SIGGRAPH 2022 R2E2 paper shows that once a scene's own data no longer fits on one device, rendering rejoins the rest of this book's distributed-systems concerns, needing genuine cross-node communication mid-render.

## Self-Check Questions

1. Why does Section 30.1's distributed render match a single-process reference bit-for-bit at every tested P, without needing any argument about floating-point operation order the way Chapter 16 or Chapter 29 did?
2. Why can Section 30.1's row-tiles be assigned to ranks in any pattern (not just contiguous strips), while Chapter 16 and Chapter 29's halo-exchange strips could not?
3. According to Section 30.2's own cost model, how does this chapter's total communication cost behave as the number of internal rendering steps grows, and how is that different from every earlier Part 6 chapter?
4. What two real details does NVIDIA's own real Omniverse RTX renderer documentation reveal that Section 30.1's clean model did not have?
5. Why is NVIDIA's own real tile-overlap detail not a counterexample to Section 30.1's zero-communication claim, according to Section 30.2's own COMMON TRAP?
6. In Section 30.2's own rebalanced row assignment, why does rank 0 end up with the non-contiguous rows 0, 4, 8, and 11, and why was Chapter 18's own weighted-sharding technique able to produce that assignment here but not in Chapter 16's setting?
7. According to Fouladi et al.'s real cited R2E2 paper, what real constraint makes a single production scene a genuine multi-node data-decomposition problem?
8. Why does a scene that does not fit on one device's memory reintroduce communication during rendering, when Section 30.1's clean case did not need any?

## Where We Go Next

Chapter 31, "Multi-GPU Monte Carlo Risk Simulation," closes out Part 6 with a workload that looks, at first glance, like this chapter's embarrassingly parallel case -- independent random simulation paths that need no communication with each other -- but reintroduces a global reduction at the end (aggregating simulation results into a single risk estimate), giving a natural point of comparison against both this chapter's pure independence and Chapter 29's own halo-exchange-plus-reduction shape.

## Worked Solutions

1. Every pixel's color in Section 30.1's model is computed as a pure function of the fixed scene description and that pixel's own coordinates, with no dependency on any other pixel's value and no combination/reduction step of any kind. Floating-point operation-order questions only arise when several values are combined (added, averaged) in an order that could differ between a distributed and a single-process run; since nothing here is ever combined across pixels or ranks, there is no such order to compare.
2. A row-tile in Chapter 16 or Chapter 29 needs its neighboring rank's boundary row to correctly compute its own edge cells, so the strips must stay contiguous and adjacent for the halo exchange to make sense. Section 30.1's row-tiles have no such dependency -- each row's pixels are computed purely from the fixed scene and that row's own coordinates -- so any rank could be given any subset of rows with no effect on correctness.
3. Section 30.2's own model shows this chapter's total communication cost stays fixed (the size of the final assembled image) no matter how many internal computational steps (ray bounces, samples per pixel) were taken, while every earlier Part 6 chapter's own cost model grows linearly (or worse) with the number of computational steps, because each of those chapters needed a communication step at every iteration.
4. NVIDIA's own real documentation reveals that (a) real multi-GPU tiles use "a small overlap region between them" for filtering/denoising at tile edges, and (b) the renderer "automatically balances the amount of total path tracing work to be performed by each GPU," meaning real tiles are not simply equal-sized slices of the image.
5. The tile overlap exists to make a post-processing filtering/denoising step look correct at the seam between two already-fully-computed tiles; it does not feed into the numerical result of any pixel's own color computation the way Chapter 16's halo exchange feeds directly into the next iteration's stencil values. It is a cosmetic-correctness exchange, not a computational-correctness one.
6. Chapter 18's own weighted-sharding technique greedily assigns each unit of work to whichever rank currently holds the smallest accumulated weight, which naturally produces a non-contiguous, interleaved assignment when the per-unit weights vary unevenly across the sequence. That assignment is only valid in Section 30.2's setting because no rank ever needs another rank's row; the same technique could not be applied to Chapter 16's contiguous halo-exchange strips, where a rank's rows must stay adjacent to its neighbors' rows for the exchange to be meaningful.
7. Fouladi et al.'s real cited constraint is that a production scene can contain "up to a terabyte of geometry and texture data (where as little as 1/250th of the scene can fit on any one node)" -- meaning no single node has enough memory to hold the whole scene, forcing the scene's own data to be partitioned across many nodes before rendering can even begin.
8. Once the scene's own data is partitioned across nodes because it does not fit on one device, a ray traveling through the scene can reach a region of geometry that is owned by a different node than the one currently tracing that ray; that ray's state (or the needed geometry) must then be communicated across node boundaries mid-render, something Section 30.1's clean model never needed because it assumed the entire scene was already available locally to every rank.

---

**Sources cited in this chapter:**

- Steven Molnar, Michael Cox, David Ellsworth, Henry Fuchs, "A Sorting Classification of Parallel Rendering," *IEEE Computer Graphics and Applications*, Vol. 14, No. 4, July 1994, pp. 23-32.
- NVIDIA Omniverse documentation, "RTX Renderer -- Multi-GPU," https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_mgpu.html.
- Wikipedia, "Render farm," https://en.wikipedia.org/wiki/Render_farm.
- Sadjad Fouladi, Brennan Shacklett, Fait Poms, Arjun Arora, Alex Ozdemir, Deepti Raghavan, Pat Hanrahan, Kayvon Fatahalian, Keith Winstein, "R2E2: Low-Latency Path Tracing of Terabyte-Scale Scenes using Thousands of Cloud CPUs," *ACM Transactions on Graphics*, 41(4), Article 76, 2022 (SIGGRAPH 2022).
