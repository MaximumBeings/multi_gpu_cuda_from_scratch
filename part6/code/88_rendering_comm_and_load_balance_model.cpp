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
