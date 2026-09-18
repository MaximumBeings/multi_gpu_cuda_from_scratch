// Chapter 28: Distributed Breadth-First Search
// 83_direction_optimizing_and_2d_partitioning_model.cpp
//
// Section 28.2 quantified 1D partitioning's own real worst-case
// communication participant count. This section presents the two real,
// named techniques production systems actually use to avoid that worst
// case, plus a real published result showing the payoff at extreme
// scale. First, Yoo et al.'s own real 2D partitioning: "A 2D
// partitioning of a graph is a partitioning of its edges such that
// each edge is owned by one processor," reducing collective
// participants from O(P) to O(sqrt(P)). Second, Beamer, Asanovic &
// Patterson's own real "Direction-Optimizing Breadth-First Search"
// (SC12): switching between top-down (push) and bottom-up (pull)
// traversal depending on frontier size -- "the bottom-up approach is
// advantageous when a large fraction of the vertices are in the
// frontier" -- because "once a vertex has found a parent, it does not
// need to check the rest of its neighbors." This section builds one
// small illustrative model of the SECOND technique's own real
// rationale (a hypothetical frontier-size progression, not a real
// graph's measured data -- explicitly labeled as such), and closes with
// a real, cited Graph500 result showing what genuinely optimized
// distributed BFS achieves at supercomputer scale.
#include <cstdio>

int main() {
    printf("--- Beamer et al.'s own real direction-optimizing idea, "
           "illustrated (NOT a real graph's measured data) ---\n");
    printf("This chapter's own illustrative frontier-fraction progression "
           "across 6 BFS levels of a hypothetical graph (chosen only to "
           "show the real SHAPE Beamer et al. describe, summing to 1.0):\n\n");

    // Illustrative frontier fractions per level (this chapter's own
    // stated example, not measured data): starts tiny, grows to a
    // large middle level (typical of small-world/scale-free graphs
    // with low diameter), then shrinks again.
    double frontierFrac[] = {0.0002, 0.02, 0.35, 0.55, 0.075, 0.0048};
    const int LEVELS = 6;

    double cumulativeVisitedBefore = 0.0; // sum of levels STRICTLY before this one
    printf("%-8s %-16s %-16s %-24s %-24s\n", "Level", "frontier frac",
           "undiscovered frac", "top-down cost ~ f*d", "bottom-up cost ~ u*d");
    const double d = 16.0; // average vertex degree -- this chapter's
                            // own stated assumption, not tied to a
                            // specific real graph's measured value
    for (int lvl = 0; lvl < LEVELS; lvl++) {
        double f = frontierFrac[lvl];
        // "Undiscovered" excludes both earlier levels AND this level's
        // own frontier (already discovered, just not yet expanded) --
        // the real set bottom-up would have to scan looking for a
        // frontier-adjacent neighbor.
        double undiscovered = 1.0 - cumulativeVisitedBefore - f;
        double topDownCost = f * d;
        double bottomUpCost = undiscovered * d;
        printf("%-8d %-16.4f %-16.4f %-24.3f %-24.3f %s\n",
               lvl, f, undiscovered, topDownCost, bottomUpCost,
               bottomUpCost < topDownCost ? "<- bottom-up cheaper here" : "");
        cumulativeVisitedBefore += f;
    }

    printf("\nAt the levels where the frontier is LARGE, the undiscovered "
           "fraction has already shrunk enough that bottom-up's own "
           "worst-case cost (checking every undiscovered vertex) undercuts "
           "top-down's cost (checking every frontier vertex's full "
           "neighbor list) -- exactly Beamer et al.'s own real stated "
           "rationale: \"the bottom-up approach is advantageous when a "
           "large fraction of the vertices are in the frontier.\" Real "
           "bottom-up implementations do even better than this simple "
           "model shows, because \"once a vertex has found a parent, it "
           "does not need to check the rest of its neighbors\" -- an "
           "early-exit saving this illustrative model does not attempt "
           "to quantify.\n\n");

    printf("--- Yoo et al.'s own real 2D partitioning definition ---\n");
    printf("\"A 2D partitioning of a graph is a partitioning of its edges "
           "such that each edge is owned by one processor\" -- reducing "
           "collective-communication participants from O(P) (Section "
           "28.2's own 1D worst case) to O(sqrt(P)).\n\n");

    printf("--- Real, cited result at supercomputer scale (NOT computed "
           "by this program) ---\n");
    printf("Graph500 BFS benchmark, June 2024 list: Fugaku (RIKEN), "
           "SCALE 42, 166,029 GTEPS (giga-traversed-edges-per-second) -- "
           "RIKEN's own stated reason for the improvement: \"we enhanced "
           "performance to 166,029 GTEPS by developing a feature that "
           "removes unnecessary vertices.\"\n");

    return 0;
}
