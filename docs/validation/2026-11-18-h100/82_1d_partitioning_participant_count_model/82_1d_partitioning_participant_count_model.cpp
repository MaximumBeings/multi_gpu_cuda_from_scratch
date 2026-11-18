// Chapter 28: Distributed Breadth-First Search
// 82_1d_partitioning_participant_count_model.cpp
//
// Section 28.1's own simulation showed distributed BFS's real
// communication pattern changing every level, depending on the
// graph's own structure. This section quantifies the real WORST CASE
// that 1D vertex partitioning (Yoo et al.'s own real SC05 term) can
// produce, and contrasts it against Chapter 16 and Chapter 27's own
// two extremes. Yoo et al.'s own real paper states the worst case
// plainly: "for 1D partitioning, all P processors are involved in the
// communication operation." This section's own worked model shows why:
// under 1D partitioning, a rank's own frontier vertices' neighbors can
// be owned by ANY other rank, so in the worst case a single BFS level
// needs up to P-1 communication partners per rank -- the SAME order as
// Chapter 27's own mandatory all-gather, but for a genuinely different
// reason: Chapter 27's O(P) was UNAVOIDABLE, every single step, by the
// physics itself; this chapter's own O(P) is a worst-case CEILING that
// depends on the graph and the current frontier, and may never
// actually be reached for a well-structured graph.
#include <cstdio>

int main() {
    printf("%-8s %-24s %-24s %-24s\n", "P",
           "Ch16 halo partners/rank", "Ch27 all-gather partners/rank",
           "Ch28 1D-BFS WORST CASE/rank");
    printf("--------------------------------------------------------------"
           "------------------\n");

    const int Ps[] = {4, 16, 64, 256, 1024, 4096};
    for (int P : Ps) {
        int haloFixed = 2;           // Chapter 16: fixed, regardless of P
        int allGatherFixed = P - 1;  // Chapter 27: mandatory, every step
        int bfsWorstCase = P - 1;    // Chapter 28 (1D partitioning): SAME
                                      // ORDER as Ch27's worst case, but
                                      // only a CEILING, not a guarantee
        printf("%-8d %-24d %-24d %-24d\n", P, haloFixed, allGatherFixed,
               bfsWorstCase);
    }

    printf("\nChapter 16's own halo-exchange partner count stays FIXED at "
           "2, regardless of P -- it never depends on problem scale.\n");
    printf("Chapter 27's own all-gather partner count is P-1, EVERY "
           "single step, UNAVOIDABLY -- the physics (no cutoff radius) "
           "leaves no choice.\n");
    printf("This chapter's own 1D-partitioned BFS partner count can reach "
           "the SAME P-1 ceiling Chapter 27 has -- but only in the worst "
           "case, only on SOME levels, and only for graphs whose edges "
           "happen to spread across every rank -- it is graph-dependent, "
           "not physics-mandated.\n\n");

    // Yoo et al.'s own real 2D partitioning fix: O(sqrt(P)) participants
    // instead of O(P). This section computes the real reduction factor.
    printf("--- Yoo et al.'s own real fix: 2D partitioning ---\n");
    printf("%-8s %-24s %-24s %-16s\n", "P", "1D worst case: P-1",
           "2D partitioning: sqrt(P)", "reduction factor");
    for (int P : Ps) {
        double sqrtP = 0.0;
        int p = P;
        // integer sqrt via simple loop (small P values here, exactness matters)
        for (int s = 1; s * s <= p; s++) sqrtP = (double)s;
        double reduction = (double)(P - 1) / sqrtP;
        printf("%-8d %-24d %-24.1f %-16.1fx\n", P, P - 1, sqrtP, reduction);
    }
    printf("\nYoo et al.'s own real quote: \"With the 2D partitioning, the "
           "number of processes involved in collective communications is "
           "O(sqrt(P)) in contrast to O(P) of 1D partitioning.\" The "
           "reduction factor itself GROWS with P -- exactly the shape "
           "that makes 2D partitioning matter more, not less, as a "
           "cluster scales up.\n");

    return 0;
}
