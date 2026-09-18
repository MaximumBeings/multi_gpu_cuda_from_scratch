// Chapter 11: NCCL -- What It Actually Does Differently From What You
// Just Built
// 32_ring_vs_tree_model.cpp
//
// Plain host C++ -- a closed-form ROUND-COUNT model, in the same
// spirit as Chapter 2's topology model and Chapter 5's cost model:
// no fabricated timings, only real, defensible structural counts.
// This compares Chapter 9's own ring all-reduce round count against
// a double binary tree's round count, for growing device counts, to
// make the NCCL 2.4 blog post's "latency scales linearly... versus
// logarithmic latency" claim (cited in Sources) concrete as numbers.
#include <cstdio>
#include <cmath>

int main() {
    printf("%-10s %-22s %-22s %-10s\n", "N", "Ring rounds (Ch9)", "Tree rounds (up+down)", "Ratio");
    printf("%-10s %-22s %-22s %-10s\n", "-", "2*(N-1)", "2*ceil(log2(N))", "ring/tree");

    int Ns[] = {4, 8, 16, 64, 256, 1024, 8192, 24576};
    for (int N : Ns) {
        // Ring: Chapter 9's own totalSteps formula, unchanged.
        long long ringRounds = 2LL * (N - 1);

        // Double binary tree: reducing up to a root takes one round per
        // level of the tree, and broadcasting back down takes another
        // round per level -- a balanced binary tree over N leaves has
        // ceil(log2(N)) levels.
        int levels = (int)std::ceil(std::log2((double)N));
        long long treeRounds = 2LL * levels;

        double ratio = (double)ringRounds / (double)treeRounds;
        printf("%-10d %-22lld %-22lld %-10.2fx\n", N, ringRounds, treeRounds, ratio);
    }

    printf("\nRing's round count grows linearly with N (doubling N roughly\n"
           "doubles the rounds needed); the tree's grows logarithmically\n"
           "(doubling N adds only one more level in each direction). This\n"
           "is a structural count of communication ROUNDS, not a measured\n"
           "time -- it makes concrete exactly the shape of the claim this\n"
           "chapter cites from NVIDIA's own NCCL 2.4 announcement, without\n"
           "fabricating a single timing number of its own.\n");

    return 0;
}
