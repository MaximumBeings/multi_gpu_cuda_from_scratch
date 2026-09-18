// Appendix B: Practice Quiz
// 123_quiz_ring_allreduce_phases.cpp
//
// Appendix B.4, Challenge 1 -- Chapter 9 built ring all-reduce as TWO
// distinct phases: reduce-scatter (P-1 steps, after which each rank
// holds the FULLY reduced result for only ONE chunk) followed by
// all-gather (P-1 more steps, after which every rank holds the complete
// reduced result). Before compiling and running this file, predict: for
// P=4 ranks, each starting with a 4-element array where
// rank r's array is [10*(r+1)+0, 10*(r+1)+1, 10*(r+1)+2, 10*(r+1)+3],
// which ONE chunk index does rank 2 hold fully-reduced immediately after
// reduce-scatter finishes (before all-gather begins), and what is that
// chunk's value?
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 123_quiz_ring_allreduce_phases.cpp -o 123_quiz_ring_allreduce_phases
// Run:     ./123_quiz_ring_allreduce_phases
#include <cstdio>
#include <vector>

constexpr int P = 4;

void printState(const std::vector<std::vector<int>>& buf, const char* label) {
    printf("%s\n", label);
    for (int r = 0; r < P; r++) {
        printf("  rank %d: [", r);
        for (int i = 0; i < P; i++) printf("%d%s", buf[r][i], i + 1 < P ? ", " : "");
        printf("]\n");
    }
    printf("\n");
}

int main() {
    // rank r's initial array: [10*(r+1)+0, 10*(r+1)+1, 10*(r+1)+2, 10*(r+1)+3]
    std::vector<std::vector<int>> buf(P, std::vector<int>(P));
    for (int r = 0; r < P; r++)
        for (int i = 0; i < P; i++)
            buf[r][i] = 10 * (r + 1) + i;

    printState(buf, "=== initial state (before reduce-scatter) ===");

    // Ground truth: fully all-reduced (summed) value for each chunk index.
    std::vector<int> trueSum(P, 0);
    for (int i = 0; i < P; i++)
        for (int r = 0; r < P; r++)
            trueSum[i] += buf[r][i];
    printf("=== ground truth: elementwise sum across all 4 ranks ===\n  [");
    for (int i = 0; i < P; i++) printf("%d%s", trueSum[i], i + 1 < P ? ", " : "");
    printf("]\n\n");

    // Reduce-scatter: P-1 steps. Standard ring convention -- rank r sends
    // chunk (r - step + P) % P to its right neighbor, which ADDS it into
    // its own copy of that same chunk. After P-1 steps, rank r holds the
    // FULLY reduced value for chunk (r + 1) % P (the chunk immediately
    // "ahead" of it in ring order, following Chapter 9's own indexing).
    for (int step = 0; step < P - 1; step++) {
        std::vector<std::vector<int>> next = buf;
        for (int r = 0; r < P; r++) {
            int sendChunk = (r - step + P) % P;
            int recvRank = (r + 1) % P;
            next[recvRank][sendChunk] += buf[r][sendChunk];
        }
        buf = next;
    }

    printState(buf, "=== after reduce-scatter (P-1 = 3 steps) ===");
    int fullyReducedChunkOnRank2 = (2 + 1) % P;
    printf("rank 2's own fully-reduced chunk is chunk index %d, value = %d "
           "(matches ground truth [%d]=%d)\n\n",
           fullyReducedChunkOnRank2, buf[2][fullyReducedChunkOnRank2],
           fullyReducedChunkOnRank2, trueSum[fullyReducedChunkOnRank2]);

    // All-gather: P-1 more steps, circulating each rank's own fully-
    // reduced chunk around the ring so every rank ends up with all P
    // fully-reduced chunks.
    for (int step = 0; step < P - 1; step++) {
        std::vector<std::vector<int>> next = buf;
        for (int r = 0; r < P; r++) {
            int sendChunk = (r + 1 - step + P) % P;
            int recvRank = (r + 1) % P;
            next[recvRank][sendChunk] = buf[r][sendChunk];
        }
        buf = next;
    }

    printState(buf, "=== after all-gather (P-1 = 3 more steps) -- every "
                     "rank now holds the full result ===");

    bool allMatch = true;
    for (int r = 0; r < P; r++)
        for (int i = 0; i < P; i++)
            if (buf[r][i] != trueSum[i]) allMatch = false;

    printf("self-check: every rank's final buffer matches ground truth: %s\n",
           allMatch ? "YES" : "NO");
    return allMatch ? 0 : 1;
}
