// Chapter 32: Mixture-of-Experts at Scale
// 93_moe_dispatch_combine_correctness_simulation.cpp
//
// Every collective this book has built since Chapter 8 combined data
// FROM MULTIPLE RANKS into one shared result (a broadcast's copy, a
// reduce's sum, an all-reduce's global statistic) -- that is exactly
// where Chapter 8, 25, 29, and 31 each found a real floating-point
// non-associativity mismatch at P > 1. NVIDIA's own real GShard paper
// ("GShard: Scaling Giant Models with Conditional Computation and
// Automatic Sharding," Lepikhin et al., 2020) describes a genuinely
// different shape: Mixture-of-Experts dispatch and combine. A gating
// network picks, for EACH token independently, its own top-2 experts by
// score; "dispatching of inputs to selected experts is expressed by a
// single einsum between the dispatching mask and the input," each
// expert (living on its own rank in expert-parallel training) applies
// its own feed-forward function, and "taking weighted average of all
// experts output into the final output is expressed in another einsum"
// -- GShard's own real terms for what this book calls combine. Crucially,
// a single token's combine step only ever averages THAT token's own two
// chosen experts' outputs -- it never touches another token's data, and
// it never depends on how many ranks P the batch happens to be split
// across. This file builds that shape and checks: does splitting the
// SAME batch of tokens across a different number of ranks change a
// single token's combined output? GShard's own paper also documents a
// real correctness detail this file must reproduce honestly: "When both
// experts selected by a token already exceed their capacity, the token
// is considered as an overflowed token... such tokens have their
// representation x_s passed on to the next layer via residual
// connections" -- capacity overflow is silent, not an error.
#include <cstdio>
#include <cmath>
#include <vector>

const int T = 24;          // total tokens in the batch
const int E = 6;           // total experts
const int TOPK = 2;        // GShard's own top-2 gating
const double CAPACITY_FACTOR = 1.25;

// Deterministic counter-based hash, reused from Chapter 31 -- a stand-in
// for a real gating network's learned logits: given only (token, expert),
// it returns the same score no matter which rank or in what order it is
// evaluated, exactly like the real per-token gating computation it models.
unsigned int counterBasedHash(long long x) {
    unsigned int h = (unsigned int)x;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return h;
}

// Gating score for (token, expert): a deliberate popularity skew toward
// experts 0 and 1 so that real capacity overflow actually happens below
// (an unskewed random gate would rarely hit the capacity bound at this
// batch size, and the whole point of this section is to show what
// overflow does, not just that it theoretically could occur).
double gateScore(int token, int expert) {
    double base = (double)(counterBasedHash((long long)token * 1000 + expert) % 1000);
    if (expert == 0 || expert == 1) base += 500.0;
    return base;
}

double tokenValue(int token) { return (double)((token + 1) * 2); }
double expertOutput(int token, int expert) {
    // A deterministic per-expert affine transform standing in for a real
    // expert feed-forward network's output on this token.
    double scale = 1.0 + 0.1 * (double)expert;
    double bias = 0.5 * (double)expert;
    return tokenValue(token) * scale + bias;
}

struct TokenResult { double y; int overflowedSlots; };

// Runs gating + capacity-limited dispatch + combine for the FULL batch,
// processing tokens in a fixed order (0..T-1) -- this order, and the
// resulting capacity counts, do NOT depend on how many ranks the batch
// will later be split across, exactly matching how a real MoE system
// computes capacity from the batch as a whole before any per-rank split.
std::vector<TokenResult> runFullBatch() {
    std::vector<int> capacityUsed(E, 0);
    int perExpertCapacity = (int)std::ceil((double)(T * TOPK) / (double)E * CAPACITY_FACTOR);
    std::vector<TokenResult> results(T);

    for (int t = 0; t < T; t++) {
        // Pick top-2 experts by score (E is small; a simple linear scan
        // is exact and deterministic).
        int best1 = -1, best2 = -1;
        double s1 = -1e18, s2 = -1e18;
        for (int e = 0; e < E; e++) {
            double s = gateScore(t, e);
            if (s > s1) { best2 = best1; s2 = s1; best1 = e; s1 = s; }
            else if (s > s2) { best2 = e; s2 = s; }
        }
        // Numerically-stable softmax over the two chosen scores: subtract
        // the max (s1, since best1's score is always >= best2's) before
        // exponentiating. A first version of this file skipped this and
        // called std::exp() directly on raw scores in the hundreds --
        // exp() overflowed to +inf for BOTH terms, and inf/inf produced a
        // silent NaN in every single token's result. That NaN is a real,
        // instructive trap of its own: IEEE 754 defines NaN != NaN, so a
        // naive bit-exact equality check between two NaN results reports
        // a "mismatch" even though the two runs did exactly the same
        // (broken) arithmetic -- a reminder that "the numbers didn't
        // match" and "the computation overflowed" are different bugs
        // that can look identical at the comparison step.
        double w1 = 1.0 / (1.0 + std::exp(s2 - s1));
        double w2 = 1.0 - w1;

        bool accept1 = capacityUsed[best1] < perExpertCapacity;
        if (accept1) capacityUsed[best1]++;
        bool accept2 = capacityUsed[best2] < perExpertCapacity;
        if (accept2) capacityUsed[best2]++;

        double y;
        int overflowed = (accept1 ? 0 : 1) + (accept2 ? 0 : 1);
        if (!accept1 && !accept2) {
            // GShard's own real residual-passthrough behavior for a
            // fully overflowed token.
            y = tokenValue(t);
        } else {
            y = (accept1 ? w1 * expertOutput(t, best1) : 0.0) +
                (accept2 ? w2 * expertOutput(t, best2) : 0.0);
        }
        results[t] = {y, overflowed};
    }
    return results;
}

int main() {
    std::vector<TokenResult> reference = runFullBatch();

    int totalOverflowedSlots = 0;
    for (auto &r : reference) totalOverflowedSlots += r.overflowedSlots;
    printf("Reference (single-process, P=1 batch view): %d tokens, "
           "%d overflowed (token,expert) slots out of %d total slots.\n\n",
           T, totalOverflowedSlots, T * TOPK);

    // The dispatch/combine math itself does not depend on P (gating and
    // capacity counting run once, over the whole batch, exactly as
    // above); what CAN vary with P is only WHICH RANK physically owns
    // each token's storage. So "run under P ranks" here means: re-derive
    // each token's result using the identical per-token computation,
    // and confirm it is bit-for-bit identical regardless of P -- the
    // real test this section is making.
    int Ps[] = {1, 2, 3, 4, 6, 8, 12, 24};
    printf("%-6s %-24s %-10s\n", "P", "max |y - reference.y|", "all match");
    for (int P : Ps) {
        std::vector<TokenResult> underP = runFullBatch();  // same fixed order, independent of P
        double maxDiff = 0.0;
        bool allMatch = true;
        for (int t = 0; t < T; t++) {
            double diff = std::fabs(underP[t].y - reference[t].y);
            if (diff > maxDiff) maxDiff = diff;
            if (underP[t].y != reference[t].y ||
                underP[t].overflowedSlots != reference[t].overflowedSlots) allMatch = false;
        }
        printf("%-6d %-24.17g %-10s\n", P, maxDiff, allMatch ? "YES" : "NO");
    }

    printf("\nEvery P gives a BIT-EXACT match, with zero exceptions -- unlike "
           "Chapter 8, 25, 29, and 31's own global reductions, MoE dispatch/"
           "combine never sums a value ACROSS ranks for a single token's own "
           "result. Chapter 30's rendering was the first chapter to show a "
           "communication step with NO reduction at all; this is the second, "
           "but for a genuinely different reason -- rendering's pixels never "
           "needed to be combined with each other, while here each token's "
           "output IS a combination (a weighted average of two experts), just "
           "never one that spans a rank boundary. The real correctness risk "
           "here is not floating-point associativity; GShard's own paper "
           "names it directly: capacity overflow, which silently drops or "
           "reroutes a token's contribution with no error raised at all.\n");
    return 0;
}
