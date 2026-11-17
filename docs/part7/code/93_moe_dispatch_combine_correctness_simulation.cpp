// Chapter 32: Mixture-of-Experts at Scale
// 93_moe_dispatch_combine_correctness_simulation.cpp
//
// Google's GShard paper ("GShard: Scaling Giant Models with Conditional
// Computation and Automatic Sharding," Lepikhin et al., 2020) gives MoE
// its standard vocabulary: a gating network picks, for EACH token, its
// top-2 experts; "dispatching of inputs to selected experts is expressed
// by a single einsum between the dispatching mask and the input," and
// "taking weighted average of all experts output into the final output
// is expressed in another einsum" -- what this book calls combine.
//
// GShard ALSO says how expert capacity is enforced, and that is where a
// partition-dependence hides: the tokens are split into G groups that are
// processed independently, and each group gets only "a fractional
// capacity of each expert." In expert-parallel training each rank's
// local batch is such a group, so the capacity check is made per rank,
// against that rank's own share of the capacity, in that rank's own
// token order. Change the number of ranks P and you change which tokens
// share a capacity budget -- and therefore which tokens overflow.
//
// GShard's overflow rule: "When both experts selected by a token already
// exceed their capacity, the token is considered as an overflowed
// token... such tokens have their representation x_s passed on to the
// next layer via residual connections" -- silent, not an error.
//
// Part A of this file runs the SAME tokens and the SAME gate under
// P = 1, 2, 3, 4, 6, 8, 12, 24 groups and reports how many tokens' outputs
// change relative to P = 1, and how the overflow count moves.
//
// Part B isolates the floating-point side of combine. With top-2 routing
// a token's combine is a 2-term sum, and IEEE 754 addition is
// commutative, so a+b == b+a bit for bit: no ordering or grouping of the
// two contributions can change the result. With top-k for k >= 3 (DeepSeek-
// V3 routes each token to 8 experts) the grouping of the k terms DOES
// matter. Part B models a hierarchical combine -- each rank first sums the
// contributions of the experts it owns, then the per-rank partials are
// summed -- and counts how many tokens' results differ from the P = 1 sum.
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

const int T = 24;          // total tokens in the batch (Part A)
const int E = 6;           // total experts (Part A)
const int TOPK = 2;        // GShard's top-2 gating
const double CAPACITY_FACTOR = 1.25;

// Deterministic counter-based hash, reused from Chapter 31 -- a stand-in
// for a real gating network's learned logits: given only (token, expert),
// it returns the same score no matter which rank evaluates it.
unsigned int counterBasedHash(long long x) {
    unsigned int h = (unsigned int)x;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return h;
}

// Gating score with a deliberate popularity skew toward experts 0 and 1,
// so that capacity overflow actually happens at this small batch size.
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

// GShard-style capacity: the batch is split into P contiguous groups
// (one per rank); each group has its OWN per-expert capacity, sized from
// the group's own token count, and checks it in the group's own order.
// P = 1 is the "whole batch is one group" special case.
std::vector<TokenResult> runGrouped(int P) {
    std::vector<TokenResult> results(T);
    int groupSize = T / P;  // every tested P divides T
    int perExpertCapacity =
        (int)std::ceil((double)(groupSize * TOPK) / (double)E * CAPACITY_FACTOR);

    for (int g = 0; g < P; g++) {
        std::vector<int> capacityUsed(E, 0);  // fresh budget per group
        for (int t = g * groupSize; t < (g + 1) * groupSize; t++) {
            int best1 = -1, best2 = -1;
            double s1 = -1e18, s2 = -1e18;
            for (int e = 0; e < E; e++) {
                double s = gateScore(t, e);
                if (s > s1) { best2 = best1; s2 = s1; best1 = e; s1 = s; }
                else if (s > s2) { best2 = e; s2 = s; }
            }
            // Numerically-stable 2-way softmax (subtract the max first).
            // An earlier draft called exp() on raw scores in the hundreds,
            // overflowed to +inf, and produced NaN for every token -- and
            // since NaN != NaN, the equality check then reported a
            // "mismatch" for two runs that did identical arithmetic.
            double w1 = 1.0 / (1.0 + std::exp(s2 - s1));
            double w2 = 1.0 - w1;

            bool accept1 = capacityUsed[best1] < perExpertCapacity;
            if (accept1) capacityUsed[best1]++;
            bool accept2 = capacityUsed[best2] < perExpertCapacity;
            if (accept2) capacityUsed[best2]++;

            double y;
            int overflowed = (accept1 ? 0 : 1) + (accept2 ? 0 : 1);
            if (!accept1 && !accept2) {
                y = tokenValue(t);  // GShard's residual passthrough
            } else {
                y = (accept1 ? w1 * expertOutput(t, best1) : 0.0) +
                    (accept2 ? w2 * expertOutput(t, best2) : 0.0);
            }
            results[t] = {y, overflowed};
        }
    }
    return results;
}

// ---- Part B: does the combine's summation grouping matter? ----
const int TB = 4096;   // tokens
const int EB = 32;     // experts, sharded contiguously: EB/P per rank

// Hierarchical combine for one token: each rank sums (in ascending expert
// order) the weighted outputs of the chosen experts it owns, then the
// per-rank partial sums are added in rank order. P = 1 is a plain
// left-to-right sum of all k terms.
double combineHierarchical(const std::vector<int> &experts,
                           const std::vector<double> &terms, int P) {
    int perRank = EB / P;
    double total = 0.0;
    bool haveTotal = false;
    for (int r = 0; r < P; r++) {
        double partial = 0.0;
        bool havePartial = false;
        for (size_t i = 0; i < experts.size(); i++) {
            if (experts[i] / perRank != r) continue;
            partial = havePartial ? partial + terms[i] : terms[i];
            havePartial = true;
        }
        if (!havePartial) continue;
        total = haveTotal ? total + partial : partial;
        haveTotal = true;
    }
    return total;
}

int countCombineMismatches(int k, int P) {
    int mismatches = 0;
    for (int t = 0; t < TB; t++) {
        // Top-k experts by an (unskewed) hash score.
        std::vector<std::pair<double, int>> scored;
        for (int e = 0; e < EB; e++) {
            double s = (double)(counterBasedHash((long long)t * 7919 + e * 104729LL) % 1000) / 100.0;
            scored.push_back({s, e});
        }
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                          [](const std::pair<double, int> &a, const std::pair<double, int> &b) {
                              return a.first > b.first || (a.first == b.first && a.second < b.second);
                          });
        double smax = scored[0].first, denom = 0.0;
        for (int i = 0; i < k; i++) denom += std::exp(scored[i].first - smax);
        std::vector<std::pair<int, double>> chosen;  // (expert, weighted term)
        for (int i = 0; i < k; i++) {
            int e = scored[i].second;
            double w = std::exp(scored[i].first - smax) / denom;
            double x = 1.0 + (double)(t % 97) / 7.0;  // non-integer token value
            chosen.push_back({e, w * (x * (1.0 + 0.1 * e) + 0.5 * e)});
        }
        // Ascending expert order, so both paths see the same term order and
        // differ ONLY in how the terms are grouped.
        std::sort(chosen.begin(), chosen.end());
        std::vector<int> experts;
        std::vector<double> terms;
        for (auto &c : chosen) { experts.push_back(c.first); terms.push_back(c.second); }
        double ref = combineHierarchical(experts, terms, 1);
        double got = combineHierarchical(experts, terms, P);
        if (got != ref) mismatches++;
    }
    return mismatches;
}

int main() {
    // ---------------- Part A ----------------
    std::vector<TokenResult> reference = runGrouped(1);
    int refSlots = 0, refFull = 0;
    for (auto &r : reference) { refSlots += r.overflowedSlots; refFull += (r.overflowedSlots == 2); }
    printf("Part A -- GShard-style per-group capacity, %d tokens, %d experts, top-%d, "
           "capacity factor %.2f\n", T, E, TOPK, CAPACITY_FACTOR);
    printf("Reference (P=1, whole batch is one group): %d overflowed (token,expert) "
           "slots of %d, %d tokens fully overflowed.\n\n", refSlots, T * TOPK, refFull);

    int Ps[] = {1, 2, 3, 4, 6, 8, 12, 24};
    printf("%-4s %-10s %-16s %-17s %-15s %-22s\n", "P", "cap/group",
           "overflow slots", "fully overflowed", "tokens changed", "max |y - reference.y|");
    for (int P : Ps) {
        std::vector<TokenResult> underP = runGrouped(P);
        int cap = (int)std::ceil((double)((T / P) * TOPK) / (double)E * CAPACITY_FACTOR);
        int slots = 0, full = 0, changed = 0;
        double maxDiff = 0.0;
        for (int t = 0; t < T; t++) {
            slots += underP[t].overflowedSlots;
            full += (underP[t].overflowedSlots == 2);
            if (underP[t].y != reference[t].y) changed++;
            maxDiff = std::max(maxDiff, std::fabs(underP[t].y - reference[t].y));
        }
        printf("%-4d %-10d %-16d %-17d %-15d %-22.6g\n", P, cap, slots, full, changed, maxDiff);
    }

    // ---------------- Part B ----------------
    printf("\nPart B -- hierarchical combine (per-rank partial sums, then a sum of "
           "partials), %d tokens, %d experts sharded contiguously across P ranks.\n",
           TB, EB);
    printf("Tokens whose combined output differs from the P=1 left-to-right sum:\n");
    int Ks[] = {2, 3, 4, 8};
    int PBs[] = {2, 4, 8, 16, 32};
    printf("%-8s", "top-k");
    for (int P : PBs) printf("P=%-8d", P);
    printf("\n");
    for (int k : Ks) {
        printf("%-8d", k);
        for (int P : PBs) printf("%-10d", countCombineMismatches(k, P));
        printf("\n");
    }

    printf("\nPart A: once capacity is enforced per group (per rank), the SAME "
           "tokens and the SAME gate give different outputs for different P -- "
           "which tokens overflow depends on which tokens share a capacity "
           "budget. Part B: with top-2 routing a token's combine is a 2-term "
           "sum, and IEEE 754 addition is commutative, so no grouping can change "
           "it; at top-3 and beyond the grouping of the terms matters and the "
           "result depends on how experts are spread across ranks.\n");
    return 0;
}
