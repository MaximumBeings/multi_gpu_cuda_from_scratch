// Chapter 36: Multi-GPU XVA and Portfolio Risk
// 107_multi_gpu_exposure_profile_correctness_simulation.cpp
//
// This file puts Sections 36.1 and 36.2 together into the one real
// multi-GPU pipeline a production XVA engine actually runs: paths are
// split across P GPUs exactly as Chapter 31 already did (embarrassingly
// parallel, no communication needed while simulating), each GPU nets
// its OWN paths' trades locally at every observation date (Section
// 36.2's own sum-then-floor rule -- entirely local, since a netting
// set's own handful of trades always lives together on whichever rank
// owns that path, echoing this book's own "keep dependent data
// together" principle from Chapter 16/27), and only THEN does a real
// cross-rank reduction combine each rank's own local (count, mean, M2)
// summary into the portfolio's own Expected Positive Exposure (EPE) --
// repeated independently at every one of T observation dates, exactly
// as Section 36.1's own cost model predicted.
//
// The question this file asks: does the EPE at each date come out the
// same at every GPU count P? The answer depends on HOW the per-rank
// summaries are combined, so the file runs four combination methods on
// the same paths and compares each one with its own P=1 result:
//
//   A. Incremental fold: every path is folded into its rank's running
//      summary one at a time with mean += delta * nb / n (the update form
//      of Chan/Golub/LeVeque), then rank summaries are folded the same way.
//   B. The same incremental fold, but on exposures rescaled to whole
//      CENTS -- exact integers -- to test whether continuous values are
//      what causes any mismatch.
//   C. Chapter 31's own combination form: each rank computes a local mean
//      in a separate pass (sum / count), and rank summaries are combined
//      as (na*ma + nb*mb) / n.
//   D. Exact fixed-point accumulation: each path's exposure is rounded
//      ONCE to an int64 multiple of 2^-32, ranks carry (count, integer
//      sum), and the mean is divided out once, at the very end. Integer
//      addition is associative, so the grouping across ranks cannot
//      change the sum.
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

const int NUM_PATHS = 24;
const int TRADES_PER_SET = 5;
const int NUM_DATES = 6;  // small T here for a readable printed profile;
                           // Section 36.1's own cost model already used
                           // the real cited T=360 ("30Y:1M") at scale.

// Deterministic per-(path, date, trade) mark-to-market value -- the
// same kind of counter-based hash used throughout this book so every
// rank, and the reference, start from identical, reproducible data.
double tradeMarkToMarket(int path, int date, int trade) {
    unsigned int h = (unsigned int)(path * 9973 + date * 131 + trade * 17 + 3);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return (double)((long long)(h % 10000)) / 100.0 - 50.0;
}

// Section 36.2's own real rule: sum the netting set's own trades FIRST,
// then floor the net result at zero -- entirely local to one path, one
// date, one rank.
double nettedExposure(int path, int date) {
    double sumOfTrades = 0.0;
    for (int t = 0; t < TRADES_PER_SET; t++) sumOfTrades += tradeMarkToMarket(path, date, t);
    return std::max(0.0, sumOfTrades);
}

// Method B's input: the same exposure in whole cents, an exact integer.
double nettedExposureCents(int path, int date) {
    return std::round(nettedExposure(path, date) * 100.0);
}

struct Stats { long long n; double mean; double M2; };

// Method A/B: the incremental (update) form of the pairwise combination.
Stats combineIncremental(const Stats &a, const Stats &b) {
    Stats c;
    c.n = a.n + b.n;
    double delta = b.mean - a.mean;
    c.mean = a.mean + delta * (double)b.n / (double)c.n;
    c.M2 = a.M2 + b.M2 + delta * delta * (double)a.n * (double)b.n / (double)c.n;
    return c;
}

// Method C: Chapter 31's own weighted-mean form.
Stats combineWeighted(const Stats &a, const Stats &b) {
    long long n = a.n + b.n;
    double delta = b.mean - a.mean;
    double mean = ((double)a.n * a.mean + (double)b.n * b.mean) / (double)n;
    double m2 = a.M2 + b.M2 + delta * delta * (double)a.n * (double)b.n / (double)n;
    return {n, mean, m2};
}

// Method D: fixed-point quantum 2^-32 currency units.
const double QUANTUM_SCALE = 4294967296.0;  // 2^32
struct FixedSum { long long n; long long sumQ; };
FixedSum combineFixed(const FixedSum &a, const FixedSum &b) {
    return {a.n + b.n, a.sumQ + b.sumQ};  // exact, associative
}

// Rank r owns paths {r, r+P, r+2P, ...} (this book's own recurring
// striped-ownership convention). Rank summaries are combined in rank
// order 0, 1, ..., P-1.
double epeWithP(int method, int date, int P) {
    if (method == 'D') {
        std::vector<FixedSum> perRank(P, FixedSum{0, 0});
        for (int p = 0; p < NUM_PATHS; p++) {
            long long q = std::llround(nettedExposure(p, date) * QUANTUM_SCALE);
            perRank[p % P] = combineFixed(perRank[p % P], FixedSum{1, q});
        }
        FixedSum total = perRank[0];
        for (int r = 1; r < P; r++) total = combineFixed(total, perRank[r]);
        // sumQ < 2^53, so the conversion is exact and the division by 2^32
        // is exact; only the final division by n rounds -- once.
        return ((double)total.sumQ / QUANTUM_SCALE) / (double)total.n;
    }
    std::vector<Stats> perRank(P, Stats{0, 0.0, 0.0});
    if (method == 'C') {
        // Local pass 1: sum and count; pass 2: M2 around the local mean.
        std::vector<double> sum(P, 0.0);
        std::vector<long long> cnt(P, 0);
        for (int p = 0; p < NUM_PATHS; p++) { sum[p % P] += nettedExposure(p, date); cnt[p % P]++; }
        for (int r = 0; r < P; r++) perRank[r] = Stats{cnt[r], sum[r] / (double)cnt[r], 0.0};
        for (int p = 0; p < NUM_PATHS; p++) {
            double d = nettedExposure(p, date) - perRank[p % P].mean;
            perRank[p % P].M2 += d * d;
        }
        Stats total = perRank[0];
        for (int r = 1; r < P; r++) total = combineWeighted(total, perRank[r]);
        return total.mean;
    }
    for (int p = 0; p < NUM_PATHS; p++) {
        double x = (method == 'B') ? nettedExposureCents(p, date) : nettedExposure(p, date);
        // Fold one path in as a one-element summary.
        perRank[p % P] = combineIncremental(perRank[p % P], Stats{1, x, 0.0});
    }
    Stats total = perRank[0];
    for (int r = 1; r < P; r++) total = combineIncremental(total, perRank[r]);
    return total.mean;
}

int main() {
    int Ps[] = {2, 3, 4, 6, 8, 12, 24};
    const int NP = 7;
    const char *names[] = {
        "A incremental fold, continuous exposure",
        "B incremental fold, whole-cent integers",
        "C Ch31 two-pass local mean + (na*ma+nb*mb)/n",
        "D exact fixed-point (count, int64 sum)"};
    const int methods[] = {'A', 'B', 'C', 'D'};

    printf("EPE vs. the same method's own P=1 result, %d paths, %d dates, "
           "P in {2,3,4,6,8,12,24} (%d (P, date) pairs per method):\n\n",
           NUM_PATHS, NUM_DATES, NP * NUM_DATES);
    printf("%-46s %-16s %-14s\n", "Combination method", "pairs differing", "max |diff|");
    for (int m = 0; m < 4; m++) {
        int differing = 0;
        double maxDiff = 0.0;
        for (int d = 0; d < NUM_DATES; d++) {
            double ref = epeWithP(methods[m], d, 1);
            for (int i = 0; i < NP; i++) {
                double diff = std::fabs(epeWithP(methods[m], d, Ps[i]) - ref);
                if (diff != 0.0) differing++;
                maxDiff = std::max(maxDiff, diff);
            }
        }
        printf("%-46s %-16d %-14.3g\n", names[m], differing, maxDiff);
    }

    printf("\nMethod D, per date (the EPE profile itself, and the largest "
           "difference from P=1 over every tested P):\n");
    printf("%-6s %-16s %-14s\n", "Date", "EPE (P=1)", "max |diff|");
    for (int d = 0; d < NUM_DATES; d++) {
        double ref = epeWithP('D', d, 1);
        double maxDiff = 0.0;
        for (int i = 0; i < NP; i++)
            maxDiff = std::max(maxDiff, std::fabs(epeWithP('D', d, Ps[i]) - ref));
        printf("%-6d %-16.10f %-14.3g\n", d, ref, maxDiff);
    }

    printf("\nMethod B rules out continuity as the cause: whole-cent integer "
           "exposures still give P-dependent means under the incremental fold, "
           "because mean + delta*nb/n rounds at every fold and the folds are "
           "grouped differently at each P. Method C is not immune either: its "
           "local means and the products na*ma each round, so the weighted "
           "combination still depends on grouping in general. Method D carries "
           "an exact integer sum, so the grouping across ranks cannot matter, "
           "and divides once at the end: the EPE is identical at every P. The "
           "cost is one rounding of each path's exposure to 2^-32 -- the same "
           "rounding at every P -- and a variance would need its own exact "
           "second pass. With T=360 dates per profile (Section 36.1), method "
           "D makes every date of the profile reproducible across GPU counts.\n");
    return 0;
}
