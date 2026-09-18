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
// as Section 36.1's own cost model predicted. This file reuses Chapter
// 31's own real Chan/Golub/LeVeque pairwise combination formula
// unchanged -- including for the P=1 reference itself, one rank folding
// every path in order, so the ONLY thing that changes between the
// reference and any tested P is how the SAME combination formula is
// grouped, never which formula is used -- applies it once per date, and
// checks the combined EPE at every date against that reference at every
// tested P, continuing this book's own established discipline of
// checking bit-exactness empirically rather than assuming it, whatever
// the result turns out to be.
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

// Chapter 31's own real Chan/Golub/LeVeque pairwise combination formula,
// reused completely unchanged: combine two (count, mean, M2) summaries
// into one, for a single observation date.
struct Stats { long long n; double mean; double M2; };

Stats combine(const Stats &a, const Stats &b) {
    Stats c;
    c.n = a.n + b.n;
    double delta = b.mean - a.mean;
    c.mean = a.mean + delta * (double)b.n / (double)c.n;
    c.M2 = a.M2 + b.M2 + delta * delta * (double)a.n * (double)b.n / (double)c.n;
    return c;
}

// P-rank simulation at one date: rank r owns paths {r, r+P, r+2P, ...}
// (this book's own recurring striped-ownership convention), computes
// its own local (count, mean, M2) over just its own paths, then a
// sequential fold combines all P ranks' summaries into one portfolio-
// level EPE for that date.
double combinedEPEWithP(int date, int P) {
    std::vector<Stats> perRank(P, Stats{0, 0.0, 0.0});
    for (int p = 0; p < NUM_PATHS; p++) {
        int r = p % P;
        double x = nettedExposure(p, date);
        // Fold this one path into rank r's own running (count, mean, M2)
        // via the same pairwise formula, treating the single new path
        // as a one-element summary.
        perRank[r] = combine(perRank[r], Stats{1, x, 0.0});
    }
    Stats total = perRank[0];
    for (int r = 1; r < P; r++) total = combine(total, perRank[r]);
    return total.mean;
}

int main() {
    printf("Reference EPE profile (P=1: one rank folds all %d paths, in "
           "order, via Chapter 31's own real Chan/Golub/LeVeque "
           "combination formula):\n", NUM_PATHS);
    std::vector<double> reference(NUM_DATES);
    printf("%-8s %-14s\n", "Date", "Reference EPE");
    for (int d = 0; d < NUM_DATES; d++) {
        reference[d] = combinedEPEWithP(d, 1);
        printf("%-8d %-14.10f\n", d, reference[d]);
    }

    int Ps[] = {1, 2, 3, 4, 6, 8, 12, 24};
    printf("\n%-6s", "P");
    for (int d = 0; d < NUM_DATES; d++) printf(" date%-3d max|diff|", d);
    printf("\n");

    bool anyMismatch = false;
    for (int P : Ps) {
        printf("%-6d", P);
        for (int d = 0; d < NUM_DATES; d++) {
            double combined = combinedEPEWithP(d, P);
            double diff = std::fabs(combined - reference[d]);
            if (diff != 0.0) anyMismatch = true;
            printf(" %-17.3g", diff);
        }
        printf("\n");
    }

    if (!anyMismatch) {
        printf("\nEvery P reproduces the exact same EPE at every observation "
               "date, to the last bit. The per-date reduction here combines "
               "each rank's own (count, mean, M2) summary via Chapter 31's "
               "own real formula, and empirically that reassociation left "
               "the MEAN bit-exact at this problem size -- unlike Chapter "
               "31's own M2/variance finding, which mismatched at P>1. "
               "Section 36.1's own cost model already showed the real "
               "reason this matters at scale: this exact combination step "
               "runs T=360 real times (Oracle's own \"30Y:1M\" tenor "
               "structure) for one production exposure profile, not once "
               "as it did for Chapter 31's own single VaR calculation.\n");
    } else {
        printf("\nAt least one (P, date) combination produced a nonzero "
               "difference from the P=1 reference at P>1 -- a real "
               "floating-point reassociation mismatch, consistent with "
               "this book's own recurring Chapter 8/25/29/31/32 finding, "
               "but now a SHARPER version of it: Chapter 31's own payoff "
               "was deliberately kept INTEGER, so its combined path count "
               "and sum matched exactly at every P, and only its variance "
               "(M2) mismatched. Here, mark-to-market exposure is "
               "genuinely continuous from the start, so even the MEAN "
               "(EPE) itself -- not only a variance -- becomes P-dependent "
               "under this real combination formula, because computing a "
               "mean incrementally (dividing by a running count at every "
               "fold) is itself a non-associative floating-point "
               "computation, unlike a flat integer sum. Section 36.1's own "
               "cost model already showed why this matters at scale: this "
               "exact combination step runs T=360 real times (Oracle's own "
               "\"30Y:1M\" tenor structure) for one production exposure "
               "profile, not once as it did for Chapter 31's own single "
               "VaR calculation -- so any such mismatch recurs at every "
               "one of those real observation dates, not just once.\n");
    }
    return 0;
}
