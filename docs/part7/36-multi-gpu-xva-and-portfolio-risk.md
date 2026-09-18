**What you will understand after this chapter:** why computing a real bank's XVA (CVA/DVA/FVA and related valuation adjustments) is not a new multi-GPU technique but the SAME Chapter 31 Monte Carlo risk pipeline repeated independently across many future observation dates, rather than computed once at a single terminal date; why real "netting sets" require summing a group of trades together BEFORE applying a zero floor, a real, provable inequality that is the entire economic reason netting reduces counterparty exposure; and why, when Chapter 31's own real cross-rank reduction formula is reused once per observation date, even the exposure profile's own MEAN -- not only its variance -- turns out to be GPU-count-dependent, a sharper version of a finding this book has now made four times.

**What you need to know first:** Chapter 8 (broadcast/reduce, the origin of this book's floating-point associativity theme), Chapter 9 (the real round-count formula reused directly in this chapter's own cost model), Chapter 12 (data parallelism, the path-splitting technique reused unchanged), and Chapter 31 (Monte Carlo risk simulation, the real Chan/Golub/LeVeque pairwise combination formula this chapter reuses without modification).

---

Real banks do not compute a single Value-at-Risk number and stop -- regulators and internal risk desks also require XVA, a family of real "Valuation Adjustments" (CVA for counterparty credit risk, DVA, FVA, and others) that price the real cost of counterparty default, funding, and capital across a bank's entire derivatives book. Real NVIDIA-adjacent GPU-accelerated risk platforms, benchmarked publicly via the real STAC-A2 benchmark ("a risk management benchmark created by leading global banks working with... STAC"), report real multi-GPU speedups on exactly this workload -- an NVIDIA-submitted result on 8 real Tesla V100 GPUs achieved "8.9x the next best throughput," and a later result on 8 real A100 GPUs, running "Monte Carlo estimation of Heston-based Greeks for a path-dependent, multi-asset option with early exercise," achieved "3.0x the throughput" over the prior best. This chapter asks what, structurally, is actually different about this workload compared to Chapter 31's own Monte Carlo VaR calculation -- and the honest answer is: less than it looks. Real Oracle Financial Services Analytics documentation describes computing "Potential Future Exposure (PFE), Expected Positive Exposure EPE... reported as a table of values for user-specified future observation dates," using "American Monte Carlo techniques" -- a full exposure PROFILE across many future dates, not one terminal number, plus a new per-path aggregation step across a "netting set" of trades. This chapter builds both real extensions on top of Chapter 31's own real machinery, unchanged.

```text
+------------------------------------------------------------------+
| Chapter 31: ONE terminal payoff -> ONE reduction (mean/var/VaR)   |
|   [ paths across P GPUs ] -> combine once -> a single risk number |
+------------------------------------------------------------------+
| Chapter 36: a full EXPOSURE PROFILE -> T independent reductions   |
|   [ paths across P GPUs ] -> combine once PER observation date    |
|     date 1 -> EPE(1)   date 2 -> EPE(2)  ...  date T -> EPE(T)    |
+------------------------------------------------------------------+
| New within each path: NETTING across trades in a netting set,     |
| BEFORE any cross-rank reduction -- sum the trades, THEN floor     |
+------------------------------------------------------------------+
```

## 36.1 From One Reduction to a Full Exposure Profile

### Intuition

Chapter 31's own Monte Carlo risk simulation asked one question of its paths: what is the portfolio worth at a single future date, and what is the distribution of that one number across paths? Real XVA calculation asks the same question many times over -- picture a bank checking in on a loan's remaining risk not just once at maturity, but on a fixed monthly schedule for its entire life, because a counterparty could default at any point along the way, not only at the very end. Real Oracle Financial Services Analytics documentation makes the real observation-date grid explicit and concrete: "Observation dates are the future dates on which you want to calculate PFE... Specify the observation date in Period <Days, Months, Year>: Interval <Days, Months, Year> format. For example, 30Y:1M, indicates that one observation date will be generated every month till 30 years" -- a real, specific T=360 monthly checkpoints. Nothing about HOW one checkpoint's own risk number is computed changes; only the number of times it must be computed changes.

!!! warning "[COMMON TRAP] Assuming a multi-date exposure profile needs a fundamentally new parallel algorithm"
    XVA sounds, on the surface, like a much harder computational problem than Chapter 31's own single-date VaR -- and in wall-clock terms it is, simply because there is more work. But the real communication PATTERN at each individual date is identical to Chapter 31's own single reduction: paths remain independent until that date's own statistics are combined. File 105's own cost model shows the honest picture -- the real scaling variable is T (the observation-date count), not a new algorithm, and not even a larger P.

### Background

```text
+----------------------------------------------------------+
| Ch31: T=1 observation point -> 1 reduction, cost paid once  |
+----------------------------------------------------------+
| Ch36: T=360 observation dates (real cited "30Y:1M" tenor)   |
|   date 1 reduction, date 2 reduction, ... date 360 reduction|
|   same Ch9 (P-1)/P formula, same tiny per-date payload       |
|   total real cost = T x Chapter 31's own single-reduction    |
|   cost -- a new SCALING AXIS, not a new TECHNIQUE             |
+----------------------------------------------------------+
```

File 105 reuses Chapter 9's own real round-count formula (already reused once, unchanged, by Chapter 31) to show that a real full exposure profile's total communication cost is exactly T copies of Chapter 31's own single-reduction cost, using the real cited "30Y:1M" tenor structure (T=360) and real STAC-A2-scale GPU counts.

```cpp
// Chapter 36: Multi-GPU XVA and Portfolio Risk
// 105_exposure_profile_reduction_cost_model.cpp
//
// Chapter 31's own Monte Carlo risk simulation needed exactly ONE
// cross-rank reduction: paths were simulated independently, then a
// single terminal payoff was reduced (via Chan/Golub/LeVeque's own real
// pairwise formula) into one mean/variance/VaR. Real XVA calculation
// asks for something structurally larger, not structurally different:
// NVIDIA/Oracle's own real Financial Services Analytics documentation
// describes computing "Potential Future Exposure (PFE), Expected
// Positive Exposure EPE... reported as a table of values for user-
// specified future observation dates," using "American Monte Carlo
// techniques, for a user-specified number of Monte Carlo paths and set
// of future observation dates" -- and gives a real illustrative tenor
// example, "30Y:1M," meaning one observation date every month for 30
// years, T=360 dates. This file reuses Chapter 9's own real (P-1)/P
// round-count formula (Chapter 31's own tool) to show that computing a
// full exposure PROFILE costs exactly T copies of Chapter 31's own
// single-reduction cost -- the same technique this book already built,
// repeated along a new axis (time), not a new technique.
#include <cstdio>

int main() {
    // Real illustrative tenor structure, directly from Oracle's own
    // real documented example: one observation date per month for 30
    // years.
    int realObservationDates = 30 * 12;  // T = 360, "30Y:1M"

    printf("Real cited exposure-profile structure (Oracle Financial "
           "Services Analytics documentation): \"30Y:1M\" -> one "
           "observation date per month for 30 years -> T=%d real future "
           "observation dates, each needing its own EPE/PFE reduction "
           "across all simulated paths.\n\n", realObservationDates);

    // Illustrative per-date message size: each rank contributes a tiny
    // (count, mean, M2) tuple per date -- Chapter 31's own real
    // Chan/Golub/LeVeque combination payload, 3 doubles = 24 bytes.
    long long bytesPerRankPerDate = 24;

    int Ps[] = {8, 16, 32, 64};  // real STAC-A2 GPU counts (8x V100/A100 cited)
    printf("%-6s %-24s %-28s %-28s\n", "P", "Off-rank frac (Ch9)",
           "Bytes/date crossing network", "Total bytes crossing (all T dates)");
    for (int P : Ps) {
        // Chapter 9's own real formula, reused unchanged from Chapter
        // 31: in a full reduction among P ranks, (P-1)/P of the payload
        // that must be combined has to leave its origin rank.
        double offRankFraction = (P > 1) ? (double)(P - 1) / (double)P : 0.0;
        double bytesPerDate = (double)P * (double)bytesPerRankPerDate * offRankFraction;
        double totalBytes = bytesPerDate * (double)realObservationDates;
        printf("%-6d %-24.4f %-28.1f %-28.1f\n", P, offRankFraction, bytesPerDate, totalBytes);
    }

    printf("\nEvery column reuses Chapter 9's own real formula exactly as "
           "Chapter 31 already applied it -- nothing new is invented here. "
           "The ONLY new variable Chapter 31's own cost model never had is "
           "T, the observation-date count: a Chapter-31-style VaR "
           "calculation pays this reduction cost ONCE (T=1); a real XVA "
           "exposure profile pays it T=%d times, once per real cited "
           "observation date, because each date's EPE/PFE is its own "
           "independent reduction over the same P paths. The per-date "
           "message size stays tiny at any P (a handful of bytes per "
           "rank) -- what actually grows the real cost here is T, not P, "
           "a genuinely different scaling axis than any Ch8-Ch34 "
           "reduction faced.\n", realObservationDates);
    return 0;
}
```

Compile and run (a plain host `.cpp` file with no CUDA/NCCL/MPI/NVSHMEM linkage, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 105_exposure_profile_reduction_cost_model \
    105_exposure_profile_reduction_cost_model.cpp
./105_exposure_profile_reduction_cost_model
```

Locked output:

```
Real cited exposure-profile structure (Oracle Financial Services Analytics documentation): "30Y:1M" -> one observation date per month for 30 years -> T=360 real future observation dates, each needing its own EPE/PFE reduction across all simulated paths.

P      Off-rank frac (Ch9)      Bytes/date crossing network  Total bytes crossing (all T dates)
8      0.8750                   168.0                        60480.0                     
16     0.9375                   360.0                        129600.0                    
32     0.9688                   744.0                        267840.0                    
64     0.9844                   1512.0                       544320.0                    

Every column reuses Chapter 9's own real formula exactly as Chapter 31 already applied it -- nothing new is invented here. The ONLY new variable Chapter 31's own cost model never had is T, the observation-date count: a Chapter-31-style VaR calculation pays this reduction cost ONCE (T=1); a real XVA exposure profile pays it T=360 times, once per real cited observation date, because each date's EPE/PFE is its own independent reduction over the same P paths. The per-date message size stays tiny at any P (a handful of bytes per rank) -- what actually grows the real cost here is T, not P, a genuinely different scaling axis than any Ch8-Ch34 reduction faced.
```

## 36.2 Netting Sets: Why the Order of Aggregation Changes Exposure

### Intuition

Picture a bank and one counterparty holding five separate derivative contracts with each other. On some of those contracts the bank is currently ahead; on others, behind. If the counterparty defaulted tomorrow, would the bank's real loss be the sum of its losses on each LOSING contract alone, ignoring the contracts where it was ahead -- or would it be allowed to offset its gains against its losses first, and only worry about the NET shortfall? Real legal netting agreements between counterparties allow the second, better-for-the-bank outcome, and real XVA documentation is specific about the computational order this implies. NVIDIA/Oracle's own real Financial Services Analytics documentation defines a netting set as "the level at which all the counterparty risk measures are calculated directly from the results of netting exposures from different trades," and defines "Positive Netted" exposure precisely as "the sum of positive exposures calculated AT THE NETTING SET LEVEL" -- sum the trades first, floor the total at zero second. A real, independent paper on XVA confirms the same structure directly: "the derivative portfolio of the bank is partitioned into bilateral netting sets of contracts which are jointly collateralized," and "the CVA of the bank can then be computed as the sum of its CVAs restricted to each netting set."

!!! warning "[COMMON TRAP] Assuming a per-trade floor and a netting-set-level floor give the same exposure"
    It is tempting to treat "floor each trade at zero, then sum" and "sum the trades, then floor once" as two equivalent ways of writing the same computation. They are not, and the real economic value of a netting agreement depends on that difference. File 106 verifies, by direct simulation across 500 random portfolios, a real and provable inequality: max(0, a+b) is never greater than max(0,a) + max(0,b), for any real numbers a and b, because max(0, x) is a convex function. Netting first can only reduce or match gross exposure -- never increase it.

### Background

```text
+----------------------------------------------------------+
| Gross (floor-then-sum): max(0,t1)+max(0,t2)+...+max(0,t5)  |
|   each trade floored SEPARATELY, then added                |
+----------------------------------------------------------+
| Netted (sum-then-floor): max(0, t1+t2+t3+t4+t5)             |
|   trades summed FIRST, floored ONCE at the netting-set level|
+----------------------------------------------------------+
| Real convexity inequality: netted never exceeds gross         |
|   (File 106's own 500-trial simulation: 0 violations)         |
+----------------------------------------------------------+
```

File 106 generates 500 random netting sets of five trades each, computes both the gross (floor-then-sum) and netted (sum-then-floor) exposure for every set, and checks that the real convexity inequality never once reverses.

```cpp
// Chapter 36: Multi-GPU XVA and Portfolio Risk
// 106_netting_set_exposure_order_simulation.cpp
//
// Real XVA documentation is specific about WHERE aggregation happens
// before a positive-exposure floor is applied. NVIDIA/Oracle's own real
// Financial Services Analytics documentation defines a netting set as
// "the level at which all the counterparty risk measures are calculated
// directly from the results of netting exposures from different
// trades," and separately defines "Positive Netted" exposure as "the
// sum of positive exposures calculated at the netting set level" -- the
// floor is applied AFTER the trades are summed, not before. A real,
// independent arXiv paper on XVA ("XVA Analysis From the Balance
// Sheet," arXiv:2009.00368) confirms the same real structure: "the
// derivative portfolio of the bank is partitioned into bilateral
// netting sets of contracts which are jointly collateralized," and "the
// CVA of the bank can then be computed as the sum of its CVAs
// restricted to each netting set." This ordering is not a stylistic
// choice: max(0, x) is a convex function, so for any two real numbers,
// max(0, a+b) <= max(0, a) + max(0, b) always (a real, provable
// inequality, not merely typically true) -- netting FIRST, then
// flooring the NET result at zero, can never produce a LARGER exposure
// than flooring each trade separately and summing the floors. This file
// verifies that real inequality by direct simulation across many
// randomly generated netting sets, quantifies the typical real
// reduction, and checks that it never once reverses.
#include <cstdio>
#include <cmath>
#include <vector>

const int TRADES_PER_NETTING_SET = 5;
const int NUM_NETTING_SETS = 500;  // many independent random trials

// Deterministic counter-based hash standing in for a per-trade,
// per-scenario mark-to-market value -- can be positive (counterparty
// owes the bank) or negative (the bank owes the counterparty).
double tradeMarkToMarket(int setIdx, int tradeIdx) {
    unsigned int h = (unsigned int)(setIdx * 97 + tradeIdx * 13 + 1);
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    // Map to a signed range, e.g. [-50, +50), so gains and losses both
    // occur across trades within the same netting set.
    return (double)((long long)(h % 10000)) / 100.0 - 50.0;
}

int main() {
    printf("Real cited rule (Oracle FSA docs): \"Positive Netted\" exposure "
           "= sum of trades WITHIN a netting set, THEN floored at zero -- "
           "not each trade floored separately and then summed.\n\n");

    double totalGross = 0.0;      // sum(max(0, trade_i)) -- floor-then-sum
    double totalNetted = 0.0;     // max(0, sum(trade_i))  -- sum-then-floor
    int violations = 0;           // count of any case where netted > gross

    printf("%-10s %-16s %-16s %-12s\n", "Set #", "Gross exposure",
           "Netted exposure", "Reduction");
    for (int s = 0; s < NUM_NETTING_SETS; s++) {
        double sumOfTrades = 0.0;
        double sumOfFloors = 0.0;
        for (int t = 0; t < TRADES_PER_NETTING_SET; t++) {
            double v = tradeMarkToMarket(s, t);
            sumOfTrades += v;
            sumOfFloors += std::max(0.0, v);
        }
        double grossExposure = sumOfFloors;               // floor each trade, then sum
        double nettedExposure = std::max(0.0, sumOfTrades); // sum trades, then floor once

        if (nettedExposure > grossExposure + 1e-9) violations++;

        totalGross += grossExposure;
        totalNetted += nettedExposure;

        if (s < 8) {  // print only the first few sets in full, for readability
            double reduction = grossExposure > 0.0
                ? 100.0 * (1.0 - nettedExposure / grossExposure) : 0.0;
            printf("%-10d %-16.4f %-16.4f %-11.1f%%\n", s, grossExposure,
                   nettedExposure, reduction);
        }
    }

    printf("\n(remaining %d netting sets computed but not printed "
           "individually)\n\n", NUM_NETTING_SETS - 8);

    printf("Across all %d randomly generated netting sets (%d trades "
           "each):\n", NUM_NETTING_SETS, TRADES_PER_NETTING_SET);
    printf("Total gross exposure (floor-then-sum):  %.4f\n", totalGross);
    printf("Total netted exposure (sum-then-floor): %.4f\n", totalNetted);
    printf("Overall reduction from netting: %.2f%%\n",
           100.0 * (1.0 - totalNetted / totalGross));
    printf("Cases where netted exposure exceeded gross exposure: %d "
           "(the real convexity inequality max(0,a+b) <= max(0,a)+max(0,b) "
           "predicts this must always be zero).\n\n", violations);

    printf("This confirms, by direct simulation rather than assumption, "
           "the real economic claim behind a netting agreement: netting "
           "trades BEFORE applying the positive-exposure floor can only "
           "ever reduce or match counterparty exposure, never increase "
           "it. Unlike every earlier non-associativity finding in this "
           "book (Chapter 8/25/29/31/32/35 -- all about floating-point "
           "SUMMATION order across ranks or transpose phases), this "
           "chapter's own order-sensitivity is about which NONLINEAR "
           "operation (the zero-floor) is applied at which STAGE of a "
           "single path's own local computation -- a real economic "
           "distinction, not a floating-point rounding one, and the "
           "entire reason netting sets exist as a real risk-management "
           "concept.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 106_netting_set_exposure_order_simulation \
    106_netting_set_exposure_order_simulation.cpp
./106_netting_set_exposure_order_simulation
```

Locked output:

```
Real cited rule (Oracle FSA docs): "Positive Netted" exposure = sum of trades WITHIN a netting set, THEN floored at zero -- not each trade floored separately and then summed.

Set #      Gross exposure   Netted exposure  Reduction   
0          32.8400          0.0000           100.0      %
1          34.5900          20.9100          39.5       %
2          38.4400          0.0000           100.0      %
3          43.2600          0.0000           100.0      %
4          14.3100          0.0000           100.0      %
5          42.1800          0.0000           100.0      %
6          6.6300           0.0000           100.0      %
7          84.4400          68.4000          19.0       %

(remaining 492 netting sets computed but not printed individually)

Across all 500 randomly generated netting sets (5 trades each):
Total gross exposure (floor-then-sum):  31101.2000
Total netted exposure (sum-then-floor): 13026.1900
Overall reduction from netting: 58.12%
Cases where netted exposure exceeded gross exposure: 0 (the real convexity inequality max(0,a+b) <= max(0,a)+max(0,b) predicts this must always be zero).

This confirms, by direct simulation rather than assumption, the real economic claim behind a netting agreement: netting trades BEFORE applying the positive-exposure floor can only ever reduce or match counterparty exposure, never increase it. Unlike every earlier non-associativity finding in this book (Chapter 8/25/29/31/32/35 -- all about floating-point SUMMATION order across ranks or transpose phases), this chapter's own order-sensitivity is about which NONLINEAR operation (the zero-floor) is applied at which STAGE of a single path's own local computation -- a real economic distinction, not a floating-point rounding one, and the entire reason netting sets exist as a real risk-management concept.
```

## 36.3 Computing an Exposure Profile Across GPUs

### Intuition

Section 36.1 established that a real exposure profile needs Chapter 31's own reduction repeated T times. Section 36.2 established that netting a path's own trades has to happen locally, in a specific order, before that path's own contribution is ready to be combined with anyone else's. Put the two together and the real multi-GPU pipeline looks almost exactly like Chapter 31's own: split paths across P GPUs (Chapter 12's technique, unchanged), have each GPU net its own paths' own trades locally at every observation date (Section 36.2's rule -- entirely local, since a netting set's own handful of trades always lives wherever that path was assigned, this book's own recurring "keep dependent data together" principle from Chapter 16 and Chapter 27), and only then run Chapter 31's own real cross-rank combination -- once per date. Running that real combination formula once per date, rather than once total, turns out to surface something Chapter 31 itself never saw: because real exposure values are continuous from the start (unlike Chapter 31's own deliberately-integer test payoff), even the profile's own MEAN, not merely its variance, becomes GPU-count-dependent.

!!! warning "[COMMON TRAP] Assuming Chapter 31's own integer-payoff finding (sum and count match, only variance mismatches) generalizes to real, continuous exposure values"
    Chapter 31 kept its own payoff deliberately integer specifically so that its combined path count and sum would be exactly reproducible at every P, isolating the mismatch to variance alone. Real XVA exposure is not integer -- it is a continuous mark-to-market value from the start. File 107 shows that once the payoff itself is continuous, the incremental mean computation inside Chapter 31's own real Chan/Golub/LeVeque formula (dividing by a running count at every fold) is itself a non-associative floating-point computation: the EPE mean itself, not only a variance-like statistic, differs by a few ULPs across different P at several tested observation dates.

### Background

```text
+----------------------------------------------------------+
| Per rank (Ch12's own path-split, unchanged): own paths      |
|   net own trades locally, EVERY date (Section 36.2's rule)  |
+----------------------------------------------------------+
| Cross-rank (Ch31's own Chan/Golub/LeVeque formula, reused): |
|   combine (count, mean, M2) -> repeated once PER date        |
+----------------------------------------------------------+
| Found: EPE itself (not just variance) is P-dependent here    |
|   because exposure is continuous, unlike Ch31's integer test |
+----------------------------------------------------------+
```

File 107 combines Sections 36.1 and 36.2 into one simulation: 24 paths, 5 trades per netting set, 6 observation dates (illustrative, for a readable printed profile -- Section 36.1's own cost model already exercised the real cited T=360 at scale), split across P in {1, 2, 3, 4, 6, 8, 12, 24}, checked against a P=1 reference built from the exact same real combination formula.

```cpp
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
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 107_multi_gpu_exposure_profile_correctness_simulation \
    107_multi_gpu_exposure_profile_correctness_simulation.cpp
./107_multi_gpu_exposure_profile_correctness_simulation
```

Locked output:

```
Reference EPE profile (P=1: one rank folds all 24 paths, in order, via Chapter 31's own real Chan/Golub/LeVeque combination formula):
Date     Reference EPE 
0        18.9762500000 
1        26.7516666667 
2        29.5337500000 
3        26.2037500000 
4        16.2587500000 
5        38.2633333333 

P      date0   max|diff| date1   max|diff| date2   max|diff| date3   max|diff| date4   max|diff| date5   max|diff|
1      0                 0                 0                 0                 0                 0                
2      3.55e-15          3.55e-15          3.55e-15          3.55e-15          3.55e-15          0                
3      3.55e-15          3.55e-15          0                 0                 0                 7.11e-15         
4      3.55e-15          3.55e-15          3.55e-15          3.55e-15          0                 0                
6      0                 0                 0                 0                 0                 7.11e-15         
8      3.55e-15          3.55e-15          0                 0                 3.55e-15          7.11e-15         
12     3.55e-15          0                 0                 3.55e-15          0                 0                
24     0                 0                 0                 0                 0                 0                

At least one (P, date) combination produced a nonzero difference from the P=1 reference at P>1 -- a real floating-point reassociation mismatch, consistent with this book's own recurring Chapter 8/25/29/31/32 finding, but now a SHARPER version of it: Chapter 31's own payoff was deliberately kept INTEGER, so its combined path count and sum matched exactly at every P, and only its variance (M2) mismatched. Here, mark-to-market exposure is genuinely continuous from the start, so even the MEAN (EPE) itself -- not only a variance -- becomes P-dependent under this real combination formula, because computing a mean incrementally (dividing by a running count at every fold) is itself a non-associative floating-point computation, unlike a flat integer sum. Section 36.1's own cost model already showed why this matters at scale: this exact combination step runs T=360 real times (Oracle's own "30Y:1M" tenor structure) for one production exposure profile, not once as it did for Chapter 31's own single VaR calculation -- so any such mismatch recurs at every one of those real observation dates, not just once.
```

## Chapter Summary

Real XVA calculation looked, at the start of this chapter, like it might need a genuinely new multi-GPU technique -- and it does not. File 105 showed that a full exposure profile is exactly Chapter 31's own single reduction, repeated T times along a new axis (real observation dates, T=360 under Oracle's own real cited "30Y:1M" tenor structure), using Chapter 9's own real round-count formula completely unchanged. File 106 introduced this chapter's one genuinely new idea: a netting set's trades must be summed BEFORE a zero floor is applied, not floored individually and then summed, and this ordering is not arbitrary -- it follows directly from the real, provable convexity inequality max(0, a+b) <= max(0, a) + max(0, b), verified across 500 random portfolios with zero violations and a real 58% average exposure reduction from netting. File 107 combined both real extensions into the actual production pipeline -- Chapter 12's own path-splitting, Section 36.2's own local netting rule, and Chapter 31's own real combination formula reused once per date -- and found a sharper version of this book's recurring floating-point theme: because real exposure is continuous rather than Chapter 31's own deliberately integer test payoff, even the exposure profile's own mean became GPU-count-dependent, not only its variance.

## Self-Check Questions

1. What real structural extension does Chapter 36 make to Chapter 31's own Monte Carlo risk simulation, and what stays completely unchanged?
2. According to File 105's own locked output, what real, cited number does T represent, and where does it come from?
3. Why does computing a full exposure profile cost T times Chapter 31's own single-reduction cost, rather than needing a fundamentally new communication pattern?
4. What is a netting set, according to the real cited Oracle and arXiv sources, and why does the order of summing trades and applying the zero-floor matter?
5. What real mathematical property of the function max(0, x) guarantees that netted exposure can never exceed gross exposure? State it in your own words.
6. File 106 found zero violations of the netted-less-than-or-equal-to-gross inequality across 500 random trials. Was this a coincidence of the random data, or something that must always hold? Explain.
7. Why does the netting-set aggregation in File 107 need no cross-rank communication at all, even in a real multi-GPU deployment?
8. File 107 found the EPE itself -- not just a variance-like statistic -- was P-dependent, unlike Chapter 31's own integer-payoff test. What real difference between the two files' payoffs explains this?
9. If a bank's real production system needed the exact same EPE regardless of how many GPUs it used, what would this chapter's own findings suggest they should do differently, if anything?

## Where We Go Next

Chapter 37 extends Chapter 21's own GPUDirect RDMA to capital markets, where the same bypass-the-host technique this book already built for GPU-to-GPU transfers turns out to matter for a very different real reason: shaving microseconds off market-data delivery in latency-sensitive trading.

## Worked Solutions

1. Chapter 36 extends Chapter 31 along a new time axis: instead of computing one terminal-date risk statistic, it computes a full profile of statistics across T future observation dates (a real XVA/PFE exposure profile). What stays completely unchanged is the underlying machinery -- Chapter 12's own path-splitting across GPUs and Chapter 31's own real Chan/Golub/LeVeque pairwise combination formula are both reused without modification; only the number of times that formula is invoked changes.
2. T represents the number of real future observation dates in an exposure profile, and File 105's own locked output uses the real cited value T=360, taken directly from Oracle Financial Services Analytics documentation's own real example tenor structure "30Y:1M" -- one observation date every month for 30 years.
3. Because each observation date's own EPE/PFE reduction is computed exactly the same way Chapter 31 computed its single VaR reduction -- independent paths, combined via the same real formula -- and the T different dates do not interact with or depend on one another's own reduction. Repeating an unchanged operation T independent times costs T times as much, without requiring any new communication PATTERN.
4. A netting set is a group of derivative trades with the same counterparty that are jointly collateralized and can be legally offset against one another upon default -- the real cited sources describe it as "the level at which... counterparty risk measures are calculated directly from the results of netting exposures from different trades" and "partitioned into bilateral netting sets... jointly collateralized." The order matters because summing the trades first (netting gains against losses) before applying the zero floor can only produce a smaller or equal result than flooring each trade separately and then summing -- exactly the real economic benefit a netting agreement is designed to provide.
5. The function max(0, x) is convex, and for any convex function f, f(a+b) is never greater than f(a) + f(b) when applied this way to a sum versus applied separately and added -- concretely, whichever of a and b is negative can only ever reduce the sum a+b before the floor is applied, whereas flooring each of a and b separately at zero first discards that offsetting effect entirely, so the separately-floored-and-summed version can only be greater than or equal to the netted version.
6. It is not a coincidence -- it is something that must always hold, because it follows from a general mathematical property (the convexity of max(0, x)) rather than from any specific property of the randomly generated trade values. The 500 trials were a verification of a claim that holds for ALL real numbers, not a search for a pattern that happened to be true for these particular random values; any set of trade values, however chosen, would produce the same zero-violation result.
7. Because Chapter 12's own path-splitting assigns each path, and everything needed to compute that path's own result, to exactly one GPU -- and a netting set's own handful of trades are all evaluated along the SAME path, so they are always already co-located on whichever single rank owns that path. There is nothing to communicate because the data that needs to be combined was never split across ranks in the first place, echoing the same "keep dependent data together" principle Chapter 16 and Chapter 27 both relied on.
8. Chapter 31 deliberately used an integer-valued test payoff specifically so that combined path counts and sums would be exactly reproducible at every P by ordinary integer arithmetic, isolating any mismatch to the variance (M2) term's own floating-point computation. File 107's real exposure values are continuous (floating-point) mark-to-market values from the very first step, so even the MEAN'S own computation -- built by repeatedly dividing by a running count inside Chapter 31's own real combination formula -- becomes a non-associative floating-point computation whose result can differ by a few ULPs depending on how paths are grouped across ranks.
9. This chapter's own findings suggest the mismatch is small (a few ULPs, around 1e-15 in File 107's own locked output) and would not change any real risk decision, but if a bank's own audit or reconciliation process required BITWISE reproducibility regardless of GPU count, they would need to fix a single canonical combination order (for example, always combining rank results in the same sequence in the same tree shape) rather than relying on "any P produces the same real answer" -- the same practical lesson Chapter 25, Chapter 29, and Chapter 31 each already drew from their own real non-associativity findings.

---

**Sources cited in this chapter:**

- NVIDIA Technical Blog. "A New STAC-A2 Record," developer.nvidia.com, fetched fresh this session. (The real "STAC-A2 is a risk management benchmark created by leading global banks... in order to allow assessment of financial compute solutions" quote, the real 8x Tesla V100 GPU configuration, and the real "8.9x the next best throughput" and related cited speedup figures.)
- STAC (Securities Technology Analysis Center). "STAC Report: STAC-A2 (derivatives risk) on 8 x NVIDIA A100 80GB GPUs and Red Hat OpenShift," docs.stacresearch.com, fetched fresh this session. (The real "Monte Carlo estimation of Heston-based Greeks for a path-dependent, multi-asset option with early exercise" benchmark description and the real cited "3.0x the throughput" result on 8 real A100 GPUs in a DGX A100 server.)
- Kinetica. "How GPUs Have Transformed XVA Pricing and Risk Calculations," kinetica.com/blog, fetched fresh this session. (The real XVA component definitions -- CVA, DVA, FVA -- and the real "compute-intensive workloads like real-time adjustment calculations are perfect for high-performance GPUs" and "running a Monte Carlo simulation... is very compute heavy and much better suited to GPUs than CPUs" quotes.)
- Oracle. "Market Risk - Monte Carlo Simulation, XVA and PFE," Oracle Financial Services Analytics documentation, docs.oracle.com, fetched fresh this session. (The real "Potential Future Exposure (PFE), Expected Positive Exposure EPE... reported as a table of values for user-specified future observation dates" and "American Monte Carlo techniques" quotes, the real "30Y:1M" tenor example, and the real netting-set and "Positive Netted... sum of positive exposures calculated at the netting set level" definitions.)
- Albanese, C. et al. "XVA Analysis From the Balance Sheet." arXiv:2009.00368, fetched fresh this session. (The real "the derivative portfolio of the bank is partitioned into bilateral netting sets of contracts which are jointly collateralized" and "the CVA of the bank can then be computed as the sum of its CVAs restricted to each netting set" quotes.)
- This book's own Chapter 8 (broadcast/reduce, the origin of this book's own floating-point associativity theme), Chapter 9 (the real round-count formula reused directly in File 105), Chapter 12 (data parallelism, the path-splitting technique reused unchanged in File 107), Chapter 16 and Chapter 27 (the "keep dependent data together" principle underlying Section 36.3's own no-communication netting step), and Chapter 31 (Monte Carlo risk simulation, whose real Chan/Golub/LeVeque combination formula and deliberately-integer test payoff this chapter reuses and directly contrasts against).
