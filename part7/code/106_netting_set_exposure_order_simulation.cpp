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
