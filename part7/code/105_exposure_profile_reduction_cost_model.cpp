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
