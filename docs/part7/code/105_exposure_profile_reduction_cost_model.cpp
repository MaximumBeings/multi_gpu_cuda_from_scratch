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
// years, T=360 dates. This file asks what computing a full exposure
// PROFILE costs in communication, compared with Chapter 31's single
// reduction -- both if the T per-date reductions are issued one at a
// time, and if they are batched into one reduction of a T-entry vector.
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

    // Per-date message size: each rank contributes a tiny (count, mean,
    // M2) tuple per date -- 3 x 8 bytes = 24 bytes.
    long long bytesPerRankPerDate = 24;

    int Ps[] = {8, 16, 32, 64};  // real STAC-A2 GPU counts (8x V100/A100 cited)
    printf("%-6s %-18s %-22s %-24s %-22s\n", "P", "Off-rank fraction",
           "Bytes/date on network", "Unbatched: T reductions", "Batched: 1 reduction");
    for (int P : Ps) {
        // In a reduction among P ranks, the (P-1) non-root ranks' tuples
        // must leave their origin rank: (P-1)/P of the P*24 bytes.
        double offRankFraction = (P > 1) ? (double)(P - 1) / (double)P : 0.0;
        double bytesPerDate = (double)P * (double)bytesPerRankPerDate * offRankFraction;
        printf("%-6d %-18.4f %-22.1f %-24s %-22s\n", P, offRankFraction, bytesPerDate,
               "360 x 24 B msgs/rank", "1 x 8640 B msg/rank");
    }

    printf("\nTwo things this table shows. First, the bytes on the network "
           "per date grow LINEARLY with P (24 x (P-1): 168 B at P=8, 1512 B "
           "at P=64), because every additional rank contributes its own "
           "tuple. Second, the T dates are independent reductions over the "
           "same P ranks, so nothing forces them to be separate collectives: "
           "packing all T=%d (count, mean, M2) tuples into one %d-byte "
           "vector per rank turns the profile into ONE reduction. Issued "
           "one date at a time, the profile pays the per-collective latency "
           "T=%d times; batched, it pays it once, and the extra bytes are "
           "still tiny. So a full exposure profile is not \"T times Chapter "
           "31's cost\" in any implementation that batches the dates -- "
           "the real cost of XVA is the T-fold larger SIMULATION work, "
           "not the reduction.\n", realObservationDates,
           realObservationDates * (int)bytesPerRankPerDate, realObservationDates);
    return 0;
}
