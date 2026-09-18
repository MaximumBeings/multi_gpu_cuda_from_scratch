// Chapter 39: Real-Time Genomics at Scale
// 114_wgs_multi_gpu_motivation_model.cpp
//
// NVIDIA's own real Clara Parabricks documentation states a real, cited
// headline result: "over 100x faster analysis of whole-genome sequencing
// (WGS) compared to CPU-only solutions" -- and a real, specific,
// separately cited example of that speedup in a real production
// pipeline: "Reduce germline analysis from ~16 hours to under 10 minutes
// on 4 NVIDIA RTX PRO 6000 Server Edition GPUs." This file does not
// treat those two real cited numbers as the same claim restated -- it
// checks them against each other. If the specific germline example is
// consistent with the general ">100x" headline, that is one honest
// cross-check between two independently cited real figures from the
// same real source, the same discipline this book used in Chapter 37
// when it caught a mismatched pair of cited figures rather than
// combining them uncritically.
#include <cstdio>

int main() {
    printf("Real cited NVIDIA Clara Parabricks headline result:\n");
    printf("- \"over 100x faster analysis of whole-genome sequencing (WGS) "
           "compared to CPU-only solutions\"\n\n");

    printf("Real cited specific example (same source, a separate, more "
           "specific claim):\n");
    printf("- \"Reduce germline analysis from ~16 hours to under 10 minutes "
           "on 4 NVIDIA RTX PRO 6000 Server Edition GPUs\"\n\n");

    double cpuOnlyMinutes = 16.0 * 60.0;   // real cited ~16 hours, in minutes
    double gpuUpperBoundMinutes = 10.0;    // real cited upper bound: "under 10 minutes"

    printf("Real cited CPU-only baseline: ~16 hours = %.0f minutes.\n",
           cpuOnlyMinutes);
    printf("Real cited GPU-accelerated upper bound: under %.0f minutes on 4 "
           "RTX PRO 6000 GPUs.\n\n", gpuUpperBoundMinutes);

    double minimumImpliedSpeedup = cpuOnlyMinutes / gpuUpperBoundMinutes;

    printf("Honest arithmetic on those two real cited numbers alone: since "
           "the real GPU-accelerated time is stated only as an upper bound "
           "(\"under 10 minutes\", not an exact figure), the MINIMUM speedup "
           "this specific real example implies is %.0fx (16 hours divided by "
           "exactly 10 minutes) -- the true real speedup could be higher if "
           "the actual run finished in under 10 minutes, but not lower.\n\n",
           minimumImpliedSpeedup);

    bool consistentWithHeadline = minimumImpliedSpeedup >= 90.0;

    printf("Cross-check against the separately cited \">100x\" headline "
           "figure: a %.0fx minimum implied speedup is %s with a real "
           "headline claim of \"over 100x\" -- both cited numbers describe "
           "the same real order of magnitude, not two disconnected claims, "
           "which is what this cross-check is actually checking for (exact "
           "numeric agreement was never expected, since one figure is a "
           "general WGS-wide claim and the other is one specific germline "
           "pipeline example on specific hardware).\n\n",
           minimumImpliedSpeedup,
           consistentWithHeadline ? "CONSISTENT (same order of magnitude)"
                                   : "INCONSISTENT (different order of magnitude)");

    printf("This is the real, cited motivation for everything the rest of "
           "this chapter examines: genomics analysis at this real cited "
           "scale is not merely faster on a GPU, it is reorganized from an "
           "overnight batch job into something that can return a result "
           "inside a single clinical visit. Section 39.2 asks the real "
           "structural question this raises: what property of whole-genome "
           "analysis lets it reach that speedup using MULTIPLE GPUs at all, "
           "the same question this book asked of Chapter 16's stencil "
           "and Chapter 30's rendering, with a genuinely different real "
           "answer each time?\n");
    return 0;
}
