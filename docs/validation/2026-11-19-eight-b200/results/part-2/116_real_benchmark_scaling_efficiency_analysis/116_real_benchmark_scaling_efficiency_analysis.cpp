// Chapter 39: Real-Time Genomics at Scale
// 116_real_benchmark_scaling_efficiency_analysis.cpp
//
// Section 39.2 established that Parabricks' own real pipeline stages are
// ALGORITHMICALLY embarrassingly parallel, by the real cited nature of
// independent reads and independent genomic regions -- zero cross-worker
// communication is required, at any P. This file asks the honest
// follow-up question: does that real algorithmic property alone deliver
// near-ideal real-world 2-GPU-to-4-GPU scaling? NVIDIA's own real
// Clara Parabricks documentation publishes a benchmark table of measured
// wall-clock runtimes (in minutes) for seven real tools on real RTX PRO
// 6000 Server Edition GPUs, at 2 GPUs and at 4 GPUs, explicitly caveated
// by NVIDIA itself: "Speeds may vary depending on the data set, GPU
// instance, host CPU, memory availability, and other factors." This file
// uses those real published numbers DIRECTLY -- no simulation, unlike
// every other file in this chapter and most of this book -- and computes
// each real tool's own observed speedup and what percentage of the ideal
// 2x speedup (doubling the GPU count) it actually achieved.
#include <cstdio>
#include <cstring>

struct BenchmarkRow {
    const char* toolName;
    double minutesAt2Gpu;
    double minutesAt4Gpu;
};

int main() {
    // Real cited benchmark data (NVIDIA Clara Parabricks documentation,
    // "About Parabricks" performance table, RTX PRO 6000 Server Edition
    // column, fetched fresh this session). Minutes, lower is faster.
    BenchmarkRow rows[] = {
        {"FQ2BAM (BWA-MEM, Paired End)",  6.93,  4.68},
        {"Giraffe (Single End)",         13.60, 11.30},
        {"Giraffe (Paired End)",         43.90, 30.62},
        {"DeepVariant (Short-Read)",      7.12,  4.80},
        {"Minimap2",                     15.68, 15.60},
        {"FQ2BAM_Meth (BWA-Meth)",       24.77, 14.68},
        {"RNA_fq2bam (STAR, Melanoma)",   6.05,  5.88},
    };
    int numRows = (int)(sizeof(rows) / sizeof(rows[0]));

    printf("Real cited NVIDIA Clara Parabricks v4.7.0 benchmark table "
           "(RTX PRO 6000 Server Edition, minutes, lower=faster):\n");
    printf("%-32s %10s %10s %12s %14s\n", "Tool", "2 GPUs", "4 GPUs",
           "Speedup", "Pct of ideal 2x");

    double minEfficiency = 1e9, maxEfficiency = -1e9;
    const char* minEfficiencyTool = nullptr;
    const char* maxEfficiencyTool = nullptr;
    double sumEfficiency = 0.0;

    for (int i = 0; i < numRows; i++) {
        double speedup = rows[i].minutesAt2Gpu / rows[i].minutesAt4Gpu;
        double idealSpeedup = 2.0;
        double efficiencyPct = 100.0 * speedup / idealSpeedup;

        printf("%-32s %10.2f %10.2f %11.3fx %13.1f%%\n", rows[i].toolName,
               rows[i].minutesAt2Gpu, rows[i].minutesAt4Gpu, speedup,
               efficiencyPct);

        sumEfficiency += efficiencyPct;
        if (efficiencyPct < minEfficiency) {
            minEfficiency = efficiencyPct;
            minEfficiencyTool = rows[i].toolName;
        }
        if (efficiencyPct > maxEfficiency) {
            maxEfficiency = efficiencyPct;
            maxEfficiencyTool = rows[i].toolName;
        }
    }

    double meanEfficiency = sumEfficiency / numRows;

    printf("\nAcross these %d real cited tools, going from 2 to 4 GPUs "
           "(doubling the GPU count) achieved a MEAN %.1f%% of the ideal "
           "2x speedup -- ranging from a low of %.1f%% (%s) to a high of "
           "%.1f%% (%s).\n\n", numRows, meanEfficiency, minEfficiency,
           minEfficiencyTool, maxEfficiency, maxEfficiencyTool);

    printf("NVIDIA's own real caveat on this exact table: \"Speeds may "
           "vary depending on the data set, GPU instance, host CPU, "
           "memory availability, and other factors.\" This file's own "
           "honest reading of that real published spread: Section 39.2's "
           "real algorithmic finding (independent reads, independent "
           "genomic regions, zero required cross-worker communication) is "
           "NECESSARY for good multi-GPU scaling, but this real published "
           "data shows it is not SUFFICIENT on its own -- %s comes within "
           "%.1f%% of the algorithm's own theoretical ceiling of ideal "
           "linear scaling, while %s manages only %.1f%%, despite both "
           "resting on the exact same real embarrassingly-parallel "
           "foundation Section 39.2 verified. The real difference must lie "
           "outside the algorithm itself -- in real engineering factors "
           "such as fixed per-run setup cost, file I/O, or host-side "
           "bottlenecks that do not shrink just because the GPU count "
           "doubled -- the same real lesson Chapter 16 and Chapter 34 each "
           "found in their own different form (\"more ranks is not free\"), "
           "now confirmed directly from NVIDIA's own published real "
           "numbers rather than from a host-side simulation, a first for "
           "this book's own Part 7 case studies.\n",
           maxEfficiencyTool, maxEfficiency, minEfficiencyTool, minEfficiency);
    return 0;
}
