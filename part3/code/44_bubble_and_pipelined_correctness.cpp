// Chapter 15: Pipeline Parallelism -- Splitting Layers Across Devices, and the Bubble That Costs You
// 44_bubble_and_pipelined_correctness.cpp
//
// Plain host C++, two checks in one file. Part A: a closed-form model
// -- not fabricated timings -- applying the exact idle-fraction
// formula Section 15.1's own direct counting already confirmed,
// (K-1)/(M+K-1), to realistic K values this book has ALREADY used for
// real (Chapter 13's own 2-way and 35-way model-parallel splits), and
// checking GPipe's own cited empirical claim that overhead is
// negligible once M >= 4*K. Part B: Chapter 13's own toy 8-layer
// network and layer-partition formula, reused verbatim, now run
// through Section 15.1's interleaved pipeline schedule instead of one
// micro-batch at a time -- checking that interleaving execution in
// TIME never changes any individual micro-batch's own result, the
// same guarantee Chapter 13 established for a single micro-batch,
// now extended to several running through the pipeline at once.
#include <cstdio>

// ---------------------------------------------------------------
// Part A: the bubble fraction, closed-form.
// ---------------------------------------------------------------
double bubbleFraction(int K, int M) {
    return (double)(K - 1) / (double)(M + K - 1);
}

// ---------------------------------------------------------------
// Part B: Chapter 13's own toy network, verbatim (same LAYERS, same
// applyLayer weights, same computeLayerRange formula).
// ---------------------------------------------------------------
const int LAYERS = 8;

void applyLayer(int layerIdx, double x0, double x1, double* y0, double* y1) {
    double w00 = 0.50 + 0.01 * layerIdx, w01 = 0.10;
    double w10 = 0.15,                   w11 = 0.55 + 0.01 * layerIdx;
    double b0 = 0.02 * layerIdx, b1 = 0.01 * layerIdx;
    double r0 = w00 * x0 + w01 * x1 + b0;
    double r1 = w10 * x0 + w11 * x1 + b1;
    *y0 = r0 > 0.0 ? r0 : 0.0;
    *y1 = r1 > 0.0 ? r1 : 0.0;
}

void computeLayerRange(int totalLayers, int worldSize, int rank, int* start, int* end) {
    int perDevice = totalLayers / worldSize;
    *start = rank * perDevice;
    *end = *start + perDevice;
}

// Chapter 13's own UNSPLIT, UN-PIPELINED reference: every layer,
// one micro-batch, applied in one unbroken loop.
void referenceForward(double x0, double x1, double* out0, double* out1) {
    double cur0 = x0, cur1 = x1;
    for (int layer = 0; layer < LAYERS; ++layer) {
        double n0, n1;
        applyLayer(layer, cur0, cur1, &n0, &n1);
        cur0 = n0; cur1 = n1;
    }
    *out0 = cur0; *out1 = cur1;
}

// Section 15.1's own schedule function, reused verbatim.
int microbatchAt(int device, int timeStep) {
    int m = timeStep - device;
    return (m >= 0) ? m : -1;
}

int main() {
    // ---------------- Part A ----------------
    printf("Part A: bubble fraction (K-1)/(M+K-1), for real K values this\n"
           "book has already used, checked at GPipe's own empirical\n"
           "threshold M = 4xK:\n\n");
    printf("%-28s %6s %10s %14s\n", "K (source)", "K", "M=4xK", "bubble frac.");
    struct KCase { const char* label; int k; };
    const KCase kcases[] = {
        {"7B-class model (Ch13)",  2},
        {"typical pipeline depth", 4},
        {"typical pipeline depth", 8},
        {"GPT-3 175B (Ch13)",     35},
    };
    for (const auto& kc : kcases) {
        int M = 4 * kc.k;
        double frac = bubbleFraction(kc.k, M);
        printf("%-28s %6d %10d %13.2f%%\n", kc.label, kc.k, M, frac * 100.0);
    }

    // ---------------- Part B ----------------
    const int K = 4;              // pipeline devices (Section 15.1's own K)
    const int TOTAL_TIME_STEPS_UPPER_BOUND = 32; // generous; actual span computed per M

    struct Case { double x0, x1; };
    const Case cases[] = { {1.0, 1.0}, {2.0, -1.0}, {0.5, 3.0} }; // Chapter 13's own 3 inputs
    const int M = sizeof(cases) / sizeof(cases[0]);
    const int totalTimeSteps = M + K - 1;

    double curBuf[M][2];
    for (int m = 0; m < M; ++m) { curBuf[m][0] = cases[m].x0; curBuf[m][1] = cases[m].x1; }

    printf("\nPart B: running Chapter 13's own %d-layer network for %d "
           "micro-batches\nthrough Section 15.1's interleaved K=%d pipeline "
           "schedule (totalTimeSteps=%d):\n\n", LAYERS, M, K, totalTimeSteps);

    for (int t = 0; t < totalTimeSteps && t < TOTAL_TIME_STEPS_UPPER_BOUND; ++t) {
        for (int d = 0; d < K; ++d) {
            int m = microbatchAt(d, t);
            if (m < 0 || m >= M) continue; // idle cell, matches Section 15.1's schedule
            int layerStart, layerEnd;
            computeLayerRange(LAYERS, K, d, &layerStart, &layerEnd);
            for (int layer = layerStart; layer < layerEnd; ++layer) {
                double n0, n1;
                applyLayer(layer, curBuf[m][0], curBuf[m][1], &n0, &n1);
                curBuf[m][0] = n0; curBuf[m][1] = n1;
            }
            printf("  t=%d: device %d processes micro-batch %d (layers [%d,%d))\n",
                   t, d, m, layerStart, layerEnd);
        }
    }

    bool allMatch = true;
    printf("\nComparing each micro-batch's PIPELINED result against Chapter "
           "13's own UN-PIPELINED reference for the same input:\n");
    for (int m = 0; m < M; ++m) {
        double refOut0, refOut1;
        referenceForward(cases[m].x0, cases[m].x1, &refOut0, &refOut1);
        bool matches = (refOut0 == curBuf[m][0]) && (refOut1 == curBuf[m][1]);
        allMatch = allMatch && matches;
        printf("  micro-batch %d, input (%.2f, %.2f): reference = (%.10f, %.10f), "
               "pipelined = (%.10f, %.10f) -- %s\n",
               m, cases[m].x0, cases[m].x1, refOut0, refOut1, curBuf[m][0], curBuf[m][1],
               matches ? "PASS (bit-identical)" : "FAIL");
    }
    printf("\nAll %d micro-batch(es) bit-identical between the pipelined "
           "schedule and Chapter 13's own un-pipelined reference: %s\n",
           M, allMatch ? "PASS" : "FAIL");

    return allMatch ? 0 : 1;
}
