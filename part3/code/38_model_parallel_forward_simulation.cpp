// Chapter 13: Model Parallelism -- When One GPU Can't Hold the Weights
// 38_model_parallel_forward_simulation.cpp
//
// Plain host C++ -- this chapter's real verification, since no real
// forward pass can run without a real device. A tiny 8-layer toy
// network (2-element vectors, one small affine transform + ReLU per
// layer) stands in for a real model deep enough that no single
// device could hold every layer. Two host arrays stand in for two
// simulated "devices" worth of memory holding different layer
// ranges (Section 13.2's own layer-partition formula, reused here),
// exchanging the SAME buffer instead of a real cudaMemcpyPeer() --
// the point being checked is whether the ANSWER changes when the
// computation is split across a partition boundary, not how the
// bytes physically move.
#include <cstdio>
#include <cmath>

const int LAYERS = 8;

// One layer's transform: y = ReLU(W_i * x + b_i), for a 2-element
// vector. Weights are small, deterministic, and distinct per layer
// index -- there is nothing to "hand-check" about the specific
// numbers; what this section checks is that calling this SAME
// function, in the SAME order, produces the SAME output whether it
// is called from one unbroken loop or from four separate simulated
// devices' loops.
void applyLayer(int layerIdx, double x0, double x1, double* y0, double* y1) {
    double w00 = 0.50 + 0.01 * layerIdx, w01 = 0.10;
    double w10 = 0.15,                   w11 = 0.55 + 0.01 * layerIdx;
    double b0 = 0.02 * layerIdx, b1 = 0.01 * layerIdx;

    double r0 = w00 * x0 + w01 * x1 + b0;
    double r1 = w10 * x0 + w11 * x1 + b1;

    *y0 = r0 > 0.0 ? r0 : 0.0; // ReLU
    *y1 = r1 > 0.0 ? r1 : 0.0; // ReLU
}

// Structurally identical to Section 13.2's computeLayerRange() (and
// Chapter 12's computeShard() before that) -- pure host arithmetic,
// needs no device.
void computeLayerRange(int totalLayers, int worldSize, int rank, int* start, int* end) {
    int perDevice = totalLayers / worldSize;
    *start = rank * perDevice;
    *end = *start + perDevice;
}

// The UNSPLIT reference: every layer applied in one unbroken loop,
// as if a single device held the entire 8-layer model.
void referenceForward(double x0, double x1, double* out0, double* out1) {
    double cur0 = x0, cur1 = x1;
    for (int layer = 0; layer < LAYERS; ++layer) {
        double n0, n1;
        applyLayer(layer, cur0, cur1, &n0, &n1);
        cur0 = n0; cur1 = n1;
    }
    *out0 = cur0; *out1 = cur1;
}

// The SPLIT forward pass: WORLD_SIZE simulated devices, each holding
// a different contiguous range of layers (Section 13.2's formula).
// Every rank applies only ITS OWN layers, in order, to whatever
// buffer the previous rank produced -- the same real dependency a
// real cudaMemcpyPeer() handoff would enforce, just without a real
// transfer call moving the bytes.
void splitForward(double x0, double x1, int worldSize, double* out0, double* out1) {
    double cur0 = x0, cur1 = x1;
    for (int rank = 0; rank < worldSize; ++rank) {
        int start, end;
        computeLayerRange(LAYERS, worldSize, rank, &start, &end);
        for (int layer = start; layer < end; ++layer) {
            double n0, n1;
            applyLayer(layer, cur0, cur1, &n0, &n1);
            cur0 = n0; cur1 = n1;
        }
        // Section 13.2's cudaMemcpyPeer() call is what would carry
        // (cur0, cur1) from rank's device to (rank+1)'s device here,
        // on real hardware. The buffer itself does not change value
        // when it crosses that boundary -- only where it physically
        // lives does.
    }
    *out0 = cur0; *out1 = cur1;
}

int main() {
    const int WORLD_SIZE = 4; // 8 layers / 4 devices = 2 layers/device, matching Section 13.2

    struct Case { double x0, x1; };
    const Case cases[] = { {1.0, 1.0}, {2.0, -1.0}, {0.5, 3.0} };

    bool allMatch = true;
    for (const auto& c : cases) {
        double refOut0, refOut1, splitOut0, splitOut1;
        referenceForward(c.x0, c.x1, &refOut0, &refOut1);
        splitForward(c.x0, c.x1, WORLD_SIZE, &splitOut0, &splitOut1);

        bool matches = (refOut0 == splitOut0) && (refOut1 == splitOut1);
        allMatch = allMatch && matches;

        printf("input (%.2f, %.2f): unsplit reference = (%.10f, %.10f), "
               "split across %d devices = (%.10f, %.10f) -- %s\n",
               c.x0, c.x1, refOut0, refOut1, WORLD_SIZE, splitOut0, splitOut1,
               matches ? "PASS (bit-identical)" : "FAIL");
    }

    printf("\nAll %zu input(s) bit-identical between the unsplit reference "
           "and the %d-device split forward pass: %s\n",
           sizeof(cases) / sizeof(cases[0]), WORLD_SIZE,
           allMatch ? "PASS" : "FAIL");

    printf("\nUnlike Chapter 8's floating-point reduce (order-sensitive --\n"
           "explicitly NOT verified bit-identical there), splitting a\n"
           "strictly SEQUENTIAL computation across devices never reorders\n"
           "any operation -- it only relocates WHERE each already-ordered\n"
           "step runs. That is why this section's match is bit-identical,\n"
           "not merely equal up to floating-point rounding.\n");

    return allMatch ? 0 : 1;
}
