// Chapter 26: Multi-GPU LLM Inference: Tensor and Pipeline Parallelism in
// Practice
// 75_tp_pp_combined_correctness_simulation.cpp
//
// Chapter 14 built Megatron-LM's real tensor-parallel MLP split (column-
// parallel GEMM1, no communication, then row-parallel GEMM2, ONE
// ncclAllReduce()) in isolation. Chapter 15 built GPipe's real pipeline
// schedule -- layers split across stages, microbatches flowing through
// them -- also in isolation. A real production LLM inference system
// combines BOTH, at the same time, over the SAME device grid: the
// Megatron-LM paper's own words for why: "tensor model parallelism
// should generally be used up to degree g when using g-GPU servers, and
// then pipeline model parallelism can be used to scale up to larger
// models across servers." This file builds that combined shape as a
// real host-side simulation -- a 2D device grid, PP_STAGES x TP_DEGREE
// -- and checks its ROUTING, the way Chapter 24's own SUMMA simulation
// did: using long long INTEGER matrices (Chapter 8's own non-
// associativity caution -- this is a correctness check on the routing,
// not a claim about floating-point summation order), so the combined
// result can be compared bit-exact against a naive, unsplit reference,
// with no rounding ambiguity possible.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

const int D = 4;             // per-layer input/output dimension
const int H_TOTAL = 8;       // per-layer hidden dimension
const int NUM_LAYERS = 4;    // total MLP layers in this toy model
const int TP_DEGREE = 2;     // tensor-parallel devices PER stage
const int PP_STAGES = 2;     // pipeline stages
const int LAYERS_PER_STAGE = NUM_LAYERS / PP_STAGES;
const int H_PER_TP = H_TOTAL / TP_DEGREE;
const int M = 4;             // microbatches (this chapter is INFERENCE --
                              // forward passes only, no backward pass)

using Vec = std::vector<long long>;
using Mat = std::vector<long long>; // row-major

long long relu(long long x) { return x > 0 ? x : 0; }

// Deterministic real-valued (integer) weights for layer l -- Chapter
// 14's own W1 (D x H_TOTAL, column-parallel) and W2 (H_TOTAL x D,
// row-parallel) shapes, reused unchanged.
Mat makeW1(int l) {
    Mat W(D * H_TOTAL);
    for (int i = 0; i < D; i++)
        for (int h = 0; h < H_TOTAL; h++)
            W[i * H_TOTAL + h] = ((l + 1) * 13 + i * 7 + h * 3) % 7 - 3;
    return W;
}
Mat makeW2(int l) {
    Mat W(H_TOTAL * D);
    for (int h = 0; h < H_TOTAL; h++)
        for (int j = 0; j < D; j++)
            W[h * D + j] = ((l + 1) * 17 + h * 5 + j * 11) % 7 - 3;
    return W;
}

// --- Naive reference: one process, no TP, no PP, one flat pass through
// every layer, exactly the way an unsplit forward pass would run. ---
Vec naiveForward(Vec x, const std::vector<Mat> &W1s, const std::vector<Mat> &W2s) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        Vec hidden(H_TOTAL, 0);
        for (int h = 0; h < H_TOTAL; h++) {
            long long s = 0;
            for (int i = 0; i < D; i++) s += x[i] * W1s[l][i * H_TOTAL + h];
            hidden[h] = relu(s);
        }
        Vec out(D, 0);
        for (int j = 0; j < D; j++) {
            long long s = 0;
            for (int h = 0; h < H_TOTAL; h++) s += hidden[h] * W2s[l][h * D + j];
            out[j] = s;
        }
        x = out;
    }
    return x;
}

// --- Combined TP+PP: one layer's real Megatron-style TP split, run
// entirely within ONE pipeline stage's own TP group. Column-parallel
// GEMM1 needs NO communication (each TP rank computes its own hidden
// shard); row-parallel GEMM2 needs exactly ONE all-reduce, SCOPED TO
// THIS STAGE'S OWN TP GROUP ONLY -- never crossing a pipeline-stage
// boundary, which is the real property this file's own COMMON TRAP is
// built around. ---
Vec layerForwardTP(Vec x, const Mat &W1, const Mat &W2) {
    std::vector<Vec> localOut(TP_DEGREE, Vec(D, 0));
    for (int r = 0; r < TP_DEGREE; r++) {
        // Column-parallel GEMM1: rank r owns hidden columns
        // [r*H_PER_TP, (r+1)*H_PER_TP) -- no communication needed here.
        Vec hiddenShard(H_PER_TP, 0);
        for (int hh = 0; hh < H_PER_TP; hh++) {
            int h = r * H_PER_TP + hh;
            long long s = 0;
            for (int i = 0; i < D; i++) s += x[i] * W1[i * H_TOTAL + h];
            hiddenShard[hh] = relu(s);
        }
        // Row-parallel GEMM2: rank r owns the H_PER_TP rows of W2
        // matching its own hidden shard, producing a PARTIAL D-vector.
        for (int j = 0; j < D; j++) {
            long long s = 0;
            for (int hh = 0; hh < H_PER_TP; hh++) {
                int h = r * H_PER_TP + hh;
                s += hiddenShard[hh] * W2[h * D + j];
            }
            localOut[r][j] = s;
        }
    }
    // ONE real all-reduce (SUM), scoped to this stage's TP group of
    // TP_DEGREE ranks -- Chapter 14's own real row-parallel pattern,
    // reused unchanged.
    Vec combined(D, 0);
    for (int r = 0; r < TP_DEGREE; r++)
        for (int j = 0; j < D; j++) combined[j] += localOut[r][j];
    return combined;
}

Vec combinedForward(Vec x, const std::vector<Mat> &W1s, const std::vector<Mat> &W2s) {
    for (int stage = 0; stage < PP_STAGES; stage++) {
        for (int li = 0; li < LAYERS_PER_STAGE; li++) {
            int l = stage * LAYERS_PER_STAGE + li;
            x = layerForwardTP(x, W1s[l], W2s[l]);
        }
        // PP handoff: the stage's OWN fully-combined activation (every
        // TP rank in this stage already holds the identical value,
        // after the all-reduce above) is what crosses to the next
        // stage -- Chapter 13's own cudaMemcpyPeer()-style activation
        // transfer, simulated here as a plain array copy. No partial,
        // not-yet-all-reduced value ever crosses a stage boundary.
    }
    return x;
}

int main() {
    std::vector<Mat> W1s(NUM_LAYERS), W2s(NUM_LAYERS);
    for (int l = 0; l < NUM_LAYERS; l++) { W1s[l] = makeW1(l); W2s[l] = makeW2(l); }

    printf("Grid: PP_STAGES=%d x TP_DEGREE=%d = %d total simulated devices, "
           "%d layers (%d per stage), %d microbatches (inference: forward only)\n\n",
           PP_STAGES, TP_DEGREE, PP_STAGES * TP_DEGREE, NUM_LAYERS,
           LAYERS_PER_STAGE, M);

    bool allExact = true;
    for (int m = 0; m < M; m++) {
        Vec x0(D);
        for (int d = 0; d < D; d++) x0[d] = (m * 3 + d) % 5 - 2;

        Vec refOut = naiveForward(x0, W1s, W2s);
        Vec combOut = combinedForward(x0, W1s, W2s);

        bool exact = (refOut == combOut);
        allExact = allExact && exact;
        printf("Microbatch %d: naive=[", m);
        for (int d = 0; d < D; d++) printf("%lld%s", refOut[d], d + 1 < D ? "," : "");
        printf("]  combined-TP+PP=[");
        for (int d = 0; d < D; d++) printf("%lld%s", combOut[d], d + 1 < D ? "," : "");
        printf("]  %s\n", exact ? "EXACT MATCH" : "MISMATCH");
    }

    printf("\nAll %d microbatches EXACT MATCH between naive and combined "
           "TP+PP: %s\n", M, allExact ? "YES" : "NO");

    // A real, direct forward-only pipeline-schedule count (NOT Chapter
    // 15's own training bubble formula -- see this section's own COMMON
    // TRAP for why that formula does not apply here): M microbatches
    // through PP_STAGES stages, one stage-slot per microbatch per
    // stage, take exactly M + PP_STAGES - 1 total time steps to drain.
    int totalSteps = M + PP_STAGES - 1;
    int totalSlots = totalSteps * PP_STAGES;
    int busySlots = M * PP_STAGES;
    int idleSlots = totalSlots - busySlots;
    printf("\nForward-only inference pipeline schedule: %d microbatches, "
           "%d stages -> %d total time steps, %d/%d stage-slots busy, "
           "%d idle (%.1f%% bubble)\n",
           M, PP_STAGES, totalSteps, busySlots, totalSlots, idleSlots,
           100.0 * idleSlots / totalSlots);

    return allExact ? 0 : 1;
}
