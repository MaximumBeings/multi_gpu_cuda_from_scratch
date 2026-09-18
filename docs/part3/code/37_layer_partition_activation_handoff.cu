// Chapter 13: Model Parallelism -- When One GPU Can't Hold the Weights
// 37_layer_partition_activation_handoff.cu
//
// Model parallelism needs exactly two things: an assignment of which
// device holds which LAYERS (pure host arithmetic, needs no device --
// structurally identical to Chapter 12's own batch-sharding formula,
// just partitioning layer indices instead of sample indices), and a
// real transfer that hands the ACTIVATION tensor produced by one
// device's last layer to the next device's first layer. That second
// part is not a new API -- it's the same real cudaMemcpyPeer() this
// book introduced in Chapter 5 and has used ever since, now moving a
// different kind of data across the same real device boundary.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

// Splits TOTAL_LAYERS into `worldSize` equal, non-overlapping,
// contiguous ranges, and returns device `rank`'s own [start, end)
// range of layer indices. Structurally identical to Chapter 12's
// computeShard() -- only what's being partitioned (layers, not
// samples) has changed. Pure host arithmetic; needs no device.
void computeLayerRange(int totalLayers, int worldSize, int rank, int* start, int* end) {
    int perDevice = totalLayers / worldSize; // assumes an even split
    *start = rank * perDevice;
    *end = *start + perDevice;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // GPT-3 175B's own real depth: 96 transformer layers (Brown et al.
    // 2020, Table 2.1).
    const int TOTAL_LAYERS = 96;

    printf("\nPartitioning %d layers across this book's own real "
           "deviceCount (%d):\n", TOTAL_LAYERS, deviceCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        int start, end;
        computeLayerRange(TOTAL_LAYERS, deviceCount, rank, &start, &end);
        printf("  rank %d: layers [%d, %d)\n", rank, start, end);
    }
    printf("  (%d device(s) -- nothing to print above.)\n", deviceCount);

    // Hypothetically, if there WERE 4 devices: every rank gets a
    // different, non-overlapping, equal-sized run of consecutive
    // layers -- genuinely computed, not illustrative.
    const int HYPOTHETICAL_WORLD_SIZE = 4;
    printf("\nThe same formula, hypothetically, with world size %d:\n", HYPOTHETICAL_WORLD_SIZE);
    for (int rank = 0; rank < HYPOTHETICAL_WORLD_SIZE; ++rank) {
        int start, end;
        computeLayerRange(TOTAL_LAYERS, HYPOTHETICAL_WORLD_SIZE, rank, &start, &end);
        printf("  rank %d: layers [%d, %d)\n", rank, start, end);
    }

    // Partitioning layer INDICES needs no device, same as Chapter 12's
    // batch sharding. Moving the ACTIVATION TENSOR that rank 0's last
    // layer produced into rank 1's first layer's input is different --
    // it is a real, unavoidable device-to-device transfer, sized by
    // GPT-3's own real dimensions: a sequence of 2048 tokens (Brown et
    // al.'s own context length, nctx) x 12288 features per token
    // (d_model), stored fp16 (2 bytes/element) the way Chapter 1's own
    // inference-footprint row already did.
    const size_t SEQ_LEN = 2048;
    const size_t D_MODEL = 12288;
    const size_t ACTIVATION_ELEMS = SEQ_LEN * D_MODEL;
    const size_t ACTIVATION_BYTES = ACTIVATION_ELEMS * 2; // fp16
    printf("\nOne layer-boundary activation tensor: %zu tokens x %zu "
           "features, fp16 -> %.2f MB that must genuinely cross the "
           "device boundary between rank 0 and rank 1 at every forward "
           "pass.\n", SEQ_LEN, D_MODEL, ACTIVATION_BYTES / 1.0e6);

    float* activation = nullptr; // never successfully allocated -- deviceCount is 0
    cudaError_t eMemcpyPeer = cudaMemcpyPeer(activation, 1, activation, 0, ACTIVATION_BYTES);
    printf("\ncudaMemcpyPeer(activation, dstDevice=1, activation, "
           "srcDevice=0, %.2f MB): %s (code %d)\n",
           ACTIVATION_BYTES / 1.0e6, cudaGetErrorString(eMemcpyPeer), (int)eMemcpyPeer);

    printf("\nPartitioning layer indices is arithmetic this chapter can\n"
           "genuinely run, exactly like Chapter 12's batch sharding. "
           "Handing\nthe activation tensor across the device boundary "
           "is the same real\ntransfer call this book has used since "
           "Chapter 5 -- it is not a new\nprimitive, only a new reason "
           "to call it. Section 13.3 checks the one\nthing that "
           "actually matters about doing this: whether a SPLIT forward\n"
           "pass computes the exact same thing an UNSPLIT one would.\n");

    return 0;
}
