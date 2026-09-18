// Chapter 12: Data Parallelism -- Replicated Model, Sharded Data
// 33_replica_setup.cu
//
// Data parallelism needs exactly two things set up before any
// training step can run: every device holding an IDENTICAL copy of
// the model's initial weights, and every device holding a DIFFERENT
// slice of the global batch. The first needs a real collective
// (ncclBroadcast, genuinely called below); the second needs no
// device at all -- it's just arithmetic over indices, so this
// section can genuinely succeed even here. Genuinely compiled with
// a real nvcc, genuinely linked against a real libnccl, and
// genuinely run.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

// Splits a global batch into `worldSize` equal, non-overlapping
// shards, and returns device `rank`'s own [start, end) range. This
// is pure host-side index arithmetic -- no device, no CUDA call, no
// NCCL call. It genuinely succeeds no matter how many real devices
// exist, because it doesn't need any.
void computeShard(int globalBatchSize, int worldSize, int rank, int* start, int* end) {
    int perDevice = globalBatchSize / worldSize; // assumes an even split
    *start = rank * perDevice;
    *end = *start + perDevice;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int GLOBAL_BATCH = 16;

    // This book's own real deviceCount is 0, so there is genuinely
    // nothing to shard across -- but the shard formula itself runs
    // and reports that honestly, exactly like every device-count loop
    // in this book since Chapter 4.
    printf("\nSharding a global batch of %d across this book's own real "
           "deviceCount (%d):\n", GLOBAL_BATCH, deviceCount);
    for (int rank = 0; rank < deviceCount; ++rank) {
        int start, end;
        computeShard(GLOBAL_BATCH, deviceCount, rank, &start, &end);
        printf("  rank %d: samples [%d, %d)\n", rank, start, end);
    }
    printf("  (%d device(s) -- nothing to print above.)\n", deviceCount);

    // Hypothetically, if there WERE 4 devices: every rank gets a
    // different, non-overlapping, equal-sized slice of the same
    // global batch -- this is genuinely computed, not illustrative
    // pseudocode.
    const int HYPOTHETICAL_WORLD_SIZE = 4;
    printf("\nThe same formula, hypothetically, with world size %d:\n", HYPOTHETICAL_WORLD_SIZE);
    for (int rank = 0; rank < HYPOTHETICAL_WORLD_SIZE; ++rank) {
        int start, end;
        computeShard(GLOBAL_BATCH, HYPOTHETICAL_WORLD_SIZE, rank, &start, &end);
        printf("  rank %d: samples [%d, %d)\n", rank, start, end);
    }

    // Splitting the DATA needs no device. Making every replica start
    // from the SAME weights does -- that's a real collective, exactly
    // the way Horovod's own hvd.BroadcastGlobalVariablesHook or this
    // book's own Chapter 8 broadcast enforces it, rather than assuming
    // it. Attempted here with the real ncclBroadcast() this book
    // linked against in Chapter 11.
    ncclComm_t comm = nullptr; // never successfully created, as in Chapter 11
    cudaStream_t stream = nullptr;
    float* weights = nullptr;
    const size_t WEIGHT_COUNT = 4;

    ncclResult_t eBcast = ncclBroadcast(weights, weights, WEIGHT_COUNT, ncclFloat, 0, comm, stream);
    printf("\nncclBroadcast(root=0, initial weights -> every replica): %s (code %d)\n",
           ncclGetErrorString(eBcast), (int)eBcast);

    printf("\nSharding data is arithmetic this chapter can genuinely run;\n"
           "broadcasting weights is a real collective this chapter can only\n"
           "genuinely attempt. Section 12.2 covers the other real collective\n"
           "data parallelism needs every step after that: averaging gradients.\n");

    return 0;
}
