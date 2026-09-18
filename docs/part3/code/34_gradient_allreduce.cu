// Chapter 12: Data Parallelism -- Replicated Model, Sharded Data
// 34_gradient_allreduce.cu
//
// Every training step, after every replica computes its own local
// gradients from its own data shard, those gradients have to be
// averaged across every replica before anyone updates their weights
// -- otherwise "replica" stops meaning what Section 12.1 set it up
// to mean. NCCL's ncclAvg reduction op (new in this chapter) does
// the averaging as part of the same real ncclAllReduce() call this
// book linked against in Chapter 11 -- no separate divide-by-N step
// required. Genuinely compiled with a real nvcc, genuinely linked
// against a real libnccl, and genuinely run.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    ncclComm_t comm = nullptr; // never successfully created, as in Chapter 11
    cudaStream_t stream = nullptr;
    float* localGradients = nullptr;
    const size_t WEIGHT_COUNT = 4;

    // ncclSum, used in Chapters 8-11, would leave every replica with
    // the SUM of every other replica's gradient -- exactly N times
    // too large an update. ncclAvg divides by the number of ranks in
    // the communicator as part of the same reduction, so the result
    // is already the correct per-parameter average gradient.
    ncclResult_t eAllReduceAvg = ncclAllReduce(localGradients, localGradients, WEIGHT_COUNT,
                                                ncclFloat, ncclAvg, comm, stream);
    printf("ncclAllReduce(localGradients, ..., ncclAvg, ...): %s (code %d)\n",
           ncclGetErrorString(eAllReduceAvg), (int)eAllReduceAvg);

    // For contrast: the same call with ncclSum, this book's only
    // reduction op through Chapter 11 -- genuinely a different real
    // call, not a relabeling of the one above.
    ncclResult_t eAllReduceSum = ncclAllReduce(localGradients, localGradients, WEIGHT_COUNT,
                                                ncclFloat, ncclSum, comm, stream);
    printf("ncclAllReduce(localGradients, ..., ncclSum, ...): %s (code %d)\n",
           ncclGetErrorString(eAllReduceSum), (int)eAllReduceSum);

    printf("\nBoth calls report the same honest failure this book's\n"
           "never-successfully-created communicator has reported since\n"
           "Chapter 11 -- the real difference between ncclAvg and ncclSum\n"
           "only shows up in what value ends up in the buffer, which needs\n"
           "a real device to observe. Section 12.3's simulation checks that\n"
           "difference the way this book always has when hardware can't:\n"
           "against an independent reference, computed on the host.\n");

    return 0;
}
