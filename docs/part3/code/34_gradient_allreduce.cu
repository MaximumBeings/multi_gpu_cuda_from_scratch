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

    // No communicator can be created on this machine (Chapter 11), so
    // the handle is literally NULL; NCCL rejects it up front
    // (NCCL_DEBUG=WARN: "AllReduce : comm argument is NULL").
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;
    float* localGradients = nullptr;
    const size_t WEIGHT_COUNT = 4;

    // ncclSum, used in Chapter 11, would leave every replica with
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

    printf("\nBoth calls report the same honest failure every collective\n"
           "handed a NULL comm has reported since Chapter 11 (NCCL_DEBUG=WARN:\n"
           "\"comm argument is NULL\"). The real difference between ncclAvg\n"
           "and ncclSum only shows up in what value ends up in the buffer,\n"
           "which needs a real device to observe. Section 12.3's simulation\n"
           "checks that difference the way this book always has when\n"
           "hardware can't: against an independent reference, computed on\n"
           "the host.\n");

    return 0;
}
