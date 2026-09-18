// Chapter 19: Stragglers, Failures, and Fault-Tolerant Collectives
// 54_blocking_vs_nonblocking_comm.cu
//
// Every real NCCL call this book has made since Chapter 11 has
// returned an honest, IMMEDIATE error code -- ncclInvalidArgument,
// ncclUnhandledCudaError -- because this environment has never had a
// real device to begin with. A real cluster's real failure mode is
// different, and worse: NVIDIA's own documentation on building
// fault-tolerant NCCL applications states it plainly -- "even healthy
// ranks should expect NCCL to either return an error OR HANG on any
// collective operation." A rank that genuinely dies mid-collective
// doesn't make every other rank see an error; it makes every other
// rank hang, forever, at whatever collective (or Chapter 17's own
// barrier) they were waiting on with it. This section builds the real
// fix: a NON-BLOCKING communicator, configured with a real
// ncclConfig_t, so a hang can be DETECTED instead of endured.
// Genuinely compiled with a real nvcc against the real installed
// libnccl.
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    ncclUniqueId uniqueId;
    ncclResult_t eUniqueId = ncclGetUniqueId(&uniqueId);
    printf("\nncclGetUniqueId(): %s (code %d)\n",
           ncclGetErrorString(eUniqueId), (int)eUniqueId);

    // The DEFAULT communicator every earlier chapter's ncclCommInitAll()
    // implicitly built: blocking. Every NCCL call on it either finishes
    // or, on a real cluster with a genuinely dead peer, never returns
    // at all -- there is no way to poll it, no way to time it out.
    ncclConfig_t blockingConfig = NCCL_CONFIG_INITIALIZER;
    blockingConfig.blocking = 1;
    ncclComm_t blockingComm = nullptr;
    ncclResult_t eBlocking = ncclCommInitRankConfig(&blockingComm, 1, uniqueId, 0, &blockingConfig);
    printf("ncclCommInitRankConfig(blocking=1): %s (code %d)\n",
           ncclGetErrorString(eBlocking), (int)eBlocking);

    // The FIX: a real, documented NCCL feature -- config.blocking = 0.
    // NVIDIA's own words: "NCCL communicators can be configured to be
    // non-blocking so that initialization functions may continue in
    // the background," specifically "so that we may detect and react
    // to timeouts." Section 19.2 uses exactly this communicator.
    ncclConfig_t nonBlockingConfig = NCCL_CONFIG_INITIALIZER;
    nonBlockingConfig.blocking = 0;
    ncclComm_t nonBlockingComm = nullptr;
    ncclResult_t eNonBlocking = ncclCommInitRankConfig(&nonBlockingComm, 1, uniqueId, 0, &nonBlockingConfig);
    printf("ncclCommInitRankConfig(blocking=0): %s (code %d)\n",
           ncclGetErrorString(eNonBlocking), (int)eNonBlocking);

    printf("\nBoth calls above return immediately in THIS environment, "
           "honestly reporting that initialization on a never-successful "
           "communicator failed to progress. That's this environment's "
           "own honest failure mode (no device, ever) -- it is NOT the "
           "same as a real cluster's hang-on-peer-failure mode, which "
           "this section's whole point is to name correctly rather than "
           "let this book's own always-instant errors quietly stand in "
           "for it. Section 19.2 builds the real detect-and-recover "
           "sequence a non-blocking communicator like the second one "
           "above makes possible.\n");

    return 0;
}
