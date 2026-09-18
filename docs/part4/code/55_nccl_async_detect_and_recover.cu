// Chapter 19: Stragglers, Failures, and Fault-Tolerant Collectives
// 55_nccl_async_detect_and_recover.cu
//
// Section 19.1 named the real problem: a blocking NCCL communicator
// gives an application no way to notice a hung collective, because
// the call that would tell it so is itself the thing that's hanging.
// This file builds NVIDIA's own real, documented fix in full: create
// the communicator NON-BLOCKING (config.blocking = 0), then instead
// of blocking on the init/collective call itself, POLL
// ncclCommGetAsyncError() in a bounded loop -- NVIDIA's own example
// code shows exactly this pattern: "NCCL_CHECK(ncclCommGetAsyncError(
// comm, &asyncError)); while (asyncError == ncclInProgress)". Once a
// real problem is detected, NVIDIA's docs describe the escape
// directly: "Typical recovery for the healthy ranks starts with
// ncclCommAbort on the existing communicator, followed by
// ncclCommInit" -- ncclCommAbort() is not recovery on its own; it
// "will exit any operation currently in progress, and destroy the
// communicator." The application still has to rebuild a fresh
// communicator afterward (not shown here -- that's just
// ncclCommInitRankConfig() again, already built in 19.1).
// Genuinely compiled with a real nvcc against the real installed
// libnccl.
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

// The real polling pattern NVIDIA's fault-tolerance docs describe:
// call ncclCommGetAsyncError() in a loop, bounded by a max iteration
// count standing in for a real wall-clock timeout, until the
// communicator reports something other than "still in progress."
ncclResult_t pollUntilSettled(ncclComm_t comm, int maxPolls, int* pollsUsed) {
    ncclResult_t asyncErr = ncclInProgress;
    int i = 0;
    for (; i < maxPolls; ++i) {
        ncclResult_t queryErr = ncclCommGetAsyncError(comm, &asyncErr);
        if (queryErr != ncclSuccess) {
            // The query call itself failed -- e.g. comm was never a
            // valid handle to begin with, this environment's own
            // honest case, since ncclCommInitRankConfig() below never
            // completes without a device.
            *pollsUsed = i + 1;
            return queryErr;
        }
        if (asyncErr != ncclInProgress) {
            *pollsUsed = i + 1;
            return asyncErr;
        }
        // A real caller would sleep briefly here before polling
        // again; this book's own honest environment never reaches a
        // second iteration, so there's nothing to genuinely sleep on.
    }
    *pollsUsed = i;
    return ncclInProgress; // exhausted the poll budget -- a real timeout
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    ncclUniqueId uniqueId;
    ncclGetUniqueId(&uniqueId);

    // Build the non-blocking communicator 19.1 introduced.
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 0;
    ncclComm_t comm = nullptr;
    ncclResult_t eInit = ncclCommInitRankConfig(&comm, 1, uniqueId, 0, &config);
    printf("\nncclCommInitRankConfig(blocking=0): %s (code %d)\n",
           ncclGetErrorString(eInit), (int)eInit);
    printf("On a non-blocking communicator this call is allowed to return "
           "BEFORE initialization has actually finished -- that's the whole "
           "point, so the caller gets control back immediately instead of "
           "blocking. Whether init is really done yet is a separate "
           "question, answered only by polling ncclCommGetAsyncError(), "
           "never by this return value alone.\n");

    // The real detect step: poll for up to MAX_POLLS iterations, the
    // way a real application would poll for up to some wall-clock
    // timeout while a peer might be hanging.
    const int MAX_POLLS = 1000;
    int pollsUsed = 0;
    ncclResult_t settled = pollUntilSettled(comm, MAX_POLLS, &pollsUsed);
    printf("\npollUntilSettled(): settled after %d/%d poll(s) with result "
           "%s (code %d)\n",
           pollsUsed, MAX_POLLS, ncclGetErrorString(settled), (int)settled);

    if (pollsUsed < MAX_POLLS) {
        printf("The very first poll already saw a settled (non-"
               "ncclInProgress) result -- this environment's own honest "
               "failure mode is immediate, not a hang, so there was never "
               "anything for this loop to genuinely wait out. On a real "
               "cluster with a genuinely dead peer, ncclCommGetAsyncError() "
               "keeps returning ncclInProgress for as long as the peer "
               "stays dead, and THIS loop is what turns that into a bounded "
               "wait instead of an unbounded hang.\n");
    } else {
        printf("Exhausted the poll budget without settling -- on a real "
               "cluster this is the real timeout firing: a peer that "
               "never comes back. Either way, settled or timed out, the "
               "next step is the same.\n");
    }

    // The real escape: NVIDIA's own docs state it directly -- "Typical
    // recovery for the healthy ranks starts with ncclCommAbort on the
    // existing communicator, followed by ncclCommInit." Abort is not
    // recovery; it just stops the hang so the process can move on.
    ncclResult_t eAbort = ncclCommAbort(comm);
    printf("\nncclCommAbort(): %s (code %d)\n",
           ncclGetErrorString(eAbort), (int)eAbort);
    printf("ncclCommAbort() only guarantees that this call returns -- it "
           "does NOT rebuild a working communicator. A real application "
           "still has to call ncclCommInitRankConfig() again afterward "
           "(19.1's own call, unchanged) to get back into a workable "
           "state, almost always with the dead rank excluded from the new "
           "membership list.\n");

    printf("\nOpen MPI's ULFM (User-Level Failure Mitigation) extension "
           "offers a real, richer alternative to this abort-and-rebuild "
           "cycle: MPIX_Comm_revoke() invalidates a communicator across "
           "every surviving rank at once (so nobody is left waiting on a "
           "peer that already knows to give up), and MPIX_Comm_shrink() "
           "then produces a NEW communicator that simply excludes the "
           "dead rank -- survivors keep going with a smaller world, no "
           "full rebuild required. NCCL has no equivalent to shrink(): "
           "ncclCommAbort() only stops the hang, and rebuilding an NCCL "
           "communicator to exclude one dead rank means calling "
           "ncclCommInitRankConfig() again with an entirely new rank "
           "count and a new ncclUniqueId, involving every surviving rank "
           "from scratch. Chapter 20 opens Part 5 with MPI's own first "
           "dedicated chapter, where ULFM gets built and run directly, "
           "not just described.\n");

    return 0;
}
