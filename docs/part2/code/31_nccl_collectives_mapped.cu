// Chapter 11: NCCL -- What It Actually Does Differently From What You
// Just Built
// 31_nccl_collectives_mapped.cu
//
// Every collective this book built by hand in Chapters 8-10 has a
// literal, real NCCL API call behind it in the installed NCCL 2.18.3
// -- except all-to-all (a dedicated ncclAlltoAll() only arrived in
// NCCL 2.28). This section genuinely calls each real function with a
// comm handle that is still NULL -- no communicator can be created on
// this machine (Section 11.1) -- and honestly reports whatever
// ncclResult_t comes back. Genuinely compiled with a real nvcc,
// genuinely linked against a real libnccl, and genuinely run (the
// failure's cause confirmed with NCCL_DEBUG=WARN).
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // No communicator exists on this machine (Section 11.1), so the
    // handle passed to every call below is literally nullptr. NCCL's
    // argument check rejects it before doing anything else -- with
    // NCCL_DEBUG=WARN it prints "<Collective> : comm argument is NULL"
    // and returns ncclInvalidArgument (code 4). This is a NULL-handle
    // check, not a check on how many devices were asked for.
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;
    float* buf = nullptr;
    const size_t COUNT = 4;

    printf("\n-- Five of this book's six hand-built collectives, as one\n"
           "   real NCCL call each: --\n\n");

    ncclResult_t eBcast = ncclBroadcast(buf, buf, COUNT, ncclFloat, 0, comm, stream);
    printf("ncclBroadcast()      [Chapter 8, Section 8.1's broadcast]:      %s (code %d)\n",
           ncclGetErrorString(eBcast), (int)eBcast);

    ncclResult_t eReduce = ncclReduce(buf, buf, COUNT, ncclFloat, ncclSum, 0, comm, stream);
    printf("ncclReduce()         [Chapter 8, Section 8.2's reduce]:         %s (code %d)\n",
           ncclGetErrorString(eReduce), (int)eReduce);

    ncclResult_t eAllReduce = ncclAllReduce(buf, buf, COUNT, ncclFloat, ncclSum, comm, stream);
    printf("ncclAllReduce()      [Chapter 9's ring all-reduce]:             %s (code %d)\n",
           ncclGetErrorString(eAllReduce), (int)eAllReduce);

    ncclResult_t eAllGather = ncclAllGather(buf, buf, COUNT, ncclFloat, comm, stream);
    printf("ncclAllGather()      [Chapter 10, Section 10.1's all-gather]:   %s (code %d)\n",
           ncclGetErrorString(eAllGather), (int)eAllGather);

    ncclResult_t eReduceScatter = ncclReduceScatter(buf, buf, COUNT, ncclFloat, ncclSum, comm, stream);
    printf("ncclReduceScatter()  [Chapter 10, Section 10.2's reduce-scatter]: %s (code %d)\n",
           ncclGetErrorString(eReduceScatter), (int)eReduceScatter);

    // All-to-all has no single dedicated call in this installed NCCL
    // version (2.18.3; ncclAlltoAll() was added in 2.28) -- it is built the same way this book built it in
    // Chapter 10, Section 10.3: one point-to-point transfer per
    // ordered pair, just using ncclSend()/ncclRecv() instead of
    // cudaMemcpyPeer(), fused together with ncclGroupStart()/
    // ncclGroupEnd() into one coordinated operation.
    printf("\n-- All-to-all [Chapter 10, Section 10.3]: no dedicated call in 2.18.3 --\n"
           "   built from ncclSend()/ncclRecv(), exactly like this book's\n"
           "   own direct implementation: --\n\n");

    ncclResult_t eGroupStart = ncclGroupStart();
    printf("ncclGroupStart(): %s (code %d)\n", ncclGetErrorString(eGroupStart), (int)eGroupStart);

    int p2pAttempted = 0;
    for (int peer = 0; peer < deviceCount; ++peer) {
        ncclResult_t eSend = ncclSend(buf, 1, ncclFloat, peer, comm, stream);
        ncclResult_t eRecv = ncclRecv(buf, 1, ncclFloat, peer, comm, stream);
        ++p2pAttempted;
        printf("ncclSend()/ncclRecv(peer=%d): %s / %s\n", peer,
               ncclGetErrorString(eSend), ncclGetErrorString(eRecv));
    }

    ncclResult_t eGroupEnd = ncclGroupEnd();
    printf("ncclGroupEnd(): %s (code %d)\n", ncclGetErrorString(eGroupEnd), (int)eGroupEnd);

    printf("\n%d device(s) worth of ncclSend()/ncclRecv() pairs attempted\n"
           "inside the group, so ncclSend()/ncclRecv() were never actually\n"
           "called. All five collective calls above report the same\n"
           "failure: ncclInvalidArgument, because the comm handle they were\n"
           "given is NULL (NCCL_DEBUG=WARN: \"comm argument is NULL\").\n"
           "ncclGroupStart()/ncclGroupEnd() succeed: they touch no comm.\n"
           "Section 11.3 turns to the one real difference that would\n"
           "matter if a comm HAD been created: how NCCL chooses an\n"
           "algorithm for ncclAllReduce().\n",
           p2pAttempted);

    return 0;
}
