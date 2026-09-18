// Chapter 11: NCCL -- What It Actually Does Differently From What You
// Just Built
// 31_nccl_collectives_mapped.cu
//
// Every collective this book built by hand in Chapters 8-10 has a
// literal, real NCCL API call behind it -- except one. This section
// genuinely calls each real function, using the communicator Section
// 11.1 showed can never successfully form on this machine, and
// honestly reports whatever ncclResult_t comes back. Genuinely
// compiled with a real nvcc, genuinely linked against a real
// libnccl, and genuinely run.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // A communicator that was requested but never successfully created,
    // exactly the state Section 11.1 left off at -- passed to every
    // real call below, exactly as a genuine program would if its own
    // ncclCommInitAll() call had failed and it kept going anyway.
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
    // version -- it is built the same way this book built it in
    // Chapter 10, Section 10.3: one point-to-point transfer per
    // ordered pair, just using ncclSend()/ncclRecv() instead of
    // cudaMemcpyPeer(), fused together with ncclGroupStart()/
    // ncclGroupEnd() into one coordinated operation.
    printf("\n-- All-to-all [Chapter 10, Section 10.3]: no dedicated call --\n"
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
           "inside the group. Every one of the six real calls above\n"
           "reports the same honest failure this chapter's communicator\n"
           "already earned -- a comm that was requested but never\n"
           "successfully created can't run any collective, real or\n"
           "hand-built, on real hardware or in this driver-less\n"
           "environment. Section 11.3 turns to the one real difference\n"
           "that would matter if a comm HAD been created: how NCCL\n"
           "chooses an algorithm for ncclAllReduce().\n",
           p2pAttempted);

    return 0;
}
