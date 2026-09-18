// Appendix C: NCCL and NVSHMEM -- The Standard Libraries You Get for Free
// 127_nccl_point_to_point.cu
//
// Chapter 11 already introduced NCCL's own real point-to-point API --
// ncclSend()/ncclRecv(), fused inside ncclGroupStart()/ncclGroupEnd() to
// build Chapter 10's own all-to-all without a dedicated NCCL call at all
// -- and its own Self-Check Question 4 asked what NCCL's documented
// ordering guarantee promises for two point-to-point calls inside one
// group: calls to the SAME peer execute in the order they were issued;
// calls to DIFFERENT peers carry no such guarantee and may progress
// concurrently. Chapter 11's own body text quoted that guarantee but did
// not build a standalone demonstration of it. This section gathers
// ncclSend/ncclRecv/ncclGroupStart/ncclGroupEnd as a consolidated
// reference (confirming all four real symbols resolve and link) and then
// builds exactly the demonstration Chapter 11's own Self-Check pointed
// at but left as an exercise: a same-peer message queue that preserves
// issue order, contrasted with different-peer queues that don't need to.
//
// Compile: nvcc -arch=sm_80 127_nccl_point_to_point.cu -o 127_nccl_point_to_point -lnccl
// Run:     ./127_nccl_point_to_point
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>

int main() {
    printf("=== Section C.2: NCCL point-to-point (ncclSend/ncclRecv), "
           "consolidated from Chapter 11 ===\n\n");

    void* pSend = (void*)&ncclSend;
    void* pRecv = (void*)&ncclRecv;
    void* pGroupStart = (void*)&ncclGroupStart;
    void* pGroupEnd = (void*)&ncclGroupEnd;
    printf("real symbols resolved: ncclSend=%p ncclRecv=%p "
           "ncclGroupStart=%p ncclGroupEnd=%p\n\n",
           pSend, pRecv, pGroupStart, pGroupEnd);

    printf("Chapter 11's own real recipe, restated as a standalone "
           "reference (this is Chapter 10's own all-to-all, built the "
           "SAME way Chapter 11 found real NCCL itself builds it -- no "
           "dedicated ncclAllToAll() call exists):\n\n");
    printf("  ncclGroupStart();\n");
    printf("  for (int peer = 0; peer < nranks; peer++) {\n");
    printf("      ncclSend(sendbuf + peer*chunkSize, chunkSize, type, peer, comm, stream);\n");
    printf("      ncclRecv(recvbuf + peer*chunkSize, chunkSize, type, peer, comm, stream);\n");
    printf("  }\n");
    printf("  ncclGroupEnd();\n\n");

    printf("=== demonstrating Chapter 11's own Self-Check Q4 answer: "
           "same-peer calls stay ordered, different-peer calls don't "
           "need to ===\n\n");

    // A tiny host-side model of NCCL's own documented ordering guarantee:
    // one FIFO queue PER PEER. Two ncclSend() calls issued to the SAME
    // peer inside one group are modeled as two pushes onto that peer's
    // own queue -- FIFO order guarantees they are delivered in the order
    // issued. Two calls to DIFFERENT peers go to DIFFERENT queues, so
    // nothing about their relative order is constrained at all.
    const int NPEERS = 3;
    std::vector<std::vector<int>> perPeerQueue(NPEERS);

    struct Call { int peer; int tag; };
    std::vector<Call> issued = {
        {0, 100}, {0, 101}, {0, 102},   // three sends to the SAME peer (0)
        {1, 200},                        // one send to a DIFFERENT peer (1)
        {2, 300}, {2, 301},              // two sends to a THIRD peer (2)
    };

    printf("issuing, inside one ncclGroupStart()/ncclGroupEnd() group:\n");
    for (const auto& c : issued) {
        printf("  ncclSend(..., peer=%d, ...) tag=%d\n", c.peer, c.tag);
        perPeerQueue[c.peer].push_back(c.tag);
    }

    printf("\nper-peer delivery order (each peer's own queue is FIFO -- "
           "same-peer calls preserve issue order by construction):\n");
    bool samePeerOrdered = true;
    for (int p = 0; p < NPEERS; p++) {
        printf("  peer %d receives, in order: [", p);
        for (size_t i = 0; i < perPeerQueue[p].size(); i++)
            printf("%d%s", perPeerQueue[p][i], i + 1 < perPeerQueue[p].size() ? ", " : "");
        printf("]\n");
    }
    // Same-peer ordering check: peer 0's own queue must equal [100,101,102]
    // in exactly that order (the order they were issued in).
    int expectedPeer0[3] = {100, 101, 102};
    for (int i = 0; i < 3; i++)
        if (perPeerQueue[0][i] != expectedPeer0[i]) samePeerOrdered = false;

    printf("\nnote what this model does NOT constrain: peer 1's own "
           "single call and peer 2's own two calls could be scheduled "
           "and delivered in any relative order with respect to peer "
           "0's own three calls -- NCCL's documented guarantee covers "
           "ONLY same-peer ordering, exactly Chapter 11's own Self-Check "
           "Q4 answer, never cross-peer ordering.\n");

    printf("\nself-check: peer 0's own three same-peer sends were "
           "delivered in exactly the order they were issued (100, 101, "
           "102): %s\n", samePeerOrdered ? "confirmed" : "MISMATCH");

    printf("\n=== attempting a genuine communicator for a real "
           "ncclSend/ncclRecv call ===\n\n");
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm;
    ncclResult_t initErr = ncclCommInitRank(&comm, 1, id, 0);
    printf("ncclCommInitRank() -> %s (%s)\n", ncclGetErrorString(initErr),
           initErr == ncclSuccess ? "succeeded" :
           "same real, honest zero-physical-GPU limitation as Section C.1 "
           "and every device-touching file since Chapter 2");

    return samePeerOrdered ? 0 : 1;
}
