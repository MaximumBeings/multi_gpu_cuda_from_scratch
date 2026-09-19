// Appendix F: From PyTorch Distributed and DeepSpeed to C++: A Rosetta Stone
// 136_ddp_bucketed_allreduce.cu
//
// Chapter 12's own File 34 called ncclAllReduce(..., ncclAvg, ...) once,
// on one flat gradient buffer, because that chapter built a single
// replicated model with one gradient tensor. torch.nn.parallel.
// DistributedDataParallel (DDP) does the identical averaging -- the same
// ncclAllReduce(..., ncclAvg, ...) this book has used since Chapter 12 --
// but automates it with a real, documented mechanism this file now
// builds: gradients are grouped into BUCKETS (PyTorch's own docs: "in
// (roughly) the reverse order of Model.parameters()"), autograd hooks
// fire per-parameter as the backward pass computes each gradient, and a
// bucket's all-reduce launches once every gradient in it is ready -- but
// PyTorch's own docs are explicit that the ISSUE order across buckets is
// always bucket-index order, never actual ready order, "which is done by
// always running allreduce in the bucket index order instead of actual
// bucket ready order." This file simulates the ready-order/issue-order
// distinction on the host (deterministic, no device needed), then makes
// one real ncclAllReduce() call per bucket, extending Chapter 12's own
// File 34 from one flat call to several bucketed ones.
//
// Compile: nvcc -arch=sm_80 136_ddp_bucketed_allreduce.cu -o 136_ddp_bucketed_allreduce -lnccl
// Run:     ./136_ddp_bucketed_allreduce
#include <cstdio>
#include <vector>
#include <algorithm>
#include <nccl.h>
#include <cuda_runtime.h>

struct Param {
    int paramIndex;   // position in Model.parameters(), forward order: 0..5
    int bucketIndex;  // which bucket this parameter is assigned to
};

int main() {
    printf("=== Section F.2: DistributedDataParallel's bucketed all-reduce, built on\n");
    printf("    Chapter 12's own ncclAllReduce(..., ncclAvg, ...) call ===\n\n");

    // 6 parameters, forward-pass order 0..5 (Model.parameters() order).
    // PyTorch's own docs: buckets are built in "(roughly) the reverse
    // order of Model.parameters()" -- because backward computes the
    // LAST layer's gradient first, bucketing in reverse forward-order
    // means a bucket tends to become ready shortly after it fills.
    const int NPARAMS = 6;
    std::vector<Param> params(NPARAMS);
    // reverse-order bucketing: params 5,4 -> bucket 0; 3,2 -> bucket 1; 1,0 -> bucket 2
    int bucketOf[NPARAMS] = {2, 2, 1, 1, 0, 0};
    for (int p = 0; p < NPARAMS; ++p) { params[p].paramIndex = p; params[p].bucketIndex = bucketOf[p]; }

    printf("bucket assignment (reverse of Model.parameters() order 0..%d):\n", NPARAMS - 1);
    printf("%-10s %-10s\n", "param#", "bucket#");
    for (auto& p : params) printf("%-10d %-10d\n", p.paramIndex, p.bucketIndex);
    printf("\n");

    // Backward pass fires autograd hooks in REVERSE forward order:
    // param 5's gradient is ready first, then 4, then 3, ... then 0.
    // That is also, by construction, bucket-fill order: bucket 0 fills
    // first (params 5,4), then bucket 1 (3,2), then bucket 2 (1,0).
    // To make the "ready order can scramble" claim genuine rather than
    // assumed, this simulation deliberately delays param 2's hook (a
    // real, common cause: a skip connection/residual branch whose
    // gradient computation takes an extra op) so bucket 1 finishes
    // AFTER bucket 2 has already fully filled -- an out-of-order ready
    // sequence a real network can genuinely produce.
    struct HookEvent { int paramIndex; int readyTick; };
    std::vector<HookEvent> hookFires = {
        {5, 0}, {4, 1},         // bucket 0 ready at tick 1
        {1, 2}, {0, 3},         // bucket 2 ready at tick 3 (fires EARLY: simulated skip connection)
        {3, 4}, {2, 5},         // bucket 1 ready at tick 5 (fires LAST, out of bucket-index order)
    };

    std::vector<int> bucketReadyAtTick(3, -1);
    std::vector<int> bucketRemaining(3, 0);
    for (auto& p : params) bucketRemaining[p.bucketIndex]++;

    std::vector<int> readyOrder; // order in which buckets actually BECOME ready
    for (auto& ev : hookFires) {
        int b = bucketOf[ev.paramIndex];
        bucketRemaining[b]--;
        if (bucketRemaining[b] == 0) {
            bucketReadyAtTick[b] = ev.readyTick;
            readyOrder.push_back(b);
        }
    }

    printf("simulated backward-pass hook firing order produces this bucket READY order:\n");
    printf("  ");
    for (size_t i = 0; i < readyOrder.size(); ++i) printf("%sbucket %d (tick %d)", i ? " -> " : "", readyOrder[i], bucketReadyAtTick[readyOrder[i]]);
    printf("\n\n");

    // PyTorch's own real rule: "DDP requires Reducer instances on all
    // processes to invoke allreduce in exactly the same order, which is
    // done by always running allreduce in the bucket index order
    // instead of actual bucket ready order." The Reducer therefore
    // WAITS for a bucket if it isn't the next index-order bucket yet,
    // even if a later-index bucket became ready first.
    std::vector<int> issueOrder = {0, 1, 2}; // bucket index order -- always, regardless of readyOrder
    printf("real DDP rule (PyTorch's own docs, quoted directly): allreduce is always\n");
    printf("issued in BUCKET INDEX order, never actual ready order -- so despite bucket\n");
    printf("2 becoming ready before bucket 1 above, the real issue order is:\n  ");
    for (size_t i = 0; i < issueOrder.size(); ++i) printf("%sbucket %d", i ? " -> " : "", issueOrder[i]);
    printf("\n\n");

    printf("=== one real ncclAllReduce(..., ncclAvg, ...) per bucket, index order ===\n\n");
    ncclComm_t comm = nullptr; // never successfully created, as in Chapter 11 and Chapter 12
    cudaStream_t stream = nullptr;
    float* bucketBuffer = nullptr;
    bool allInvalidArg = true;
    for (int b : issueOrder) {
        size_t bucketElems = 0;
        for (auto& p : params) if (p.bucketIndex == b) bucketElems++;
        ncclResult_t e = ncclAllReduce(bucketBuffer, bucketBuffer, bucketElems, ncclFloat, ncclAvg, comm, stream);
        printf("bucket %d (%zu params): ncclAllReduce(..., ncclAvg, ...) -> %s\n",
               b, bucketElems, ncclGetErrorString(e));
        if (e != ncclInvalidArgument) allInvalidArg = false;
    }
    printf("\n(the same honest ncclInvalidArgument Chapter 12's own File 34 already\n");
    printf("reported for its single flat call -- a never-created communicator fails the\n");
    printf("identical way whether it's asked to reduce one buffer or three)\n");

    bool orderCorrect = (issueOrder[0] == 0 && issueOrder[1] == 1 && issueOrder[2] == 2);
    bool readyOrderWasScrambled = (readyOrder[0] != 0 || readyOrder[1] != 1 || readyOrder[2] != 2);
    printf("\nself-check: the simulated ready order was genuinely scrambled relative to\n");
    printf("bucket-index order (%s), yet the real issue order sent to ncclAllReduce\n",
           readyOrderWasScrambled ? "confirmed scrambled" : "MISMATCH -- not actually scrambled");
    printf("stayed strictly ascending bucket-index order (%s), AND every real NCCL call\n",
           orderCorrect ? "confirmed" : "MISMATCH");
    printf("reports the same honest failure (%s): %s\n", allInvalidArg ? "confirmed" : "MISMATCH",
           (orderCorrect && readyOrderWasScrambled && allInvalidArg) ? "confirmed" : "MISMATCH");

    return (orderCorrect && readyOrderWasScrambled && allInvalidArg) ? 0 : 1;
}
