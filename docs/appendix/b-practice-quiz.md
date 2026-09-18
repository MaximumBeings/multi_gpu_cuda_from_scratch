# Appendix B: Practice Quiz

Every chapter in this book already ends with its own Self-Check Questions and Worked Solutions, testing that chapter's own material in isolation. This appendix is different: it tests whether the ideas actually connect across chapters -- whether you can look at Chapter 40's own backprojection/forward-projection split and recognize it as the same "same partition, different operation, different communication need" shape Chapter 34's gated-versus-fixed all-to-all raises, or explain why Chapter 27's own communication volume grows with GPU count while Chapter 16's shrinks. B.2 asks sixteen conceptual questions, two from each of this book's eight Parts, with answers in B.3. B.4 raises the stakes: three short, genuinely compiled programs, each built around a real finding this book established somewhere -- read the setup, commit to a prediction, then compile and run it yourself before reading the revealed output.

## B.1 How to Use This Quiz

Answer every question in B.2 in your own words -- out loud, or on paper -- before reading B.3's answer. For B.4, do the same with actual code: read the setup, write down what you believe the program will print, and only then compile it and compare. A prediction you get wrong tells you exactly which chapter to reread; a prediction you get right by guessing, without being able to explain why, is worth treating the same way. Every one of B.4's programs compiles and runs in seconds, entirely on the host, with no GPU required -- this book's entire method has been "genuinely compiled, genuinely run, never assumed," and this quiz is not an exception.

## B.2 Conceptual Review Questions

**Part 0 -- GPU Foundations and Topology (Chapters 1-3)**

1. Chapter 1 introduced both a "memory wall" and a "compute wall" as reasons a single GPU is not enough. What is the difference between the two, and why doesn't simply building a bigger single GPU solve either one indefinitely?
2. Chapter 2 established that PCIe, NVLink, and NVSwitch offer very different real bandwidth between GPUs, and Chapter 3 established that CUDA tracks "current device" per host thread rather than globally. Why do both facts matter for the same underlying reason?

**Part 1 -- Peer Access and Host-Orchestrated Transfers (Chapters 4-7)**

3. What does UVA (Unified Virtual Addressing, Chapter 4) actually give a program that plain P2P memory access alone doesn't?
4. Chapter 6 built cross-device synchronization using `cudaStreamWaitEvent()`. Why can't a simple `cudaDeviceSynchronize()` on the producing device alone guarantee correctness when a second device's kernel depends on the first device's result?

**Part 2 -- Collectives From Scratch, Then NCCL (Chapters 8-11)**

5. Chapter 9 built ring all-reduce out of two distinct phases (reduce-scatter, then all-gather) rather than one single pass. What would go wrong with a naive "everyone sends their full array to everyone else, then locally sums" approach at scale, and how does splitting into these two phases fix it?
6. NCCL (Chapter 11) doesn't provide a dedicated Barrier primitive. Chapter 17 later built one anyway. What did it use, and why is that the smallest real collective NCCL does provide that can stand in for a barrier?

**Part 3 -- Parallelization Strategies (Chapters 12-16)**

7. Chapter 13 (model parallelism) and Chapter 14 (tensor parallelism) both split one model across multiple GPUs, but they split it along fundamentally different axes. What's the difference?
8. Chapter 16's domain decomposition uses halo/ghost cells, and Chapter 15's pipeline parallelism uses micro-batching to reduce a "bubble." What kind of inefficiency is each one specifically fighting, and why are they different problems even though both involve idle time?

**Part 4 -- Synchronization, Load Balancing, and Fault Tolerance (Chapters 17-19)**

9. Chapter 18 built load balancing around each GPU's own `cudaGetDeviceProperties()`, rather than assuming every GPU in a cluster is identical. Why does this matter even in a cluster where every GPU is nominally "the same model"?
10. Chapter 19 described handling a straggler or failed rank using `ncclCommGetAsyncError()` and `ncclCommAbort()` rather than anything resembling error recovery. Why can NCCL only offer "detect and escape," not genuine fault tolerance, the way Chapter 20's MPI ULFM discussion contrasts against?

**Part 5 -- Scaling Beyond One Node (Chapters 20-23)**

11. Chapter 20 built a naive MPI ring send/recv that deadlocks at scale, fixed with `MPI_Sendrecv()`. What specifically causes the naive version to deadlock, and why does combining send and receive into one call fix it rather than merely reordering the existing calls?
12. NVSHMEM (Chapter 22) let a CUDA kernel call `nvshmem_int_p()` to write directly into another GPU's memory from device code. How is this fundamentally different from every earlier chapter's own communication, including GPUDirect RDMA (Chapter 21)?

**Part 6 -- Case Studies I (Chapters 24-31)**

13. Chapter 27 (N-body) and Chapter 16's halo-based domain decomposition move in opposite directions as GPU count grows -- Chapter 27's own communication volume GROWS with P, while Chapter 16's shrinks (or stays bounded). Why the opposite behavior?
14. Chapter 30's ray tracing case study found a "sort-last" communication pattern with literally zero communication during the actual rendering computation. What later forces communication back into the picture, and how is that different from every earlier chapter's own reason for needing communication?

**Part 7 -- Industrial and Healthcare Case Studies (Chapters 32-40)**

15. Chapter 32 (MoE/DeepEP) and Chapter 34 (recommendation systems/HugeCTR) both use all-to-all communication, but for structurally different reasons. What's the difference?
16. Chapter 40's medical imaging case study found that the exact same z-slab domain decomposition needs zero communication for backprojection but real cross-node communication for forward projection. Since it's the same partition of the same volume, why do the two operations differ so sharply?

## B.3 Conceptual Review Answers

**1.** The memory wall is that a model's own size (parameters plus optimizer state, per Chapter 1's own ZeRO memory formula) can outgrow a single GPU's own HBM capacity regardless of how fast that GPU computes; the compute wall is that training the largest real models (GPT-3-class, per Chapter 1's own cited FLOPs and H100 throughput figures) would take an impractically long wall-clock time on one GPU even if that GPU somehow had unlimited memory. Building a bigger single GPU pushes both walls further out but never removes them, because real model and dataset sizes have historically grown faster than any single chip's own capacity or throughput -- multiple GPUs working together is the only way to keep memory and compute scaling roughly in step with model growth.

**2.** Both facts are about the same underlying reality: "the GPU" is not one monolithic resource a program can address uniformly. Chapter 2's topology fact means the exact same collective operation can run dramatically faster or slower purely depending on which physical link two participating GPUs happen to be connected by, something invisible to the algorithm's own correctness. Chapter 3's per-thread current-device fact means a host thread's own calls (kernel launches, `cudaMalloc`, etc.) silently target whichever device that thread most recently selected, so two threads can each believe they're operating on "the GPU" while actually operating on two different physical devices. Both facts share the same lesson: a multi-GPU program must be explicit about which physical device and which link any given operation actually uses, because nothing about a naive single-GPU mental model implies any of that by default.

**3.** P2P access is the CAPABILITY for one GPU to read or write another GPU's memory at all, once explicitly enabled via `cudaDeviceEnablePeerAccess()`. UVA is what makes a single pointer VALUE meaningful regardless of which device it was allocated on or which device is currently dereferencing it, because every device's own address space is mapped into one shared virtual address range. Without UVA, a program would need to track, separately from the pointer itself, which device that pointer belongs to, and explicitly select that device before using it; with UVA, `cudaMemcpyPeer()` and P2P kernel access can treat a pointer opaquely, and the runtime resolves which physical device it actually lives on.

**4.** `cudaDeviceSynchronize()` blocks the HOST thread until that one device's own enqueued work finishes, but it does nothing to order operations already enqueued on a DIFFERENT device's own stream -- if the consuming device's kernel was already launched (even if it hasn't started running yet), nothing guarantees the GPU scheduler won't run it before the producing device's write has actually landed in memory the consumer can see, unless an explicit device-side dependency exists. `cudaStreamWaitEvent()` creates that explicit dependency directly between the two devices' own streams (make this stream wait for an event recorded on the other device), enforced by the GPU hardware and driver itself, rather than relying on the host thread's own synchronous blocking as a proxy for a cross-device ordering guarantee it cannot actually provide.

**5.** A naive "send full array to everyone, then locally sum" approach moves the entire array O(P) times per rank (once to each of the other P-1 ranks), so total data moved scales up badly as P grows. Ring reduce-scatter and all-gather instead move each rank's own data only ONCE around the ring per phase, and each message is only 1/P of the full array rather than the whole thing, so total data moved per rank converges to roughly 2(P-1)/P times the array size regardless of how many ranks participate -- Chapter 9's own real cost formula. Splitting into phases is what makes that reduced volume possible: reduce-scatter's own job is purely to get each chunk fully reduced exactly once, never redundantly, and all-gather's own job is purely to redistribute those already-final chunks, rather than repeatedly re-transmitting not-yet-final partial sums the way a naive approach would.

**6.** Chapter 17 built a barrier from a throwaway single-element `ncclAllReduce()` call, immediately followed by a real `cudaStreamSynchronize()` -- every rank must contribute before the collective can complete, and waiting for that shared collective's completion is exactly the property a barrier needs (no rank proceeds past it until every rank has reached it). `ncclAllReduce()` is the smallest fit because it's the one collective every rank genuinely must participate in symmetrically (unlike, say, `ncclBroadcast()`, which has one distinguished root and doesn't force every other rank to contribute data), so its own inherent all-participate requirement does the barrier's actual job for free -- the reduced value itself is simply discarded.

**7.** Model parallelism (Chapter 13) splits a model DEPTH-wise -- different layers (or groups of layers) live on different GPUs, and activations are handed off, via `cudaMemcpyPeer()`, from one GPU to the next as data flows forward through the network, so at any instant each GPU is doing a different part of the SAME forward pass. Tensor parallelism (Chapter 14) instead splits WITHIN a single layer -- for example, splitting one large matrix multiply's own columns or rows across GPUs -- so every participating GPU computes on the same layer simultaneously, each owning only a slice of that layer's own weights, requiring a collective (`ncclAllReduce()`, for row-parallel) to combine partial results back together before the next layer can proceed.

**8.** Chapter 16's halo and ghost cells exist to minimize COMMUNICATION VOLUME for a spatially local computation -- a stencil update needs only its immediate neighbors' boundary data, not the whole remote domain -- so the inefficiency being fought is wasted bandwidth from over-communicating. Chapter 15's pipeline bubble is a COMPUTE-UTILIZATION problem -- with only one microbatch in flight, most pipeline stages sit idle waiting for data to reach them -- so the inefficiency being fought is wasted GPU time from under-utilization, fixed by keeping multiple microbatches in flight simultaneously (the `(K-1)/(M+K-1)` bubble fraction Chapter 15 derived, which shrinks as more microbatches M are used). One is a communication-volume problem; the other is a compute-idle-time problem, and the two don't trade off against each other in general.

**9.** In practice, even nominally-identical GPUs can have real, measurable performance differences from thermal throttling, differing power limits, differing firmware or driver versions, or simply age-related clock degradation -- and a heterogeneous cluster explicitly mixing GPU generations, an increasingly common real deployment shape, makes the differences even starker. Chapter 18's weighted-sharding approach queries each device's OWN reported capability directly and assigns work proportionally, rather than assuming a naive equal split is optimal -- an equal split leaves the slowest device as the critical-path bottleneck while faster devices sit idle waiting for it, exactly the avoidable inefficiency real production multi-GPU clusters are built to avoid.

**10.** Once a NCCL communicator's own internal collective state becomes inconsistent -- a rank died mid-collective, say -- there is no way to resynchronize the surviving ranks' own view of that collective's progress; NCCL's communicators are not designed to be repaired, only destroyed and, at best, rebuilt from scratch with a smaller or different rank set. `ncclCommAbort()` is explicitly a one-way escape hatch: it tears the broken communicator down immediately rather than attempting to complete or roll back the in-flight collective. MPI's ULFM (User-Level Fault Mitigation) extension, by contrast, is specifically designed to let a surviving set of ranks revalidate a communicator and continue operating with the failed ranks excised -- a genuinely different design goal NCCL's own API was never built around.

**11.** A naive ring where every rank calls a BLOCKING `MPI_Send()` before its own `MPI_Recv()` deadlocks because `MPI_Send()` itself can block until a matching receive is posted on the other side -- if every rank in the ring is simultaneously stuck waiting inside its own `MPI_Send()`, none of them will ever reach the `MPI_Recv()` call that would unblock its neighbor, a genuine circular wait. Simply reordering (receive-then-send on alternating ranks) can dodge the deadlock for particular ring sizes but is fragile and doesn't generalize; `MPI_Sendrecv()` instead performs both operations as one atomic MPI operation that the MPI implementation itself is responsible for completing safely, removing the deadlock risk structurally rather than by careful manual ordering.

**12.** Every earlier communication mechanism in this book -- `cudaMemcpyPeer()`, NCCL collectives, MPI messages, even GPUDirect RDMA's own optimized data path -- is initiated by the HOST, meaning a CPU thread issues the API call that starts the transfer, even when the actual bytes move GPU-to-GPU without touching host memory. NVSHMEM's `nvshmem_int_p()` is called from INSIDE a running GPU kernel, by a GPU thread, with no host involvement in initiating that specific write at all -- the GPU itself decides, mid-kernel, to write into a remote GPU's memory, opening up communication patterns (fine-grained, data-dependent point-to-point messaging deep inside a kernel) that a host-orchestrated model structurally cannot express without stopping the kernel and returning control to the host first.

**13.** Chapter 16's stencil computation is spatially LOCAL -- each cell only ever needs its immediate neighbors, so as P grows and each rank's own local domain shrinks, the halo boundary that gets communicated shrinks right along with it (though Chapter 16 itself flagged the eventual problem when the halo stops shrinking as fast as the interior does). Chapter 27's naive all-pairs N-body force calculation is the opposite: every particle's force depends on EVERY other particle in the whole system, regardless of physical proximity, so as P grows, each rank needs a full copy of every other rank's own particle set to compute its own particles' forces -- an all-gather whose total volume genuinely grows with the number of ranks being gathered from, unlike a fixed-size local halo.

**14.** During rendering itself, each GPU independently ray-traces its own assigned tile of the same fully-replicated scene, needing no data from any other GPU at all -- communication only reappears at the very end, to composite each GPU's own rendered tile into one final image, a form of gather rather than a computational dependency. What forces communication back DURING computation, per Chapter 30's own real R2E2 terabyte-scale-scene citation, is when the SCENE ITSELF stops fitting on one GPU's own memory -- once no single GPU can hold the whole scene, rendering a ray that crosses into another GPU's own scene partition requires fetching that partition's geometry data mid-render, a fundamentally different reason for communication (data doesn't fit) than every earlier chapter's reason (an algorithm's own mathematical dependency structure).

**15.** Chapter 32's MoE all-to-all is GATED -- which expert, and therefore which GPU, each token is routed to depends on a learned gating function's own per-token decision, so the exact communication pattern varies from batch to batch and even includes a real capacity-overflow and residual-passthrough case when an expert receives more tokens than it has room for. Chapter 34's HugeCTR embedding all-to-all is FIXED -- every batch reshards embedding lookups along the same predetermined schedule with no gating and no possibility of overflow, because which embedding table shard a given feature belongs to is a static property of the model's own architecture, not a per-batch learned decision. Both are the same primitive (all-to-all), but one has a dynamic, data-dependent routing table and the other doesn't.

**16.** Backprojection's own real cited property is that "each node locally stores the part of the detector data needed to perform a BP operation, so this can be performed locally and independently" -- every voxel a given rank owns can be fully reconstructed using only that same rank's own already-local detector data, with no dependency on any other rank's slab at all. Forward projection's own geometry is different: computing a projection ray's contribution on the detector can require volume data from multiple z-slabs at once, because a single ray can pass through voxels owned by different ranks as it traverses the volume -- the partition is identical, but which OPERATION is applied to that partition determines whether a given rank's own local data is sufficient by itself, or whether the operation's own geometry inherently reaches across partition boundaries.

## B.4 Predict-the-Output Challenges

### Challenge 1: Reduce-Scatter's Own Intermediate State

Chapter 9 built ring all-reduce as two distinct phases, and it's easy to blur them together in memory the way an inclusive and an exclusive scan can blur together. Before compiling and running the program below, predict: for P=4 ranks, each starting with a 4-element array where rank r's array is `[10*(r+1)+0, 10*(r+1)+1, 10*(r+1)+2, 10*(r+1)+3]`, which ONE chunk index does rank 2 hold fully-reduced immediately after reduce-scatter finishes (before all-gather begins), and what is that chunk's value?

```cpp
// 123_quiz_ring_allreduce_phases.cpp
//
// Appendix B.4, Challenge 1 -- Chapter 9 built ring all-reduce as TWO
// distinct phases: reduce-scatter (P-1 steps, after which each rank
// holds the FULLY reduced result for only ONE chunk) followed by
// all-gather (P-1 more steps, after which every rank holds the complete
// reduced result). Before compiling and running this file, predict: for
// P=4 ranks, each starting with a 4-element array where
// rank r's array is [10*(r+1)+0, 10*(r+1)+1, 10*(r+1)+2, 10*(r+1)+3],
// which ONE chunk index does rank 2 hold fully-reduced immediately after
// reduce-scatter finishes (before all-gather begins), and what is that
// chunk's value?
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 123_quiz_ring_allreduce_phases.cpp -o 123_quiz_ring_allreduce_phases
// Run:     ./123_quiz_ring_allreduce_phases
#include <cstdio>
#include <vector>

constexpr int P = 4;

void printState(const std::vector<std::vector<int>>& buf, const char* label) {
    printf("%s\n", label);
    for (int r = 0; r < P; r++) {
        printf("  rank %d: [", r);
        for (int i = 0; i < P; i++) printf("%d%s", buf[r][i], i + 1 < P ? ", " : "");
        printf("]\n");
    }
    printf("\n");
}

int main() {
    // rank r's initial array: [10*(r+1)+0, 10*(r+1)+1, 10*(r+1)+2, 10*(r+1)+3]
    std::vector<std::vector<int>> buf(P, std::vector<int>(P));
    for (int r = 0; r < P; r++)
        for (int i = 0; i < P; i++)
            buf[r][i] = 10 * (r + 1) + i;

    printState(buf, "=== initial state (before reduce-scatter) ===");

    // Ground truth: fully all-reduced (summed) value for each chunk index.
    std::vector<int> trueSum(P, 0);
    for (int i = 0; i < P; i++)
        for (int r = 0; r < P; r++)
            trueSum[i] += buf[r][i];
    printf("=== ground truth: elementwise sum across all 4 ranks ===\n  [");
    for (int i = 0; i < P; i++) printf("%d%s", trueSum[i], i + 1 < P ? ", " : "");
    printf("]\n\n");

    // Reduce-scatter: P-1 steps. Standard ring convention -- rank r sends
    // chunk (r - step + P) % P to its right neighbor, which ADDS it into
    // its own copy of that same chunk. After P-1 steps, rank r holds the
    // FULLY reduced value for chunk (r + 1) % P (the chunk immediately
    // "ahead" of it in ring order, following Chapter 9's own indexing).
    for (int step = 0; step < P - 1; step++) {
        std::vector<std::vector<int>> next = buf;
        for (int r = 0; r < P; r++) {
            int sendChunk = (r - step + P) % P;
            int recvRank = (r + 1) % P;
            next[recvRank][sendChunk] += buf[r][sendChunk];
        }
        buf = next;
    }

    printState(buf, "=== after reduce-scatter (P-1 = 3 steps) ===");
    int fullyReducedChunkOnRank2 = (2 + 1) % P;
    printf("rank 2's own fully-reduced chunk is chunk index %d, value = %d "
           "(matches ground truth [%d]=%d)\n\n",
           fullyReducedChunkOnRank2, buf[2][fullyReducedChunkOnRank2],
           fullyReducedChunkOnRank2, trueSum[fullyReducedChunkOnRank2]);

    // All-gather: P-1 more steps, circulating each rank's own fully-
    // reduced chunk around the ring so every rank ends up with all P
    // fully-reduced chunks.
    for (int step = 0; step < P - 1; step++) {
        std::vector<std::vector<int>> next = buf;
        for (int r = 0; r < P; r++) {
            int sendChunk = (r + 1 - step + P) % P;
            int recvRank = (r + 1) % P;
            next[recvRank][sendChunk] = buf[r][sendChunk];
        }
        buf = next;
    }

    printState(buf, "=== after all-gather (P-1 = 3 more steps) -- every "
                     "rank now holds the full result ===");

    bool allMatch = true;
    for (int r = 0; r < P; r++)
        for (int i = 0; i < P; i++)
            if (buf[r][i] != trueSum[i]) allMatch = false;

    printf("self-check: every rank's final buffer matches ground truth: %s\n",
           allMatch ? "YES" : "NO");
    return allMatch ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 123_quiz_ring_allreduce_phases.cpp -o 123_quiz_ring_allreduce_phases
./123_quiz_ring_allreduce_phases
```

**Revealed output:**

```text
=== initial state (before reduce-scatter) ===
  rank 0: [10, 11, 12, 13]
  rank 1: [20, 21, 22, 23]
  rank 2: [30, 31, 32, 33]
  rank 3: [40, 41, 42, 43]

=== ground truth: elementwise sum across all 4 ranks ===
  [100, 104, 108, 112]

=== after reduce-scatter (P-1 = 3 steps) ===
  rank 0: [10, 104, 86, 56]
  rank 1: [30, 21, 108, 79]
  rank 2: [60, 52, 32, 112]
  rank 3: [100, 93, 74, 43]

rank 2's own fully-reduced chunk is chunk index 3, value = 112 (matches ground truth [3]=112)

=== after all-gather (P-1 = 3 more steps) -- every rank now holds the full result ===
  rank 0: [100, 104, 108, 112]
  rank 1: [100, 104, 108, 112]
  rank 2: [100, 104, 108, 112]
  rank 3: [100, 104, 108, 112]

self-check: every rank's final buffer matches ground truth: YES
```

Only chunk index 3 is fully reduced on rank 2 right after reduce-scatter -- every other entry in rank 2's own buffer at that point is still a partial, in-progress sum, not yet the final answer, exactly the distinction the two-phase design exists to make precise.

### Challenge 2: Reduction Order Is Not Free

Chapters 25, 29, and 31 each independently rediscovered that floating-point addition is not associative -- summing the same values in a different order, a real and unavoidable consequence of which rank's partial result a reduction combines first, can produce a genuinely different final bit pattern. Before compiling and running the program below, predict: given the four doubles `a=1e16, b=1.0, c=-1e16, d=1.0`, does the left-to-right sequential sum `((a+b)+c)+d` equal the pairwise sum `(a+c)+(b+d)`? Both orders add the exact same four numbers.

```cpp
// 124_quiz_reassociation_cancellation.cpp
//
// Appendix B.4, Challenge 2 -- Chapters 25, 29, and 31 each independently
// rediscovered that floating-point addition is not associative: summing
// the SAME set of values in a different ORDER (a real, unavoidable
// consequence of which rank's partial result a reduction combines first)
// can produce a genuinely different final bit pattern, not just a
// hypothetical rounding worry. Before compiling and running this file,
// predict: given the four doubles a=1e16, b=1.0, c=-1e16, d=1.0, does
// the LEFT-TO-RIGHT sequential sum ((a+b)+c)+d equal the PAIRWISE sum
// (a+c)+(b+d)? Both orders add the exact same four numbers.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 124_quiz_reassociation_cancellation.cpp -o 124_quiz_reassociation_cancellation
// Run:     ./124_quiz_reassociation_cancellation
#include <cstdio>

int main() {
    double a = 1e16, b = 1.0, c = -1e16, d = 1.0;

    printf("=== summing the same four doubles in two different orders ===\n");
    printf("a=%.1f, b=%.1f, c=%.1f, d=%.1f\n\n", a, b, c, d);

    // Sequential, left-to-right (the order Chapter 25's naive baseline
    // combines ring-forwarded partial sums in).
    double step1 = a + b;
    double step2 = step1 + c;
    double sequential = step2 + d;
    printf("sequential ((a+b)+c)+d:\n");
    printf("  a+b       = %.1f  (b=1.0 is far smaller than a's own ULP at "
           "this magnitude -- it vanishes in the rounding)\n", step1);
    printf("  (a+b)+c   = %.1f\n", step2);
    printf("  ((a+b)+c)+d = %.1f\n\n", sequential);

    // Pairwise, the order Chapter 31's tree-style combine groups values in.
    double left = a + c;
    double right = b + d;
    double pairwise = left + right;
    printf("pairwise (a+c)+(b+d):\n");
    printf("  a+c       = %.1f  (equal magnitude, opposite sign -- exact "
           "cancellation, no rounding loss)\n", left);
    printf("  b+d       = %.1f\n", right);
    printf("  (a+c)+(b+d) = %.1f\n\n", pairwise);

    printf("sequential result = %.1f\n", sequential);
    printf("pairwise result   = %.1f\n", pairwise);
    printf("difference        = %.1f\n\n", pairwise - sequential);

    bool match = (sequential == pairwise);
    printf("self-check: do the two orders agree? %s -- both orders sum "
           "the IDENTICAL four numbers, so any difference is entirely a "
           "property of ROUND-OFF ORDER, not of the data itself. This is "
           "the exact same real mechanism Chapters 25/29/31 each found: "
           "a ring's own sequential forwarding order and a tree-style "
           "pairwise combine order are NOT required to produce bit-"
           "identical sums, even though both are mathematically \"the "
           "same\" reduction.\n",
           match ? "YES (unexpected for these values)" : "NO");
    return 0;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 124_quiz_reassociation_cancellation.cpp -o 124_quiz_reassociation_cancellation
./124_quiz_reassociation_cancellation
```

**Revealed output:**

```text
=== summing the same four doubles in two different orders ===
a=10000000000000000.0, b=1.0, c=-10000000000000000.0, d=1.0

sequential ((a+b)+c)+d:
  a+b       = 10000000000000000.0  (b=1.0 is far smaller than a's own ULP at this magnitude -- it vanishes in the rounding)
  (a+b)+c   = 0.0
  ((a+b)+c)+d = 1.0

pairwise (a+c)+(b+d):
  a+c       = 0.0  (equal magnitude, opposite sign -- exact cancellation, no rounding loss)
  b+d       = 2.0
  (a+c)+(b+d) = 2.0

sequential result = 1.0
pairwise result   = 2.0
difference        = 1.0

self-check: do the two orders agree? NO -- both orders sum the IDENTICAL four numbers, so any difference is entirely a property of ROUND-OFF ORDER, not of the data itself. This is the exact same real mechanism Chapters 25/29/31 each found: a ring's own sequential forwarding order and a tree-style pairwise combine order are NOT required to produce bit-identical sums, even though both are mathematically "the same" reduction.
```

The two orders don't merely round differently in some tiny last digit -- they disagree by a full `1.0`, because `a` and `c` cancel EXACTLY when added directly to each other, but `b`'s own contribution is lost entirely when added to `a` first, since `b` falls below `a`'s own representable precision at that magnitude. This is deliberately the least subtle possible version of the same mechanism Chapters 25/29/31 found as a small, easy-to-miss single-bit difference -- the underlying cause is identical.

### Challenge 3: The Ghost-Cell Trap

Chapters 16 and 40 both warned that a halo or ghost-cell exchange must genuinely happen before a boundary cell is computed -- skipping it does not crash the program, it silently computes a wrong answer using stale data. Before compiling and running the program below, predict: for an 8-cell 1D array split across 2 ranks (4 interior cells each), how many of the 8 total cells does a version that never synchronizes ghost cells get wrong compared to a single-process reference -- and are they the interior cells, or specifically the ones nearest the rank boundary?

```cpp
// 125_quiz_ghost_cell_trap.cpp
//
// Appendix B.4, Challenge 3 -- Chapters 16 and 40 both warned that a
// halo/ghost-cell exchange must genuinely happen before a boundary cell
// is computed -- skipping it does not crash the program, it silently
// computes a WRONG answer using stale data. This file splits an 8-cell
// 1D array across 2 ranks (4 interior cells each), computes a 3-point
// smoothing average at every cell, and runs the SAME computation twice:
// once with a correct ghost-cell exchange across the rank boundary, and
// once where that exchange is skipped, leaving each rank's own ghost
// cell at its uninitialized starting value of 0. Before compiling and
// running this file, predict: how many of the 8 total cells does the
// no-sync version get WRONG compared to the single-process reference --
// and are they the interior cells, or specifically the ones nearest the
// rank boundary?
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 125_quiz_ghost_cell_trap.cpp -o 125_quiz_ghost_cell_trap
// Run:     ./125_quiz_ghost_cell_trap
#include <cstdio>
#include <vector>

constexpr int NX = 8;   // global array size
constexpr int P = 2;    // ranks
constexpr int LOCAL = NX / P;  // 4 interior cells per rank

double globalValue(int i) {
    // Deterministic, arbitrary values for cell i.
    return 10.0 + i;
}

double referenceSmooth(int i) {
    double sum = globalValue(i);
    int count = 1;
    if (i - 1 >= 0)  { sum += globalValue(i - 1); count++; }
    if (i + 1 < NX)  { sum += globalValue(i + 1); count++; }
    return sum / count;
}

int main() {
    printf("=== 8-cell array split across 2 ranks, 4 interior cells each "
           "===\n\n");

    // --- Correct version: ghost cells genuinely synchronized. ---
    std::vector<double> correct(NX);
    for (int rank = 0; rank < P; rank++) {
        int start = rank * LOCAL;
        for (int local = 0; local < LOCAL; local++) {
            int i = start + local;
            double left  = (i - 1 >= 0)  ? globalValue(i - 1) : globalValue(i);
            double right = (i + 1 < NX) ? globalValue(i + 1) : globalValue(i);
            int count = 1;
            double sum = globalValue(i);
            if (i - 1 >= 0)  { sum += left;  count++; }
            if (i + 1 < NX) { sum += right; count++; }
            correct[i] = sum / count;
        }
    }

    // --- Buggy version: each rank's own ghost cell is never fetched
    // from its neighbor -- it stays at its uninitialized starting value
    // of 0.0, exactly the "hard delete" of a real synchronization step.
    std::vector<double> buggy(NX);
    for (int rank = 0; rank < P; rank++) {
        int start = rank * LOCAL;
        int end = start + LOCAL;
        for (int local = 0; local < LOCAL; local++) {
            int i = start + local;
            double leftGhost = 0.0, rightGhost = 0.0;   // never synced
            double left, right;
            bool useLeftGhost = (i - 1 < start) && (i - 1 >= 0);
            bool useRightGhost = (i + 1 >= end) && (i + 1 < NX);
            left  = useLeftGhost  ? leftGhost  : ((i - 1 >= 0) ? globalValue(i - 1) : globalValue(i));
            right = useRightGhost ? rightGhost : ((i + 1 < NX) ? globalValue(i + 1) : globalValue(i));
            int count = 1;
            double sum = globalValue(i);
            if (i - 1 >= 0)  { sum += left;  count++; }
            if (i + 1 < NX) { sum += right; count++; }
            buggy[i] = sum / count;
        }
    }

    printf("%-6s %-14s %-14s %-10s\n", "cell", "reference", "correct-sync", "buggy(no-sync)");
    int mismatches = 0;
    for (int i = 0; i < NX; i++) {
        double ref = referenceSmooth(i);
        bool correctMatches = (correct[i] == ref);
        bool buggyMatches = (buggy[i] == ref);
        if (!buggyMatches) mismatches++;
        printf("%-6d %-14.4f %-14s %-10s\n", i, ref,
               correctMatches ? "matches" : "MISMATCH",
               buggyMatches ? "matches" : "MISMATCH (stale ghost=0)");
    }

    printf("\ntotal cells where the no-sync version disagrees with the "
           "single-process reference: %d out of %d\n", mismatches, NX);
    printf("those mismatches occur ONLY at the two rank-boundary cells "
           "(index 3 and index 4) -- every purely interior cell (0,1,2 "
           "and 5,6,7) never reads a neighbor across the rank boundary "
           "at all, so it is correct with or without the ghost-cell "
           "exchange. This is exactly Chapters 16 and 40's own real "
           "finding: a missing halo sync does not corrupt the whole "
           "domain, it silently corrupts precisely the cells whose "
           "correctness depends on it -- the boundary, and nowhere "
           "else.\n");
    return 0;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 125_quiz_ghost_cell_trap.cpp -o 125_quiz_ghost_cell_trap
./125_quiz_ghost_cell_trap
```

**Revealed output:**

```text
=== 8-cell array split across 2 ranks, 4 interior cells each ===

cell   reference      correct-sync   buggy(no-sync)
0      10.5000        matches        matches
1      11.0000        matches        matches
2      12.0000        matches        matches
3      13.0000        matches        MISMATCH (stale ghost=0)
4      14.0000        matches        MISMATCH (stale ghost=0)
5      15.0000        matches        matches
6      16.0000        matches        matches
7      16.5000        matches        matches

total cells where the no-sync version disagrees with the single-process reference: 2 out of 8
those mismatches occur ONLY at the two rank-boundary cells (index 3 and index 4) -- every purely interior cell (0,1,2 and 5,6,7) never reads a neighbor across the rank boundary at all, so it is correct with or without the ghost-cell exchange. This is exactly Chapters 16 and 40's own real finding: a missing halo sync does not corrupt the whole domain, it silently corrupts precisely the cells whose correctness depends on it -- the boundary, and nowhere else.
```

Exactly 2 of the 8 cells are wrong, and both are the cells immediately adjacent to the rank boundary -- not a handful of scattered errors, and not the entire domain. A missing synchronization step tends to look like this in practice: a small, localized, boundary-shaped wrongness that is easy to miss in a quick spot-check of the interior, and easy to misdiagnose as "probably fine" precisely because most of the output IS fine.

## Appendix Summary

B.2 and B.3 asked whether this book's individual ideas actually connect across Parts -- whether a reader can recognize the same underlying shape (a communication volume that shrinks with more ranks versus one that grows, a fixed schedule versus a gated one, a partition whose correctness depends on the operation built on top of it) recurring in case studies that look nothing alike on the surface, from GPU foundations all the way through Part 7's industrial and healthcare chapters. B.4 made three of the book's most consequential real findings concrete and checkable: confusing reduce-scatter's own intermediate state with the finished all-reduce result, underestimating how dramatically reduction order can change a floating-point answer, and skipping a halo synchronization step that silently corrupts only the cells nearest a rank boundary. All three compile and run in seconds, entirely on the host -- if a prediction did not match the revealed output, that mismatch is worth chasing back to the chapter that first built the idea, which this appendix has named at every step along the way.
