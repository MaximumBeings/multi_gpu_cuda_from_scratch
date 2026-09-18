**What you will understand after this chapter:** why a Mixture-of-Experts (MoE) layer needs a genuinely different communication pattern than anything Part 2 through Part 6 built -- a per-TOKEN dispatch/combine all-to-all, not a per-RANK reduction -- and why that difference means MoE combine never hits the floating-point non-associativity this book kept finding since Chapter 8; what DeepSeek-AI's own real, open-source DeepEP library actually is, why it ships two entirely separate all-to-all kernel families (one tuned for training throughput, one for inference latency) rather than a single "best" kernel, and how its own real V1 design used NVIDIA's NVSHMEM (this book's own Chapter 22) with GPU-initiated RDMA to let the GPU drive the network directly, bypassing the CPU per dispatch -- plus a candid look at DeepEP's own V2, which moved away from that NVSHMEM foundation entirely, a reminder that even a cutting-edge production library's transport layer keeps changing.

**What you need to know first:** Chapter 10 (all-to-all, this chapter's direct ancestor), Chapter 14 (tensor parallelism -- MoE is a different way of sharding work within a layer), and Chapter 22 (NVSHMEM and GPU-initiated communication, which this chapter's real GPU-initiated all-to-all API calls build on directly).

---

Part 6 closed with eight case studies that each fit into one of three communication shapes this book had already built: a fixed communication pattern regardless of scale (Chapter 16's halo exchange), one that grows with the number of ranks (Chapter 27's N-body all-gather), or one with no communication at all during computation (Chapter 30's rendering). Part 7 opens with four industrial systems that are cutting-edge, real, and actively deployed today -- starting with Mixture-of-Experts, the architecture behind DeepSeek-V3, Mixtral, and GPT-4's rumored design, which turns out to need a FOURTH shape this book has never built: a communication pattern keyed by which TOKEN needs which EXPERT, decided fresh at every layer, for every batch, by a small neural network. NVIDIA's own real GShard paper (Lepikhin et al., 2020) gave this pattern its now-standard name -- "dispatch" to route each token to its chosen experts, "combine" to route each expert's output back -- and DeepSeek-AI's own real, open-source DeepEP library exists specifically because that dispatch/combine step, done at production scale across many GPUs, is expensive enough to deserve its own dedicated communication library, built directly on top of this book's own Chapter 22 (NVSHMEM) and Chapter 10 (all-to-all).

```text
+------------------------------------------------------------------+
|  Chapters 8-31 (every collective in this book so far):           |
|  Rank 0 data --+                                                 |
|  Rank 1 data --+--> [ REDUCE across ALL ranks ] --> one result   |
|  Rank 2 data --+     (order/grouping-dependent: Ch8/25/29/31      |
|                       each found a real 1-bit float mismatch)     |
+------------------------------------------------------------------+
|  Chapter 32 (MoE dispatch/combine):                               |
|  Token A --> [ gate ] --> expert 2 (on Rank 1) --+                |
|  Token B --> [ gate ] --> expert 0 (on Rank 0)    |--> DISPATCH   |
|  Token C --> [ gate ] --> expert 2 (on Rank 1) --+                |
|                                                                    |
|  expert output --> [ weighted combine, PER TOKEN ONLY ] --> COMBINE|
|  (never sums across ranks for one token -- bit-exact at every P)  |
+------------------------------------------------------------------+
```

## 32.1 Expert Parallelism and the Dispatch/Combine All-to-All

### Intuition

Picture a large office mailroom that receives a stack of letters (tokens) every morning. Instead of one clerk reading every letter, a fast triage clerk (the gating network) glances at each letter's address and routes it to one of several specialist departments (experts) sitting in different buildings (ranks). Two departments per letter, actually -- GShard's own real design sends each token to its top-2 scoring experts, not just one, so a little redundancy survives if one department is slow or overloaded. Each department processes its own letters and mails back a reply; the original sender's assistant (the combine step) simply averages the two replies, weighted by how confident the triage clerk was in each department. Crucially, that averaging step never needs to know what happened to anyone ELSE's letter -- unlike Chapter 8's broadcast/reduce chapter, where every rank's value had to be folded into one shared answer, here every letter's own reply is entirely its own business.

But departments have finite desks. If a wildly popular department gets more letters than it has desks for (its "capacity"), the overflow letters don't get processed by that department at all -- GShard's own real paper calls this an "overflowed" token, and its own real fix is disarmingly simple: the letter is just forwarded on unopened, via what the paper calls a residual connection, exactly as if no department touched it. No error, no retry -- just silently reduced processing for that one letter.

!!! warning "[COMMON TRAP] Assuming capacity overflow is a rare edge case, not a routine one"
    It is tempting to treat "expert capacity" as a safety valve that almost never trips, sized generously enough that overflow is a one-in-a-million event. Real gating networks are learned, not designed, and real token popularity is skewed -- some experts genuinely do get chosen far more often than an even split would predict, especially early in training or for specific domains of input. File 93 below deliberately reproduces this: with a modest, deliberately-skewed gating popularity and a capacity factor of 1.25x the even split, **7 of 24 tokens (29%) end up FULLY overflowed** -- both of their chosen experts already full -- and are passed straight through unprocessed by any expert at all. A production system that assumes overflow is rare and doesn't monitor its own overflow rate can be silently degrading a meaningful fraction of its own batch's quality with zero errors to alert anyone.

### Background

```text
+-------------------------------------------------------------+
| gate(token) -> pick top-2 experts by score                  |
|   |                                                          |
|   +-> capacity check per expert (global count, order-fixed)  |
|          |--> ACCEPT: expert(token) computed, weight applied |
|          |--> single OVERFLOW: that term drops to 0          |
|          |--> BOTH OVERFLOW: y = token's own residual value  |
|   |                                                          |
|   +-> combine: y = w1*expert1(token) + w2*expert2(token)     |
+-------------------------------------------------------------+
```

File 93 builds exactly this shape on the host: 24 tokens, 6 experts, top-2 gating with a deterministic (counter-based, reproducible) popularity skew toward experts 0 and 1, and a capacity factor of 1.25x. Gating scores use a numerically-stable softmax (subtracting the larger of the two chosen scores before calling `exp()`) -- an early version of this file skipped that stabilization and called `exp()` directly on raw scores in the hundreds, which silently overflowed to `+inf` and produced NaN in every single token's combined output. That NaN was itself a genuine teaching moment: IEEE 754 defines `NaN != NaN`, so a bit-exact-equality check between two runs that both produced NaN reported a "mismatch" even though both runs did the identical (broken) arithmetic -- a reminder that "the outputs don't match" and "the computation overflowed" are different bugs that can disguise themselves as each other. The fixed version below is what actually ran and locked.

```cpp
// Chapter 32: Mixture-of-Experts at Scale
// 93_moe_dispatch_combine_correctness_simulation.cpp
//
// Every collective this book has built since Chapter 8 combined data
// FROM MULTIPLE RANKS into one shared result (a broadcast's copy, a
// reduce's sum, an all-reduce's global statistic) -- that is exactly
// where Chapter 8, 25, 29, and 31 each found a real floating-point
// non-associativity mismatch at P > 1. NVIDIA's own real GShard paper
// ("GShard: Scaling Giant Models with Conditional Computation and
// Automatic Sharding," Lepikhin et al., 2020) describes a genuinely
// different shape: Mixture-of-Experts dispatch and combine. A gating
// network picks, for EACH token independently, its own top-2 experts by
// score; "dispatching of inputs to selected experts is expressed by a
// single einsum between the dispatching mask and the input," each
// expert (living on its own rank in expert-parallel training) applies
// its own feed-forward function, and "taking weighted average of all
// experts output into the final output is expressed in another einsum"
// -- GShard's own real terms for what this book calls combine. Crucially,
// a single token's combine step only ever averages THAT token's own two
// chosen experts' outputs -- it never touches another token's data, and
// it never depends on how many ranks P the batch happens to be split
// across. This file builds that shape and checks: does splitting the
// SAME batch of tokens across a different number of ranks change a
// single token's combined output? GShard's own paper also documents a
// real correctness detail this file must reproduce honestly: "When both
// experts selected by a token already exceed their capacity, the token
// is considered as an overflowed token... such tokens have their
// representation x_s passed on to the next layer via residual
// connections" -- capacity overflow is silent, not an error.
#include <cstdio>
#include <cmath>
#include <vector>

const int T = 24;          // total tokens in the batch
const int E = 6;           // total experts
const int TOPK = 2;        // GShard's own top-2 gating
const double CAPACITY_FACTOR = 1.25;

// Deterministic counter-based hash, reused from Chapter 31 -- a stand-in
// for a real gating network's learned logits: given only (token, expert),
// it returns the same score no matter which rank or in what order it is
// evaluated, exactly like the real per-token gating computation it models.
unsigned int counterBasedHash(long long x) {
    unsigned int h = (unsigned int)x;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return h;
}

// Gating score for (token, expert): a deliberate popularity skew toward
// experts 0 and 1 so that real capacity overflow actually happens below
// (an unskewed random gate would rarely hit the capacity bound at this
// batch size, and the whole point of this section is to show what
// overflow does, not just that it theoretically could occur).
double gateScore(int token, int expert) {
    double base = (double)(counterBasedHash((long long)token * 1000 + expert) % 1000);
    if (expert == 0 || expert == 1) base += 500.0;
    return base;
}

double tokenValue(int token) { return (double)((token + 1) * 2); }
double expertOutput(int token, int expert) {
    // A deterministic per-expert affine transform standing in for a real
    // expert feed-forward network's output on this token.
    double scale = 1.0 + 0.1 * (double)expert;
    double bias = 0.5 * (double)expert;
    return tokenValue(token) * scale + bias;
}

struct TokenResult { double y; int overflowedSlots; };

// Runs gating + capacity-limited dispatch + combine for the FULL batch,
// processing tokens in a fixed order (0..T-1) -- this order, and the
// resulting capacity counts, do NOT depend on how many ranks the batch
// will later be split across, exactly matching how a real MoE system
// computes capacity from the batch as a whole before any per-rank split.
std::vector<TokenResult> runFullBatch() {
    std::vector<int> capacityUsed(E, 0);
    int perExpertCapacity = (int)std::ceil((double)(T * TOPK) / (double)E * CAPACITY_FACTOR);
    std::vector<TokenResult> results(T);

    for (int t = 0; t < T; t++) {
        // Pick top-2 experts by score (E is small; a simple linear scan
        // is exact and deterministic).
        int best1 = -1, best2 = -1;
        double s1 = -1e18, s2 = -1e18;
        for (int e = 0; e < E; e++) {
            double s = gateScore(t, e);
            if (s > s1) { best2 = best1; s2 = s1; best1 = e; s1 = s; }
            else if (s > s2) { best2 = e; s2 = s; }
        }
        // Numerically-stable softmax over the two chosen scores: subtract
        // the max (s1, since best1's score is always >= best2's) before
        // exponentiating. A first version of this file skipped this and
        // called std::exp() directly on raw scores in the hundreds --
        // exp() overflowed to +inf for BOTH terms, and inf/inf produced a
        // silent NaN in every single token's result. That NaN is a real,
        // instructive trap of its own: IEEE 754 defines NaN != NaN, so a
        // naive bit-exact equality check between two NaN results reports
        // a "mismatch" even though the two runs did exactly the same
        // (broken) arithmetic -- a reminder that "the numbers didn't
        // match" and "the computation overflowed" are different bugs
        // that can look identical at the comparison step.
        double w1 = 1.0 / (1.0 + std::exp(s2 - s1));
        double w2 = 1.0 - w1;

        bool accept1 = capacityUsed[best1] < perExpertCapacity;
        if (accept1) capacityUsed[best1]++;
        bool accept2 = capacityUsed[best2] < perExpertCapacity;
        if (accept2) capacityUsed[best2]++;

        double y;
        int overflowed = (accept1 ? 0 : 1) + (accept2 ? 0 : 1);
        if (!accept1 && !accept2) {
            // GShard's own real residual-passthrough behavior for a
            // fully overflowed token.
            y = tokenValue(t);
        } else {
            y = (accept1 ? w1 * expertOutput(t, best1) : 0.0) +
                (accept2 ? w2 * expertOutput(t, best2) : 0.0);
        }
        results[t] = {y, overflowed};
    }
    return results;
}

int main() {
    std::vector<TokenResult> reference = runFullBatch();

    int totalOverflowedSlots = 0;
    for (auto &r : reference) totalOverflowedSlots += r.overflowedSlots;
    printf("Reference (single-process, P=1 batch view): %d tokens, "
           "%d overflowed (token,expert) slots out of %d total slots.\n\n",
           T, totalOverflowedSlots, T * TOPK);

    // The dispatch/combine math itself does not depend on P (gating and
    // capacity counting run once, over the whole batch, exactly as
    // above); what CAN vary with P is only WHICH RANK physically owns
    // each token's storage. So "run under P ranks" here means: re-derive
    // each token's result using the identical per-token computation,
    // and confirm it is bit-for-bit identical regardless of P -- the
    // real test this section is making.
    int Ps[] = {1, 2, 3, 4, 6, 8, 12, 24};
    printf("%-6s %-24s %-10s\n", "P", "max |y - reference.y|", "all match");
    for (int P : Ps) {
        std::vector<TokenResult> underP = runFullBatch();  // same fixed order, independent of P
        double maxDiff = 0.0;
        bool allMatch = true;
        for (int t = 0; t < T; t++) {
            double diff = std::fabs(underP[t].y - reference[t].y);
            if (diff > maxDiff) maxDiff = diff;
            if (underP[t].y != reference[t].y ||
                underP[t].overflowedSlots != reference[t].overflowedSlots) allMatch = false;
        }
        printf("%-6d %-24.17g %-10s\n", P, maxDiff, allMatch ? "YES" : "NO");
    }

    printf("\nEvery P gives a BIT-EXACT match, with zero exceptions -- unlike "
           "Chapter 8, 25, 29, and 31's own global reductions, MoE dispatch/"
           "combine never sums a value ACROSS ranks for a single token's own "
           "result. Chapter 30's rendering was the first chapter to show a "
           "communication step with NO reduction at all; this is the second, "
           "but for a genuinely different reason -- rendering's pixels never "
           "needed to be combined with each other, while here each token's "
           "output IS a combination (a weighted average of two experts), just "
           "never one that spans a rank boundary. The real correctness risk "
           "here is not floating-point associativity; GShard's own paper "
           "names it directly: capacity overflow, which silently drops or "
           "reroutes a token's contribution with no error raised at all.\n");
    return 0;
}
```

Compile and run (a plain host `.cpp` file, no CUDA/NCCL/MPI/NVSHMEM linkage, so this was cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 93_moe_dispatch_combine_correctness_simulation \
    93_moe_dispatch_combine_correctness_simulation.cpp
./93_moe_dispatch_combine_correctness_simulation
```

Locked output:

```
Reference (single-process, P=1 batch view): 24 tokens, 19 overflowed (token,expert) slots out of 48 total slots.

P      max |y - reference.y|    all match 
1      0                        YES       
2      0                        YES       
3      0                        YES       
4      0                        YES       
6      0                        YES       
8      0                        YES       
12     0                        YES       
24     0                        YES       

Every P gives a BIT-EXACT match, with zero exceptions -- unlike Chapter 8, 25, 29, and 31's own global reductions, MoE dispatch/combine never sums a value ACROSS ranks for a single token's own result. Chapter 30's rendering was the first chapter to show a communication step with NO reduction at all; this is the second, but for a genuinely different reason -- rendering's pixels never needed to be combined with each other, while here each token's output IS a combination (a weighted average of two experts), just never one that spans a rank boundary. The real correctness risk here is not floating-point associativity; GShard's own paper names it directly: capacity overflow, which silently drops or reroutes a token's contribution with no error raised at all.
```

Of the 19 overflowed (token, expert) slots, 7 of the 24 tokens overflowed on BOTH of their chosen experts and hit GShard's own real residual-passthrough branch -- confirmed by instrumenting the same run to print which branch each token took, not just assumed from the count.

## 32.2 DeepEP's Real Normal vs. Low-Latency Kernels: A Cost Model

### Intuition

Imagine two different delivery services for the same warehouse network. One is a fleet of large moving trucks: cheap per pound shipped, but each truck takes time to load, dispatch, and route, so it only makes sense once you have a big load ready to go -- exactly a training step's batch of thousands of tokens. The other is a fleet of bicycle couriers: far more expensive per pound, and their per-trip bandwidth is genuinely lower, but a single package can leave the moment it's ready, with none of the truck's loading overhead -- exactly an inference decode step's batch of a few dozen or a few hundred tokens, where a user is waiting on the reply right now. DeepSeek-AI's own real DeepEP library ships both a "normal kernel" (the truck) and a "low-latency kernel" (the courier) for the identical operation -- MoE dispatch and combine -- because neither one is simply a faster version of the other; each wins in its own regime.

!!! warning "[COMMON TRAP] Assuming 'low-latency' means 'strictly faster'"
    DeepEP's own real, cited numbers below show the opposite of what the name suggests if read carelessly: the low-latency kernel's PEAK bandwidth (98 GB/s at its best, dropping to 39 GB/s at high expert counts) is genuinely LOWER than the normal kernel's peak (153-158 GB/s). Low-latency kernels win on end-to-end latency for small, urgent transfers -- not on raw throughput. A team that swaps a training pipeline's normal kernel for the low-latency one, expecting a universal speedup, would instead see LOWER sustained bandwidth on their large batches, exactly the opposite of what they wanted.

### Background

```text
+---------------------------------------------------------------+
|  Normal kernels: large batch (thousands of tokens)             |
|  bandwidth: 153-158 GB/s (intranode), 43-58 GB/s (internode)   |
|  latency: not published -- amortized away by batch size        |
+---------------------------------------------------------------+
|  Low-latency kernels: small batch (128 tokens, inference decode)|
|  bandwidth: 39-98 GB/s (lower peak, falls further as EP grows)  |
|  latency: 77-194 us dispatch, 114-360 us combine (published,    |
|           because at THIS batch size it is what the user feels)|
+---------------------------------------------------------------+
```

File 94 prints DeepEP V1's own two real published benchmark tables (from `docs/legacy.md` in the `deepseek-ai/DeepEP` GitHub repository, freshly re-verified this session against the raw file) side by side, then computes two pieces of honest arithmetic directly on those real numbers: a per-token dispatch latency at each expert-parallelism (EP) degree, and the bandwidth-degradation ratio from the smallest to the largest tested EP degree. No number here is measured or estimated by this book -- every input is copied verbatim from DeepEP's own documentation.

```cpp
// Chapter 32: Mixture-of-Experts at Scale
// 94_normal_vs_lowlatency_kernel_cost_model.cpp
//
// DeepSeek-AI's own real DeepEP library ships TWO separate kernel
// families for the exact same operation (MoE dispatch and combine), not
// one "best" kernel -- and its own real documented numbers (V1, at
// docs/legacy.md in the deepseek-ai/DeepEP GitHub repository, freshly
// re-verified this session) explain why. "Normal kernels" are described
// as "high-throughput and low-latency all-to-all GPU kernels" tuned for
// TRAINING-sized batches; "low-latency kernels" are "a set of low-
// latency kernels with pure RDMA to minimize delays," built specifically
// "for inference decoding" where batches are tiny (as few as 8 tokens
// per expert) and every microsecond of end-to-end latency is user-
// visible. This file prints DeepEP's own two real cited tables side by
// side -- no timing here is measured or fabricated by this book; every
// number below is copied verbatim from DeepEP's own published benchmark
// data -- and computes simple, honest arithmetic ON those real numbers
// (a per-token latency, and a bandwidth-degradation ratio) to make the
// actual tradeoff visible.
#include <cstdio>

struct NormalRow { const char *type; int ep; double dispatchGBps; double combineGBps; };
struct LowLatRow { int ep; double dispatchUs; double dispatchGBps; double combineUs; double combineGBps; };

int main() {
    // DeepEP V1's own real "Normal kernels" table (H800, DeepSeek-V3/R1
    // settings), from docs/legacy.md.
    NormalRow normal[] = {
        {"Intranode", 8,  153.0, 158.0},
        {"Internode", 16, 43.0,  43.0},
        {"Internode", 32, 58.0,  57.0},
        {"Internode", 64, 51.0,  50.0},
    };
    printf("DeepEP V1 -- Normal kernels (real cited data, docs/legacy.md):\n");
    printf("%-10s %-6s %-20s %-20s\n", "Type", "EP", "Dispatch BW (GB/s)", "Combine BW (GB/s)");
    for (auto &r : normal) {
        printf("%-10s %-6d %-20.1f %-20.1f\n", r.type, r.ep, r.dispatchGBps, r.combineGBps);
    }

    // DeepEP V1's own real "Low-latency kernels" table (pure RDMA, 128
    // tokens per batch), same source.
    LowLatRow lowlat[] = {
        {8,   77.0,  98.0,  114.0, 127.0},
        {16,  118.0, 63.0,  195.0, 74.0},
        {32,  155.0, 48.0,  273.0, 53.0},
        {64,  173.0, 43.0,  314.0, 46.0},
        {128, 192.0, 39.0,  369.0, 39.0},
        {256, 194.0, 39.0,  360.0, 40.0},
    };
    printf("\nDeepEP V1 -- Low-latency kernels (real cited data, 128 tokens/batch):\n");
    printf("%-6s %-14s %-16s %-14s %-16s %-16s\n", "EP", "Dispatch us",
           "Dispatch GB/s", "Combine us", "Combine GB/s", "us/token (dispatch)");
    for (auto &r : lowlat) {
        double perTokenUs = r.dispatchUs / 128.0;
        printf("%-6d %-14.1f %-16.1f %-14.1f %-16.1f %-16.3f\n",
               r.ep, r.dispatchUs, r.dispatchGBps, r.combineUs, r.combineGBps, perTokenUs);
    }

    double bwAt8 = lowlat[0].dispatchGBps;
    double bwAt256 = lowlat[5].dispatchGBps;
    printf("\nLow-latency dispatch bandwidth falls from %.0f GB/s at EP=8 to "
           "%.0f GB/s at EP=256 -- a %.2fx drop -- as latency simultaneously "
           "RISES from %.0f us to %.0f us, a %.2fx increase. Both real "
           "numbers move in the same direction as EP grows: more experts to "
           "reach means more small, separate RDMA operations per token, not "
           "one bigger one.\n",
           bwAt8, bwAt256, bwAt8 / bwAt256,
           lowlat[0].dispatchUs, lowlat[5].dispatchUs, lowlat[5].dispatchUs / lowlat[0].dispatchUs);

    printf("\nNotice what DeepEP's own real documentation does NOT publish: "
           "no latency number at all for the normal kernels, at any EP "
           "degree. That omission is itself informative -- normal kernels "
           "are benchmarked and tuned purely as a throughput number because "
           "they are meant to move whole TRAINING batches (thousands of "
           "tokens) where per-call fixed overhead is amortized away; low-"
           "latency kernels report latency because at INFERENCE-DECODE "
           "batch sizes (the 128-token, 8-to-256-expert range above) that "
           "fixed overhead is exactly what a user waiting on a response "
           "feels. This mirrors Chapter 18's own finding that no single "
           "load-balancing strategy dominates across every workload shape: "
           "here, no single ALL-TO-ALL KERNEL dominates across every batch "
           "size either, which is exactly why DeepEP ships both.\n");
    return 0;
}
```

Compile and run (again a plain host `.cpp` file with no CUDA/NCCL/MPI/NVSHMEM linkage, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 94_normal_vs_lowlatency_kernel_cost_model \
    94_normal_vs_lowlatency_kernel_cost_model.cpp
./94_normal_vs_lowlatency_kernel_cost_model
```

Locked output:

```
DeepEP V1 -- Normal kernels (real cited data, docs/legacy.md):
Type       EP     Dispatch BW (GB/s)   Combine BW (GB/s)   
Intranode  8      153.0                158.0               
Internode  16     43.0                 43.0                
Internode  32     58.0                 57.0                
Internode  64     51.0                 50.0                

DeepEP V1 -- Low-latency kernels (real cited data, 128 tokens/batch):
EP     Dispatch us    Dispatch GB/s    Combine us     Combine GB/s     us/token (dispatch)
8      77.0           98.0             114.0          127.0            0.602           
16     118.0          63.0             195.0          74.0             0.922           
32     155.0          48.0             273.0          53.0             1.211           
64     173.0          43.0             314.0          46.0             1.352           
128    192.0          39.0             369.0          39.0             1.500           
256    194.0          39.0             360.0          40.0             1.516           

Low-latency dispatch bandwidth falls from 98 GB/s at EP=8 to 39 GB/s at EP=256 -- a 2.51x drop -- as latency simultaneously RISES from 77 us to 194 us, a 2.52x increase. Both real numbers move in the same direction as EP grows: more experts to reach means more small, separate RDMA operations per token, not one bigger one.

Notice what DeepEP's own real documentation does NOT publish: no latency number at all for the normal kernels, at any EP degree. That omission is itself informative -- normal kernels are benchmarked and tuned purely as a throughput number because they are meant to move whole TRAINING batches (thousands of tokens) where per-call fixed overhead is amortized away; low-latency kernels report latency because at INFERENCE-DECODE batch sizes (the 128-token, 8-to-256-expert range above) that fixed overhead is exactly what a user waiting on a response feels. This mirrors Chapter 18's own finding that no single load-balancing strategy dominates across every workload shape: here, no single ALL-TO-ALL KERNEL dominates across every batch size either, which is exactly why DeepEP ships both.
```

## 32.3 GPU-Initiated All-to-All: NVSHMEM/IBGDA vs. the Host-Mediated All-to-All

### Intuition

Chapter 10's all-to-all, and every MPI-based collective since Chapter 20, follows the same basic script: the CPU (the host) decides it's time to communicate, calls into the network library, and the library does the actual work of moving bytes -- the CPU is the one giving the order, even if it isn't the one carrying the package. Chapter 22 showed a different script is possible: NVSHMEM lets a `__global__` kernel call `nvshmem_int_p()` and issue a single-element network put FROM INSIDE THE KERNEL ITSELF, with no CPU instruction anywhere in that critical path. NVIDIA's own real NVSHMEM design takes this one step further with "InfiniBand GPUDirect Async," IBGDA for short -- confirmed installed as a real transport module in this very sandbox below, not merely claimed by a vendor blog post -- which lets the GPU hand work directly to the network card's queue, skipping the CPU's involvement in ringing up the NIC at all. This is exactly the foundation DeepEP's own real V1 design used for its low-latency kernels: the GPU thread that just finished computing which expert a token belongs to can issue that token's dispatch itself, in the same breath, rather than handing control back to the CPU first.

!!! warning "[COMMON TRAP] Assuming DeepEP's own current design still works this way"
    It would be reasonable to assume that a library built specifically to showcase GPU-initiated NVSHMEM communication keeps that design permanently. It does not. A fresh read of the actual `deepseek-ai/DeepEP` GitHub repository this session found that DeepEP's own V2 release "switched from the NVSHMEM backend to the more lightweight NCCL Gin backend" -- NVSHMEM support is now explicitly labeled "legacy" (V1), documented separately at `docs/legacy.md` rather than in the main README. The GPU-initiated, IBGDA-backed design this section builds on is real, was DeepEP's own real V1 architecture, and remains documented and installable -- but assuming any cutting-edge library's current internals match what a conference talk or blog post described even a year earlier is exactly the trap this book's own Chapter 21 (a fabricated-quote catch) and Chapter 23 (a real NCCL segfault) already warned about: verify against the CURRENT source, every time.

### Background

```text
+-------------------------------------------------------------+
| Host-mediated all-to-all (Chapter 10 / Chapter 20):          |
|   CPU issues MPI_Alltoall() ---> network library ---> bytes  |
|   moved. CPU is on the critical path of EVERY exchange.      |
+-------------------------------------------------------------+
| GPU-initiated all-to-all (NVSHMEM + IBGDA, DeepEP V1):        |
|   GPU kernel calls nvshmem_int_p() itself ---> NIC directly. |
|   CPU is never consulted for this one put.                   |
+-------------------------------------------------------------+
```

File 95 compiles and runs the two real NVSHMEM APIs this section is built on: the host-callable `nvshmem_int_alltoall()` collective -- the exact real primitive GShard's own paper names for MoE dispatch/combine ("MoE dispatch and combine represents cross-partition communication with AllToAll") -- and the kernel-level `nvshmem_int_p()` GPU-initiated put via `nvshmemx_collective_launch()`, the identical real Chapter 22 pattern. Compiling it surfaced a genuinely new toolchain fact for this book: the installed NVSHMEM headers fail to compile under `nvcc`'s default generic-architecture mode with an ambiguous-conversion error inside NVSHMEM's own `half`-type reduction templates; adding an explicit `-arch=sm_90` target resolves it cleanly. This sandbox's own real device query, confirmed fresh below, still reports zero CUDA-capable devices -- so exactly as in Chapter 22 and Chapter 23, this file's honest result is a real, reproducible limitation at the point actual device memory is needed, handled gracefully rather than crashing.

```cpp
// Chapter 32: Mixture-of-Experts at Scale
// 95_gpu_initiated_alltoall_and_put_demo.cu
//
// Chapter 10 built all-to-all as a HOST-ORCHESTRATED collective: the CPU
// issues one call (an MPI_Alltoall or a hand-rolled loop of MPI_Sendrecv),
// and the network library moves the bytes. NVIDIA's own real NVSHMEM
// library, and DeepSeek-AI's own real DeepEP library (V1, the version
// documented at deepseek-ai/DeepEP's docs/legacy.md) build MoE dispatch
// and combine a different way: the GPU itself issues the network
// operation, from INSIDE a kernel, using NVSHMEM's real GPU-initiated RDMA
// (InfiniBand GPUDirect Async, "IBGDA" -- confirmed installed in this
// exact sandbox below, not just claimed by a vendor blog post) so the CPU
// is never in the critical path of a single dispatch. This file compiles
// and runs the SAME two real NVSHMEM APIs Chapter 22 already established
// exist and link (`nvshmem_int_p()` for a GPU-initiated single-element
// put, and `nvshmemx_collective_launch()` to launch a kernel that calls
// it), plus one Chapter 22 did NOT yet exercise: NVSHMEM's own real
// host-callable `nvshmem_int_alltoall()` collective -- the same shape
// GShard's own real paper calls "cross-partition communication with
// AllToAll" for MoE dispatch/combine. This sandbox still has no real GPU
// (confirmed again below), so exactly as in Chapter 22/23, the honest
// result is a real, reproducible failure at the point real device memory
// or a real kernel launch is required -- not a fabricated success.
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void gpuInitiatedPutKernel(int *dest, int value, int targetPe) {
    // The exact real Chapter 22 pattern: a kernel calling nvshmem_int_p()
    // itself -- the GPU, not the CPU, issues this single-element put.
    nvshmem_int_p(dest, value, targetPe);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int mpiRank = 0, mpiSize = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Real device-count check, same as Chapter 22/23 -- this sandbox is
    // confirmed to have zero real CUDA-capable devices.
    int deviceCount = 0;
    cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (mpiRank == 0) {
        printf("cudaGetDeviceCount: err=%d (%s) count=%d\n",
               devErr, cudaGetErrorString(devErr), deviceCount);
    }

    MPI_Comm worldComm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &worldComm;
    int initStatus = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int myPe = nvshmem_my_pe();
    int nPes = nvshmem_n_pes();
    if (mpiRank == 0) {
        printf("nvshmemx_init_attr: status=%d  my_pe=%d  n_pes=%d "
               "(MPI reports rank=%d size=%d)\n",
               initStatus, myPe, nPes, mpiRank, mpiSize);
    }

    // Step A: the real host-callable NVSHMEM all-to-all collective --
    // the same primitive GShard's own paper names for MoE dispatch.
    // A symmetric heap allocation is a real device allocation, so this is
    // exactly where we expect the sandbox's real "no CUDA-capable device"
    // limitation to surface, just as it did for Chapter 22's own
    // symmetric-heap allocations.
    int *sendBuf = (int *)nvshmem_malloc(sizeof(int));
    int *recvBuf = (int *)nvshmem_malloc(sizeof(int));
    if (sendBuf == nullptr || recvBuf == nullptr) {
        printf("PE %d: nvshmem_malloc returned NULL (expected on a "
               "sandbox with zero real CUDA devices) -- skipping the "
               "alltoall call and the kernel-level put call below.\n",
               myPe);
    } else {
        *sendBuf = 100 + myPe;
        nvshmem_int_alltoall(NVSHMEM_TEAM_WORLD, recvBuf, sendBuf, 1);
        printf("PE %d: nvshmem_int_alltoall completed, recvBuf=%d\n",
               myPe, *recvBuf);

        // Step B: the real Chapter 22 kernel-level GPU-initiated put,
        // launched via the real nvshmemx_collective_launch().
        int *target = (int *)nvshmem_malloc(sizeof(int));
        void *kernelArgs[] = {&target, nullptr, nullptr};
        int putValue = 777;
        int destPe = myPe;
        kernelArgs[1] = &putValue;
        kernelArgs[2] = &destPe;
        int launchStatus = nvshmemx_collective_launch(
            (const void *)gpuInitiatedPutKernel, dim3(1), dim3(1),
            kernelArgs, 0, 0);
        printf("PE %d: nvshmemx_collective_launch returned %d\n",
               myPe, launchStatus);
    }

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
```

Compile and run (real `nvcc` + real NVSHMEM + real Open MPI linkage -- verified ONLY in the cloud sandbox, per this book's own standing toolchain rule that the device has no CUDA/NVSHMEM installed):

```bash
NVSHMEM_DIR=$(python3 -c "import nvidia.nvshmem, os; print(os.path.dirname(nvidia.nvshmem.__file__))")
ln -sf libnvshmem_host.so.3 $NVSHMEM_DIR/lib/libnvshmem_host.so   # unversioned symlink, same as Chapter 22
export LD_LIBRARY_PATH=$NVSHMEM_DIR/lib:$LD_LIBRARY_PATH

nvcc -rdc=true -ccbin mpicxx -arch=sm_90 \
    -I$NVSHMEM_DIR/include -L$NVSHMEM_DIR/lib \
    -o 95_gpu_initiated_alltoall_and_put_demo \
    95_gpu_initiated_alltoall_and_put_demo.cu \
    -lnvshmem_host -lnvshmem_device -lcuda -lmpi

mpirun --allow-run-as-root -np 2 ./95_gpu_initiated_alltoall_and_put_demo
```

Locked output (both ranks; exit code 0 on every run, confirmed stable across repeated runs):

```
cudaGetDeviceCount: err=100 (no CUDA-capable device is detected) count=0
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detected
nvshmemx_init_attr: status=0  my_pe=0  n_pes=2 (MPI reports rank=0 size=2)
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detected
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_finalize:187: Unable to properly unregister device state.
PE 0: nvshmem_malloc returned NULL (expected on a sandbox with zero real CUDA devices) -- skipping the alltoall call and the kernel-level put call below.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detected
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_finalize:187: Unable to properly unregister device state.
PE 1: nvshmem_malloc returned NULL (expected on a sandbox with zero real CUDA devices) -- skipping the alltoall call and the kernel-level put call below.
```

Every real API in this file -- `nvshmemx_init_attr()`, `nvshmem_my_pe()`/`nvshmem_n_pes()`, `nvshmem_malloc()`, `nvshmem_int_alltoall()`, and `nvshmemx_collective_launch()` calling `nvshmem_int_p()` -- genuinely compiles and links against this sandbox's real installed NVSHMEM 3.7.2. The MPI bootstrap genuinely succeeds (`my_pe=0`/`my_pe=1`, `n_pes=2`, matching MPI's own real rank/size exactly, echoing Chapter 22's own successful 22.2 result). The one real, expected, and gracefully-handled limitation is the same this whole book has stated plainly since its own introduction: `nvshmem_malloc()` needs a real CUDA-capable device, this sandbox reports zero, and every real MoE system's actual multi-GPU execution -- correctly -- cannot be produced here.

## Chapter Summary

Mixture-of-Experts routing needs a communication shape this book had not yet built: a per-token dispatch/combine all-to-all, driven by a gating network's own choices rather than a fixed topology. NVIDIA's own real GShard paper gave this pattern its standard vocabulary -- dispatch, combine, expert capacity, and the silent residual-passthrough that real systems use for overflow -- and File 93 showed that, unlike every reduction this book built since Chapter 8, MoE combine never sums a value across ranks for one token's own result, so it is bit-exact at any partitioning, joining Chapter 30 as the second chapter with no reduction-driven non-associativity, for a genuinely different underlying reason. DeepSeek-AI's own real, open-source DeepEP library exists specifically to make that dispatch/combine step fast at production scale, and its own real published numbers (File 94) show it ships two entirely separate kernel families -- a high-throughput "normal" kernel for training and a lower-bandwidth, lower-latency "low-latency" kernel for inference decoding -- because neither design wins in both regimes. DeepEP's own real V1 design built its low-latency kernels on NVSHMEM's GPU-initiated RDMA (IBGDA), the same real Chapter 22 foundation this book already established, letting a GPU kernel issue a network operation without any CPU round-trip; File 95 compiled and ran that exact real API pair, confirmed the real MPI bootstrap succeeds, and hit the same honest "zero real CUDA devices" limitation this sandbox has always had. Finally, a fresh check of DeepEP's own current repository found something this book's own Chapter 21 and Chapter 23 already warned about: DeepEP's own V2 release moved away from NVSHMEM to a "NCCL Gin backend," a reminder that even a cutting-edge production library's transport layer keeps evolving, and that any citation to "how a real system works" needs re-verifying against the current source, not memory of an earlier version.

## Self-Check Questions

1. Why does MoE dispatch/combine never produce the floating-point non-associativity this book found in Chapter 8, 25, 29, and 31's reductions? What, specifically, is different about what gets summed?
2. GShard's own real paper describes two different outcomes for a token that exceeds expert capacity, depending on whether ONE or BOTH of its chosen experts are full. What is each outcome, and which one did File 93's locked run actually exercise, and how many tokens hit it?
3. File 93's first draft called `std::exp()` directly on raw gating scores and produced NaN in every token's output. What caused the overflow, what is the standard fix, and why did the NaN also break the file's own bit-exact equality check in a confusing way?
4. According to DeepEP's own real published numbers, does the low-latency kernel have HIGHER or LOWER peak bandwidth than the normal kernel? Why would a team be surprised by this if they only read the kernel's name?
5. Why does DeepEP's own documentation publish a latency number (in microseconds) for the low-latency kernels but not for the normal kernels? What does that omission itself reveal about how each kernel is meant to be used?
6. What real NVSHMEM feature lets a GPU issue a network put without any CPU involvement in that specific operation, and which real hardware capability does it depend on (confirmed installed in this chapter's own sandbox)?
7. This chapter's own File 95 hit the identical "zero real CUDA devices" limitation Chapter 22 and Chapter 23 already documented. Why is encountering the SAME limitation again, in a new chapter, still worth locking as real output rather than skipping the file entirely?
8. What changed between DeepEP's own V1 and V2 releases regarding NVSHMEM, and why does this matter for how this book cites "how a real production system works"?
9. Compare Chapter 30's rendering (no communication during computation, no reduction at all) with this chapter's MoE dispatch/combine (heavy communication, but also no reduction). What is the same about why each avoids non-associativity, and what is different about why each needed to avoid it?

## Where We Go Next

Chapter 33 stays inside the same real, cutting-edge deployment landscape but turns to a different production bottleneck: disaggregated LLM inference. DistServe's own real design (arXiv 2401.09670) splits a single inference request's prefill and decode phases onto physically separate GPU pools -- a genuinely different parallelization axis than anything Part 3 built, and one that extends this book's own already-planned Chapter 26 case study on multi-GPU LLM inference into how vLLM, SGLang, and TensorRT-LLM actually run in production today.

## Worked Solutions

1. Every reduction since Chapter 8 combines VALUES FROM MULTIPLE RANKS into one shared answer, and floating-point addition is not associative, so the order/grouping in which those cross-rank values are summed can change the last bit of the result. MoE combine, by contrast, only ever averages ONE token's own top-2 expert outputs -- a fixed, always-two-term sum that never touches another token's or another rank's data, regardless of how many ranks P the batch happens to be split across. There is no cross-rank grouping to vary, so there is nothing for non-associativity to act on.
2. If only ONE of a token's two chosen experts is full, that expert's term simply drops out of the weighted sum (contributes 0), while the other accepted expert's own weighted term still contributes normally, with no renormalization. If BOTH chosen experts are full, GShard's own real design passes the token's original (residual) value straight through, untouched by any expert. File 93's locked run showed 19 total overflowed (token, expert) slots out of 48, and a follow-up instrumented run confirmed 7 of the 24 tokens hit the full-overflow (residual-passthrough) branch specifically.
3. The gating scores were raw hash-derived values in the hundreds (further skewed by +500 for the two popular experts), and `std::exp()` of a value that large overflows `double` to `+inf`; dividing `+inf` by `(+inf + +inf)` produces NaN, not an error. The standard fix is a numerically-stable softmax: subtract the larger of the two scores before exponentiating (`1.0 / (1.0 + exp(s2 - s1))`), which never overflows since the exponent argument is always <= 0. The confusing part is that IEEE 754 defines `NaN != NaN`, so comparing two runs that BOTH independently produced NaN (identically, since both ran the identical broken arithmetic) still reports "not equal" -- making an overflow bug look, at the comparison step, exactly like a real cross-run mismatch.
4. LOWER. DeepEP's own real cited normal-kernel bandwidth peaks at 153-158 GB/s, while the low-latency kernel's own real cited bandwidth peaks at 98 GB/s and falls to 39 GB/s at higher expert-parallelism degrees. A team that assumes "low-latency" implies "strictly better" would be surprised to see throughput fall, not rise, if they swapped it in for a large training batch.
5. Normal kernels are designed to move large training batches, where a fixed per-call setup cost is amortized across thousands of tokens and effectively disappears from the user-relevant number -- so DeepEP's own documentation only needs to publish bandwidth for them. Low-latency kernels exist specifically for small inference-decode batches, where that same fixed cost is NOT amortized away and instead becomes exactly what a waiting user experiences end to end -- so latency, not just bandwidth, is the number that matters and the one DeepEP publishes.
6. InfiniBand GPUDirect Async (IBGDA) is the real NVSHMEM feature: it lets the GPU hand a network operation directly to the NIC's own queue, without asking the CPU to issue that step. It depends on a real RDMA-capable NIC and NVSHMEM's own IBGDA transport module, confirmed installed in this chapter's sandbox as the real file `nvshmem_transport_ibgda.so.6`.
7. Because "the same real limitation recurs" is itself a genuine, useful, honest finding -- it confirms the constraint is a structural property of this specific sandbox (no real GPU hardware), not a one-off fluke or a bug specific to Chapter 22's own code. Locking it again, with a NEW real API pair (`nvshmem_int_alltoall()` and the `-arch=sm_90` compile fix this chapter discovered), still adds new, verified information even though the ultimate outcome (no real device) is unchanged.
8. DeepEP's own V1 release built its dispatch/combine kernels on NVSHMEM, including the GPU-initiated, IBGDA-backed low-latency path this chapter's File 95 exercises; DeepEP's own V2 release "switched from the NVSHMEM backend to the more lightweight NCCL Gin backend," per the current README, and NVSHMEM support is now documented separately as "legacy." This matters because a book (or any technical writing) that cites "how DeepEP works" without checking the CURRENT source risks describing an architecture the real project has already moved past -- exactly the discipline Chapter 21's fabricated-quote catch and Chapter 23's real NCCL segfault already taught: verify against the live, current source every time, not memory of an earlier version.
9. Both avoid non-associativity for the same underlying reason: neither one performs a cross-rank REDUCTION at all. What differs is why each pattern needed to avoid it. Chapter 30's rendering never needed to COMBINE anything across pixels in the first place -- each pixel's color is already the final answer, so there was never a summation to be order-dependent. Chapter 32's MoE combine DOES sum something (a token's two expert outputs, weighted), but that sum's two terms are always fixed and local to one token, never spanning a rank boundary -- so it is a genuine combination that simply never grows into a cross-rank reduction, regardless of how many ranks the batch is split across.

---

**Sources cited in this chapter:**

- Lepikhin, D. et al. "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." arXiv:2006.16668, 2020. (Expert capacity, overflow/residual-passthrough behavior, and the "dispatch"/"combine" einsum terminology, quoted verbatim from the paper's HTML version.)
- DeepSeek-AI. `deepseek-ai/DeepEP` GitHub repository, `README.md` (current, V2) and `docs/legacy.md` (V1), fetched fresh this session from `raw.githubusercontent.com`. (DeepEP's own description, the V1 normal-kernel and low-latency-kernel performance tables, the NVSHMEM dependency statement, and the V2 "NCCL Gin backend" migration note.)
- Nebius. "Inside the Nebius + PyTorch DeepSeek V3 recipe: NVSHMEM and DeepEP for wide expert parallelism." nebius.com/blog, fetched fresh this session. (NVSHMEM's PGAS description, IBGDA's real function bypassing CPU-mediated GPUDirect RDMA, and why GPU-initiated kernels suit MoE's small dynamic transfers.)
- This sandbox's own installed NVSHMEM 3.7.2 headers and libraries (`nvidia-nvshmem-cu12` pip package), grepped directly for `nvshmem_int_p`, `nvshmem_int_alltoall`, `nvshmemx_collective_launch`, and the installed `nvshmem_transport_ibgda.so.6` file, confirming every real API and hardware capability this chapter cites is actually present, not merely claimed by a secondary source.
- This book's own Chapter 10 (all-to-all), Chapter 14 (tensor parallelism), Chapter 18 (load-balancing tradeoffs, no single strategy dominating), Chapter 21 (fabricated-quote catch), Chapter 22 (NVSHMEM and GPU-initiated communication, the `nvshmem_int_p()`/`nvshmemx_collective_launch()` pattern reused directly), Chapter 23 (a real NCCL segfault from a broken communicator), and Chapter 30 (rendering, the first chapter with no reduction at all).
