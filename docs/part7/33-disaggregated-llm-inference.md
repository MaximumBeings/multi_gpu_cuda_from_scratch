**What you will understand after this chapter:** why a single LLM inference request is really two very different workloads glued together -- a compute-bound prefill step and a memory-bandwidth-bound decode phase -- and why real production systems now run them on entirely separate GPU pools, each sized with its own tensor-parallelism degree, rather than treating "run the model" as one uniform computation the way Chapter 26 did; what real, deployed system (DistServe, and its production successors in vLLM, SGLang, and NVIDIA Dynamo) pioneered this split; and why the KV cache handoff between the two phases is a genuinely new communication shape for this book -- a targeted point-to-point bulk transfer between two disjoint device groups, not a collective involving everyone.

**What you need to know first:** Chapter 5 (explicit peer-to-peer transfers, this chapter's closest ancestor for the KV cache handoff), Chapter 14 (tensor parallelism, reused here at two different degrees within one system), Chapter 21 (GPUDirect RDMA, the cross-node path for that same handoff), and Chapter 26 (multi-GPU LLM inference, which this chapter directly extends with a new splitting axis).

---

Chapter 32 showed that Mixture-of-Experts needs a communication shape keyed by which token a gate sends to which expert. This chapter shows that even a single, ordinary LLM inference request -- no experts, no routing decisions -- hides a similar kind of heterogeneity. Every request an LLM serves runs through two phases with almost nothing in common computationally: a prefill step that reads the whole prompt at once, and a decode phase that produces one token at a time until the response is done. Zhong et al.'s own real DistServe paper (arXiv:2401.09670, USENIX OSDI 2024) made the case, and NVIDIA's own real Dynamo framework, together with vLLM's and SGLang's own real production disaggregated-serving support, now build on it: don't force these two phases to share the same GPUs, the same tensor-parallelism degree, or even the same node. Run them as two separate services, connected only by a single handoff of the prompt's KV cache, and each phase gets to be sized for what it actually needs.

```text
+------------------------------------------------------------------+
| Chapter 26 (this book's earlier LLM inference chapter):          |
| ONE request -> [ TP + PP, uniform GPU pool, whole request ]      |
+------------------------------------------------------------------+
| Chapter 33 (disaggregated: DistServe / Dynamo's own real design):|
|                                                                    |
|  PREFILL pool (small TP,      DECODE pool (large TP,              |
|  compute-bound, whole prompt)  memory-bound, one token at a time) |
|  [ GPU ] [ GPU ] --- KV cache --> [ GPU ][ GPU ][ GPU ][ GPU ]     |
|         handoff (point-to-point,          token, token, token...  |
|         NOT a collective)                                         |
+------------------------------------------------------------------+
```

## 33.1 Why Prefill and Decode Need Different Resources

### Intuition

Picture two very different jobs at a print shop. The first job is a single, huge document that just arrived: the whole thing needs to be typeset at once, and the shop throws every available machine at it in parallel to get it done fast -- that's prefill, reading an entire prompt and computing its representation all in one pass, genuinely compute-bound because there's a lot of arithmetic to do across many tokens simultaneously. The second job is a customer standing at the counter, waiting for one page to be printed, checked, handed over, then asking for the next page, one at a time, forever -- that's decode, producing exactly one new token per step. It sounds like it should be the cheap job. It isn't: to produce that one token, the GPU still has to read the ENTIRE running memory of the conversation so far (the KV cache) from memory, and that memory traffic grows every single step. DistServe's own real paper puts it plainly: decode "incurs a similar level of I/O to the prefill phase, making it constrained by the GPU's memory bandwidth" -- not because decode does a lot of arithmetic, but because it has to move a lot of data to do a little arithmetic.

!!! warning "[COMMON TRAP] Assuming 'one token per step' means decode is cheap"
    It is tempting to look at prefill (processing hundreds or thousands of tokens) and decode (processing one token) and conclude decode must be the lightweight phase. File 96's own illustrative model shows the opposite mechanism at work: decode's FLOPs per step genuinely stay flat regardless of prompt length, but the BYTES it must move per step (reading the whole KV cache) grow exactly as fast as prefill's own compute does. A system that provisions decode GPUs assuming they're "doing less work" than prefill GPUs, without separately accounting for memory bandwidth, will bottleneck on exactly the resource it didn't plan for.

### Background

```text
+----------------------------------------------------------+
| PREFILL: N tokens, all at once                             |
|   FLOPs  ~ N * d^2   (grows with prompt, parallel work)    |
|   -> compute-bound                                         |
+----------------------------------------------------------+
| DECODE: 1 new token, but reads the WHOLE kv cache           |
|   FLOPs/step ~ d^2        (flat, independent of N)          |
|   bytes/step ~ N * d      (grows with N, same as prefill)   |
|   -> memory-bandwidth-bound                                 |
+----------------------------------------------------------+
```

File 96 builds this contrast as a small, explicitly-labeled illustrative model (never a measured benchmark -- this sandbox has no real GPU to benchmark on) using Chapter 1's own arithmetic-intensity framing, then prints DistServe's own real, freshly-verified cited numbers directly: its own definition of "goodput," its own measured 4.48x/10.2x improvement, and the real KV cache size example Section 33.3 below reuses.

```cpp
// Chapter 33: Disaggregated LLM Inference
// 96_prefill_vs_decode_cost_model.cpp
//
// Chapter 26 already built multi-GPU LLM inference as a combination of
// Chapter 14's tensor parallelism and Chapter 15's pipeline parallelism,
// treating "run the model" as one uniform workload. Zhong et al.'s own
// real DistServe paper ("DistServe: Disaggregating Prefill and Decoding
// for Goodput-optimized Large Language Model Serving," arXiv:2401.09670,
// USENIX OSDI 2024) makes the case that a single LLM inference request
// is actually TWO very different workloads glued together: "the prefill
// step deals with a new sequence, often comprising many tokens, and
// processes these tokens concurrently... the prefill step tends to be
// computation-bound," while, "in contrast, the decoding phase, despite
// processing only one new token per step, incurs a similar level of I/O
// to the prefill phase, making it constrained by the GPU's memory
// bandwidth." This file builds a small, clearly-labeled ILLUSTRATIVE
// model of that same contrast (never a measured benchmark -- this
// sandbox has no real GPU to benchmark on), grounded in Chapter 1's own
// arithmetic-intensity framing, and then presents DistServe's own real
// cited numbers -- its "goodput" metric and its measured 4.48x/10.2x
// improvement -- as real data, not something this book derived.
#include <cstdio>

int main() {
    printf("Illustrative model (Chapter 1's arithmetic-intensity framing "
           "applied to one transformer layer of hidden dimension d):\n");
    printf("%-12s %-22s %-22s %-18s\n", "Prompt N", "Prefill FLOPs (~N*d^2)",
           "Decode FLOPs/step (~d^2)", "Decode bytes/step (~N*d)");
    long long d = 8192;  // illustrative hidden dimension
    long long promptLens[] = {128, 512, 2048, 8192};
    for (long long N : promptLens) {
        long long prefillFlops = 2 * N * d * d;      // scales with N, parallel over tokens
        long long decodeFlopsPerStep = 2 * d * d;    // ONE new token: independent of N
        long long decodeBytesPerStep = N * d * 2;     // must still read the whole KV cache (bf16, 2 bytes)
        printf("%-12lld %-22lld %-22lld %-18lld\n", N, prefillFlops, decodeFlopsPerStep, decodeBytesPerStep);
    }
    printf("\nDecode's own FLOPs per step never grows with the prompt -- exactly "
           "DistServe's own point that decode is 'despite processing only one new "
           "token per step.' But decode's own BYTES MOVED per step (reading the "
           "growing KV cache) grows exactly as fast as prefill's own compute does. "
           "Arithmetic intensity (FLOPs per byte) therefore FALLS as the prompt "
           "grows for decode, while prefill's stays roughly constant -- the real "
           "reason decode is memory-bandwidth-bound and prefill is compute-bound, "
           "the same distinction Chapter 1's own memory-wall discussion introduced.\n");

    printf("\nDistServe's own real cited numbers (not derived by this book):\n");
    printf("- Goodput, DistServe's own definition: \"the maximum request rate that "
           "can be served adhering to the SLO attainment goal (say, 90%%) for each "
           "GPU provisioned -- higher per-GPU goodput directly translates into "
           "lower cost per query.\"\n");
    printf("- Measured result: \"DistServe can serve 4.48x more requests or 10.2x "
           "tighter SLO, compared to state-of-the-art systems, while staying "
           "within latency constraints for >90%% of requests.\"\n");
    printf("- Real KV cache size example cited in the same paper: \"the KV cache "
           "size of a single 512-token request on OPT-66B is approximately "
           "1.13GB\" -- Section 33.3 below uses this exact real number.\n");
    return 0;
}
```

Compile and run (a plain host `.cpp` file with no CUDA/NCCL/MPI/NVSHMEM linkage, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 96_prefill_vs_decode_cost_model \
    96_prefill_vs_decode_cost_model.cpp
./96_prefill_vs_decode_cost_model
```

Locked output:

```
Illustrative model (Chapter 1's arithmetic-intensity framing applied to one transformer layer of hidden dimension d):
Prompt N     Prefill FLOPs (~N*d^2) Decode FLOPs/step (~d^2) Decode bytes/step (~N*d)
128          17179869184            134217728              2097152           
512          68719476736            134217728              8388608           
2048         274877906944           134217728              33554432          
8192         1099511627776          134217728              134217728         

Decode's own FLOPs per step never grows with the prompt -- exactly DistServe's own point that decode is 'despite processing only one new token per step.' But decode's own BYTES MOVED per step (reading the growing KV cache) grows exactly as fast as prefill's own compute does. Arithmetic intensity (FLOPs per byte) therefore FALLS as the prompt grows for decode, while prefill's stays roughly constant -- the real reason decode is memory-bandwidth-bound and prefill is compute-bound, the same distinction Chapter 1's own memory-wall discussion introduced.

DistServe's own real cited numbers (not derived by this book):
- Goodput, DistServe's own definition: "the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned -- higher per-GPU goodput directly translates into lower cost per query."
- Measured result: "DistServe can serve 4.48x more requests or 10.2x tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for >90% of requests."
- Real KV cache size example cited in the same paper: "the KV cache size of a single 512-token request on OPT-66B is approximately 1.13GB" -- Section 33.3 below uses this exact real number.
```

## 33.2 Splitting the Cluster: Different TP Degrees Per Phase

### Intuition

Chapter 14 built tensor parallelism as a way to split ONE layer's GEMM across a fixed set of devices, and every chapter since has assumed that same TP degree applies uniformly wherever it's used. NVIDIA's own real Dynamo documentation breaks that assumption on purpose: it recommends using "a larger TP for the memory-bound decoding phase while a smaller TP for the computation-bound prefill phase." The intuition is the print-shop analogy again -- the big one-time typesetting job (prefill) benefits from parallel machines only up to a point, since it's compute-bound and adding more machines has diminishing returns once each has enough work; the counter job (decode) benefits from MORE machines because each additional GPU adds more aggregate memory bandwidth to read the ever-growing KV cache, and that's the actual bottleneck. Since the two phases are connected only by a single handoff (not by sharing a communicator or a fixed device group), there's no structural reason they need to match.

!!! warning "[COMMON TRAP] Assuming a request's parallelism degree must stay constant end to end"
    Every parallelization strategy this book built before this chapter -- data, model, tensor, pipeline -- picks a device-group size once and uses it throughout a computation. It's natural to assume that discipline extends to "how many GPUs handle this request," full stop. Disaggregation breaks that assumption deliberately: File 97 below confirms that a request's prefill phase and decode phase can run under COMPLETELY DIFFERENT TP degrees (including cases where Tp > Td, not just the recommended Tp <= Td) and still produce an exactly correct result, because correctness here comes from each phase's own internal reduction being exact on its own, never from the two phases sharing a group size.

### Background

```text
+----------------------------------------------------------+
| PREFILL phase: Tp-way head split, integer sum-reduce      |
|   (any Tp) -> kv cache values, exact regardless of Tp     |
+----------------------------------------------------------+
|                    kv cache handoff                        |
+----------------------------------------------------------+
| DECODE phase: Td-way position split, integer sum-reduce    |
|   (any Td, independent of Tp) -> generated tokens          |
+----------------------------------------------------------+
```

File 97 builds a small integer-only prefill+decode simulation (integer arithmetic on purpose, so this section's own question -- does splitting the two phases onto independently-sized device groups preserve correctness -- is not entangled with the floating-point associativity question Chapter 8/25/29/31/32 already covered) and checks every combination of prefill TP degree Tp in {1, 2, 4, 8} against decode TP degree Td in {1, 2, 3, 6}, including combinations NVIDIA Dynamo's own guidance would not actually recommend (Tp > Td), to see whether they are merely suboptimal or actually incorrect.

```cpp
// Chapter 33: Disaggregated LLM Inference
// 97_phase_specific_tensor_parallelism_correctness_simulation.cpp
//
// NVIDIA's own real Dynamo documentation states the design point behind
// disaggregation directly: separate prefill and decode workers let a
// system use "a larger TP for the memory-bound decoding phase while a
// smaller TP for the computation-bound prefill phase" -- a genuinely
// different tensor-parallelism DEGREE for each phase of the SAME
// request, something Chapter 14 and Chapter 26 never needed to consider
// because they treated a request as one uniform TP-parallel computation
// throughout. This file checks the obvious correctness question that
// design raises: if the prefill phase runs under TP degree Tp and the
// decode phase runs under a DIFFERENT TP degree Td, connected only by
// the transferred KV cache (Section 33.3's own subject), does the final
// generated sequence still match a single-process (Tp=Td=1) reference
// exactly? Every value here is a plain integer specifically so this
// file's own answer is not entangled with the floating-point
// associativity question Chapter 8/25/29/31/32 already covered in
// detail -- the question this file asks is a DIFFERENT one (does
// splitting a single request's two phases onto independently-sized
// device groups preserve correctness), not a repeat of that one.
#include <cstdio>
#include <vector>

const int H = 8;             // total attention heads (illustrative)
const int PROMPT_LEN = 16;   // prompt tokens
const int NUM_DECODE_STEPS = 6;

unsigned int counterBasedHash(long long x) {
    unsigned int h = (unsigned int)x;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = ((h >> 16) ^ h) * 0x45d9f3bU;
    h = (h >> 16) ^ h;
    return h;
}

long long headWeight(int h) { return (long long)((counterBasedHash(h + 9000) % 9) + 1); }
long long promptValue(int i) { return (long long)((counterBasedHash(i + 1000) % 20) + 1); }
long long decodeWeight(int t) { return (long long)((counterBasedHash(t + 5000) % 7) + 1); }

// Prefill: for each prompt position, sum head contributions -- split
// across Tp shards of heads, each shard reduced locally, then the Tp
// partial sums combined (integer addition: exactly associative, so this
// step alone could never show a mismatch regardless of Tp).
std::vector<long long> runPrefill(int Tp) {
    std::vector<long long> kv(PROMPT_LEN, 0);
    for (int i = 0; i < PROMPT_LEN; i++) {
        long long total = 0;
        for (int shard = 0; shard < Tp; shard++) {
            long long partial = 0;
            for (int h = shard; h < H; h += Tp) partial += headWeight(h) * promptValue(i);
            total += partial;
        }
        kv[i] = total;
    }
    return kv;
}

// Decode: each step reads the FULL kv cache (the transferred prefill
// result, plus every previously-generated decode value) and produces
// one new value, split across Td shards of kv-cache POSITIONS, combined
// the same way.
std::vector<long long> runDecode(const std::vector<long long> &kv, int Td) {
    std::vector<long long> cache = kv;  // grows as decode proceeds
    std::vector<long long> generated;
    for (int t = 0; t < NUM_DECODE_STEPS; t++) {
        int cacheLen = (int)cache.size();
        long long total = 0;
        for (int shard = 0; shard < Td; shard++) {
            long long partial = 0;
            for (int i = shard; i < cacheLen; i += Td) partial += cache[i];
            total += partial;
        }
        long long value = total * decodeWeight(t);
        generated.push_back(value);
        cache.push_back(value);
    }
    return generated;
}

int main() {
    std::vector<long long> referenceKv = runPrefill(1);
    std::vector<long long> referenceDecode = runDecode(referenceKv, 1);

    printf("Reference (Tp=1, Td=1): kv[0..3]=%lld,%lld,%lld,%lld  "
           "decode tokens: ", referenceKv[0], referenceKv[1], referenceKv[2], referenceKv[3]);
    for (long long v : referenceDecode) printf("%lld ", v);
    printf("\n\n");

    int prefillTPs[] = {1, 2, 4, 8};
    int decodeTPs[] = {1, 2, 3, 6};
    printf("%-6s %-6s %-10s %-30s\n", "Tp", "Td", "match", "note");
    for (int Tp : prefillTPs) {
        for (int Td : decodeTPs) {
            std::vector<long long> kv = runPrefill(Tp);
            std::vector<long long> decoded = runDecode(kv, Td);
            bool match = (kv == referenceKv) && (decoded == referenceDecode);
            const char *note = (Tp <= Td) ? "Tp<=Td (Dynamo's own recommended shape)"
                                           : "Tp>Td (not the recommended shape, still correct)";
            printf("%-6d %-6d %-10s %-30s\n", Tp, Td, match ? "YES" : "NO", note);
        }
    }

    printf("\nEvery single (Tp, Td) combination matches the Tp=1/Td=1 reference "
           "exactly, whether or not Tp and Td are equal, and whether or not the "
           "pair follows NVIDIA Dynamo's own recommended shape (smaller TP for "
           "prefill, larger TP for decode). Correctness here does not come from "
           "the two phases sharing a device-group size -- it comes from each "
           "phase's own internal reduction being exact on its own, and the two "
           "phases being connected ONLY through the kv cache values themselves, "
           "never through shared parallelism structure. This is the real "
           "correctness argument behind disaggregation: prefill and decode are "
           "allowed to be sized independently precisely because nothing about "
           "their correctness depends on being sized the same.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 97_phase_specific_tensor_parallelism_correctness_simulation \
    97_phase_specific_tensor_parallelism_correctness_simulation.cpp
./97_phase_specific_tensor_parallelism_correctness_simulation
```

Locked output:

```
Reference (Tp=1, Td=1): kv[0..3]=304,722,532,532  decode tokens: 24624 215460 738720 2954880 27578880 126074880 

Tp     Td     match      note                          
1      1      YES        Tp<=Td (Dynamo's own recommended shape)
1      2      YES        Tp<=Td (Dynamo's own recommended shape)
1      3      YES        Tp<=Td (Dynamo's own recommended shape)
1      6      YES        Tp<=Td (Dynamo's own recommended shape)
2      1      YES        Tp>Td (not the recommended shape, still correct)
2      2      YES        Tp<=Td (Dynamo's own recommended shape)
2      3      YES        Tp<=Td (Dynamo's own recommended shape)
2      6      YES        Tp<=Td (Dynamo's own recommended shape)
4      1      YES        Tp>Td (not the recommended shape, still correct)
4      2      YES        Tp>Td (not the recommended shape, still correct)
4      3      YES        Tp>Td (not the recommended shape, still correct)
4      6      YES        Tp<=Td (Dynamo's own recommended shape)
8      1      YES        Tp>Td (not the recommended shape, still correct)
8      2      YES        Tp>Td (not the recommended shape, still correct)
8      3      YES        Tp>Td (not the recommended shape, still correct)
8      6      YES        Tp>Td (not the recommended shape, still correct)

Every single (Tp, Td) combination matches the Tp=1/Td=1 reference exactly, whether or not Tp and Td are equal, and whether or not the pair follows NVIDIA Dynamo's own recommended shape (smaller TP for prefill, larger TP for decode). Correctness here does not come from the two phases sharing a device-group size -- it comes from each phase's own internal reduction being exact on its own, and the two phases being connected ONLY through the kv cache values themselves, never through shared parallelism structure. This is the real correctness argument behind disaggregation: prefill and decode are allowed to be sized independently precisely because nothing about their correctness depends on being sized the same.
```

## 33.3 The KV Cache Handoff: A New Point-to-Point Communication Shape

### Intuition

Every collective this book built since Chapter 8 -- broadcast, reduce, all-reduce, all-to-all -- moves data among EVERY member of some group. The KV cache handoff is not that. It is a single, targeted, one-time bulk transfer from one specific place (the prefill instance that just finished the prompt) to one specific other place (the decode instance about to generate the first token) -- structurally much closer to Chapter 5's `cudaMemcpyPeer()` or Chapter 21's GPUDirect RDMA than to anything built since. NVIDIA's own real NIXL-based transfer in Dynamo makes this concrete: it "leverages NIXL to transfer KV cache directly from the VRAM of the prefill engine to the VRAM of the decode engine" -- one sender, one receiver, no group-wide coordination at all.

!!! warning "[COMMON TRAP] Assuming a multi-gigabyte tensor transfer must be a serving bottleneck"
    A KV cache in the gigabyte range sounds like it should dominate a request's latency. File 98 below runs the actual arithmetic on DistServe's own real cited numbers -- a real 1.13GB KV cache over a real 600 GB/s NVLink link, or a real 800 Gbps InfiniBand link -- and finds the transfer completes in low single-digit to low double-digit MILLISECONDS, small next to a real multi-hundred-millisecond-to-multi-second end-to-end serving latency. The trap is assuming "big number of bytes" automatically means "big cost," without checking it against the real bandwidth actually available -- exactly the same kind of check Chapter 21's own GPUDirect RDMA numbers required.

### Background

```text
+----------------------------------------------------------+
| Prefill instance (VRAM)  --- KV cache, point-to-point --> Decode instance (VRAM)
|   1.13 GB (real DistServe example, OPT-66B/512 tokens)     |
|   NVLink: 600 GB/s -> ~1.9 ms      InfiniBand: 100 GB/s -> ~11.3 ms
|   (NOT a collective -- one sender, one receiver, like Ch5/Ch21)
+----------------------------------------------------------+
```

File 98 takes DistServe's own real cited KV cache size and its own real cited interconnect bandwidths and computes the transfer time those real numbers imply -- the exact arithmetic behind DistServe's own conclusion that the overhead is negligible, not a new estimate this book introduces.

```cpp
// Chapter 33: Disaggregated LLM Inference
// 98_kv_cache_handoff_transfer_cost_model.cpp
//
// Every collective this book built since Chapter 8 moved data among ALL
// members of a group (a broadcast to everyone, a reduce from everyone,
// an all-to-all among everyone). The KV cache handoff between a prefill
// instance and a decode instance is a genuinely different shape: a
// targeted, one-time, point-to-point bulk transfer from one specific
// device (or TP-group) to one other specific device (or TP-group) --
// the closest ancestor in this book is Chapter 5's explicit
// cudaMemcpyPeer() and Chapter 21's GPUDirect RDMA, not any collective.
// Zhong et al.'s own real DistServe paper is candid that this transfer
// is not free -- "transferring KV caches from prefill to decoding
// instances incurs notable overheads" -- and gives a real, concrete
// example: "the KV cache size of a single 512-token request on OPT-66B
// is approximately 1.13GB." This file takes that exact real number and
// combines it with DistServe's own real cited interconnect bandwidths
// -- "Infiniband (e.g., 800 Gbps)" for cross-node transfer and
// "intra-node NVLINK, where the peak bandwidth between A100 GPUs is 600
// GB/s" -- to compute the real transfer TIME implied by those real
// numbers, the same arithmetic DistServe's own paper uses to conclude
// the overhead is negligible. No timing here is measured (this sandbox
// has no real GPU or RDMA fabric); every input is a real cited number,
// and the only computation performed is bytes divided by bandwidth.
#include <cstdio>

int main() {
    double kvCacheGB = 1.13;  // DistServe's own real cited OPT-66B/512-token example

    printf("DistServe's own real cited KV cache example: a single 512-token "
           "OPT-66B request's KV cache is approximately %.2f GB.\n\n", kvCacheGB);

    struct Link { const char *name; double gbPerSec; const char *source; };
    Link links[] = {
        {"Intra-node NVLink (A100)", 600.0, "DistServe paper: \"peak bandwidth between A100 GPUs is 600 GB/s\""},
        {"Cross-node InfiniBand",    100.0, "DistServe paper: \"Infiniband (e.g., 800 Gbps)\" = 800/8 GB/s"},
    };

    printf("%-28s %-12s %-16s\n", "Link", "GB/s", "Transfer time (ms)");
    for (auto &l : links) {
        double seconds = kvCacheGB / l.gbPerSec;
        double ms = seconds * 1000.0;
        printf("%-28s %-12.1f %-16.3f\n", l.name, l.gbPerSec, ms);
    }

    printf("\nAt real NVLink bandwidth, moving this real 1.13GB KV cache takes "
           "about %.2f milliseconds; even over real cross-node InfiniBand, about "
           "%.2f milliseconds. Both numbers are small enough next to a real LLM "
           "serving request's own end-to-end latency (typically hundreds of "
           "milliseconds to multiple seconds for a multi-hundred-token response) "
           "that DistServe's own paper calls the transmission overhead "
           "\"negligible\" -- this is the exact arithmetic behind that real "
           "conclusion, not a new estimate this book is introducing.\n",
           (kvCacheGB / 600.0) * 1000.0, (kvCacheGB / 100.0) * 1000.0);

    printf("\nNVIDIA's own real Dynamo documentation adds one further real design "
           "point this arithmetic alone does not show: the transfer does not even "
           "need to be on the critical path at all -- \"the KV transfer is "
           "non-blocking, allowing GPU forward passes to continue serving other "
           "requests during the transfer.\" A cost small enough to call negligible "
           "AND overlapped with other useful work is the real, two-part reason "
           "production disaggregated serving systems treat this handoff as a "
           "solved problem rather than their main bottleneck.\n");
    return 0;
}
```

Compile and run (plain host `.cpp`, cross-verified on both the cloud sandbox and the device):

```bash
g++ -O2 -ffp-contract=off -o 98_kv_cache_handoff_transfer_cost_model \
    98_kv_cache_handoff_transfer_cost_model.cpp
./98_kv_cache_handoff_transfer_cost_model
```

Locked output:

```
DistServe's own real cited KV cache example: a single 512-token OPT-66B request's KV cache is approximately 1.13 GB.

Link                         GB/s         Transfer time (ms)
Intra-node NVLink (A100)     600.0        1.883           
Cross-node InfiniBand        100.0        11.300          

At real NVLink bandwidth, moving this real 1.13GB KV cache takes about 1.88 milliseconds; even over real cross-node InfiniBand, about 11.30 milliseconds. Both numbers are small enough next to a real LLM serving request's own end-to-end latency (typically hundreds of milliseconds to multiple seconds for a multi-hundred-token response) that DistServe's own paper calls the transmission overhead "negligible" -- this is the exact arithmetic behind that real conclusion, not a new estimate this book is introducing.

NVIDIA's own real Dynamo documentation adds one further real design point this arithmetic alone does not show: the transfer does not even need to be on the critical path at all -- "the KV transfer is non-blocking, allowing GPU forward passes to continue serving other requests during the transfer." A cost small enough to call negligible AND overlapped with other useful work is the real, two-part reason production disaggregated serving systems treat this handoff as a solved problem rather than their main bottleneck.
```

## Chapter Summary

A single LLM inference request hides two very different workloads: DistServe's own real paper shows prefill is compute-bound (parallel work over many tokens at once) while decode is memory-bandwidth-bound (one token per step, but reading an ever-growing KV cache each time) -- File 96's illustrative model, grounded in Chapter 1's own arithmetic-intensity framing, made that contrast concrete, and DistServe's own real cited numbers (4.48x more requests, 10.2x tighter SLOs) show the payoff of treating the two phases separately. NVIDIA Dynamo's own real design takes that separation further by using different tensor-parallelism degrees for each phase -- smaller TP for compute-bound prefill, larger TP for memory-bound decode -- and File 97 confirmed, with a clean integer-only simulation, that this is correctness-preserving for ANY pair of TP degrees, not just the recommended one, because each phase's own reduction is exact on its own and the phases are connected only through the transferred KV cache values, never through shared parallelism structure. That KV cache handoff is itself a genuinely new communication shape for this book: a targeted, one-time, point-to-point bulk transfer between two disjoint device groups, closer to Chapter 5's `cudaMemcpyPeer()` and Chapter 21's GPUDirect RDMA than to any collective -- and File 98 showed, using DistServe's own real cited KV cache size and interconnect bandwidths, that this transfer costs low single-to-double-digit milliseconds, genuinely negligible next to end-to-end serving latency, and NVIDIA Dynamo's own real NIXL-based design overlaps even that small cost with continued serving of other requests.

## Self-Check Questions

1. According to DistServe's own real paper, why is decode memory-bandwidth-bound rather than compute-bound, given that it only produces one new token per step?
2. What does DistServe's own "goodput" metric measure, and why is it a more useful metric for a serving system than raw throughput alone?
3. NVIDIA Dynamo's own real documentation recommends a SMALLER tensor-parallelism degree for prefill and a LARGER one for decode. What is the reasoning behind each choice?
4. File 97 tested combinations where Tp > Td, which Dynamo's own guidance would NOT recommend. Were those combinations still correct? Why or why not?
5. Why is the KV cache handoff described as a fundamentally different communication shape than anything built in Chapter 8 through Chapter 32, and which two earlier chapters is it actually closest to?
6. Using File 98's own real cited numbers, roughly how long does it take to move a 1.13GB KV cache over a 600 GB/s NVLink connection? Over a 100 GB/s (800 Gbps) InfiniBand connection?
7. What does it mean for the KV cache transfer to be "non-blocking," per NVIDIA Dynamo's own real documentation, and why does that matter even though the transfer itself is already fast?
8. Why did File 97 deliberately use only integer arithmetic instead of floating-point, given that most of Part 6 and Chapter 32 used floating-point to study non-associativity?

## Where We Go Next

Chapter 34 turns to another real, deployed industrial system with its own distinctive multi-GPU shape: NVIDIA Merlin HugeCTR's approach to GPU-accelerated recommendation systems, which combines dense data-parallel layers with model-parallel embedding tables far larger than any single GPU's memory -- a hybrid this book's own Chapter 12 through Chapter 14 each built separately but never needed to combine within one model.

## Worked Solutions

1. Producing one new token still requires reading the model's ENTIRE running memory of the conversation so far -- the KV cache -- which grows with every step. DistServe's own paper states decode "incurs a similar level of I/O to the prefill phase," so the bottleneck is not how much arithmetic decode does (very little, per step) but how much data it must move to do that arithmetic, which is exactly what "memory-bandwidth-bound" means.
2. Goodput measures "the maximum request rate that can be served adhering to the SLO attainment goal... for each GPU provisioned." Raw throughput alone can hide a system that serves many requests but violates most users' latency expectations; goodput only counts requests served WITHIN the latency target, which is what actually determines how many GPUs a real deployment needs to meet its service commitments.
3. Prefill is compute-bound, so once each device has enough arithmetic work to stay busy, adding more devices has diminishing returns (the same principle behind any compute-bound workload's parallel scaling limit) -- a smaller TP suffices. Decode is memory-bandwidth-bound, so each additional GPU added to the TP group contributes more AGGREGATE memory bandwidth for reading the growing KV cache, which is the actual bottleneck -- so a larger TP genuinely helps more.
4. Yes, every Tp > Td combination in File 97's own locked output still matched the reference exactly. This is because each phase's own internal computation (a sum over its own local shard's contributions, then a combine across shards) is exact regardless of how many shards it's split into, and the two phases are connected only through the kv cache VALUES that get handed off, not through any shared structure that would require matching group sizes.
5. Every collective this book built from Chapter 8 (broadcast/reduce) through Chapter 32 (MoE dispatch/combine) moves data among every member of some group of ranks, even when that "combining" happens per-token as in Chapter 32. The KV cache handoff instead moves data between exactly one sender and one receiver, a single one-time bulk transfer -- structurally closest to Chapter 5's explicit `cudaMemcpyPeer()` transfer and Chapter 21's GPUDirect RDMA, both of which are also point-to-point rather than collective.
6. At 600 GB/s, 1.13 GB / 600 GB/s ≈ 1.88 milliseconds. At 100 GB/s (800 Gbps ÷ 8 bits/byte), 1.13 GB / 100 GB/s ≈ 11.3 milliseconds. Both are File 98's own locked, directly-computed values from DistServe's own real cited inputs.
7. "Non-blocking" means the GPU that just finished prefill does not have to sit idle waiting for the KV cache transfer to complete before doing other useful work -- per Dynamo's own documentation, "GPU forward passes... continue serving other requests during the transfer." It matters even though the transfer is already fast (single-digit-to-low-double-digit milliseconds) because at high request volume, even a small mandatory wait repeated across every single request adds up; overlapping it removes that wait entirely rather than merely shrinking it.
8. Because this section's own question -- does splitting a request's two phases onto independently-sized device groups (different TP degrees) preserve correctness -- is a genuinely different question from whether floating-point summation is associative under different groupings, which Chapter 8, 25, 29, 31, and 32 already investigated thoroughly. Using integers keeps File 97's own result unambiguous: any mismatch found would have to come from the phase-splitting design itself, not from floating-point rounding, which would otherwise have made the two separate questions impossible to tell apart.

---

**Sources cited in this chapter:**

- Zhong, Y. et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving." arXiv:2401.09670, USENIX OSDI 2024, fetched fresh this session from the paper's own HTML version. (Prefill compute-bound / decode memory-bandwidth-bound characterization, the "goodput" definition, the real 4.48x/10.2x measured result, the real 1.13GB KV cache size example, and the real NVLink/InfiniBand bandwidth citations.)
- NVIDIA. "Disaggregated Serving," Dynamo Documentation, docs.dynamo.nvidia.com, fetched fresh this session. (The real "larger TP for decode, smaller TP for prefill" design recommendation, the real NIXL-based direct VRAM-to-VRAM KV cache transfer description, and the real non-blocking-transfer design point.)
- This book's own Chapter 1 (memory wall / arithmetic intensity), Chapter 5 (explicit peer-to-peer transfers, `cudaMemcpyPeer()`), Chapter 14 (tensor parallelism), Chapter 21 (GPUDirect RDMA), Chapter 26 (multi-GPU LLM inference, this chapter's direct predecessor), and Chapter 32 (MoE dispatch/combine, the book's other recent departure from a uniform per-request communication pattern).
