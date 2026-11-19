# Content review — November 19, 2026

Three independent reviews covered chapters 1–40, Appendices B–G, worked examples and diagram labels; a separate setup review covered Getting Started and Appendix A. Confirmed corrections include device-access permissions, normalized gradients, communicator/probe scope, MPI transport staging, callback threading and dependencies, graph replay checks, overlap claims, decoding cost, tree-building dependencies, exposure sign and the scope of industry measurements. This review is not a proof that every program is free of defects.

## Confirmed findings and corrections

| Chapters | Correction |
| --- | --- |
| 1, 13 | Scope the 16-byte Adam state convention and approximate `6ND` compute model; distinguish a capacity estimate from a historical or universal model-parallelism requirement. |
| 4 | Separate peer-access capability queries from calls that enable access. |
| 7 | Complete producer writes before IPC memory publication; notify the importer after recording the intended event generation; remove stale no-second-process claims. |
| 10–12 | Use the recorded NCCL version, update obsolete probe questions, and normalize gradients according to loss/sample weighting rather than declaring SUM invalid. |
| 14 | Separate host-modeled collective counts from GPU results and qualify floating-point reproducibility. |
| 19 | Describe the actual wall-clock polling deadline and qualify idealized backup-worker overhead. |
| 20–21 | Distinguish CUDA-aware pointer support from direct transport; MPI is one NCCL bootstrap option, not a universal requirement. |
| 23 | State callback/MPI thread and dependency contracts; distinguish peer kernel loads from peer-copy nodes; match replay answers to the actual checks. |
| 25 | Distinguish a synchronized bucket correctness test from measured backward/communication overlap. |
| 33 | Count attention as well as projections; include both K and V in BF16 cache bytes. |
| 36–39 | Correct credit-exposure direction, scope deployment/timing claims, and distinguish parent-child tree dependencies from independent sibling work. |
| Appendix F | Qualify NCCL AlltoAll availability and show larger ZeRO groups/offload as alternatives to a fixed eight-way state budget. |

## IPC producer completion — GPU rerun passed

Example 17 now completes the producer’s host-to-device copy before publishing its memory handle to a separate CUDA context. CUDA permits a pageable host-to-device `cudaMemcpy` to return before the final device DMA completes. A pipe transfers the handle, not a CUDA completion dependency. The corrected source adds checked `cudaDeviceSynchronize()` before publication. Previous passing runs remain available unchanged; they tested the earlier revision. The corrected revision subsequently passed **20 full-payload executions and two clean memchecks** on a two-H100 host on November 19, 2026, with physical producer/consumer roles reversed. See the [complete system information, dates, commands and sample outputs](validation-h100-ipc.md).

Source: [CUDA API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html).

To reproduce the corrected-source check on another suitable Linux host with two peer-accessible H100 GPUs, run from the repository root into a fresh evidence directory:

```bash
python3 tools/validate_examples.py --execute --arch sm_90 \
    --match 17_ipc_mem_handle --output artifacts/ipc-completion-rerun
```

This reproduction command is distinct from the captured run linked above. Inspect your own per-example record and the full 4096-value payload check; the captured collector additionally reversed physical GPU roles and ran memcheck.


## New local CPU execution records

The following arithmetic models were corrected, compiled and executed on November 19, 2026 (America/New_York). All precise timestamps below are UTC. These runs validate the displayed arithmetic and labels, not GPU execution or an external training system.

System: **macOS-26.6-arm64-arm-64bit-Mach-O**, **arm64**; `Apple clang version 21.0.0 (clang-2100.4.1.1)`.

[System record](validation/2026-11-19-content-review/system.json) · [Manifest](validation/2026-11-19-content-review/results.json).

## 96. Projection and dense-attention costs {#96}

Compiled UTC: `2026-11-19T15:11:53.690942+00:00`. Executed UTC: `2026-11-19T15:11:53.938160+00:00`. Compile/run exits: **0 / 0**.

Current source SHA-256: `c8c50f0009153ac6179a699d95fd51108e30de5e3ebda36a0a19b16c9bc0cbb7`.

[Source](validation/2026-11-19-content-review/96_prefill_vs_decode_cost_model/96_prefill_vs_decode_cost_model.cpp) · [Compile log](validation/2026-11-19-content-review/96_prefill_vs_decode_cost_model/compile.log) · [Run log](validation/2026-11-19-content-review/96_prefill_vs_decode_cost_model/run.log).

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -Werror 96_prefill_vs_decode_cost_model.cpp -o ./example
./example
```

**Captured sample output**

```text
Illustrative single-layer MHA terms; multiply-add counts as 2 FLOPs.
One d-by-d projection: prefill=2*N*d*d; decode=2*d*d.
Dense decode attention QK+PV=4*N*d FLOPs; BF16 K+V cache=4*N*d bytes.
N projection_prefill_FLOPs projection_decode_FLOPs attention_decode_FLOPs KV_read_bytes
128 17179869184 134217728 4194304 4194304
512 68719476736 134217728 16777216 16777216
2048 274877906944 134217728 67108864 67108864
8192 1099511627776 134217728 268435456 268435456
Attention work grows with context. These are selected terms, not total layer FLOPs or traffic.
For GQA/MQA, cache bytes=2*N*Hkv*Dh*element_bytes; attention work still depends on query heads.
Weights, other projections, MLPs, softmax, batch size and reuse must enter a full roofline model.
MODEL CHECK PASS: projection/attention separation and both K/V byte counts
```

## 113. Tree-node bounds and dependencies {#113}

Compiled UTC: `2026-11-19T15:11:54.139274+00:00`. Executed UTC: `2026-11-19T15:11:54.206481+00:00`. Compile/run exits: **0 / 0**.

Current source SHA-256: `18e5e05c417bbf13800cd371a4f27936b95770336011b21791e37b03bbd84479`.

[Source](validation/2026-11-19-content-review/113_distributed_tree_building_sequential_reduction_model/113_distributed_tree_building_sequential_reduction_model.cpp) · [Compile log](validation/2026-11-19-content-review/113_distributed_tree_building_sequential_reduction_model/compile.log) · [Run log](validation/2026-11-19-content-review/113_distributed_tree_building_sequential_reduction_model/run.log).

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -Werror 113_distributed_tree_building_sequential_reduction_model.cpp -o ./example
./example
```

**Captured sample output**

```text
XGBoost max_depth default: 6; illustrative ensemble: 500 trees.
The native xgboost.train API defaults to 10 boosting rounds; this model selects 500 trees.
Full depth-six binary tree: at most 63 internal split nodes; ensemble: at most 31500.
Node counts do not count MPI/NCCL calls: histograms may be batched, cached or derived by subtraction.
A root-to-leaf path has at most 6 dependent splits. Independent siblings may overlap.
Ordinary boosting rounds depend on prior predictions; rounds need not equal trees for every objective/configuration.
MODEL CHECK PASS: split-node bound and dependency scope; no training/performance claim
```

## 137. ZeRO state budgets and alternatives {#137}

Compiled UTC: `2026-11-19T15:11:54.322580+00:00`. Executed UTC: `2026-11-19T15:11:54.425775+00:00`. Compile/run exits: **0 / 0**.

Current source SHA-256: `de11d07097ce18f89496894fabe8074a32c60c2de75b31b33b7c2d8bd15bfcd8`.

[Source](validation/2026-11-19-content-review/137_zero_stage_memory_model/137_zero_stage_memory_model.cpp) · [Compile log](validation/2026-11-19-content-review/137_zero_stage_memory_model/compile.log) · [Run log](validation/2026-11-19-content-review/137_zero_stage_memory_model/run.log).

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -Werror 137_zero_stage_memory_model.cpp -o ./example
./example
```

**Captured sample output**

```text
=== Section F.3: DeepSpeed's ZeRO stages, derived from Chapter 13's own
    16-bytes-per-parameter breakdown, not a new formula ===

Chapter 13's own real per-parameter breakdown (mixed-precision Adam):
  2 bytes fp16 parameter   + 2 bytes fp16 gradient   = 4 bytes  (unsharded by Pos/Pos+g)
  4 bytes fp32 parameter + 4 bytes momentum + 4 bytes variance = 12 bytes  (DeepSpeed's own "optimizer states")
  total: 16 bytes/parameter, exactly Chapter 13's own already-locked figure

DeepSpeed's own real documented stages (deepspeed.readthedocs.io, quoted):
  stage 1 (Pos):     "optimizer states...are partitioned across the
                      processes, so that each process updates only its
                      partition"
  stage 2 (Pos+g):   "the reduced 16-bit gradients...are also partitioned
                      such that each process retains only the gradients
                      corresponding to its portion of the optimizer states"
  stage 3 (Pos+g+p): "the 16-bit model parameters are partitioned across
                      the processes. ZeRO-3 will automatically collect and
                      partition them during the forward and backward
                      passes"
  config: DeepSpeedZeroConfig's own "stage" field, 1/2/3, inside the
  zero_optimization dict -- a real, documented config key, not renamed here.

self-consistency check against the ZeRO paper's own published 7.5B-param,
Nd=64 worked example (31.4 / 16.6 / 1.9 GB):
  Pos     = 31.4 GB
  Pos+g   = 16.6 GB
  Pos+g+p = 1.9 GB
  matches published figures: PASS

applying the SAME validated formula to Chapter 13's own GPT-3 figure,
sharded across Chapter 12's own REPLICAS=8 data-parallel group:

configuration                GB/replica
no ZeRO (Ch13 baseline)          2800.0
ZeRO stage 1 (Pos)                962.5
ZeRO stage 2 (Pos+g)              656.2
ZeRO stage 3 (Pos+g+p)            350.0

Fixed eight-way ZeRO-3 persistent state: 350.0 GB/rank; exceeds an 80 GB device.
Larger ZeRO groups, tensor/pipeline parallelism or supported state offload are alternatives.
A 36-way group gives 77.778 GB/rank of persistent state only, not a complete memory budget.
Include activations, live parameter gathers, communication buffers and temporary workspaces.
MODEL CHECK PASS: declared ZeRO state formulas; no universal parallelism requirement
```

## Primary references

- [NVIDIA attention arithmetic and memory traffic](https://developer.nvidia.com/blog/co-designing-ai-model-attention-for-fast-interactive-long-context-inference/).
- [XGBoost native training API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.train).
