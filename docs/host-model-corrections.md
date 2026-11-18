# Corrected host models — November 18, 2026

Three host-only models were corrected after adversarial review, then compiled and executed locally. These runs check the stated arithmetic and corrected labels; they do not benchmark a GPU, network or trained model.

System: **macOS-26.6-arm64-arm-64bit-Mach-O**, **arm64**; `Apple clang version 21.0.0 (clang-2100.4.1.1)`. All times below are UTC.

[System record](validation/2026-11-18-host-corrections/system.json) · [Manifest](validation/2026-11-18-host-corrections/results.json).

## 98. kv cache handoff transfer cost model {#98}

Compiled UTC: `2026-11-18T13:04:45.776856+00:00`. Executed UTC: `2026-11-18T13:04:46.048350+00:00`. Compile/run exits: **0 / 0**.

Current source SHA-256: `cac5907c6d28de3b1410b790e7eedb3a9fef625db9f6557075f0248bc9a79ea7`.

[Source](validation/2026-11-18-host-corrections/98_kv_cache_handoff_transfer_cost_model/98_kv_cache_handoff_transfer_cost_model.cpp) · [Compile log](validation/2026-11-18-host-corrections/98_kv_cache_handoff_transfer_cost_model/compile.log) · [Run log](validation/2026-11-18-host-corrections/98_kv_cache_handoff_transfer_cost_model/run.log).

Commands below use local names; captured logs retain exact paths.

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -Werror 98_kv_cache_handoff_transfer_cost_model.cpp -o ./example
./example
```

**Captured sample output**

```text
Ideal one-way transfer bounds for 1.13 decimal GB:
A100 NVLink: 300 GB/s one-way ceiling -> 3.767 ms
800 Gbit/s InfiniBand: 100 GB/s line rate -> 11.300 ms
Protocol overhead, topology, tensor-parallel sharding and contention change achieved latency.
Overlap can hide work for other requests; this request still depends on its KV arrival.
MODEL CHECK PASS: directional bandwidth arithmetic
```

## 110. gdrcopy fixed overhead vs latency budget model {#110}

Compiled UTC: `2026-11-18T13:04:50.437894+00:00`. Executed UTC: `2026-11-18T13:04:50.602580+00:00`. Compile/run exits: **0 / 0**.

Current source SHA-256: `e820cb923d511b84415fa45426a5d08f28aeb198bb126a6abf764fb96cb176d6`.

[Source](validation/2026-11-18-host-corrections/110_gdrcopy_fixed_overhead_vs_latency_budget_model/110_gdrcopy_fixed_overhead_vs_latency_budget_model.cpp) · [Compile log](validation/2026-11-18-host-corrections/110_gdrcopy_fixed_overhead_vs_latency_budget_model/compile.log) · [Run log](validation/2026-11-18-host-corrections/110_gdrcopy_fixed_overhead_vs_latency_budget_model/run.log).

Commands below use local names; captured logs retain exact paths.

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -Werror 110_gdrcopy_fixed_overhead_vs_latency_budget_model.cpp -o ./example
./example
```

**Captured sample output**

```text
Published figures from distinct experiments (not a controlled comparison):
cudaMemcpy per-call overhead illustration: 6.0-7.0 us; not a round-trip measurement
A40 GDRCopy CPU/GPU ping-pong: 1.08762 us round-trip; not a CPU read-latency figure
GH200 four LSTM_A instances: 4.61 us p99; RTX PRO 6000 small model: 4.30 us p99
Arithmetic ratios to those separate budgets: memcpy 130.2%/139.5%; ping-pong 23.6%/25.3%
Ratios do not predict a replacement's latency, throughput or tail percentile.
CPU writes through mapped GPU memory and CPU reads have different costs; GDRCopy warns device-to-host reads can be slow.
MODEL CHECK PASS: distinct-operation comparison
```

## 111. gnn embedding fraud accuracy model {#111}

Compiled UTC: `2026-11-18T13:04:50.773231+00:00`. Executed UTC: `2026-11-18T13:04:50.865961+00:00`. Compile/run exits: **0 / 0**.

Current source SHA-256: `1edc5663412a05aaec3856478160730e749dfac25c0cd9e940fc333068637094`.

[Source](validation/2026-11-18-host-corrections/111_gnn_embedding_fraud_accuracy_model/111_gnn_embedding_fraud_accuracy_model.cpp) · [Compile log](validation/2026-11-18-host-corrections/111_gnn_embedding_fraud_accuracy_model/compile.log) · [Run log](validation/2026-11-18-host-corrections/111_gnn_embedding_fraud_accuracy_model/run.log).

Commands below use local names; captured logs retain exact paths.

```bash
clang++ -std=c++17 -O2 -Wall -Wextra -Werror 111_gnn_embedding_fraud_accuracy_model.cpp -o ./example
./example
```

**Captured sample output**

```text
NVIDIA's TabFormer fraud workflow: XGBoost test AUPRC
With GNN embeddings: 0.90; without embeddings: 0.79
Absolute AUPRC gain: 0.11; relative metric gain: 13.9%
MAG240M is a separate academic-paper graph benchmark, not the source of these fraud AUPRC values.
AUPRC improvement is not a measured fraud-catch rate, false-positive reduction or dollar saving.
MODEL CHECK PASS: AUPRC metric arithmetic and dataset attribution
```

## Primary references

- [NVIDIA Ampere Tuning Guide — NVLink bandwidth](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#third-generation-nvlink).
- [NVIDIA GDRCopy README — CPU mappings and ping-pong example](https://github.com/NVIDIA/gdrcopy).
- [NVIDIA fraud-detection workflow — TabFormer and the separate MAG240M benchmark](https://developer.nvidia.com/blog/optimizing-fraud-detection-in-financial-services-with-graph-neural-networks-and-nvidia-gpus/).
