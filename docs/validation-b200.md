# Eight-B200 execution results — November 19, 2026

**145 programs compiled; 142 completed successfully. Three eight-rank NVSHMEM examples did not complete within their time limits.** These are mixed results, not an all-pass B200 certification.

The independent eight-device correctness harness passed all **56 directed peer-copy pairs**, **four collective patterns** and **three halo generations**, with zero numerical errors. The corrected IPC source also passed its full 4096-value check under Compute Sanitizer, which reported zero errors. All five documented Makefile recipes compiled for `sm_100`.

## System, scope and dependency setup

- Eight **NVIDIA B200** GPUs, CC **10.0**, with local **NV18** peer paths; one Linux host, not eight nodes.
- Ubuntu **24.04.5 LTS**, kernel **6.8.0-1064-nvidia**, driver **580.105.08**.
- NVCC **12.8.93**, GCC **13.3.0**, NCCL **2.26.2**, NVSHMEM **3.8.0**.
- The system image's managed UCX package conflicted with Ubuntu's MPI dependency selection. Tests used privately extracted **Ubuntu Open MPI 4.1.6** packages; system driver and UCX libraries were not replaced.
- The final bounded NVSHMEM retries selected same-node MPI `self,vader,tcp` transport and the `ob1` PML. Their raw environment and failed prior attempts remain available.

Four workers compiled/ran disjoint source groups to shorten the rental. Individual chapter examples retain their own participant counts; MPI multi-rank cases used eight ranks except the single-PE initializer. The independent eight-device harness ran after the initial chapter workers. All figures are correctness checks, not controlled bandwidth or scaling benchmarks. CPU models remain CPU models; API probes are not numerical workloads. No cross-node RDMA or physical-failure recovery was tested.

[GPU information](validation/2026-11-19-eight-b200/results/part-0/gpus.log) · [Topology](validation/2026-11-19-eight-b200/results/part-0/topology.log) · [Compiler](validation/2026-11-19-eight-b200/results/part-0/nvcc.log) · [Host compiler](validation/2026-11-19-eight-b200/host-gcc.log) · [OS](validation/2026-11-19-eight-b200/host-os.log) · [Private MPI packages](validation/2026-11-19-eight-b200/mpi-packages.json) · [Final retry environment](validation/2026-11-19-eight-b200/final-environment.json).

## Outcome counts

| Classification | Programs | Meaning |
| --- | ---: | --- |
| Numerical/transport checks completed | 31 | Captured passing reference checks |
| API checks completed | 13 | Captured passing API probes |
| Other executions completed | 98 | Exit-zero programs; includes host models and build/link demonstrations |
| Runtime timeouts | 3 | Compiled, but did not complete; not counted as validation passes |

[All final records](validation/2026-11-19-eight-b200/results/final-results.json) · [Integrity hashes](validation/2026-11-19-eight-b200/evidence-integrity.json) · [Initial outcome summary](validation/2026-11-19-eight-b200/run-summary.json) · [Private-MPI retry records](validation/2026-11-19-eight-b200/results/latest-results.json).

## Incomplete NVSHMEM cases

On the **eight-B200 host**, the remaining three programs—**64 — MPI bootstrap**, **65 — GPU-initiated put**, and **128 — NVSHMEM API reference**—printed successful checks on all eight GPUs but then timed out before exiting. **Their payload/bootstrap checks passed; their complete execution did not.** Each used eight MPI ranks. In every final 60-second retry, all eight PEs printed their expected checks and rank 0 printed `VALIDATION PASS: 8 PEs, errors=0`. These three results remain classified as timeouts, not completed executions.

The check messages occur before `nvshmem_finalize()` and `MPI_Finalize()` in the sources. The captured logs do not establish which call or rank stalled, or whether the cause was the runtime setup or program behavior. A missing PMIx `help-pcompress.txt` diagnostic also appears; it does not establish the cause of the timeout. The separate NVSHMEM all-to-all/put example **95 completed with eight PEs** in the longer retry.

Testing stopped to limit the cost of the eight-B200 instance. This report records the observed outcome; no further B200 rerun is scheduled. The original attempts, final retries, commands and raw logs are preserved unchanged.

## Independent eight-device harness

Compiled UTC: `2026-11-19T21:56:10.332221+00:00`. Executed UTC: `2026-11-19T21:56:11.849784+00:00`. Compile/run exits: **0 / 0**.

The successful compile command uses nvcc's host-compiler forwarding syntax:

```bash
nvcc -std=c++17 -O2 -arch=sm_100 tools/validate_eight_gpus.cu \
    -lnccl -Xcompiler=-pthread -o eight-gpu-check
./eight-gpu-check
```

The earlier bare `-pthread` invocation was rejected by nvcc and is retained in the logs. The corrected command compiled and executed; no numerical harness algorithm was changed.

[Compile log](validation/2026-11-19-eight-b200/eight-compile-fixed.log) · [Complete harness output](validation/2026-11-19-eight-b200/eight-run-fixed.log) · [Makefile build log](validation/2026-11-19-eight-b200/setup-record/make.log) · [IPC memcheck](validation/2026-11-19-eight-b200/ipc-memcheck.log).

**Actual harness output excerpt**

```text
ALL_PEER_PAIRS: PASS; pairs=56; values_checked=229544; errors=0
ALL_REDUCE: PASS; 32792 values checked across 8 ranks; errors=0
ALL_GATHER: PASS; 33088 values checked across 8 ranks; errors=0
REDUCE_SCATTER: PASS; 4136 values checked across 8 ranks; errors=0
GROUPED_ALL_TO_ALL: PASS; 16448 values checked across 8 ranks; errors=0
HALO_GENERATION_1: PASS; 8264 values checked across 8 ranks; errors=0
HALO_GENERATION_2: PASS; 8264 values checked across 8 ranks; errors=0
HALO_GENERATION_3: PASS; 8264 values checked across 8 ranks; errors=0
EIGHT_GPU_AUDIT: PASS; numerical_errors=0; cleanup=PASS
```

## Every program: execution date and sample-output log

All precise dates below are **UTC, November 19, 2026**, also November 19 in America/New_York. Each linked folder preserves the tested source, compile log and execution output.

| Program | Final status | Executed UTC | Captured output |
| --- | --- | --- | --- |
| 120_toolchain_version_check.cu | executed_unverified | `2026-11-19T21:53:28.020701+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/120_toolchain_version_check/run.log) |
| 121_multi_gpu_stack_version_check.cpp | executed_unverified | `2026-11-19T21:55:28.387033+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-0/121_multi_gpu_stack_version_check/run.log) |
| 122_multi_device_runtime_api_check.cu | executed_unverified | `2026-11-19T21:53:28.020201+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/122_multi_device_runtime_api_check/run.log) |
| 123_quiz_ring_allreduce_phases.cpp | executed_unverified | `2026-11-19T21:53:27.319644+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/123_quiz_ring_allreduce_phases/run.log) |
| 124_quiz_reassociation_cancellation.cpp | executed_unverified | `2026-11-19T21:53:28.087211+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/124_quiz_reassociation_cancellation/run.log) |
| 125_quiz_ghost_cell_trap.cpp | executed_unverified | `2026-11-19T21:53:27.270548+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/125_quiz_ghost_cell_trap/run.log) |
| 126_nccl_collective_reference.cu | validation_pass | `2026-11-19T21:53:41.779049+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/126_nccl_collective_reference/run.log) |
| 127_nccl_point_to_point.cu | validation_pass | `2026-11-19T21:53:28.438472+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/127_nccl_point_to_point/run.log) |
| 128_nvshmem_api_reference.cu | runtime_timeout | `2026-11-19T21:59:05.315051+00:00` | [Log](validation/2026-11-19-eight-b200/results/final/128_nvshmem_api_reference/run.log) |
| 129_graph_exec_update.cu | validation_pass | `2026-11-19T21:53:28.039140+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/129_graph_exec_update/run.log) |
| 130_cooperative_multi_device.cu | api_probe_pass | `2026-11-19T21:53:47.671917+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/130_cooperative_multi_device/run.log) |
| 131_modern_composed_equivalent.cu | validation_pass | `2026-11-19T21:53:47.410747+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/131_modern_composed_equivalent/run.log) |
| 132_round_count_vs_timing.cpp | executed_unverified | `2026-11-19T21:53:28.257748+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/132_round_count_vs_timing/run.log) |
| 133_multi_device_timing.cu | validation_pass | `2026-11-19T21:53:42.549132+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/133_multi_device_timing/run.log) |
| 134_nccl_debug_and_nsys.cu | api_probe_pass | `2026-11-19T21:53:54.866646+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/134_nccl_debug_and_nsys/run.log) |
| 135_torch_distributed_collectives_mapped.cu | api_probe_pass | `2026-11-19T21:53:56.208585+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/135_torch_distributed_collectives_mapped/run.log) |
| 136_ddp_bucketed_allreduce.cu | validation_pass | `2026-11-19T21:53:29.526477+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/136_ddp_bucketed_allreduce/run.log) |
| 137_zero_stage_memory_model.cpp | executed_unverified | `2026-11-19T21:53:49.443819+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/137_zero_stage_memory_model/run.log) |
| 138_collective_order_deadlock.c | executed_unverified | `2026-11-19T21:55:28.337731+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-2/138_collective_order_deadlock/run.log) |
| 139_stream_wait_enqueue_order.cu | validation_pass | `2026-11-19T21:54:02.251475+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/139_stream_wait_enqueue_order/run.log) |
| 140_topology_assumption_cost.cpp | executed_unverified | `2026-11-19T21:53:45.741279+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/140_topology_assumption_cost/run.log) |
| hello_host.cpp | executed_unverified | `2026-11-19T21:53:49.509879+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/hello_host/run.log) |
| hello_kernel.cu | executed_unverified | `2026-11-19T21:53:59.408699+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/hello_kernel/run.log) |
| hello_mpi.cpp | executed_unverified | `2026-11-19T21:55:29.217645+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-0/hello_mpi/run.log) |
| hello_nccl.cu | executed_unverified | `2026-11-19T21:53:46.709731+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/hello_nccl/run.log) |
| hello_nvshmem.cu | executed_unverified | `2026-11-19T21:57:24.340392+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-1/hello_nvshmem/run.log) |
| 01_device_query.cu | executed_unverified | `2026-11-19T21:53:59.976533+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/01_device_query/run.log) |
| 02_memory_wall.cpp | executed_unverified | `2026-11-19T21:54:05.590058+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/02_memory_wall/run.log) |
| 03_compute_wall.cpp | executed_unverified | `2026-11-19T21:53:46.775764+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/03_compute_wall/run.log) |
| 04_topology_bandwidth_model.cpp | executed_unverified | `2026-11-19T21:53:49.730805+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/04_topology_bandwidth_model/run.log) |
| 05_peer_access_query.cu | executed_unverified | `2026-11-19T21:54:10.330212+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/05_peer_access_query/run.log) |
| 06_single_device_setup.cu | executed_unverified | `2026-11-19T21:54:06.207578+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/06_single_device_setup/run.log) |
| 07_multi_device_loop_pattern.cu | executed_unverified | `2026-11-19T21:53:47.543363+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/07_multi_device_loop_pattern/run.log) |
| 08_pointer_attributes.cu | executed_unverified | `2026-11-19T21:53:50.298014+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/08_pointer_attributes/run.log) |
| 09_peer_access_setup.cu | executed_unverified | `2026-11-19T21:54:11.812011+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/09_peer_access_setup/run.log) |
| 10_p2p_simulation.cpp | executed_unverified | `2026-11-19T21:54:08.741931+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/10_p2p_simulation/run.log) |
| 11_memcpy_peer.cu | validation_pass | `2026-11-19T21:54:01.352279+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/11_memcpy_peer/run.log) |
| 12_staged_host_transfer.cu | validation_pass | `2026-11-19T21:53:55.438317+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/12_staged_host_transfer/run.log) |
| 13_when_each_wins_model.cpp | executed_unverified | `2026-11-19T21:54:20.611386+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/13_when_each_wins_model/run.log) |
| 14_event_basics.cu | api_probe_pass | `2026-11-19T21:54:09.309447+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/14_event_basics/run.log) |
| 15_cross_device_wait.cu | api_probe_pass | `2026-11-19T21:54:04.638407+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/15_cross_device_wait/run.log) |
| 16_overlap_query.cu | api_probe_pass | `2026-11-19T21:53:58.825076+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/16_overlap_query/run.log) |
| 17_ipc_mem_handle.cu | validation_pass | `2026-11-19T21:54:21.429399+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/17_ipc_mem_handle/run.log) |
| 18_ipc_event_handle.cu | validation_pass | `2026-11-19T21:54:12.796383+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/18_ipc_event_handle/run.log) |
| 19_ipc_handoff_simulation.cpp | executed_unverified | `2026-11-19T21:54:07.924562+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/19_ipc_handoff_simulation/run.log) |
| 20_broadcast_by_hand.cu | validation_pass | `2026-11-19T21:54:00.206942+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/20_broadcast_by_hand/run.log) |
| 21_reduce_by_hand.cu | validation_pass | `2026-11-19T21:54:26.319712+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/21_reduce_by_hand/run.log) |
| 22_collectives_simulation.cpp | executed_unverified | `2026-11-19T21:54:17.786908+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/22_collectives_simulation/run.log) |
| 23_naive_allreduce.cu | validation_pass | `2026-11-19T21:54:08.792726+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/23_naive_allreduce/run.log) |
| 24_ring_step_by_hand.cu | validation_pass | `2026-11-19T21:54:06.098905+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/24_ring_step_by_hand/run.log) |
| 25_ring_allreduce_simulation.cpp | executed_unverified | `2026-11-19T21:54:30.157982+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/25_ring_allreduce_simulation/run.log) |
| 26_allgather_by_hand.cu | validation_pass | `2026-11-19T21:54:18.655016+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/26_allgather_by_hand/run.log) |
| 27_reducescatter_by_hand.cu | validation_pass | `2026-11-19T21:54:15.736862+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/27_reducescatter_by_hand/run.log) |
| 28_alltoall_by_hand.cu | validation_pass | `2026-11-19T21:54:12.592525+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/28_alltoall_by_hand/run.log) |
| 29_collectives_simulation.cpp | executed_unverified | `2026-11-19T21:54:30.424808+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/29_collectives_simulation/run.log) |
| 30_nccl_comm_init.cu | api_probe_pass | `2026-11-19T21:54:25.550287+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/30_nccl_comm_init/run.log) |
| 31_nccl_collectives_mapped.cu | api_probe_pass | `2026-11-19T21:54:22.029337+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/31_nccl_collectives_mapped/run.log) |
| 32_ring_vs_tree_model.cpp | executed_unverified | `2026-11-19T21:54:18.134365+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/32_ring_vs_tree_model/run.log) |
| 33_replica_setup.cu | validation_pass | `2026-11-19T21:54:31.693643+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/33_replica_setup/run.log) |
| 34_gradient_allreduce.cu | validation_pass | `2026-11-19T21:54:42.320427+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/34_gradient_allreduce/run.log) |
| 35_data_parallel_consistency_simulation.cpp | executed_unverified | `2026-11-19T21:54:22.099101+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/35_data_parallel_consistency_simulation/run.log) |
| 36_model_shard_memory_model.cpp | executed_unverified | `2026-11-19T21:54:18.250917+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/36_model_shard_memory_model/run.log) |
| 37_layer_partition_activation_handoff.cu | validation_pass | `2026-11-19T21:54:39.392200+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/37_layer_partition_activation_handoff/run.log) |
| 38_model_parallel_forward_simulation.cpp | executed_unverified | `2026-11-19T21:54:50.267954+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/38_model_parallel_forward_simulation/run.log) |
| 39_column_parallel_linear.cu | executed_unverified | `2026-11-19T21:54:22.716639+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/39_column_parallel_linear/run.log) |
| 40_row_parallel_allreduce.cu | validation_pass | `2026-11-19T21:54:19.570383+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/40_row_parallel_allreduce/run.log) |
| 41_tensor_parallel_mlp_simulation.cpp | executed_unverified | `2026-11-19T21:54:46.085943+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/41_tensor_parallel_mlp_simulation/run.log) |
| 42_pipeline_schedule.cpp | executed_unverified | `2026-11-19T21:54:50.334288+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/42_pipeline_schedule/run.log) |
| 43_pipeline_stage_sync.cu | api_probe_pass | `2026-11-19T21:54:24.499542+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/43_pipeline_stage_sync/run.log) |
| 44_bubble_and_pipelined_correctness.cpp | executed_unverified | `2026-11-19T21:54:32.025439+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/44_bubble_and_pipelined_correctness/run.log) |
| 45_domain_decomposition_row_range.cu | executed_unverified | `2026-11-19T21:54:46.653584+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/45_domain_decomposition_row_range/run.log) |
| 46_halo_exchange.cu | validation_pass | `2026-11-19T21:54:51.202072+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/46_halo_exchange/run.log) |
| 47_stencil_decomposition_simulation.cpp | executed_unverified | `2026-11-19T21:54:26.684284+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/47_stencil_decomposition_simulation/run.log) |
| 48_per_device_sync_is_not_a_barrier.cu | executed_unverified | `2026-11-19T21:54:32.592951+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/48_per_device_sync_is_not_a_barrier/run.log) |
| 49_nccl_allreduce_as_barrier.cu | validation_pass | `2026-11-19T21:54:48.587157+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/49_nccl_allreduce_as_barrier/run.log) |
| 50_barrier_consistency_simulation.cpp | executed_unverified | `2026-11-19T21:54:56.242667+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/50_barrier_consistency_simulation/run.log) |
| 51_heterogeneous_capability_query.cu | executed_unverified | `2026-11-19T21:54:27.251836+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/51_heterogeneous_capability_query/run.log) |
| 52_weighted_shard_partition.cpp | executed_unverified | `2026-11-19T21:54:35.428565+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/52_weighted_shard_partition/run.log) |
| 53_makespan_simulation.cpp | executed_unverified | `2026-11-19T21:54:59.541928+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/53_makespan_simulation/run.log) |
| 54_blocking_vs_nonblocking_comm.cu | api_probe_pass | `2026-11-19T21:54:57.412145+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/54_blocking_vs_nonblocking_comm/run.log) |
| 55_nccl_async_detect_and_recover.cu | api_probe_pass | `2026-11-19T21:54:29.335644+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/55_nccl_async_detect_and_recover/run.log) |
| 56_backup_workers_simulation.cpp | executed_unverified | `2026-11-19T21:54:35.695285+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/56_backup_workers_simulation/run.log) |
| 57_mpi_ring_send_recv.c | executed_unverified | `2026-11-19T21:55:35.083054+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-2/57_mpi_ring_send_recv/run.log) |
| 58_mpi_cuda_aware_query_and_fallback.cpp | validation_pass | `2026-11-19T21:55:35.211173+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-0/58_mpi_cuda_aware_query_and_fallback/run.log) |
| 59_mpi_errhandler_return.c | executed_unverified | `2026-11-19T21:57:28.128902+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-1/59_mpi_errhandler_return/run.log) |
| 60_hybrid_mpi_nccl_bootstrap.cpp | executed_unverified | `2026-11-19T21:55:39.273028+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-2/60_hybrid_mpi_nccl_bootstrap/run.log) |
| 61_gpudirect_rdma_query.cu | api_probe_pass | `2026-11-19T21:55:00.111636+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/61_gpudirect_rdma_query/run.log) |
| 62_staged_vs_rdma_cost_model.cpp | executed_unverified | `2026-11-19T21:55:07.115882+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/62_staged_vs_rdma_cost_model/run.log) |
| 63_nvshmem_init_single_pe.cu | validation_pass | `2026-11-19T21:56:39.911184+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-0/63_nvshmem_init_single_pe/run.log) |
| 64_nvshmem_mpi_bootstrap.cu | runtime_timeout | `2026-11-19T21:59:05.315797+00:00` | [Log](validation/2026-11-19-eight-b200/results/final/64_nvshmem_mpi_bootstrap/run.log) |
| 65_gpu_initiated_put.cu | runtime_timeout | `2026-11-19T21:59:05.316214+00:00` | [Log](validation/2026-11-19-eight-b200/results/final/65_gpu_initiated_put/run.log) |
| 66_cross_device_graph_capture.cu | validation_pass | `2026-11-19T21:55:07.733574+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/66_cross_device_graph_capture/run.log) |
| 67_nccl_captured_in_graph.cu | validation_pass | `2026-11-19T21:57:00.288861+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-0/67_nccl_captured_in_graph/run.log) |
| 68_host_node_and_multi_process_graphs.cu | api_probe_pass | `2026-11-19T21:54:36.319552+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/68_host_node_and_multi_process_graphs/run.log) |
| 69_gemm_memory_wall_model.cpp | executed_unverified | `2026-11-19T21:55:04.004084+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/69_gemm_memory_wall_model/run.log) |
| 70_summa_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:14.477519+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/70_summa_correctness_simulation/run.log) |
| 71_cublasxt_multi_gpu_gemm.cu | validation_pass | `2026-11-19T21:54:34.889124+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/71_cublasxt_multi_gpu_gemm/run.log) |
| 72_bucketed_gradient_allreduce.cu | validation_pass | `2026-11-19T21:58:45.309196+00:00` | [Log](validation/2026-11-19-eight-b200/results/retry-1/72_bucketed_gradient_allreduce/run.log) |
| 73_distributed_sgd_convergence_simulation.cpp | executed_unverified | `2026-11-19T21:55:04.371038+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/73_distributed_sgd_convergence_simulation/run.log) |
| 74_ring_allreduce_communication_cost_model.cpp | executed_unverified | `2026-11-19T21:55:14.544063+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/74_ring_allreduce_communication_cost_model/run.log) |
| 75_tp_pp_combined_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:54:37.825027+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/75_tp_pp_combined_correctness_simulation/run.log) |
| 76_tp_pp_scale_cost_model.cpp | executed_unverified | `2026-11-19T21:54:37.856407+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/76_tp_pp_scale_cost_model/run.log) |
| 77_kv_cache_tp_scaling_model.cpp | executed_unverified | `2026-11-19T21:55:04.437394+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/77_kv_cache_tp_scaling_model/run.log) |
| 78_all_pairs_nbody_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:14.761395+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/78_all_pairs_nbody_correctness_simulation/run.log) |
| 79_nbody_vs_halo_communication_model.cpp | executed_unverified | `2026-11-19T21:54:37.891449+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/79_nbody_vs_halo_communication_model/run.log) |
| 80_barnes_hut_complexity_crossover_model.cpp | executed_unverified | `2026-11-19T21:54:37.972787+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/80_barnes_hut_complexity_crossover_model/run.log) |
| 81_distributed_bfs_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:04.804396+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/81_distributed_bfs_correctness_simulation/run.log) |
| 82_1d_partitioning_participant_count_model.cpp | executed_unverified | `2026-11-19T21:55:14.827812+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/82_1d_partitioning_participant_count_model/run.log) |
| 83_direction_optimizing_and_2d_partitioning_model.cpp | executed_unverified | `2026-11-19T21:54:37.958038+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/83_direction_optimizing_and_2d_partitioning_model/run.log) |
| 84_distributed_jacobi_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:54:38.289693+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/84_distributed_jacobi_correctness_simulation/run.log) |
| 85_allreduce_check_frequency_cost_model.cpp | executed_unverified | `2026-11-19T21:55:04.921072+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/85_allreduce_check_frequency_cost_model/run.log) |
| 86_jacobi_convergence_rate_model.cpp | executed_unverified | `2026-11-19T21:55:14.944537+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/86_jacobi_convergence_rate_model/run.log) |
| 87_independent_tile_rendering_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:54:38.174765+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/87_independent_tile_rendering_correctness_simulation/run.log) |
| 88_rendering_comm_and_load_balance_model.cpp | executed_unverified | `2026-11-19T21:54:38.456540+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/88_rendering_comm_and_load_balance_model/run.log) |
| 89_out_of_core_scene_scale_model.cpp | executed_unverified | `2026-11-19T21:55:04.987383+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/89_out_of_core_scene_scale_model/run.log) |
| 90_independent_path_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:15.161829+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/90_independent_path_correctness_simulation/run.log) |
| 91_rng_independence_and_reduction_topology_model.cpp | executed_unverified | `2026-11-19T21:54:38.441732+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/91_rng_independence_and_reduction_topology_model/run.log) |
| 92_var_percentile_and_allgather_cost_model.cpp | executed_unverified | `2026-11-19T21:54:38.673613+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/92_var_percentile_and_allgather_cost_model/run.log) |
| 100_embedding_alltoall_resharding_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:05.304384+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/100_embedding_alltoall_resharding_correctness_simulation/run.log) |
| 101_embedding_alltoall_comm_volume_cost_model.cpp | executed_unverified | `2026-11-19T21:55:15.228321+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/101_embedding_alltoall_comm_volume_cost_model/run.log) |
| 102_fourcastnet_data_parallel_training_model.cpp | executed_unverified | `2026-11-19T21:54:38.508132+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/102_fourcastnet_data_parallel_training_model/run.log) |
| 103_distributed_fft_transpose_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:54:39.090928+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/103_distributed_fft_transpose_correctness_simulation/run.log) |
| 104_distributed_fft_communication_cost_model.cpp | executed_unverified | `2026-11-19T21:55:05.370702+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/104_distributed_fft_communication_cost_model/run.log) |
| 105_exposure_profile_reduction_cost_model.cpp | executed_unverified | `2026-11-19T21:55:15.294816+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/105_exposure_profile_reduction_cost_model/run.log) |
| 106_netting_set_exposure_order_simulation.cpp | executed_unverified | `2026-11-19T21:54:38.674646+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/106_netting_set_exposure_order_simulation/run.log) |
| 107_multi_gpu_exposure_profile_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:54:39.358113+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/107_multi_gpu_exposure_profile_correctness_simulation/run.log) |
| 108_kernel_bypass_market_data_baseline_model.cpp | executed_unverified | `2026-11-19T21:55:05.437197+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/108_kernel_bypass_market_data_baseline_model/run.log) |
| 109_gpudirect_rdma_market_data_projection_model.cpp | executed_unverified | `2026-11-19T21:55:15.361531+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/109_gpudirect_rdma_market_data_projection_model/run.log) |
| 110_gdrcopy_fixed_overhead_vs_latency_budget_model.cpp | executed_unverified | `2026-11-19T21:54:38.791188+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/110_gdrcopy_fixed_overhead_vs_latency_budget_model/run.log) |
| 111_gnn_embedding_fraud_accuracy_model.cpp | executed_unverified | `2026-11-19T21:54:39.474801+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/111_gnn_embedding_fraud_accuracy_model/run.log) |
| 112_graph_partitioned_neighbor_fetch_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:05.603849+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/112_graph_partitioned_neighbor_fetch_correctness_simulation/run.log) |
| 113_distributed_tree_building_sequential_reduction_model.cpp | executed_unverified | `2026-11-19T21:55:15.427892+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/113_distributed_tree_building_sequential_reduction_model/run.log) |
| 114_wgs_multi_gpu_motivation_model.cpp | executed_unverified | `2026-11-19T21:54:38.857800+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/114_wgs_multi_gpu_motivation_model/run.log) |
| 115_embarrassingly_parallel_genomics_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:54:39.641613+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/115_embarrassingly_parallel_genomics_correctness_simulation/run.log) |
| 116_real_benchmark_scaling_efficiency_analysis.cpp | executed_unverified | `2026-11-19T21:55:05.670081+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/116_real_benchmark_scaling_efficiency_analysis/run.log) |
| 117_zslab_ghost_cell_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:15.544514+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/117_zslab_ghost_cell_correctness_simulation/run.log) |
| 118_backprojection_zero_comm_and_real_scaling_arithmetic.cpp | executed_unverified | `2026-11-19T21:54:38.974527+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/118_backprojection_zero_comm_and_real_scaling_arithmetic/run.log) |
| 119_forward_projection_communication_wall_cost_model.cpp | executed_unverified | `2026-11-19T21:54:39.758351+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/119_forward_projection_communication_wall_cost_model/run.log) |
| 93_moe_dispatch_combine_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:06.137436+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/93_moe_dispatch_combine_correctness_simulation/run.log) |
| 94_normal_vs_lowlatency_kernel_cost_model.cpp | executed_unverified | `2026-11-19T21:55:15.610824+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/94_normal_vs_lowlatency_kernel_cost_model/run.log) |
| 95_gpu_initiated_alltoall_and_put_demo.cu | validation_pass | `2026-11-19T22:00:05.481638+00:00` | [Log](validation/2026-11-19-eight-b200/results/final/95_gpu_initiated_alltoall_and_put_demo/run.log) |
| 96_prefill_vs_decode_cost_model.cpp | executed_unverified | `2026-11-19T21:54:39.824764+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-1/96_prefill_vs_decode_cost_model/run.log) |
| 97_phase_specific_tensor_parallelism_correctness_simulation.cpp | executed_unverified | `2026-11-19T21:55:06.416925+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-2/97_phase_specific_tensor_parallelism_correctness_simulation/run.log) |
| 98_kv_cache_handoff_transfer_cost_model.cpp | executed_unverified | `2026-11-19T21:55:15.727584+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-3/98_kv_cache_handoff_transfer_cost_model/run.log) |
| 99_embedding_capacity_and_parallelism_choice_model.cpp | executed_unverified | `2026-11-19T21:54:39.045414+00:00` | [Log](validation/2026-11-19-eight-b200/results/part-0/99_embedding_capacity_and_parallelism_choice_model/run.log) |

Earlier [H100](validation-h100-rerun.md) and [A100](validation-a100.md) records remain intact with their own source revisions and scopes.
