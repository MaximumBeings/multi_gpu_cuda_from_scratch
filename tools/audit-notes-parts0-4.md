# Parts 0–4 content and implementation audit

Scope: Chapters 1–19, Getting Started and the index. No commit or publication performed by this audit. GPU execution is delegated to the recorded hardware audit; the checks below do not claim GPU results.

## Implemented source fixes

`audit-source-changes-parts0-4.json` records the 20 current source hashes. Matching complete source fences in the chapters were synchronized.

- 11/12: Correct per-device allocation ownership; genuine peer-copy and pinned-host-staged transfers, 4,096 nonuniform values checked, checked errors and hardware prerequisites.
- 14/15/43: Initialized handles, guarded event/stream lifecycle, nonzero failures, two-device eligibility where applicable. These demonstrate API ordering, not numerical compute or overlap.
- 17/18: Actual POSIX two-process memory/event IPC. Fork precedes CUDA initialization. Pipe transport exchanges opaque handles; producer lifetime extends through importer exit; 4,096 nonuniform exact payload values are checked. Legacy IPC examples target Linux. The event variant is a lifecycle/readiness demonstration, not an overlap benchmark.
- 20/21/23/24/26/27/28: Real device buffers and complete nonuniform integer oracles for broadcast, host-staged reduce, host-reduce/peer-broadcast AllReduce, one ring step, AllGather, host-staged ReduceScatter and destination-specific All-to-All. Distinct send/receive storage prevents ring overwrite races. CPU reduction arithmetic is explicitly scoped.
- 33/34: Real NCCL initial-weight broadcast and Avg/Sum gradient AllReduce, grouped rank participation, completed stream work, full reference checks.
- 37: Actual 50,331,648-byte activation-sized handoff; all 25,165,824 binary16-sized words checked. This does not execute a transformer forward pass.
- 40: Real GPU feature-sliced partial linear products (3×7 times 7×5), including uneven feature shards, followed by NCCL summation and a full independent CPU reference.
- 46: Real halo exchange on 2–4 GPUs over three generations, poisoned halo slots and varying interiors; every interior, neighbor halo and physical boundary value checked.
- 49: Valid NCCL dummy-element AllReduce with all ranks participating, completed streams and checked results. CPU phase-boundary semantics are explicit.

All changed workloads require at least two GPUs when their architecture needs a device pair. Missing hardware returns 2; API or oracle errors return nonzero. Successful paths release resources. Early process failure relies on process teardown for remaining CUDA allocations rather than continuing with invalid handles.

## Content corrections

- Historical authoring outputs and simulations are distinguished from successful current-source GPU execution. Host simulations do not prove CUDA visibility, asynchronous buffer lifetimes, scheduling-dependent deadlocks or physical-failure recovery.
- NVSwitch peer reachability does not allocate a separate full endpoint bandwidth to every simultaneous peer. H100’s aggregate 900 GB/s bidirectional endpoint budget is shared; ideal topology matrices are models, not measurements.
- UVA pointer metadata does not grant peer permissions. A supported peer route may traverse PCIe switches or NVSwitch; a dedicated physical wire is not required.
- Event operations have overhead; event recording generations matter. Copy/kernel overlap and kernel/kernel concurrency are different capabilities. Pinned host-memory requirements do not apply to transfers entirely between device allocations.
- A sole host controller can enforce a phase boundary by enqueue-all/wait-all/launch-next. Independent submitters need additional coordination. A queued NCCL operation orders its stream even if the host has not explicitly waited.
- Equal averaging of rank-local mean gradients assumes equal sample weights; uneven shards require weighted normalization.
- Exact equality in unchanged scalar host models does not establish architecture-independent GPU bit identity. Kernel choice, fused arithmetic and precision modes can differ.
- Static SM/clock properties are heuristics, not measured workload speed or live throttling telemetry.
- NCCL host blocking/progress and GPU collective completion are distinguished; peer failure may surface as an immediate error, blocked call or stalled device work.
- Toolchain and architecture prerequisites are per-example. H100 `sm_90` and B200 `sm_100` builds and their execution evidence must remain separate.

## Primary references consulted

- NVIDIA CUDA Runtime API: device, event, stream, peer-memory and memory-management contracts; API synchronization behavior.
- NVIDIA CUDA Programming Guide: multi-device context management and legacy interprocess communication.
- NVIDIA Blackwell Compatibility Guide.
- NVIDIA Hopper architecture description.
- NVIDIA NCCL communicator, stream and reduction-operation documentation.
- NVIDIA CUDA Best Practices Guide: numerical accuracy and precision.
- NVIDIA CUDA-aware MPI introduction.

## Local verification

- All 20 modified standalone sources matched their embedded chapter fences (ignoring only the terminating newline).
- Strict MkDocs build passed after the changes.
- `git diff --check` passed.
- Fresh publisher/verifier CPU-mock preparation covered all 145 sources without writing published or chapter files; output tampering was rejected. This test is explicitly not CUDA compilation or execution.

## Publication tools

`publish_execution.py` is the strict publisher. `publish_validation.py` is its compatibility entry point; incomplete evidence is not publishable. `verify_execution.py` checks current-source hashes, downloadable source/log inventory, captured commands, timestamp scopes, displayed output and evidence classification. Fresh publication remains gated on all 145 current source hashes and successful recorded execution statuses.
