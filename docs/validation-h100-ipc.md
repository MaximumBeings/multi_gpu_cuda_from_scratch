# H100 IPC correction rerun — November 19, 2026

**PASS:** the corrected example 17 compiled and passed **20 normal executions** (ten in each physical GPU direction) and **two Compute Sanitizer memcheck executions**, with `--target-processes all` requested. Each execution checks all **4096 nonuniform payload values** in the importing process. This is a same-node transport/correctness check, not a bandwidth benchmark or a cross-node test.

The source adds checked `cudaDeviceSynchronize()` after the producer’s pageable host-to-device copy and before publishing its IPC memory handle. These results validate that exact corrected revision; the earlier datasets retain their original sources and outputs.

## System and source

- Hardware: **2 × NVIDIA H100 80GB HBM3**, local **NV18** peer path.
- OS: **Ubuntu 24.04.5 LTS**, kernel **6.8.0-1064-nvidia**.
- Driver: **580.105.08**; CUDA compiler: **12.8.93**; GCC: **13.3.0**; target: **sm_90**.
- Source SHA-256: `0a6c761e152bb3d0015b3f96bfb93cf7fa61764b90cde1f23aafdf11539a6388`.

GPU identity and direction are explicit: each command sets `CUDA_VISIBLE_DEVICES` using physical UUIDs, so logical device 0 is the producer and logical device 1 the consumer. The reverse runs swap those UUIDs.

```text
Forward: GPU-610e96a4-c1f5-1b82-3d12-788e61020da2,GPU-2712078d-8293-d97d-7122-4601c5d7e628
Reverse: GPU-2712078d-8293-d97d-7122-4601c5d7e628,GPU-610e96a4-c1f5-1b82-3d12-788e61020da2
```

[Tested source](validation/2026-11-19-h100-ipc/17_ipc_mem_handle.cu) · [GPU identity](validation/2026-11-19-h100-ipc/gpu.log) · [Topology](validation/2026-11-19-h100-ipc/topology.log) · [Compiler](validation/2026-11-19-h100-ipc/nvcc.log) · [Host compiler](validation/2026-11-19-h100-ipc/gcc.log) · [OS](validation/2026-11-19-h100-ipc/os.log) · [Kernel](validation/2026-11-19-h100-ipc/kernel.log) · [Sanitizer version](validation/2026-11-19-h100-ipc/sanitizer-version.log).

[Complete command manifest](validation/2026-11-19-h100-ipc/commands.json) · [Results](validation/2026-11-19-h100-ipc/results.json) · [Integrity hashes](validation/2026-11-19-h100-ipc/integrity.json) · [Collector](validation/2026-11-19-h100-ipc/ipc-rerun-20261119.py).

## Commands and captured outputs

Compiled UTC: `2026-11-19T15:34:12.720447+00:00`. Compiler exit: **0**. [Exact compile log](validation/2026-11-19-h100-ipc/compile.log).

Run in a scratch directory containing the tested source. Set the physical UUID order for the intended direction, as in the manifest above.

```bash
nvcc -std=c++17 -O2 -arch=sm_90 17_ipc_mem_handle.cu -o example
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-610e96a4-c1f5-1b82-3d12-788e61020da2,GPU-2712078d-8293-d97d-7122-4601c5d7e628
./example
compute-sanitizer --tool memcheck --target-processes all --error-exitcode 3 ./example
```

**Actual normal-run sample output**

```text
Consumer: 4096 / 4096 exact payload values correct.
PASS: two-process IPC memory, 4096 nonuniform values checked.
```

**Actual forward memcheck output**

```text
========= COMPUTE-SANITIZER
Consumer: 4096 / 4096 exact payload values correct.
PASS: two-process IPC memory, 4096 nonuniform values checked.
========= ERROR SUMMARY: 0 errors
```

The reversed-direction memcheck also exited zero and reported zero errors. Both runs include the importing process’s complete payload comparison. The tool output supplies an aggregate error summary; no separate per-process instrumentation count or kernel-race coverage is inferred from it. The explicit producer completion call supplies the required ordering contract; repeated passing executions alone are not a proof of all possible schedules.

## All execution dates and logs

All times below are UTC on **November 19, 2026** (also November 19 in America/New_York). All 22 executions exited **0**.

| Run | Executed UTC | Raw log |
| --- | --- | --- |
| forward-01 | `2026-11-19T15:34:13.886174+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-01.log) |
| forward-02 | `2026-11-19T15:34:15.502455+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-02.log) |
| forward-03 | `2026-11-19T15:34:16.117384+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-03.log) |
| forward-04 | `2026-11-19T15:34:16.732362+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-04.log) |
| forward-05 | `2026-11-19T15:34:17.347387+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-05.log) |
| forward-06 | `2026-11-19T15:34:17.962376+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-06.log) |
| forward-07 | `2026-11-19T15:34:18.577250+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-07.log) |
| forward-08 | `2026-11-19T15:34:19.192212+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-08.log) |
| forward-09 | `2026-11-19T15:34:19.807193+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-09.log) |
| forward-10 | `2026-11-19T15:34:20.422100+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-10.log) |
| forward-memcheck | `2026-11-19T15:34:21.037084+00:00` | [Output](validation/2026-11-19-h100-ipc/forward-memcheck.log) |
| reverse-01 | `2026-11-19T15:34:23.905820+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-01.log) |
| reverse-02 | `2026-11-19T15:34:24.520888+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-02.log) |
| reverse-03 | `2026-11-19T15:34:25.135921+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-03.log) |
| reverse-04 | `2026-11-19T15:34:25.750891+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-04.log) |
| reverse-05 | `2026-11-19T15:34:26.365898+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-05.log) |
| reverse-06 | `2026-11-19T15:34:26.980996+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-06.log) |
| reverse-07 | `2026-11-19T15:34:27.595960+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-07.log) |
| reverse-08 | `2026-11-19T15:34:28.210997+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-08.log) |
| reverse-09 | `2026-11-19T15:34:28.826020+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-09.log) |
| reverse-10 | `2026-11-19T15:34:29.441001+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-10.log) |
| reverse-memcheck | `2026-11-19T15:34:30.055965+00:00` | [Output](validation/2026-11-19-h100-ipc/reverse-memcheck.log) |

[Chapter 7: IPC memory and events](part1/07-cuda-inter-process-communication.md) · [Content-review findings](content-review-2026-11-19.md).
