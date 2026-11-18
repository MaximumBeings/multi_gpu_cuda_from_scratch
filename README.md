# Multi-GPU Programming in CUDA C++

A 40-chapter book covering device ownership, peer transfers, collectives, parallel decomposition, multi-node communication and industrial case studies, with seven appendices.

Read the [public book](https://maximumbeings.github.io/multi_gpu_cuda_from_scratch/) or start with [Getting Started](docs/getting-started.md). GPU programs, CPU reference simulations, cost models and API probes have different validation scopes; the execution records state which was checked.

The final review corrected the staged MPI example (58): device-buffer initialization, local-rank GPU binding, error checks and a full receive oracle. **144 current sources match their execution records; example 58 awaits GPU revalidation.** Its dated output below belongs to the archived revision.

## Recorded validation

[H100 records](docs/validation-h100.md) cover two H100 GPUs; [A100 records](docs/validation-a100.md) cover an eight-A100 host with eight-process MPI runs. All 145 recorded source revisions compiled and executed in each pass. Each dataset distinguishes 33 numerical/transport checks, 13 API validations and 99 other executions. The A100 supplement also checks all 56 ordered peer-copy pairs, four eight-rank collective patterns and three halo generations. B200 results are pending actual hardware.

## Build the website

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs build --strict
python3 tools/verify_site_links.py
mkdocs serve
```

## Compile and execute examples

Use a Linux GPU host with CUDA, NCCL, MPI and NVSHMEM installed for the examples that require them. The runner discovers headers/libraries, captures exact source hashes, compiler commands, UTC timestamps, topology and raw logs, and applies timeouts. It never treats exit status alone as a numerical correctness proof.

```bash
# Compile; no workloads execute without --execute.
python3 tools/validate_examples.py --arch sm_90 --output artifacts/h100-compile

# Execute on Hopper; output directories must be fresh.
python3 tools/validate_examples.py --execute --arch sm_90 --output artifacts/h100-run

# Later Blackwell validation uses separate evidence and a suitable toolkit.
python3 tools/validate_examples.py --execute --arch sm_100 --output artifacts/b200-run
```

Use `--mpi-ranks 8` on an eight-GPU host for the multi-rank examples (the explicit single-PE example remains at one). Individual examples retain their own documented device counts; use `tools/validate_eight_gpus.cu` for the dedicated all-eight-device checks.

Use repeatable `--match` and `--exclude` filters for targeted reruns. Set `CUDA_HOME`, `CUDA_INCDIR`, `CUDA_LIBDIR`, `NVSHMEM_HOME` and `NVSHMEM_LIB` when dependencies live outside normal search paths. Use the current per-program record for additional link flags and MPI process counts. A single-node test does not establish RDMA, IBGDA or cluster-scale performance.

## Original illustrations

The book includes 49 opening illustrations and 99 technical SVGs with embedded Inter, distributed under its included SIL Open Font License. The artwork uses a 760-unit reading-column canvas, selectable text and different layouts for topologies, matrix partitions, chunk exchanges, pipelines and ownership protocols.

```bash
pip install -r tools/illustrations/requirements.txt
python3 tools/illustrations/build.py
python3 tools/illustrations/validate.py
```

The generator preserves source-code fences and uses checked insertion markers. SVG links open the original vector artwork. Regeneration does not compile or execute code.

## Documentation display checks

The web listings use `tools/approved_display.py` for an explicit filename-and-literal allowlist of stale comment and diagnostic-wording cleanups. Algorithm statements, queries and calculations stay unchanged. Downloadable tested sources, full logs, timestamps and SHA-256 records remain immutable. Output marked “excerpt” selects actual recorded lines; it never substitutes newly written diagnostic text. `python3 tools/test_approved_display.py` checks this boundary and rejects unapproved algorithm changes.
