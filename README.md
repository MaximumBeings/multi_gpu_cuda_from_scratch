# Multi-GPU Programming in CUDA C++

A 40-chapter book covering device ownership, peer transfers, collectives, parallel decomposition, multi-node communication and industrial case studies, with seven appendices.

Read the [public book](https://maximumbeings.github.io/multi_gpu_cuda_from_scratch/) or start with [Getting Started](docs/getting-started.md). GPU programs, CPU reference simulations, cost models and API probes have different validation scopes; the execution records state which was checked.

The corrected staged-MPI example (58) passed the November 18 two-H100 numerical rerun and memory check. **All 145 current sources have successful execution records.** See the [additional dated H100 results](docs/validation-h100-rerun.md) for commands, system information and sample outputs.

## Recorded validation

[H100 records](docs/validation-h100.md) cover two H100 GPUs; [A100 records](docs/validation-a100.md) cover an eight-A100 host with eight-process MPI runs. All 145 recorded source revisions compiled and executed in each pass. The historical datasets distinguish 33 numerical/transport checks, 13 API validations and 99 other executions; the November 18 rerun records 34, 13 and 98 respectively after the staged-MPI correction. Three subsequent host-only model corrections have [separate local execution records](docs/host-model-corrections.md). The A100 supplement also checks all 56 ordered peer-copy pairs, four eight-rank collective patterns and three halo generations. The [eight-B200 results](docs/validation-b200.md) record 145 successful compilations, 142 completed executions and three unresolved eight-rank NVSHMEM timeouts. These timeouts are not validation passes; their cause has not yet been isolated.

## Build the website

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/test_formatter_lookup.py
python -m mkdocs build --strict
python3 tools/verify_site_links.py
python -m mkdocs serve
```

Use the same activated environment for installation and MkDocs on macOS and Linux. The pinned `clang-format` wheel contains its binary; the build hook locates that binary inside the installed Python package, so it does not require a global Homebrew installation or the environment’s scripts directory on `PATH`. To publish manually after validation, run `python -m mkdocs gh-deploy` from this environment. A system `mkdocs` command can select a different Python environment from the one where requirements were installed.

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

## Code listing presentation

Website builds format C, C++ and CUDA listings with pinned clang-format 19.1.7: four-space indentation, 68-column layout and expanded control-flow blocks. The build rejects changes to lexical tokens, literals or comment contents. Only displayed listings are formatted; stored sources, execution hashes, commands and captured outputs remain unchanged. Long literals wrap visually without inserting copied newlines. Style settings are in `.clang-format`; the build hook is `hooks/code_formatting.py`.
