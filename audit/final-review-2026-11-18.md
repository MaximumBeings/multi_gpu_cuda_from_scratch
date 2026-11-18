# Final review — November 18, 2026 (UTC)

The independent content review, fresh SVG renders, strict documentation build and internal-link review cover all 40 chapters, seven appendices, setup material and both recorded execution datasets.

## Corrected code

Example 58 previously copied an uninitialized device allocation into the MPI send buffer. Exit zero did not establish payload correctness. The current source initializes 4096 rank-specific values, binds local ranks to devices, checks CUDA/MPI operations, validates every received value, frees resources and returns failure on a mismatch. The November 18 rerun recognized its numerical validation marker and verified both ranks’ payloads, with zero-error memcheck records.

The exact old/current source hashes are pinned in `tools/gpu_revalidation.json`. Earlier H100/A100 sources and logs remain unchanged. The November 18 two-H100 dataset verifies the corrected source: current coverage is 145 compiled/executed sources, comprising 34 numerical/transport checks, 13 API checks and 98 other executions. `tools/verify_rerun.py` checks the supplemental results and original failure/retry records.

## Clarified explanations and coverage

- Peer-copy synchronization guarantees and concurrent-transfer scope.
- NVSwitch topology and bottleneck assumptions.
- Floating-point reproducibility, collective ordering and tolerance claims.
- Uniform-input cuBLAS-Xt GEMM checks cannot detect every index permutation.
- IPC event execution covers event lifecycle and payload, rather than an outstanding-producer wait.

The last two points describe the existing tests' scope; their recorded outputs do not establish the stronger cases. Future tests should add nonuniform matrix inputs and a producer event recorded while work remains pending.

## Documentation checks

All 148 SVGs freshly rendered and reviewed; no blocking visual issue identified. Strict MkDocs and internal links pass. Evidence checks retain raw-source/log inventories and distinguish verified corrections from historical execution. Regression checks reject unregistered changes to the corrected source and archived hash.

## Additional adversarial review and host-model corrections

The subsequent review corrected Chapter 8’s signed-integer-overflow explanation, Chapter 33’s one-way NVLink bandwidth interpretation, Chapter 37’s distinction between CPU mappings, copy-call overhead and CPU/GPU ping-pong measurements, and Chapter 38’s TabFormer-versus-MAG240M dataset attribution. The three corresponding CPU-only programs (98, 110 and 111) were compiled and executed on macOS arm64 with Apple Clang 21.0.0 on November 18 UTC; commands, timestamps, source hashes and captured outputs are published in `docs/host-model-corrections.md`. These runs validate model arithmetic and corrected labels, not GPU performance or fraud-model training.

The three historical GPU-host datasets retain all 145 original source revisions and raw logs, including the earlier versions of these three models. Current sources 98, 110 and 111 are matched to the separate host-correction dataset through `tools/host_model_corrections.json`; `tools/verify_host_corrections.py` validates its inventory, hashes, successful compile/run records and visible corrections. Thus the latest revision’s coverage is 142 sources backed by their matching GPU-host records plus three locally revalidated host-only models. No new GPU execution is implied by the host corrections. Relevant SVG catalog entries were reviewed and already use conceptual labels without the incorrect numbers or dataset attribution.
