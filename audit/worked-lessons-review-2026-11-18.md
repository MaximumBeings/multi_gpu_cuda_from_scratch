# Worked lessons and rotated adversarial review — November 18, 2026

## Scope

All 40 chapters and seven appendices now introduce a concrete worked problem before the longer topical sections. Each includes declared inputs/units, numbered intermediate steps, an illustrated ownership or calculation table, an expected check, a pitfall and a guided exercise with an explained answer. The homepage provides reading routes and explains how to distinguish a derived example from captured hardware output.

Three authors covered separate groups; reviewers then rotated to groups they had not authored: Chapters 1–14/A–C, Chapters 15–28/D–E, and Chapters 29–40/F–G. Reviews examined both new walkthroughs and existing prose for contradictions, arithmetic, ownership, API scope, clarity and flow. Review findings were corrected in place.

## Corrections from review

- Peer-copy success does not identify a physical route, host-asynchronous does not promise immediate return, and a two-leg model is not a guaranteed upper bound on driver staging costs.
- Ring all-reduce is one algorithm, not the universal implementation of every training collective. Chapter 9's title, navigation and opener now describe reduce-scatter/all-gather; its existing page URL and previous heading anchor remain available.
- CUDA stream/pinning prerequisites now identify the specific overlap scenario. Completion and safe storage reuse are separate dependencies.
- NCCL group ordering, stream completion and barrier scope are distinguished. Recovery guidance is scoped to tested NCCL 2.26.2, with later communicator-shrink capabilities identified separately.
- MPI/local-rank/CUDA-visible ordinal assumptions are explicit. NVSHMEM initialization descriptions and exercise answers match the current explicit-MPI sources; issuer completion, consumer readiness and collective launch requirements are separated.
- NCCL graph capture/replay participation and valid buffer lifetimes are explicit. Analytical timing examples are not described as measured launch overhead.
- Column-vector conventions, tensor/pipeline group ownership and floating-point identity assumptions are stated.
- Application case studies distinguish toy arithmetic from production-library or domain validation. Exact quantiles need not gather every value; RNG draw ranges need a nonoverlap policy; potential tree split counts are not measured collective counts. Genomics stages and local ASTRA backprojection now state their actual dependency/input assumptions. Numerical partitioning, aggregation, completion and communication assumptions are made explicit.

## Validation and execution scope

The 47 new SVGs have embedded licensed fonts, responsive 760-unit canvases, numerical tables and at least 14-unit text (main text is larger). All were rendered for visual review. `tools/illustrations/validate_worked.py` checks coverage, glyph coverage in the actual embedded font, canvas bounds and explained exercises; it runs in deployment CI. The previous 148 illustrations retain their own validator. Total: 195 SVGs.

The existing 145 program sources and raw execution datasets were not changed or rerun for this documentation pass. A comparison to baseline commit `4f1d70a` preserved all 1,242 original fenced blocks in order and all 435 complete captured execution blocks byte-for-byte. New numerical models are teaching examples, not additional GPU measurements. The existing source/evidence, approved-display and pending-revalidation checks continue to enforce the recorded execution scope.

Validation: strict MkDocs build; internal targets and anchors; 145 displayed source listings with tokens/comments preserved; execution/rerun/host-correction validators; original and worked SVG validators; whitespace checks. Publication is verified separately after the source commit is pushed.

## Page and illustration coverage

| Page | Worked illustration |
| --- | --- |
| [01-why-one-gpu-is-not-enough](../docs/part0/01-why-one-gpu-is-not-enough.md) | [ch01](../docs/images/worked/ch01.svg) |
| [02-multi-gpu-hardware-topology](../docs/part0/02-multi-gpu-hardware-topology.md) | [ch02](../docs/images/worked/ch02.svg) |
| [03-the-cuda-multi-gpu-programming-model](../docs/part0/03-the-cuda-multi-gpu-programming-model.md) | [ch03](../docs/images/worked/ch03.svg) |
| [04-peer-to-peer-memory-access-and-uva](../docs/part1/04-peer-to-peer-memory-access-and-uva.md) | [ch04](../docs/images/worked/ch04.svg) |
| [05-explicit-transfers-cudamemcpypeer-staged-host-transfers](../docs/part1/05-explicit-transfers-cudamemcpypeer-staged-host-transfers.md) | [ch05](../docs/images/worked/ch05.svg) |
| [06-streams-events-and-cross-device-synchronization](../docs/part1/06-streams-events-and-cross-device-synchronization.md) | [ch06](../docs/images/worked/ch06.svg) |
| [07-cuda-inter-process-communication](../docs/part1/07-cuda-inter-process-communication.md) | [ch07](../docs/images/worked/ch07.svg) |
| [08-broadcast-and-reduce-the-first-two-collectives-by-hand](../docs/part2/08-broadcast-and-reduce-the-first-two-collectives-by-hand.md) | [ch08](../docs/images/worked/ch08.svg) |
| [09-ring-all-reduce-the-algorithm-behind-every-multi-gpu-training-job](../docs/part2/09-ring-all-reduce-the-algorithm-behind-every-multi-gpu-training-job.md) | [ch09](../docs/images/worked/ch09.svg) |
| [10-all-gather-reduce-scatter-and-all-to-all](../docs/part2/10-all-gather-reduce-scatter-and-all-to-all.md) | [ch10](../docs/images/worked/ch10.svg) |
| [11-nccl-what-it-actually-does-differently](../docs/part2/11-nccl-what-it-actually-does-differently.md) | [ch11](../docs/images/worked/ch11.svg) |
| [12-data-parallelism-replicated-model-sharded-data](../docs/part3/12-data-parallelism-replicated-model-sharded-data.md) | [ch12](../docs/images/worked/ch12.svg) |
| [13-model-parallelism-when-one-gpu-cant-hold-the-weights](../docs/part3/13-model-parallelism-when-one-gpu-cant-hold-the-weights.md) | [ch13](../docs/images/worked/ch13.svg) |
| [14-tensor-parallelism-splitting-a-single-matrix-multiply-across-devices](../docs/part3/14-tensor-parallelism-splitting-a-single-matrix-multiply-across-devices.md) | [ch14](../docs/images/worked/ch14.svg) |
| [15-pipeline-parallelism-splitting-layers-across-devices-and-the-bubble-that-costs-you](../docs/part3/15-pipeline-parallelism-splitting-layers-across-devices-and-the-bubble-that-costs-you.md) | [ch15](../docs/images/worked/ch15.svg) |
| [16-domain-decomposition-for-scientific-computing-halo-exchange-and-distributed-stencils](../docs/part3/16-domain-decomposition-for-scientific-computing-halo-exchange-and-distributed-stencils.md) | [ch16](../docs/images/worked/ch16.svg) |
| [17-barriers-and-global-synchronization-across-devices](../docs/part4/17-barriers-and-global-synchronization-across-devices.md) | [ch17](../docs/images/worked/ch17.svg) |
| [18-load-balancing-across-heterogeneous-gpus](../docs/part4/18-load-balancing-across-heterogeneous-gpus.md) | [ch18](../docs/images/worked/ch18.svg) |
| [19-stragglers-failures-and-fault-tolerant-collectives](../docs/part4/19-stragglers-failures-and-fault-tolerant-collectives.md) | [ch19](../docs/images/worked/ch19.svg) |
| [20-mpi-from-scratch-then-cuda-aware-mpi-scaling-beyond-one-node](../docs/part5/20-mpi-from-scratch-then-cuda-aware-mpi-scaling-beyond-one-node.md) | [ch20](../docs/images/worked/ch20.svg) |
| [21-gpudirect-rdma-bypassing-the-host-entirely](../docs/part5/21-gpudirect-rdma-bypassing-the-host-entirely.md) | [ch21](../docs/images/worked/ch21.svg) |
| [22-nvshmem-and-gpu-initiated-communication](../docs/part5/22-nvshmem-and-gpu-initiated-communication.md) | [ch22](../docs/images/worked/ch22.svg) |
| [23-cuda-graphs-across-multiple-gpus-and-multiple-nodes](../docs/part5/23-cuda-graphs-across-multiple-gpus-and-multiple-nodes.md) | [ch23](../docs/images/worked/ch23.svg) |
| [24-multi-gpu-dense-matrix-multiplication-at-scale](../docs/part6/24-multi-gpu-dense-matrix-multiplication-at-scale.md) | [ch24](../docs/images/worked/ch24.svg) |
| [25-distributed-training-of-a-neural-network-data-parallelism-and-ring-all-reduce](../docs/part6/25-distributed-training-of-a-neural-network-data-parallelism-and-ring-all-reduce.md) | [ch25](../docs/images/worked/ch25.svg) |
| [26-multi-gpu-llm-inference-tensor-and-pipeline-parallelism-in-practice](../docs/part6/26-multi-gpu-llm-inference-tensor-and-pipeline-parallelism-in-practice.md) | [ch26](../docs/images/worked/ch26.svg) |
| [27-n-body-simulation-at-scale](../docs/part6/27-n-body-simulation-at-scale.md) | [ch27](../docs/images/worked/ch27.svg) |
| [28-distributed-breadth-first-search](../docs/part6/28-distributed-breadth-first-search.md) | [ch28](../docs/images/worked/ch28.svg) |
| [29-distributed-jacobi-solver](../docs/part6/29-distributed-jacobi-solver.md) | [ch29](../docs/images/worked/ch29.svg) |
| [30-multi-gpu-ray-tracing-and-rendering](../docs/part6/30-multi-gpu-ray-tracing-and-rendering.md) | [ch30](../docs/images/worked/ch30.svg) |
| [31-multi-gpu-monte-carlo-risk-simulation](../docs/part6/31-multi-gpu-monte-carlo-risk-simulation.md) | [ch31](../docs/images/worked/ch31.svg) |
| [32-mixture-of-experts-at-scale](../docs/part7/32-mixture-of-experts-at-scale.md) | [ch32](../docs/images/worked/ch32.svg) |
| [33-disaggregated-llm-inference](../docs/part7/33-disaggregated-llm-inference.md) | [ch33](../docs/images/worked/ch33.svg) |
| [34-gpu-recommendation-systems](../docs/part7/34-gpu-recommendation-systems.md) | [ch34](../docs/images/worked/ch34.svg) |
| [35-ai-weather-forecasting-at-scale](../docs/part7/35-ai-weather-forecasting-at-scale.md) | [ch35](../docs/images/worked/ch35.svg) |
| [36-multi-gpu-xva-and-portfolio-risk](../docs/part7/36-multi-gpu-xva-and-portfolio-risk.md) | [ch36](../docs/images/worked/ch36.svg) |
| [37-gpudirect-rdma-in-capital-markets](../docs/part7/37-gpudirect-rdma-in-capital-markets.md) | [ch37](../docs/images/worked/ch37.svg) |
| [38-real-time-fraud-detection-at-payment-scale](../docs/part7/38-real-time-fraud-detection-at-payment-scale.md) | [ch38](../docs/images/worked/ch38.svg) |
| [39-real-time-genomics-at-scale](../docs/part7/39-real-time-genomics-at-scale.md) | [ch39](../docs/images/worked/ch39.svg) |
| [40-medical-imaging-reconstruction-at-scale](../docs/part7/40-medical-imaging-reconstruction-at-scale.md) | [ch40](../docs/images/worked/ch40.svg) |
| [a-installation-and-setup](../docs/appendix/a-installation-and-setup.md) | [appendix-a](../docs/images/worked/appendix-a.svg) |
| [b-practice-quiz](../docs/appendix/b-practice-quiz.md) | [appendix-b](../docs/images/worked/appendix-b.svg) |
| [c-nccl-and-nvshmem](../docs/appendix/c-nccl-and-nvshmem.md) | [appendix-c](../docs/images/worked/appendix-c.svg) |
| [d-cuda-graphs-and-cooperative-multi-device-kernels](../docs/appendix/d-cuda-graphs-and-cooperative-multi-device-kernels.md) | [appendix-d](../docs/images/worked/appendix-d.svg) |
| [e-profiling-and-benchmarking-multi-gpu-communication](../docs/appendix/e-profiling-and-benchmarking-multi-gpu-communication.md) | [appendix-e](../docs/images/worked/appendix-e.svg) |
| [f-pytorch-distributed-and-deepspeed-to-cpp-rosetta-stone](../docs/appendix/f-pytorch-distributed-and-deepspeed-to-cpp-rosetta-stone.md) | [appendix-f](../docs/images/worked/appendix-f.svg) |
| [g-common-failure-modes-deadlocks-silent-corruption-topology-mismatches](../docs/appendix/g-common-failure-modes-deadlocks-silent-corruption-topology-mismatches.md) | [appendix-g](../docs/images/worked/appendix-g.svg) |

## Primary references used for corrections

- [CUDA 12.8 runtime memory APIs](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/group__CUDART__MEMORY.html) and [API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html).
- [NCCL collective semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html); individual version/capture/recovery references are linked beside their corrections in the chapters.
- [cuRAND device API](https://docs.nvidia.com/cuda/curand/device-api-overview.html).
- [XGBoost GPU support](https://xgboost.readthedocs.io/en/stable/gpu/).
- [DeepVariant implementation details, r1.10](https://github.com/google/deepvariant/blob/r1.10/docs/deepvariant-details.md).
- [Distributed ASTRA methods and evaluation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5143361/).
