# Multi-GPU Programming in CUDA C++ -- Table of Contents (planning document)

Subtitle: From Peer-to-Peer Memory to Distributed Training Across the Cluster

This is the full planned outline. Check this file first before starting any new chapter or appendix. Mirrors the sibling books' own build-verify-lock discipline (see getting-started.md for this book's own specific honesty discipline around no real multi-GPU hardware).

## Part 0 -- Why Multi-GPU, and the Hardware Underneath It
1. Why One GPU Is Not Enough
2. Multi-GPU Hardware Topology: PCIe, NVLink, and NVSwitch
3. The CUDA Multi-GPU Programming Model: Devices, Contexts, and Streams Across Them

## Part 1 -- Getting Data Between GPUs
4. Peer-to-Peer Memory Access and Unified Virtual Addressing
5. Explicit Transfers: cudaMemcpyPeer, Staged Host Transfers, and When Each Wins
6. Streams, Events, and Cross-Device Synchronization
7. CUDA Inter-Process Communication: Sharing Memory and Events Across Processes

## Part 2 -- Collective Communication, Built From Scratch
8. Broadcast and Reduce: The First Two Collectives, By Hand
9. Ring All-Reduce: The Algorithm Behind Every Multi-GPU Training Job
10. All-Gather, Reduce-Scatter, and All-to-All
11. NCCL: What It Actually Does Differently From What You Just Built

## Part 3 -- Parallelization Strategies
12. Data Parallelism: Replicated Model, Sharded Data
13. Model Parallelism: When One GPU Can't Hold the Weights
14. Tensor Parallelism: Splitting a Single Matrix Multiply Across Devices
15. Pipeline Parallelism: Splitting Layers Across Devices, and the Bubble That Costs You
16. Domain Decomposition for Scientific Computing: Halo Exchange and Distributed Stencils

## Part 4 -- Synchronization, Load Balancing, and Failure
17. Barriers and Global Synchronization Across Devices
18. Load Balancing Across Heterogeneous GPUs
19. Stragglers, Failures, and Fault-Tolerant Collectives

## Part 5 -- Scaling Beyond One Node
20. MPI From Scratch, Then CUDA-Aware MPI: Scaling Beyond One Node
21. GPUDirect RDMA: Bypassing the Host Entirely
22. NVSHMEM and GPU-Initiated Communication
23. CUDA Graphs Across Multiple GPUs and Multiple Nodes

## Part 6 -- Case Studies
24. Multi-GPU Dense Matrix Multiplication at Scale
25. Distributed Training of a Neural Network: Data Parallelism and Ring All-Reduce
26. Multi-GPU LLM Inference: Tensor and Pipeline Parallelism in Practice
27. Multi-GPU N-Body Simulation
28. Multi-GPU Graph Processing: Distributed Breadth-First Search
29. A Distributed Jacobi Solver: Multi-GPU Scientific Computing With Halo Exchange
30. Multi-GPU Ray Tracing and Rendering
31. Multi-GPU Monte Carlo Risk Simulation

## Appendices
- A. Installation and Setup -- Multi-GPU Development Without a Multi-GPU Machine
- B. Practice Quiz
- C. NCCL and NVSHMEM: The Standard Libraries You Get for Free
- D. CUDA Graphs and Cooperative Multi-Device Kernels
- E. Profiling and Benchmarking Multi-GPU Communication
- F. From PyTorch Distributed and DeepSpeed to C++: A Rosetta Stone
- G. Common Failure Modes: Deadlocks, Silent Corruption From Missed Synchronization, and Topology Mismatches

## Status
Scaffolded 2026-09-17: mkdocs.yml, docs/index.md, docs/getting-started.md, this file.
Chapter 1 ("Why One GPU Is Not Enough") written 2026-09-17: docs/part0/01-why-one-gpu-is-not-enough.md, with code in docs/part0/code/ (01_device_query.cu, 02_memory_wall.cpp, 03_compute_wall.cpp). All three genuinely compiled and run (nvcc 12.0 via `apt-get install nvidia-cuda-toolkit` -- the pip nvcc route does not actually provide a working nvcc binary, see the corrected getting-started.md).
Chapter 2 ("Multi-GPU Hardware Topology: PCIe, NVLink, and NVSwitch") written 2026-09-17: docs/part0/02-multi-gpu-hardware-topology.md, with code in docs/part0/code/ (04_topology_bandwidth_model.cpp, 05_peer_access_query.cu). Cited real PCIe/NVLink/NVSwitch bandwidth figures (sources at end of chapter file); topology bandwidth model genuinely compiled/run (g++) and deterministic; peer-access query genuinely compiled with real nvcc and run, honestly reporting cudaErrorNoDevice. Chapter 3 ("The CUDA Multi-GPU Programming Model: Devices, Contexts, and Streams Across Them") written 2026-09-17: docs/part0/03-the-cuda-multi-gpu-programming-model.md, with code in docs/part0/code/ (06_single_device_setup.cu, 07_multi_device_loop_pattern.cu). Covers per-thread current-device state, implicit contexts, stream/event device binding (kernel-launch-vs-memcpy enforcement asymmetry), the idiomatic multi-device loop pattern, and the new-thread-defaults-to-device-0 pitfall -- cited to the CUDA Programming Guide's multi-GPU chapter and an NVIDIA dev blog post. Both code files genuinely compiled with real nvcc and run, deterministic. Chapter 4 ("Peer-to-Peer Memory Access and Unified Virtual Addressing") written 2026-09-17: docs/part1/04-peer-to-peer-memory-access-and-uva.md, with code in docs/part1/code/ (08_pointer_attributes.cu, 09_peer_access_setup.cu, 10_p2p_simulation.cpp). This is Part 1's first chapter and the book's first use of the host-side N-buffer simulation technique (see getting-started.md), used in Section 4.3 to verify P2P transfer routing logic (direct write vs. host-staged fallback) against an independent reference, since real peer-to-peer transfer performance cannot be honestly measured without a second real device. Sections 4.1-4.2 cover Unified Virtual Addressing (cudaPointerGetAttributes) and the real, unidirectional peer-access setup sequence (cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess), cited to the CUDA Programming Guide's multi-GPU chapter and CUDA Runtime API peer-access docs. Both .cu files genuinely compiled with real nvcc and run, honestly reporting cudaErrorNoDevice; the .cpp simulation genuinely compiled with g++ and run, deterministic PASS/PASS. Written in the DSA-book page format (header box, Intuition/Background split, one COMMON TRAP per section, full code, Chapter Summary/Self-Check Questions/Where We Go Next/Worked Solutions/Sources cited) established during the Chapter 1-3 reformat. Chapter 5 ("Explicit Transfers: cudaMemcpyPeer, Staged Host Transfers, and When Each Wins") written 2026-09-17: docs/part1/05-explicit-transfers-cudamemcpypeer-staged-host-transfers.md, with code in docs/part1/code/ (11_memcpy_peer.cu, 12_staged_host_transfer.cu, 13_when_each_wins_model.cpp). Covers the real cudaMemcpyPeer()/cudaMemcpyPeerAsync() calls and what they fall back to when peer access isn't enabled (5.1), the staged host-fallback made explicit as two cudaMemcpy() calls through a pinned buffer (5.2), and a plain host C++ closed-form cost model reusing Chapter 2's own already-cited PCIe/NVLink bandwidth figures to show peer-direct winning by 2x-3.17x when available, and being the only option at all when it isn't (5.3). Cited to the CUDA Programming Guide's multi-GPU chapter, the CUDA Runtime API reference for cudaMemcpyPeer, and the Asynchronous Execution page for pinned-memory semantics. Both .cu files genuinely compiled with real nvcc and run, honestly reporting cudaErrorNoDevice; the .cpp cost model genuinely compiled with g++ and run, deterministic. Written in the established DSA-book page format from the start. Next: Part 1, Chapter 6 (Streams, Events, and Cross-Device Synchronization).
