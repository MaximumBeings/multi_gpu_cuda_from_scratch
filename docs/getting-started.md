# Getting Started

This book targets CUDA C++ compiled with `nvcc` for real NVIDIA GPU architectures -- `sm_70` (Volta) and newer covers everything this book builds -- plus MPI (OpenMPI or MPICH) for Part 5's multi-node material. A real multi-GPU machine with a matching driver is what you need to *run* the code in this book end to end; `nvcc` itself only needs a CUDA toolkit and will happily compile for architectures your current machine doesn't have, and MPI programs compile and run fine as multiple local ranks on a single machine with no GPU at all.

## Installing a toolchain

A full NVIDIA CUDA Toolkit install (from developer.nvidia.com, or via your platform's package manager, e.g. `apt-get install nvidia-cuda-toolkit` on Debian/Ubuntu) gives you a genuine, working `nvcc`, `ptxas`, `cicc`, and every header this book uses, with no GPU or driver required to *compile* -- only to run device code. This is what this book's own examples are authored and verified with.

The lighter-weight `pip install nvidia-cuda-nvcc-cu12` route some CUDA guides suggest is **verified here not to work as a drop-in `nvcc` replacement**: as of the versions checked while writing this book (12.9.86 and 12.4.131, on both x86_64 and aarch64), that wheel installs only `ptxas`, the `nvvm` backend, and CUDA C++ headers -- it does *not* include the `nvcc` driver script or the `cicc` front-end compiler, so there is no `nvcc` binary to invoke afterward. This matches multiple reports on NVIDIA's own developer forums of the same gap. If a lighter-than-full-toolkit option matters to you, check for an `nvcc` binary under the installed package's `bin/` directory before relying on it -- don't assume the package name implies a working compiler. For Part 5's MPI material, an ordinary OpenMPI install (`apt-get install libopenmpi-dev openmpi-bin` or your platform's equivalent) is enough to compile and run real, multi-rank MPI programs on one machine -- CUDA-aware MPI specifically (an OpenMPI build configured `--with-cuda`) is the one piece of this book's own toolchain that a driver-less, device-less environment cannot exercise for real, and Chapter 20 and Appendix A both say exactly which parts of that chapter's code are genuinely run as ordinary MPI ranks versus checked by simulation for that reason.

```bash
nvcc --version   # compiler, needed to build anything
nvcc -arch=sm_80 hello_cuda.cu -o hello_cuda   # pick the arch flag matching your GPU
mpirun --version   # for Part 5 onward
```

## The honesty discipline this book follows

This book's reference implementations were authored and compiled in an environment with a full CUDA and MPI toolchain but **no NVIDIA GPU, no NVIDIA driver, and no real second device of any kind** -- a more severe version of the constraint a single-GPU CUDA book already has, since a multi-GPU book's own central claims are about what happens *between* devices, not just on one. This book handles that limitation the same way throughout, stated once here rather than repeated on every page:

- **Plain C++ host code** (no `__global__`, no `__device__`, no CUDA Runtime API calls) is genuinely compiled with `g++` and genuinely run. Its output is locked into the page and re-verified by a fresh recompile and rerun before publication.
- **CUDA kernels that need a device to execute** are genuinely compiled with `nvcc` for a real architecture, then checked by a **host-side simulation**: ordinary C++ that walks the exact same grid, block, and thread loop nest the kernel's own launch configuration would produce, checked against an independent scalar reference.
- **Multi-device orchestration** -- a peer-to-peer transfer, a collective like ring all-reduce, a cross-device barrier -- is checked by a **host-side simulation of every participating device**: N real in-memory buffers standing in for N real devices, exchanging data with the exact same message pattern (same ring order, same tree shape, same round count) the real collective would use, with the final result checked against an independently computed reference (an ordinary sequential sum, for instance, checked against a simulated ring all-reduce's result). This is a real, exhaustive verification of the collective's *logic* -- where nearly every genuine multi-GPU bug actually lives (an off-by-one in a ring's neighbor index, a race between a send and the buffer it reads from, a barrier one rank never reaches) -- not a substitute for measuring it on real hardware.
- **CUDA Runtime API calls that can be genuinely made without a device** (`cudaGetDeviceCount`, `cudaMalloc`, `cudaDeviceCanAccessPeer`, and similar) are genuinely called, and this environment honestly reports back whatever a driver-less, device-less machine actually reports -- typically a device count of zero and `cudaErrorNoDevice` or `cudaErrorInsufficientDriver` -- rather than a fabricated success.
- **Plain MPI code** (point-to-point sends/receives, communicators, collectives) genuinely runs as multiple real local ranks (`mpirun -np 4 ...`) on a single machine, with no GPU involved at all -- this is real, not simulated, since MPI itself needs no special hardware.
- **Real hardware facts** this book states -- NVLink or PCIe generation bandwidth figures, a specific GPU's real interconnect topology, a real cluster's real numbers -- are cited to real, current vendor documentation rather than assumed from memory, the same discipline this book's own sibling volume follows for edge-hardware bandwidth figures.
- **Timing and throughput numbers are never fabricated.** Where a chapter's argument depends on which of two approaches is faster rather than merely which is correct, this book measures a genuinely computed, deterministic quantity instead of a wall-clock number -- a message count, a round count, a byte count moved -- specifically because a wall-clock number captured once on one machine is not reproducible on a rerun, let alone on a reader's own cluster, the way an exact count is.

Every chapter states which of the above applies to each piece of its own code, so nothing is left for a reader to guess about how a claim was actually established.

## Compile-line conventions

- Plain C++ host files (`.cpp`): `g++ -std=c++17 -Wall -Wextra -O2 file.cpp -o binary`
- CUDA files with real device code (`.cu`): `nvcc -arch=sm_80 file.cu -o binary`
- Files that call the CUDA Runtime API from the host add `-lcudart` and the toolchain's include/library paths, shown inline wherever they're used.
- MPI files (`.cpp`, built with the MPI wrapper compiler): `mpic++ -std=c++17 -Wall -Wextra -O2 file.cpp -o binary`, run as `mpirun -np N ./binary`.
- CUDA-aware MPI files combine both: `mpic++ -std=c++17 file.cpp -I$CUDA_HOME/include -L$CUDA_HOME/lib64 -lcudart -o binary`.

## Prerequisites

This book assumes working knowledge of C++ (structs, templates, pointers, RAII) and prior single-GPU CUDA experience at the level of "you have written and launched a kernel before" -- it does not re-derive the CUDA execution and memory model from nothing the way a first CUDA book would, though Part 0 does rebuild everything specific to *multiple* devices from scratch. No prior MPI experience is assumed; Part 5's own Chapter 20 builds plain MPI from nothing before it ever adds CUDA to the picture.
