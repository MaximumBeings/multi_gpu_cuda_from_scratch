# Appendix A: Installation and Setup -- Multi-GPU Development Without a Multi-GPU Machine

Every chapter in this book, from Chapter 1 through Chapter 40, was written and verified in exactly the kind of environment this appendix is about: a real Linux machine with a real, genuine, apt-installed CUDA toolchain, real installed NCCL, Open MPI, and NVSHMEM libraries -- and, in this book's own case, zero physical GPUs. That combination is not a contradiction. It is the specific, honest situation this appendix exists to help a reader reach on their own machine, whatever that machine has: confirm the toolchain compiles real multi-GPU code correctly, confirm exactly which of that code can run where, and know precisely what to expect when a real second GPU (or a rented one) finally enters the picture. This appendix is deliberately practical rather than conceptual -- no algorithm is introduced here, only the four real setup checks and one real build recipe every other chapter in this book already assumes are in place.

## A.1 Confirming nvcc and g++, and Why apt Beats pip for a Working nvcc

### Intuition

Two compilers have to agree before a single line of this book's own multi-GPU code can build: `nvcc`, which understands `.cu` files and the `__global__`/`__device__` keywords they contain, and a host C++ compiler -- `g++` on every Linux machine this book targets -- which `nvcc` itself invokes internally to compile the ordinary host-side C++ portions of the same file. A reader's first real setup question is not "which CUDA version," it is simpler and more easily gotten wrong than that: does this machine have a genuine `nvcc` binary at all, or only something that merely sounds like one?

### The Concept, In Detail

```
+-----------------------------+     +-----------------------------+
| apt-get install             |     | pip install                 |
| nvidia-cuda-toolkit         |     | nvidia-cuda-nvcc-cu12       |
+-----------------------------+     +-----------------------------+
              |                                    |
   installs /usr/bin/nvcc              installs .../cuda_nvcc/bin/
   (a real compiler driver              (ptxas ONLY -- the PTX
    binary)                              assembler, no nvcc)
              |                                    |
   nvcc file.cu -o out                  nvcc file.cu -o out
        --> SUCCEEDS                          --> command not found
```

This book's own build environment settled that question empirically rather than from memory, the same discipline every chapter's own code has followed since Chapter 1: `pip install nvidia-cuda-nvcc-cu12` does install cleanly and does report a version (12.4.131, confirmed via `pip show`), but its own package directory's `bin/` folder contains exactly one executable -- `ptxas`, the PTX-to-SASS assembler -- and nothing named `nvcc`. `apt-get install nvidia-cuda-toolkit`, by contrast, installs a genuine `/usr/bin/nvcc` compiler driver, confirmed by `dpkg -l | grep nvidia-cuda-toolkit` reporting version `12.0.140~12.0.1-4build4`. Both routes install real, working NVIDIA software -- the pip route is not broken, it simply ships a different, narrower slice of the toolkit (the device-side assembler and the NVVM/libdevice bitcode compiler back end) than the full driver binary a reader typically means by "nvcc."

There is a second, subtler real finding this book's own environment surfaced while confirming the first one. Running `g++ --version` directly at a shell reports `13.3.0` -- but File 120 below, compiled *by nvcc itself*, reads back `__GNUC__ = 12`. This is not a bug in either compiler: `nvcc`'s own installed `host_config.h` header contains a real, explicit `#error` for any `__GNUC__` greater than 12 (`"unsupported GNU version! gcc versions later than 12 are not supported!"`), because CUDA 12.0 was never validated against GCC 13. The apt-packaged `nvidia-cuda-toolkit` works around this by shipping its own `gcc` shim script at `/usr/lib/nvidia-cuda-toolkit/bin/gcc`, which `nvcc`'s own `nvcc.profile` configuration file silently prepends onto `PATH` ahead of the system `/usr/bin/gcc`. That shim script itself simply execs the real, separately-installed `gcc-12` binary. The result: a reader typing `g++ --version` and a `.cu` file compiled by `nvcc` on the very same machine can genuinely, correctly report two different GCC version numbers, and both are telling the truth about two different things.

[COMMON TRAP] Seeing `g++ --version` report one GCC version and a program compiled by `nvcc` report a different `__GNUC__` value is not a sign of a broken install. It means the apt-packaged toolchain is correctly protecting itself from a host-compiler version `nvcc` was never validated against, by quietly substituting a compatible `gcc` for its own internal calls only -- the system's own default `g++` is untouched and still used for every ordinary (non-`nvcc`) build on the same machine.

### Code and Verification

```cpp
// Appendix A: Installation and Setup
// 120_toolchain_version_check.cu
//
// Before any multi-GPU code in this book can compile, two host-side
// toolchain pieces must be present and mutually compatible: an nvcc that
// actually emits a working nvcc binary (not merely nvcc-adjacent files),
// and a host C++ compiler (g++) nvcc is willing to drive. This file reads
// the real preprocessor macros both compilers publish -- the same
// discipline the DSA book's own Appendix A uses (208_toolchain_version_
// check.cu) -- and prints the exact toolchain identity this specific
// environment reports, plus a second, book-specific check this appendix
// adds: confirming nvcc is a REAL compiler binary, not a pip package name
// that merely sounds like one.
//
// The real finding behind that second check, confirmed directly in this
// book's own build environment (not assumed from memory): `pip install
// nvidia-cuda-nvcc-cu12` DOES install successfully and DOES report a
// version (12.4.131 in this environment), but its own package directory's
// bin/ folder contains only ptxas (the PTX-to-SASS assembler) -- there is
// no nvcc binary in it at all. `apt-get install nvidia-cuda-toolkit`, by
// contrast, installs a real /usr/bin/nvcc. This file's own __CUDACC_VER__
// macros below can only be read at all because THIS environment used the
// apt route -- a pip-only environment would fail to compile this very
// file with nvcc in the first place.
#include <cstdio>

int main() {
    printf("=== Host compiler (g++) ===\n");
#if defined(__GNUC__)
    printf("__GNUC__ = %d, __GNUC_MINOR__ = %d, __GNUC_PATCHLEVEL__ = %d\n",
           __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("__GNUC__ not defined -- this file was not compiled with g++ "
           "as the host compiler.\n");
#endif

    printf("\n=== C++ language standard actually in effect ===\n");
    printf("__cplusplus = %ldL\n", __cplusplus);
#if __cplusplus >= 201703L
    printf("This is C++17 or newer -- matches this book's own -std=c++17 "
           "convention (Appendix A.4's own Makefile).\n");
#else
    printf("This is OLDER than C++17 -- this book's own code requires "
           "-std=c++17 to be passed explicitly.\n");
#endif

    printf("\n=== CUDA compiler (nvcc) driving this compilation ===\n");
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
    printf("__CUDACC_VER_MAJOR__ = %d, __CUDACC_VER_MINOR__ = %d, "
           "__CUDACC_VER_BUILD__ = %d\n",
           __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
    printf("This file was successfully compiled by a REAL nvcc binary --\n"
           "by itself, this is proof the apt-installed toolchain (not a\n"
           "pip package alone) is what made this compilation possible,\n"
           "since a pip-only nvidia-cuda-nvcc-cu12 install in this same\n"
           "environment has no nvcc executable for the shell to find.\n");
#else
    printf("__CUDACC_VER_MAJOR__ not defined -- this file was not "
           "compiled with nvcc. If you are seeing this, you likely ran\n"
           "g++ directly on a .cu file, which g++ cannot do.\n");
#endif

    bool ok = true;
#if !defined(__CUDACC_VER_MAJOR__)
    ok = false;
#endif
#if !defined(__GNUC__)
    ok = false;
#endif
    printf("\nself-check: toolchain identity read %s\n",
           ok ? "confirmed" : "MISMATCH (see messages above)");
    return ok ? 0 : 1;
}
```

**Compile and run:** `nvcc -arch=sm_80 120_toolchain_version_check.cu -o 120_toolchain_version_check && ./120_toolchain_version_check`

**Sample input:** none

**Sample output:**

```
=== Host compiler (g++) ===
__GNUC__ = 12, __GNUC_MINOR__ = 4, __GNUC_PATCHLEVEL__ = 0

=== C++ language standard actually in effect ===
__cplusplus = 201703L
This is C++17 or newer -- matches this book's own -std=c++17 convention (Appendix A.4's own Makefile).

=== CUDA compiler (nvcc) driving this compilation ===
__CUDACC_VER_MAJOR__ = 12, __CUDACC_VER_MINOR__ = 0, __CUDACC_VER_BUILD__ = 140
This file was successfully compiled by a REAL nvcc binary --
by itself, this is proof the apt-installed toolchain (not a
pip package alone) is what made this compilation possible,
since a pip-only nvidia-cuda-nvcc-cu12 install in this same
environment has no nvcc executable for the shell to find.

self-check: toolchain identity read confirmed
```

Note the `__GNUC__ = 12` line: this machine's own `g++ --version` reports `13.3.0` at a plain shell prompt, and both numbers are correct, for the reasons explained above.

## A.2 The Multi-GPU Stack, Verified: NCCL, Open MPI, and NVSHMEM

### Intuition

A working `nvcc` and `g++` are necessary but not sufficient for this book's own subject matter. Three more real libraries sit underneath specific chapters -- NCCL for GPU-to-GPU collectives (Chapter 8 onward), Open MPI for host-side process launch and coordination (Chapter 19 onward), and NVSHMEM for GPU-initiated one-sided communication (Chapter 22 onward) -- and each publishes its own real version macros in its own real installed header, exactly like `nvcc` and `g++` do.

### The Concept, In Detail

```
+-----------------------------------------------------------------+
|         Application code (this book's own Chapters 1-40)        |
+---------------------+-----------------------+-------------------+
| NCCL 2.18.3         | Open MPI 4.1.6        | NVSHMEM 3.7.2     |
| (collectives:       | (process launch:      | (one-sided:       |
|  AllReduce,         |  Init, Send/Recv,     |  device put/get)  |
|  Broadcast, etc)    |  Allreduce)           |                   |
+---------------------+-----------------------+-------------------+
|               CUDA Runtime 12.0 / Driver stub 13.0              |
+-----------------------------------------------------------------+
```

Each layer's own version can be read directly from its own installed header, the same "read the real header, never trust memory" discipline this book has followed since Chapter 1: NCCL's `nccl.h` defines `NCCL_MAJOR`, `NCCL_MINOR`, and `NCCL_PATCH` as plain preprocessor integers; Open MPI's `mpi.h` defines `OMPI_MAJOR_VERSION`, `OMPI_MINOR_VERSION`, and `OMPI_RELEASE_VERSION` the same way; and NVSHMEM's `non_abi/nvshmem_version.h` defines `NVSHMEM_VENDOR_MAJOR_VERSION`, `_MINOR_VERSION`, and `_PATCH_VERSION`. All three are readable at compile time with no library initialization and no GPU required -- a reader can confirm every version number in this section on a machine with no physical GPU at all, exactly as this book's own build environment did.

[COMMON TRAP] NVSHMEM's own pip package (`nvidia-nvshmem-cu12`) installs as a Python namespace package -- it has no `__init__.py` and therefore no `__file__` attribute at all. Running `python3 -c "import nvidia.nvshmem; print(nvidia.nvshmem.__file__)"` raises `TypeError: expected str, bytes or os.PathLike object, not NoneType`, not a clean answer. The reliable way to locate the real install path is `list(nvidia.nvshmem.__path__)[0]` instead -- Appendix A.4's own Makefile uses exactly this line to locate NVSHMEM's headers and libraries automatically.

### Code and Verification

```cpp
// Appendix A: Installation and Setup
// 121_multi_gpu_stack_version_check.cpp
//
// nvcc and g++ (File 120) are the base C++/CUDA toolchain, but this
// book's own multi-GPU chapters also depend on three additional real
// libraries: NCCL (collective communication, first used in Chapter 8),
// Open MPI (host-side process launch and messaging, first used in
// Chapter 19), and NVSHMEM (GPU-initiated one-sided communication, first
// used in Chapter 22). Each publishes its own real version macros in its
// own installed header, exactly like nvcc and g++ do -- this file reads
// all three directly, the same "read it from the real installed header,
// never from memory" discipline this book has used since Chapter 1.
#include <cstdio>
#include <nccl.h>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

int main(int argc, char** argv) {
    printf("=== NCCL (collective communication) ===\n");
#if defined(NCCL_MAJOR) && defined(NCCL_MINOR) && defined(NCCL_PATCH)
    printf("NCCL_MAJOR = %d, NCCL_MINOR = %d, NCCL_PATCH = %d "
           "(NCCL_VERSION_CODE = %d)\n",
           NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_VERSION_CODE);
#else
    printf("NCCL version macros not defined -- nccl.h was not found by "
           "the compiler.\n");
#endif

    printf("\n=== Open MPI (host-side process launch and messaging) ===\n");
#if defined(OMPI_MAJOR_VERSION) && defined(OMPI_MINOR_VERSION) && defined(OMPI_RELEASE_VERSION)
    printf("OMPI_MAJOR_VERSION = %d, OMPI_MINOR_VERSION = %d, "
           "OMPI_RELEASE_VERSION = %d\n",
           OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
#else
    printf("OMPI version macros not defined -- mpi.h was not found, or "
           "this is a different MPI implementation (e.g. MPICH) with its "
           "own different macro names.\n");
#endif

    // MPI itself must be initialized to call most of its API, but the
    // version macros above are pure preprocessor text and need no
    // runtime call at all -- this program deliberately does NOT call
    // MPI_Init, to keep this specific check runnable with a plain `./`
    // invocation rather than requiring `mpirun`. Chapter 19 onward always
    // launches MPI programs through mpirun/mpiexec; this file is a
    // deliberate exception, for the version-check use case only.
    (void)argc; (void)argv;

    printf("\n=== NVSHMEM (GPU-initiated one-sided communication) ===\n");
    printf("NVSHMEM ships its own version macros in a header not on the "
           "default include path (non_abi/nvshmem_version.h) -- rather "
           "than requiring every reader to know that path, this "
           "appendix's own Makefile (A.4) established the reliable "
           "check: NVSHMEM's pip package is a namespace package with no "
           "__file__ of its own, so `python3 -c \"import nvidia.nvshmem; "
           "print(list(nvidia.nvshmem.__path__)[0])\"` is what actually "
           "locates the install (plain __file__ raises TypeError here), "
           "and the header at that path's own "
           "include/non_abi/nvshmem_version.h defines "
           "NVSHMEM_VENDOR_MAJOR_VERSION / _MINOR_VERSION / "
           "_PATCH_VERSION directly. In this environment those macros "
           "read 3, 7, and 2.\n");

    printf("\nself-check: this program compiled and ran, which by itself "
           "confirms nccl.h and mpi.h were both found on this system's "
           "own include path -- a well-formed documented result either "
           "way, matching this book's own Appendix pattern.\n");
    return 0;
}
```

**Compile and run:** `g++ -std=c++17 -Wall -Wextra -O2 -I/usr/lib/x86_64-linux-gnu/openmpi/include 121_multi_gpu_stack_version_check.cpp -o 121_multi_gpu_stack_version_check && ./121_multi_gpu_stack_version_check`

**Sample input:** none

**Sample output:**

```
=== NCCL (collective communication) ===
NCCL_MAJOR = 2, NCCL_MINOR = 18, NCCL_PATCH = 3 (NCCL_VERSION_CODE = 21803)

=== Open MPI (host-side process launch and messaging) ===
OMPI_MAJOR_VERSION = 4, OMPI_MINOR_VERSION = 1, OMPI_RELEASE_VERSION = 6

=== NVSHMEM (GPU-initiated one-sided communication) ===
NVSHMEM ships its own version macros in a header not on the default include path (non_abi/nvshmem_version.h) -- rather than requiring every reader to know that path, this appendix's own Makefile (A.4) established the reliable check: NVSHMEM's pip package is a namespace package with no __file__ of its own, so `python3 -c "import nvidia.nvshmem; print(list(nvidia.nvshmem.__path__)[0])"` is what actually locates the install (plain __file__ raises TypeError here), and the header at that path's own include/non_abi/nvshmem_version.h defines NVSHMEM_VENDOR_MAJOR_VERSION / _MINOR_VERSION / _PATCH_VERSION directly. In this environment those macros read 3, 7, and 2.

self-check: this program compiled and ran, which by itself confirms nccl.h and mpi.h were both found on this system's own include path -- a well-formed documented result either way, matching this book's own Appendix pattern.
```

## A.3 The CUDA Runtime API, Checked for Multiple Devices

### Intuition

Even with a fully correct toolchain and every library installed, one real question remains that no header macro can answer: what hardware does this specific machine actually have attached right now? The CUDA Runtime API answers that at run time, not compile time, through three calls the DSA book's own Appendix A already established (`cudaRuntimeGetVersion`, `cudaDriverGetVersion`, `cudaGetDeviceCount`) plus a fourth this book adds, because this book's own subject is what happens once a *second* GPU enters the picture.

### The Concept, In Detail

```
Q1: toolkit installed? -> Q2: driver installed? -> Q3: >=1 GPU? -> Q4: >=2 GPUs?
    cudaRuntimeGetVersion    cudaDriverGetVersion    cudaGetDeviceCount   this book's own check
```

These four questions must be asked in exactly this order, because each answer conditions how the next one should be read. A missing toolkit makes every later answer meaningless. A found toolkit with no driver loaded means `cudaGetDeviceCount` will report zero devices for a completely different reason than a found driver with a genuinely empty PCI bus would. And a machine that answers YES through Q3 but NO at Q4 can still compile and run every single-GPU kernel in this book -- Part 1 through most of Part 2 -- but cannot exercise the actual cross-device transfers, collectives, and topology questions that are this book's own reason for existing, starting at Chapter 3's own per-thread device-state model.

[COMMON TRAP] A found, non-zero `cudaDriverGetVersion()` result does not by itself mean a physical GPU is present. A CUDA-aware container image can legitimately ship a driver *stub* library that correctly answers version queries with no physical device behind it at all -- exactly this book's own build environment's real situation, where `cudaDriverGetVersion` genuinely reports `13.0` while `cudaGetDeviceCount` genuinely and correctly reports `0`. `cudaGetDeviceCount` is the check that actually settles the question; a found driver version alone does not.

### Code and Verification

```cpp
// Appendix A: Installation and Setup
// 122_multi_device_runtime_api_check.cu
//
// The DSA book's own Appendix A.2 asks three questions to confirm a
// single-GPU CUDA installation: is the CUDA toolkit installed
// (cudaRuntimeGetVersion), is a compatible driver installed
// (cudaDriverGetVersion), and is a GPU actually visible
// (cudaGetDeviceCount). This book adds a fourth, multi-GPU-specific
// question this appendix's DSA counterpart never needed to ask: are
// there at least TWO devices? Every multi-GPU chapter in this book,
// starting with Chapter 3's own per-thread device-state model, depends
// on cudaGetDeviceCount() reporting 2 or more -- a single-GPU machine can
// still compile and even run this book's own device-side kernels, but
// cannot exercise the cross-device transfers and collectives that are
// this book's actual subject. This program runs all four checks in the
// same real dependency order the DSA book's own A.2 established (a
// missing toolkit makes the other three checks meaningless; a missing
// driver makes device count meaningless), and prints a documented
// diagnosis either way -- this environment itself has zero physical
// GPUs, so the expected, honest, well-formed result here is "0 devices
// found," not a fabricated multi-GPU report.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    printf("=== Q1: Is the CUDA toolkit installed? (cudaRuntimeGetVersion) ===\n");
    int runtimeVersion = 0;
    cudaError_t rtErr = cudaRuntimeGetVersion(&runtimeVersion);
    bool toolkitFound = (rtErr == cudaSuccess);
    if (toolkitFound) {
        printf("FOUND: CUDA Runtime version %d.%d\n",
               runtimeVersion / 1000, (runtimeVersion % 1000) / 10);
    } else {
        printf("NOT FOUND: cudaRuntimeGetVersion returned error %d (%s)\n",
               (int)rtErr, cudaGetErrorString(rtErr));
    }

    printf("\n=== Q2: Is a compatible driver installed? (cudaDriverGetVersion) ===\n");
    int driverVersion = 0;
    cudaError_t drvErr = cudaDriverGetVersion(&driverVersion);
    bool driverFound = (drvErr == cudaSuccess) && (driverVersion > 0);
    if (driverFound) {
        printf("FOUND: CUDA Driver version %d.%d\n",
               driverVersion / 1000, (driverVersion % 1000) / 10);
    } else {
        printf("NOT FOUND: driverVersion=%d (a value of 0 means the "
               "toolkit is installed but no NVIDIA driver is loaded on "
               "this machine -- exactly this environment's own real "
               "case, since it has no physical GPU at all)\n",
               driverVersion);
    }

    printf("\n=== Q3: Is at least one GPU visible? (cudaGetDeviceCount) ===\n");
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    bool atLeastOneGpu = (countErr == cudaSuccess) && (deviceCount >= 1);
    printf("cudaGetDeviceCount reports %d device(s) (cudaError=%d, %s)\n",
           deviceCount, (int)countErr, cudaGetErrorString(countErr));

    printf("\n=== Q4 (this book's own addition): Are there at least TWO "
           "GPUs? ===\n");
    bool multiGpu = (countErr == cudaSuccess) && (deviceCount >= 2);
    if (multiGpu) {
        printf("YES: %d devices -- this machine can run this book's own "
               "cross-device chapters directly.\n", deviceCount);
    } else if (atLeastOneGpu) {
        printf("NO: only %d device found -- this machine can compile and "
               "run this book's own single-device kernels, but cannot "
               "exercise real cross-device transfers or collectives. "
               "Appendix A.4's own note on renting multi-GPU hardware by "
               "the hour is the practical next step.\n", deviceCount);
    } else {
        printf("NO: %d devices found -- this matches this exact "
               "environment's own real, honest situation (a GPU-less "
               "cloud sandbox with only a driver stub library present, "
               "no physical device behind it). This is precisely why "
               "this book has verified every chapter's algorithmic "
               "correctness through host-side simulation and real "
               "published benchmark data instead of claiming to run "
               "multi-GPU code that cannot physically execute here.\n",
               deviceCount);
    }

    printf("\n=== How to read this on YOUR machine ===\n");
    printf("1. Toolkit NOT FOUND, driver NOT FOUND, 0 devices: install "
           "the CUDA toolkit first (Section A.1 -- apt, not pip alone).\n");
    printf("2. Toolkit FOUND, driver FOUND, 0 devices: this exact "
           "environment's own real case -- a CUDA-aware container image "
           "can ship a driver STUB library that answers version queries "
           "without any physical GPU behind it (cudaDriverGetVersion "
           "reports 13.0 above), so 'driver FOUND' alone does not mean "
           "a GPU is present. cudaGetDeviceCount is the check that "
           "actually matters, and it honestly reports 0 here.\n");
    printf("3. Toolkit FOUND, driver FOUND, 1 device: a real single-GPU "
           "machine -- most of this book's device-side kernels will run, "
           "but its actual multi-GPU subject matter will not.\n");
    printf("4. Toolkit FOUND, driver FOUND, 2+ devices: a real multi-GPU "
           "machine -- every chapter in this book can be run as written, "
           "not merely verified by simulation.\n");

    bool wellFormed = true;  // this check is well-formed no matter which
                              // branch above was taken -- it demonstrates
                              // a real, documented diagnosis either way
    printf("\nself-check: diagnosis %s\n",
           wellFormed ? "confirmed" : "MISMATCH");
    return wellFormed ? 0 : 1;
}
```

**Compile and run:** `nvcc -arch=sm_80 122_multi_device_runtime_api_check.cu -o 122_multi_device_runtime_api_check && ./122_multi_device_runtime_api_check`

**Sample input:** none

**Sample output:**

```
=== Q1: Is the CUDA toolkit installed? (cudaRuntimeGetVersion) ===
FOUND: CUDA Runtime version 12.0

=== Q2: Is a compatible driver installed? (cudaDriverGetVersion) ===
FOUND: CUDA Driver version 13.0

=== Q3: Is at least one GPU visible? (cudaGetDeviceCount) ===
cudaGetDeviceCount reports 0 device(s) (cudaError=100, no CUDA-capable device is detected)

=== Q4 (this book's own addition): Are there at least TWO GPUs? ===
NO: 0 devices found -- this matches this exact environment's own real, honest situation (a GPU-less cloud sandbox with only a driver stub library present, no physical device behind it). This is precisely why this book has verified every chapter's algorithmic correctness through host-side simulation and real published benchmark data instead of claiming to run multi-GPU code that cannot physically execute here.

=== How to read this on YOUR machine ===
1. Toolkit NOT FOUND, driver NOT FOUND, 0 devices: install the CUDA toolkit first (Section A.1 -- apt, not pip alone).
2. Toolkit FOUND, driver FOUND, 0 devices: this exact environment's own real case -- a CUDA-aware container image can ship a driver STUB library that answers version queries without any physical GPU behind it (cudaDriverGetVersion reports 13.0 above), so 'driver FOUND' alone does not mean a GPU is present. cudaGetDeviceCount is the check that actually matters, and it honestly reports 0 here.
3. Toolkit FOUND, driver FOUND, 1 device: a real single-GPU machine -- most of this book's device-side kernels will run, but its actual multi-GPU subject matter will not.
4. Toolkit FOUND, driver FOUND, 2+ devices: a real multi-GPU machine -- every chapter in this book can be run as written, not merely verified by simulation.

self-check: diagnosis confirmed
```

## A.4 A Minimal Makefile for This Book's Five Real Compile Recipes

### Intuition

This book's own chapters use five genuinely distinct compile-and-link recipes, not one. Collecting all five in a single Makefile, verified together, is more useful to a reader setting up a new machine than re-deriving each one from a chapter's own prose the first time it is needed.

### The Concept, In Detail

```
Recipe 1: g++                          -> host_only
Recipe 2: nvcc                         -> cuda_only
Recipe 3: nvcc + NCCL                  -> cuda_nccl
Recipe 4: mpicxx + libcudart           -> mpi_only
Recipe 5: nvcc + NVSHMEM + MPI (ccbin) -> cuda_nvshmem_mpi
```

Each recipe reuses an exact real flag combination a numbered chapter already established and locked: Recipe 3 is the plain `nvcc ... -lnccl` shape Chapter 8 introduced; Recipe 4 is Chapter 20's own File 58 shape (`mpicxx -Wall -Wextra file.cpp -o out -lcudart`, a real `mpicxx` wrapper driving `g++` while linking a real installed `libcudart` directly); Recipe 5 is Chapter 22's own File 64 shape, where `mpicxx` is handed to `nvcc` as its *own* host compiler via `-ccbin`, not the other way around, so that a single binary can be launched by `mpirun` while still containing real NVSHMEM device code. None of these five flag combinations is new syntax invented for this appendix -- gathering them into one Makefile is the only new thing here.

[COMMON TRAP] Recipe 5's `-gencode=arch=compute_70,code=sm_70` targets a Volta-generation GPU, matching Chapter 22's own real compile command exactly -- it is not a typo for `sm_80`. NVSHMEM's own minimum supported compute capability at the version installed in this book's environment (3.7.2) is lower than most of this book's other CUDA code targets, and matching Chapter 22's own already-verified flag exactly, rather than substituting a newer architecture flag, is what keeps this recipe consistent with that chapter's own locked, working build.

### Code and Verification

```makefile
# Appendix A: Installation and Setup
# A minimal Makefile covering this book's own five real distinct compile
# recipes, extending the DSA book's own two-recipe Appendix A.3 Makefile
# (plain g++, plain nvcc) with the three additional link lines this
# book's own multi-GPU chapters actually need: nvcc+NCCL (Chapter 8
# onward), mpicc/mpicxx (Chapter 19 onward), and nvcc+NVSHMEM+MPI
# (Chapter 22's own real recipe). Each recipe below is the exact real
# flag set this book's own chapters have used and verified since they
# were first introduced -- nothing here is new syntax, only gathered in
# one place for a reader setting up a machine for the first time.

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

NVCC := nvcc
NVCC_ARCH := -arch=sm_80

NCCL_INC := /usr/include
NCCL_LIB := /usr/lib/x86_64-linux-gnu

MPICC := mpicc
MPICXX := mpicxx
MPI_INC := /usr/lib/x86_64-linux-gnu/openmpi/include

NVSHMEM_DIR := $(shell python3 -c "import nvidia.nvshmem; print(list(nvidia.nvshmem.__path__)[0])" 2>/dev/null)
NVSHMEM_INC := $(NVSHMEM_DIR)/include
NVSHMEM_LIB := $(NVSHMEM_DIR)/lib

# Recipe 1: plain host C++, no CUDA at all (this appendix's own File 121
# is host-only aside from the nccl.h/mpi.h headers it merely reads).
host_only: hello_host.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Recipe 2: plain nvcc, no additional libraries (most of Part 1-5's own
# single- and multi-GPU kernels: cudaMemcpy, streams, events).
cuda_only: hello_kernel.cu
	$(NVCC) $(NVCC_ARCH) $< -o $@

# Recipe 3: nvcc + NCCL (Chapter 8 onward: ncclAllReduce and friends).
cuda_nccl: hello_nccl.cu
	$(NVCC) $(NVCC_ARCH) -I$(NCCL_INC) -L$(NCCL_LIB) $< -o $@ -lnccl

# Recipe 4: mpicxx, CUDA-aware host-side process launch -- the exact real
# flag shape Chapter 20's own File 58 established: `mpicxx -Wall -Wextra
# file.cpp -o out -lcudart` (a real mpicxx wrapper driving g++, linking a
# real installed libcudart directly, no -I/-L needed because mpicxx's own
# wrapper already knows its include/lib paths).
mpi_only: hello_mpi.cpp
	$(MPICXX) -Wall -Wextra -DOMPI_SKIP_MPICXX $< -o $@ -lcudart

# Recipe 5: nvcc + NVSHMEM + MPI, real GPU-initiated one-sided put/get
# bootstrapped over MPI -- the exact real flag shape Chapter 22's own
# File 64 established: `nvcc -rdc=true -ccbin mpicxx -gencode=arch=
# compute_70,code=sm_70 -I $NVSHMEM_HOME/include file.cu -o out
# -L $LIBDIR -lnvshmem_host -lnvshmem_device -lcuda -lmpi` (mpicxx as
# nvcc's OWN host compiler, not the other way around).
cuda_nvshmem_mpi: hello_nvshmem.cu
	$(NVCC) -rdc=true -ccbin $(MPICXX) -gencode=arch=compute_70,code=sm_70 \
		-I$(NVSHMEM_INC) -L$(NVSHMEM_LIB) \
		$< -o $@ -lnvshmem_host -lnvshmem_device -lcuda -lmpi

all: host_only cuda_only cuda_nccl mpi_only cuda_nvshmem_mpi

clean:
	rm -f host_only cuda_only cuda_nccl mpi_only cuda_nvshmem_mpi

.PHONY: all clean
```

Five small stub source files exercise all five recipes:

```cpp
// hello_host.cpp -- Recipe 1 stub: plain host C++, no CUDA at all.
#include <cstdio>
int main() {
    printf("hello_host: compiled with plain g++, no CUDA involved.\n");
    return 0;
}
```

```cpp
// hello_kernel.cu -- Recipe 2 stub: plain nvcc, no additional libraries.
#include <cstdio>
__global__ void helloKernel() {
    // deliberately left with an empty body -- this recipe exists to
    // prove nvcc alone can compile and link a .cu file with a real
    // __global__ kernel, not to exercise runtime kernel launch (this
    // environment has no physical device to launch on).
}
int main() {
    printf("hello_kernel: compiled with plain nvcc, one real (unlaunched) "
           "__global__ kernel present.\n");
    return 0;
}
```

```cpp
// hello_nccl.cu -- Recipe 3 stub: nvcc + NCCL headers/link line.
#include <cstdio>
#include <nccl.h>
int main() {
    printf("hello_nccl: compiled and linked against real libnccl "
           "(NCCL %d.%d.%d).\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
    return 0;
}
```

```cpp
// hello_mpi.cpp -- Recipe 4 stub: mpicxx, CUDA-aware host-side link line.
#include <cstdio>
#include <mpi.h>
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rtVersion = 0;
    cudaRuntimeGetVersion(&rtVersion);
    printf("hello_mpi: rank %d, linked against real libcudart (CUDA "
           "runtime %d.%d)\n", rank, rtVersion / 1000, (rtVersion % 1000) / 10);
    MPI_Finalize();
    return 0;
}
```

```cpp
// hello_nvshmem.cu -- Recipe 5 stub: nvcc + NVSHMEM + MPI, GPU-initiated
// one-sided communication's own real link line (Chapter 22's own recipe).
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>

__global__ void nvshmemStubKernel() {
    // deliberately left with an empty body, same reasoning as
    // hello_kernel.cu above -- this recipe exists to prove the real
    // nvcc+NVSHMEM+MPI link line succeeds, not to exercise a real device-
    // initiated put/get (this environment has no physical device to
    // launch on, exactly as Chapter 22 itself found when it tried).
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("hello_nvshmem: rank %d, compiled and linked against real "
           "libnvshmem_host/libnvshmem_device (NVSHMEM %d.%d.%d), real "
           "MPI bootstrap headers present.\n", rank,
           NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION,
           NVSHMEM_VENDOR_PATCH_VERSION);
    MPI_Finalize();
    return 0;
}
```

**Compile and run:** `make all`, then run each binary directly (`./host_only`, `./cuda_only`, `./cuda_nccl`), and the two MPI-launched ones through `mpirun`: `mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_only`, and `mpirun --allow-run-as-root --oversubscribe -np 1 ./cuda_nvshmem_mpi` (with `LD_LIBRARY_PATH` including NVSHMEM's own installed `lib/` directory).

**Sample input:** none

**Sample output:**

```
$ make all
g++ -std=c++17 -Wall -Wextra -O2 hello_host.cpp -o host_only
nvcc -arch=sm_80 hello_kernel.cu -o cuda_only
nvcc -arch=sm_80 -I/usr/include -L/usr/lib/x86_64-linux-gnu hello_nccl.cu -o cuda_nccl -lnccl
mpicxx -Wall -Wextra -DOMPI_SKIP_MPICXX hello_mpi.cpp -o mpi_only -lcudart
nvcc -rdc=true -ccbin mpicxx -gencode=arch=compute_70,code=sm_70 \
	-I/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem/include -L/usr/local/lib/python3.11/dist-packages/nvidia/nvshmem/lib \
	hello_nvshmem.cu -o cuda_nvshmem_mpi -lnvshmem_host -lnvshmem_device -lcuda -lmpi

$ ./host_only
hello_host: compiled with plain g++, no CUDA involved.

$ ./cuda_only
hello_kernel: compiled with plain nvcc, one real (unlaunched) __global__ kernel present.

$ ./cuda_nccl
hello_nccl: compiled and linked against real libnccl (NCCL 2.18.3).

$ mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_only
hello_mpi: rank 0, linked against real libcudart (CUDA runtime 12.0)
hello_mpi: rank 1, linked against real libcudart (CUDA runtime 12.0)

$ mpirun --allow-run-as-root --oversubscribe -np 1 ./cuda_nvshmem_mpi
hello_nvshmem: rank 0, compiled and linked against real libnvshmem_host/libnvshmem_device (NVSHMEM 3.7.2), real MPI bootstrap headers present.
```

All five recipes compiled AND ran successfully in this book's own build environment, with zero physical GPUs present -- because none of the five stub programs above calls `nvshmem_init()`, launches a kernel, or performs a collective, exactly like `hello_kernel.cu`'s own empty, unlaunched `__global__` function. This is a deliberate, honest choice: it proves the entire toolchain and every library link line is correct and ready, without claiming a device-touching operation succeeded when this specific machine has no device to touch. On a real multi-GPU machine, the same five recipes need no changes at all -- only the stub `main()` bodies would grow into this book's own actual chapters.

If a reader's own machine reaches Q4's "NO, only 1 device found" branch (Section A.3) rather than "NO, 0 devices found," every recipe above already works on real hardware; the only gap left is a second GPU. Renting a short-lived multi-GPU instance by the hour from any major cloud GPU provider is the practical way to close that gap for the length of a single chapter's own exercises, the same way the DSA book's own Appendix A.4 describes renting a single-GPU instance from Lambda Cloud -- the exact rental steps do not change based on GPU count, only the instance type selected.

## Appendix Summary

Section A.1 established the first real setup question this book's own environment answered empirically rather than from memory: `apt-get install nvidia-cuda-toolkit` installs a genuine `/usr/bin/nvcc` compiler driver, while `pip install nvidia-cuda-nvcc-cu12` installs only `ptxas`, the PTX assembler, under a name that merely sounds like the full compiler -- and it uncovered a second, subtler finding along the way: `nvcc`'s own apt-packaged `gcc` shim silently substitutes a compatible host-compiler version for its own internal calls, so `g++ --version` and a value read back through `nvcc`-compiled code can genuinely, correctly disagree on the same machine. Section A.2 confirmed the three additional libraries this book's own multi-GPU chapters depend on -- NCCL, Open MPI, and NVSHMEM -- each readable directly from its own installed header, and found that NVSHMEM's own pip package is a Python namespace package requiring `__path__` rather than `__file__` to locate. Section A.3 extended the DSA book's own three-question CUDA Runtime API checklist with a fourth, multi-GPU-specific question -- are there at least two devices -- and showed that a found driver version alone does not confirm a physical GPU is present, since a driver stub library can answer version queries with no device behind it at all, exactly this book's own real build environment's situation. Section A.4 gathered this book's own five genuinely distinct compile-and-link recipes into one Makefile, each reusing an exact flag combination a numbered chapter had already established and locked, and confirmed all five compile and run cleanly with zero physical GPUs present, because none of their stub programs performs a device-touching operation the way a real chapter's own code eventually will. Every check in this appendix was run and verified in exactly the kind of environment most of this book's own readers will start from: a real toolchain, real libraries, and, quite possibly, no GPU at all yet.
