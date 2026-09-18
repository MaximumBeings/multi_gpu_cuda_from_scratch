# Chapter 22: NVSHMEM and GPU-Initiated Communication

**What you will understand by the end of this chapter:**

- What NVSHMEM actually is: a real implementation of OpenSHMEM's partitioned global address space (PGAS) model for GPU clusters, and how its symmetric heap differs from every earlier chapter's own device memory.
- Why NVSHMEM still needs a bootstrap mechanism -- and how it reuses this book's own real MPI infrastructure (Chapter 20) to solve the exact discovery problem Chapter 21 already solved for NCCL.
- What makes NVSHMEM structurally different from every collective this book has built since Chapter 8: a CUDA kernel can call `nvshmem_put()`/`nvshmem_get()` directly, from device code, while it is still running -- no host round-trip, no separate collective call enqueued from the CPU.
- How to genuinely install and link against real NVSHMEM in an environment with no GPU, and what actually happens, in real detail, when a real GPU-initiated call is attempted with no device to run it on.

**What you need to know first:**

- Chapter 20's real `MPI_Init`/`MPI_Comm_rank`/`MPI_Comm_size` and `MPI_Comm` handle.
- Chapter 21's real hybrid MPI+NCCL bootstrap (`ncclGetUniqueId()` + `MPI_Bcast()` + `ncclCommInitRank()`) -- NVSHMEM's own bootstrap options mirror this directly.
- Chapter 11's NCCL communicator basics, for contrast: NCCL's calls are host-initiated: the CPU enqueues them.

---

Every collective this book has built since Chapter 8 shares one property NVSHMEM breaks: a CPU thread issues the call. `ncclAllReduce()`, `MPI_Send()`, Chapter 21's own hybrid bootstrap -- all of them are the host telling the GPU (or another process) what to do next; the GPU itself never decides to communicate. A recent system-level analysis of NVSHMEM states the older model plainly: "NCCL was host-driven: the CPU enqueues collective operations, and the library selects algorithms and schedules." NVSHMEM's own model is different in kind, not just in API surface: "communication and synchronization are launched directly from GPU code rather than through a host-managed control path." This chapter builds that model for real, in three steps: 22.1 gets a genuinely installed NVSHMEM running at all, in the simplest single-process shape it falls back to with no bootstrap; 22.2 reuses this book's own real MPI infrastructure to give NVSHMEM a real multi-process world, exactly as Chapter 21 reused MPI for NCCL; 22.3 writes the one thing no earlier chapter's API could offer -- a CUDA kernel that puts data into a remote PE's memory itself, from inside a running thread -- and reports, in full, what a real GPU-initiated call does when there is genuinely no GPU underneath it.

```text
Host-initiated (every chapter since Ch8):
+--------+     +--------------------+     +------------------+
|  CPU   | --> | ncclAllReduce(...) | --> |  GPU executes    |
| thread |     | MPI_Send(...)      |     |  what the CPU    |
|        |     | Ch21 hybrid boot   |     |  already sent    |
+--------+     +--------------------+     +------------------+
  The CPU decides every time -- the GPU never decides on its own.

Device-initiated (this chapter):
+--------+     +----------------------------+     +----------------------+
|  CPU   | --> | nvshmemx_collective_       | --> | GPU kernel, still    |
| thread |     | launch(kernel)             |     | running, issues      |
+--------+     +----------------------------+     | nvshmem_put(...)     |
                                                   | itself -- no host   |
                                                   | round-trip at all   |
                                                   +----------------------+
```

## 22.1 NVSHMEM's PGAS Model and the Symmetric Heap

### Intuition

Every device memory allocation this book has used since Chapter 4 -- `cudaMalloc()`, peer-accessible or not -- belongs to exactly one device, and any other device that wants to touch it needs a real mechanism (`cudaMemcpyPeer()`, IPC handles, GPUDirect RDMA) to reach across. NVSHMEM's own memory model is built differently from the start. OpenSHMEM, the real community standard NVSHMEM implements for GPU clusters, "provides a partitioned global address space (PGAS) parallel programming model" -- every participating process (a PE, OpenSHMEM's own term, short for Processing Element) allocates from a real, documented structure called the Symmetric Heap: "NVSHMEM dynamic memory allocation routines (e.g., nvshmem_malloc) allow collective allocation of Symmetric Data Objects on a special memory region called the Symmetric Heap." Every PE calls `nvshmem_malloc()` together, with the same size, and every PE ends up with an object at the corresponding location in its own copy of that heap -- "symmetric," because the layout is guaranteed to line up across every PE, which is exactly what lets any PE compute where a REMOTE PE's copy of that same object lives, without ever asking that PE first. Before any of that can matter, though, NVSHMEM has to answer the same real question Chapter 20 opened this book's whole multi-process arc with: how many PEs are there, and which one is this process? Without a real answer, there is no symmetric heap to speak of at all -- just one process, alone.

```text
Every cudaMalloc() since Ch4:               NVSHMEM's own nvshmem_malloc():

+-----------------------+                   +-------------------------------+
| Device 0's own memory |                   | PE 0's symmetric heap slot X  |
| reachable elsewhere   |                   | PE 1's symmetric heap slot X  |  (same X)
| only via              |                   | PE 2's symmetric heap slot X  |  (same X)
| cudaMemcpyPeer/IPC/    |                   | ...                           |
| GPUDirect RDMA, built  |                   +-------------------------------+
| explicitly call by    |                   Every PE allocated together, same
| call                  |                   size -- any PE can compute a remote
+-----------------------+                   PE's address at slot X without ever
                                             asking that PE first.
```

### Background

The code below calls `nvshmem_init()` with no bootstrap plugin selected at all -- NVSHMEM's own default path when nothing else (a launcher like Hydra or Slurm, or this chapter's own later MPI bootstrap) has told it how to find other processes.

```cpp
// Chapter 22: NVSHMEM and GPU-Initiated Communication
// 63_nvshmem_init_single_pe.cu
//
// Every collective this book has built since Chapter 8 was HOST-
// initiated: a CPU thread calls ncclAllReduce()/MPI_Send()/etc., and
// that call enqueues work for the GPU to do later. NVSHMEM is built on
// a different model entirely -- OpenSHMEM's own real definition: "a
// community standard, one-sided communication API that provides a
// partitioned global address space (PGAS) parallel programming
// model." Every NVSHMEM process is a PE (Processing Element, PGAS's
// own term for a rank). nvshmem_malloc() allocates from a real,
// documented structure: "a special memory region called the Symmetric
// Heap," created so that "Symmetric Data Objects" (same type, size,
// and layout) exist at every PE at once. This section builds the
// plainest possible real NVSHMEM program: nvshmem_init() with no
// bootstrap plugin requested at all, genuinely installed via the pip
// package nvidia-nvshmem-cu12 (which -- unlike the incomplete pip nvcc
// package Chapter 1's own toolchain note already warned about --
// genuinely ships real headers, a real host library, and real device
// bitcode).
// Genuinely compiled with a real nvcc against the real installed
// NVSHMEM 3.7.2.
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
    // No bootstrap flags at all -- NVSHMEM's own default init path.
    // With no launcher (Hydra, Slurm, or this chapter's own later MPI
    // bootstrap) telling it otherwise, NVSHMEM genuinely falls back to
    // treating this one process as the entire, one-PE world.
    nvshmem_init();

    int myPe = nvshmem_my_pe();
    int nPes = nvshmem_n_pes();
    printf("nvshmem_init() returned; my_pe=%d n_pes=%d\n", myPe, nPes);
    printf("No bootstrap plugin was requested, so NVSHMEM had no way to "
           "discover any other real process -- this is NOT the same "
           "honest failure this book has shown since Chapter 3 "
           "(cudaErrorNoDevice, ncclUnhandledCudaError, etc.). Those "
           "calls always at least TRIED to find a device and reported "
           "exactly why they couldn't. nvshmem_init() with no bootstrap "
           "never tried to find a peer at all -- n_pes=1 is a correct, "
           "honest answer to the question it was actually asked.\n");

    // nvshmem_malloc() needs a real symmetric heap, which needs a real
    // CUDA context on a real device -- this environment has neither.
    void *sym = nvshmem_malloc(sizeof(int));
    printf("nvshmem_malloc(sizeof(int)) = %p\n", sym);

    nvshmem_finalize();
    return 0;
}
```

Compiled with `nvcc -rdc=true -ccbin g++ -gencode=arch=compute_70,code=sm_70 -I $NVSHMEM_HOME/include 63_nvshmem_init_single_pe.cu -o 63_nvshmem_init_single_pe -L $LIBDIR -lnvshmem_host -lnvshmem_device -lcuda` (real dynamic linking against the real installed NVSHMEM, following NVIDIA's own install guide's exact command shape) and genuinely run with `LD_LIBRARY_PATH` pointed at the real installed library directory. Locked output:

```text
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detectednvshmem_init() returned; my_pe=0 n_pes=1
No bootstrap plugin was requested, so NVSHMEM had no way to discover any other real process -- this is NOT the same honest failure this book has shown since Chapter 3 (cudaErrorNoDevice, ncclUnhandledCudaError, etc.). Those calls always at least TRIED to find a device and reported exactly why they couldn't. nvshmem_init() with no bootstrap never tried to find a peer at all -- n_pes=1 is a correct, honest answer to the question it was actually asked.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detectednvshmem_malloc(sizeof(int)) = (nil)
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_finalize:187: Unable to properly unregister device state.
```

!!! warning "[COMMON TRAP] Treating n_pes=1 as an error, or as the same kind of honest failure every earlier chapter has shown"
    Every device-touching call since Chapter 3 -- `cudaGetDeviceCount()`, `ncclCommInitAll()`, `MPIX_Query_cuda_support()` -- genuinely tried to find something real (a device, a peer, a capability) and reported, honestly, that it couldn't. `nvshmem_init()` called with no bootstrap plugin is a different case entirely: it was never ASKED to discover other processes, so `n_pes=1` is not a degraded or failed answer -- it is the exactly correct answer to a single-process program that requested nothing more. Reading `n_pes=1` as "NVSHMEM is broken here" would miss the actual lesson of this section, which Section 22.2 answers directly: NVSHMEM's own real multi-PE behavior needs a bootstrap plugin explicitly requested, the same way Chapter 21's own NCCL communicator needed an explicit `ncclUniqueId` distributed by MPI rather than assuming one process could see every rank on its own.

## 22.2 Real Multi-PE NVSHMEM via MPI Bootstrap

### Intuition

Section 22.1's single-PE fallback is what NVSHMEM does when nothing tells it how to find other processes. Chapter 20 already solved that exact discovery problem once, for MPI itself; Chapter 21 reused MPI a second time, to bootstrap NCCL's own `ncclUniqueId`. NVSHMEM ships the same real answer as a first-class option: `NVSHMEMX_INIT_WITH_MPI_COMM`, set via `nvshmemx_set_attr_mpi_comm_args()`, hands NVSHMEM a real `MPI_Comm` this book has already built genuine multi-process programs on top of. (NVSHMEM also ships a second, structurally different bootstrap path, `NVSHMEMX_INIT_WITH_UNIQUEID` -- a uniqueid generated once and distributed to every PE, exactly like Chapter 21's own `ncclUniqueId` + `MPI_Bcast()` pattern; this section uses the MPI path because MPI is already present and simpler to reach for.) Once NVSHMEM has a real `MPI_Comm` to query, `nvshmem_my_pe()` and `nvshmem_n_pes()` stop answering "1" and start answering with the real number of processes MPI itself discovered -- the same real multi-process world every earlier Part 5 chapter has built, now handed to a third real library.

```text
Section 22.1 (no bootstrap):
+----------------+     +------------------+
| nvshmem_init() | --> | n_pes=1, my_pe=0 |  (correct -- but the whole real
+----------------+     +------------------+   world really is just 1 process)

Section 22.2 (MPI bootstrap):
+--------------------------------+
| mpirun -np 2 ./program         |   (Ch20's own launch)
+----------------+---------------+
                 |
       +---------+---------+
       |                   |
   rank 0 (MPI)         rank 1 (MPI)
       |                   |
   nvshmemx_set_attr_mpi_comm_args(&MPI_COMM_WORLD)
       |                   |
   nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM)
       |                   |
   my_pe=0, n_pes=2     my_pe=1, n_pes=2
```

### Background

```cpp
// Chapter 22: NVSHMEM and GPU-Initiated Communication
// 64_nvshmem_mpi_bootstrap.cu
//
// File 63's own single-PE fallback is what happens when NVSHMEM has no
// way to discover other processes. Chapter 20/21 already solved
// exactly this discovery problem once, for MPI and for NCCL's own
// bootstrap -- NVSHMEM ships the SAME real answer as a first-class
// option: NVSHMEMX_INIT_WITH_MPI_COMM, set via
// nvshmemx_set_attr_mpi_comm_args(), reuses a real MPI_Comm this book
// has already built real multi-process programs on top of since
// Chapter 20. (NVSHMEM also ships NVSHMEMX_INIT_WITH_UNIQUEID, a
// second bootstrap path structurally identical to Chapter 21's own
// ncclUniqueId + MPI_Bcast pattern -- not used here, since the MPI
// path is simpler when MPI is already present.)
// Genuinely compiled with a real mpicxx (linked against the real
// installed NVSHMEM 3.7.2, real Open MPI, and real CUDA runtime) and
// genuinely run with a real mpirun.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm mpiComm = MPI_COMM_WORLD;

    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpiComm;
    int rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int myPe = nvshmem_my_pe();
    int nPes = nvshmem_n_pes();
    printf("PE %d of %d: nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM) rc=%d\n",
           myPe, nPes, rc);

    if (myPe == 0) {
        printf("\nCompare this n_pes=%d to file 63's own n_pes=1: the ONLY "
               "difference between these two files is which bootstrap path "
               "was requested. Real MPI (Chapter 20's own MPI_Comm_rank()/"
               "MPI_Comm_size() underneath this call) genuinely discovered "
               "%d real processes and handed that world to NVSHMEM -- the "
               "exact same discovery problem Chapter 21's own hybrid "
               "bootstrap solved for NCCL, now solved for NVSHMEM instead.\n",
               nPes, nPes);
    }

    void *sym = nvshmem_malloc(sizeof(int));
    printf("PE %d: nvshmem_malloc(sizeof(int)) = %p (honestly NULL -- no "
           "real device to host a symmetric heap on)\n", myPe, sym);

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
```

Compiled with `nvcc -rdc=true -ccbin mpicxx -gencode=arch=compute_70,code=sm_70 -I $NVSHMEM_HOME/include 64_nvshmem_mpi_bootstrap.cu -o 64_nvshmem_mpi_bootstrap -L $LIBDIR -lnvshmem_host -lnvshmem_device -lcuda -lmpi` (the same real `mpicxx`-as-host-compiler trick Chapter 20/21 already established, now driving `nvcc` itself rather than the other way around) and genuinely run with `mpirun --allow-run-as-root --oversubscribe -np 2 ./64_nvshmem_mpi_bootstrap`. Locked output:

```text
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detectedPE 0 of 2: nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM) rc=0

Compare this n_pes=2 to file 63's own n_pes=1: the ONLY difference between these two files is which bootstrap path was requested. Real MPI (Chapter 20's own MPI_Comm_rank()/MPI_Comm_size() underneath this call) genuinely discovered 2 real processes and handed that world to NVSHMEM -- the exact same discovery problem Chapter 21's own hybrid bootstrap solved for NCCL, now solved for NVSHMEM instead.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detectedPE 1 of 2: nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM) rc=0
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detectedPE 0: nvshmem_malloc(sizeof(int)) = (nil) (honestly NULL -- no real device to host a symmetric heap on)
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:580 Cuda failure. Status = CUDA_ERROR_NO_DEVICE. Description = no CUDA-capable device is detectedPE 1: nvshmem_malloc(sizeof(int)) = (nil) (honestly NULL -- no real device to host a symmetric heap on)
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_finalize:187: Unable to properly unregister device state.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_finalize:187: Unable to properly unregister device state.
```

!!! warning "[COMMON TRAP] Assuming nvshmemx_init_attr()'s rc=0 means the symmetric heap is usable"
    `rc=0` from `nvshmemx_init_attr()` reports only that NVSHMEM's OWN bootstrap step succeeded -- MPI genuinely discovered 2 processes and NVSHMEM genuinely accepted that world. It says nothing about whether the underlying CUDA context that the symmetric heap depends on is usable, and this section's own locked output proves the gap directly: `rc=0` on both PEs, immediately followed by `nvshmem_malloc()` honestly returning `NULL` on both. A real program that checks only `nvshmemx_init_attr()`'s return value and assumes `nvshmem_malloc()` will therefore succeed is checking the wrong layer -- bootstrap success and symmetric-heap availability are two separate real facts, exactly the same kind of two-layer gap Chapter 21 named between `cudaDevAttrGPUDirectRDMASupported` (device capability) and PCIe topology (whether that capability is actually fast).

## 22.3 GPU-Initiated Communication: A Kernel That Puts to a Remote PE, By Itself

### Intuition

Sections 22.1 and 22.2 only ever called NVSHMEM from the host -- `nvshmem_init()`, `nvshmem_malloc()`, `nvshmem_finalize()`. Every one of those calls could, in principle, be a slightly stranger `cudaMalloc()`. This section builds the one capability no earlier chapter's API could offer at all, host or device: a CUDA kernel that issues a PUT to another PE's memory itself, from device code, while that thread is still executing. NVSHMEM's own documentation is direct about this: "Device-side APIs can be called by CUDA kernel threads to efficiently access locations in symmetric memory through one-sided read (get), write (put), and atomic update API calls." A plain `<<<>>>` launch is not enough for a kernel that will call NVSHMEM's own synchronizing device APIs -- `nvshmemx_collective_launch()` exists specifically to coordinate with NVSHMEM's internal state before handing control to the GPU, the real reason this section's own kernel is launched through it rather than through the syntax every earlier chapter has used since Chapter 3.

```text
Every kernel launch since Ch3:                 This section's own launch:

+----------------------------+                 +----------------------------------+
| plain CUDA launch syntax   |                 | nvshmemx_collective_launch(       |
| (triple angle-bracket      |                 |   (const void*)putKernel, grid,   |
| launch), no coordination   |                 |   block, args, sharedMem, stream) |
| with any communication     |                 +----------------------------------+
| library at all             |                 -- NVSHMEM's own launch path, required
+----------------------------+                    because the kernel body calls a real
                                                   device-side NVSHMEM API (nvshmem_int_p)
```

### Background

```cpp
// Chapter 22: NVSHMEM and GPU-Initiated Communication
// 65_gpu_initiated_put.cu
//
// Files 63-64 only ever called NVSHMEM from the HOST -- init, malloc,
// finalize. This section builds the one thing no earlier chapter's
// API could offer at all: a CUDA kernel that issues a PUT to another
// PE's memory ITSELF, from device code, while that thread is still
// running. NVSHMEM's own documentation states this plainly: "Device-
// side APIs can be called by CUDA kernel threads to efficiently
// access locations in symmetric memory through one-sided read (get),
// write (put), and atomic update API calls." A plain <<<>>> launch
// cannot be used for a kernel that calls NVSHMEM's own synchronizing
// operations -- nvshmemx_collective_launch() exists specifically to
// launch a kernel that will use them, coordinating with NVSHMEM's own
// internal state before handing control to the GPU.
// Genuinely compiled with a real mpicxx (linked against the real
// installed NVSHMEM 3.7.2, real Open MPI, and real CUDA runtime) and
// genuinely run with a real mpirun.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void putKernel(int *dest, int val, int targetPe) {
    // GPU-initiated communication: issued by a thread that is still
    // running, no host round-trip at all -- the one call this entire
    // book has never been able to build before this chapter.
    nvshmem_int_p(dest, val, targetPe);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm mpiComm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpiComm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int myPe = nvshmem_my_pe();
    int nPes = nvshmem_n_pes();
    int peer = (myPe + 1) % nPes;

    int *dest = (int*)nvshmem_malloc(sizeof(int));
    printf("PE %d of %d: nvshmem_malloc=%p, launching putKernel targeting peer PE %d\n",
           myPe, nPes, (void*)dest, peer);

    void *args[] = {&dest, (void*)&myPe, (void*)&peer};
    dim3 grid(1), block(1);
    int rc = nvshmemx_collective_launch((const void*)putKernel, grid, block, args, 0, 0);
    printf("PE %d: nvshmemx_collective_launch() rc=%d (%s)\n",
           myPe, rc, nvshmemx_status_string(rc));

    cudaError_t syncErr = cudaDeviceSynchronize();
    printf("PE %d: cudaDeviceSynchronize()=%s\n", myPe, cudaGetErrorString(syncErr));

    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}
```

Compiled the same way as file 64, and genuinely run with `mpirun --allow-run-as-root --oversubscribe -np 2 ./65_gpu_initiated_put`. Locked output -- this is the deepest real GPU-touching call this entire book has attempted, and it genuinely fails harder than anything in any earlier chapter:

```text
PE 0 of 2: nvshmem_malloc=(nil), launching putKernel targeting peer PE 1
PE 1 of 2: nvshmem_malloc=(nil), launching putKernel targeting peer PE 0
nvshmem cuda device query failed, exiting
nvshmemi_check_state_and_init_d() failedPE 0: nvshmemx_collective_launch() rc=7 (NVSHMEMX_ERROR_INTERNAL)
PE 0: cudaDeviceSynchronize()=no CUDA-capable device is detected
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_finalize:187: Unable to properly unregister device state.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_get_device_state_ptrs:111: Unable to access device state. 100
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:1474: NULL value Unable to query pointer information.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:nvshmemid_hostlib_finalize:1566: aborting due to error in nvshmem_finalize
nvshmem cuda device query failed, exiting
nvshmemi_check_state_and_init_d() failedPE 1: nvshmemx_collective_launch() rc=7 (NVSHMEMX_ERROR_INTERNAL)
PE 1: cudaDeviceSynchronize()=no CUDA-capable device is detected
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_finalize:187: Unable to properly unregister device state.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/device/init/init_host.cu:nvshmemi_get_device_state_ptrs:111: Unable to access device state. 100
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:1474: NULL value Unable to query pointer information.
/dvs/p4/build/sw/rel/gpgpu/toolkit/r12.9/main_nvshmem/src/host/init/init.cu:nvshmemid_hostlib_finalize:1566: aborting due to error in nvshmem_finalize
```

`mpirun` itself then reports the job aborted, with exit code **255** -- unlike every earlier chapter's clean `exit 0` after an honest error, `nvshmemx_collective_launch()`'s real internal device-state check (`nvshmemi_check_state_and_init_d()`) fails outright (`rc=7`, `NVSHMEMX_ERROR_INTERNAL`), `cudaDeviceSynchronize()` separately and honestly reports no device, and even `nvshmem_finalize()` itself cannot cleanly tear down afterward -- it reports being unable to access device state and unable to query pointer information, then aborts.

!!! warning "[COMMON TRAP] Expecting a GPU-initiated call to fail as cleanly as a host-initiated one"
    Every host-initiated call this book has made since Chapter 3 -- `cudaMalloc()`, `ncclCommInitAll()`, `MPI_Send()` -- checks for a device (or a peer, or a valid communicator) BEFORE doing anything else, and returns one honest error code if it isn't there. `nvshmemx_collective_launch()` is asked to do something structurally harder: hand a kernel to a device, coordinate NVSHMEM's own internal per-device state for that launch, and only then let the kernel run. This section's own locked output shows that when the device itself doesn't exist, that coordination step fails first and hard (`NVSHMEMX_ERROR_INTERNAL`), the standard CUDA error only shows up SEPARATELY when `cudaDeviceSynchronize()` is called afterward, and `nvshmem_finalize()` itself -- normally just cleanup -- fails too, because it also needs real device state to tear down. A program that assumes a `nvshmemx_collective_launch()` failure means "no kernel ran, otherwise everything is fine" is wrong on the last point: the whole library's internal bookkeeping is now compromised, and this book's own real `mpirun` run above genuinely could not finalize cleanly as a result -- the honest lesson is that GPU-initiated communication concentrates failure into a much sharper, less recoverable moment than every earlier chapter's host-initiated calls ever did.

## Chapter Summary

This chapter left every earlier chapter's host-initiated communication model behind. Section 22.1 got a genuinely installed real NVSHMEM (via the pip package `nvidia-nvshmem-cu12`, which -- unlike the incomplete pip `nvcc` package Chapter 1 warned about -- ships complete real headers, host library, and device bitcode) running in its simplest form: no bootstrap plugin, a correct single-PE world. Section 22.2 reused this book's own real MPI infrastructure (Chapter 20) to give NVSHMEM a real multi-PE world via `NVSHMEMX_INIT_WITH_MPI_COMM`, the same real discovery problem Chapter 21 already solved for NCCL, now solved for a third library. Section 22.3 built the one genuinely new capability this book has never had before: a CUDA kernel calling `nvshmem_int_p()` directly from device code, launched through `nvshmemx_collective_launch()` -- and reported, honestly and in full, that this deepest real GPU-touching call fails harder and less cleanly than any earlier chapter's device-less error, right down to `nvshmem_finalize()` itself being unable to tear down afterward.

## Self-Check Questions

1. What does "partitioned global address space" mean, and how does NVSHMEM's symmetric heap differ from a plain `cudaMalloc()` allocation from Chapter 4 onward?
2. Section 22.1's `nvshmem_init()` call with no bootstrap plugin returns `n_pes=1`. Why is this not the same kind of honest failure as `cudaErrorNoDevice`?
3. Name the two real bootstrap paths NVSHMEM offers that this chapter discusses, and which earlier chapter's own bootstrap pattern each one mirrors.
4. In Section 22.2, `nvshmemx_init_attr()` returns `rc=0` on both PEs, yet `nvshmem_malloc()` still returns `NULL`. What does this prove about what `rc=0` actually guarantees?
5. Why does the kernel in file 65 need to be launched with `nvshmemx_collective_launch()` instead of the plain `<<<>>>` syntax every earlier chapter has used?
6. Contrast the failure Section 22.3 reports (`rc=7`, `NVSHMEMX_ERROR_INTERNAL`, exit 255) with every earlier chapter's own device-less failures (e.g. Chapter 11's `ncclUnhandledCudaError`, exit 0). What is genuinely different about it?
7. What does the quoted contrast between NCCL ("host-driven: the CPU enqueues collective operations") and NVSHMEM ("launched directly from GPU code") actually change about who decides when communication happens?
8. Why does `nvshmem_finalize()` itself fail in Section 22.3's own locked output, when it succeeded cleanly in Sections 22.1 and 22.2?

## Where We Go Next

Chapter 22 built the one thing every earlier chapter's API genuinely couldn't: communication issued by a still-running kernel, not by the host. Chapter 23 returns to something more familiar in shape -- CUDA graphs, first built single-device in earlier books this series assumes -- and asks what changes when a single graph has to span multiple GPUs and multiple nodes at once: which parts of a graph can legitimately cross a device boundary, and which of this book's own real communication primitives (peer copies, NCCL collectives, MPI calls) can actually be captured as graph nodes rather than issued one at a time.

## Worked Solutions

**1.** A partitioned global address space means every participating process (PE) contributes memory to one logical address space, even though the underlying memory is physically distributed across separate devices -- any PE can, in principle, name and address any other PE's contribution. NVSHMEM's own symmetric heap is a real implementation of this: every PE calls `nvshmem_malloc()` together with the same size, so every PE's copy of a given allocation sits at the corresponding offset in its own heap. A plain `cudaMalloc()` allocation, by contrast, belongs to exactly one device and is not automatically nameable from any other device at all -- reaching it from elsewhere always needed an explicit mechanism this book built one chapter at a time (`cudaMemcpyPeer()`, IPC handles, GPUDirect RDMA).

**2.** `cudaErrorNoDevice` is returned by a call that genuinely tried to find a device and honestly reported that none exists. `nvshmem_init()` with no bootstrap plugin was never asked to find any OTHER process at all -- it was asked to initialize NVSHMEM for whatever world already exists, and with no launcher or bootstrap telling it otherwise, that world is genuinely just the one calling process. `n_pes=1` is therefore a correct answer to the actual question asked, not a degraded or failed answer to a question about discovering peers.

**3.** `NVSHMEMX_INIT_WITH_MPI_COMM` (used in this chapter) hands NVSHMEM a real `MPI_Comm`, mirroring Chapter 20's own real `MPI_Init`/`MPI_Comm_rank`/`MPI_Comm_size` discovery. `NVSHMEMX_INIT_WITH_UNIQUEID` (mentioned but not used) generates one id and distributes it to every PE, structurally identical to Chapter 21's own `ncclGetUniqueId()` + `MPI_Bcast()` + `ncclCommInitRank()` pattern.

**4.** It proves that bootstrap success and symmetric-heap availability are two separate real facts, checked at two different layers. `rc=0` only reports that NVSHMEM's own process-discovery step (via MPI) succeeded -- both PEs agreed on a real 2-process world. It says nothing about whether the CUDA context underneath the symmetric heap is usable, and this section's own output shows that gap directly: bootstrap succeeds on both PEs, and the very next call, `nvshmem_malloc()`, honestly fails on both.

**5.** Because the kernel's body calls `nvshmem_int_p()`, a real device-side NVSHMEM API that needs NVSHMEM's own internal per-device state to be coordinated before the kernel starts running. A plain `<<<>>>` launch has no way to perform that coordination; `nvshmemx_collective_launch()` exists specifically to launch kernels that will call NVSHMEM's own synchronizing device APIs.

**6.** Every earlier chapter's device-less failure was a single honest error code, cleanly returned, with the program going on to exit 0. Section 22.3's failure is structurally different: the internal state check inside `nvshmemx_collective_launch()` itself fails first (`NVSHMEMX_ERROR_INTERNAL`), a SEPARATE standard CUDA error shows up only later when `cudaDeviceSynchronize()` is called, and even `nvshmem_finalize()` -- ordinary cleanup in every earlier section -- fails too, reporting it cannot access device state or query pointer information, and the whole real `mpirun` job aborts with a non-zero exit code (255) rather than the clean 0 every earlier chapter ended with.

**7.** It moves the DECISION of when to communicate from the host to the device. In every earlier chapter, a CPU thread decided the exact moment a collective or send call happened, and the GPU simply executed whatever the CPU had already enqueued. With NVSHMEM's device-side APIs, a GPU thread that is still running can decide, on its own, to issue a put or get -- the host's only role was launching the kernel that contains that decision, not making the decision itself.

**8.** Because `nvshmem_finalize()`'s own cleanup work depends on the SAME per-device internal state that `nvshmemx_collective_launch()`'s failed check was trying to verify in the first place. In Sections 22.1 and 22.2, that internal state was never exercised by anything beyond `nvshmem_malloc()`'s own honest `NULL` return, so finalize's cleanup path, while still logging its own device-less warning, completed. In Section 22.3, the collective-launch attempt left NVSHMEM's internal bookkeeping in a state finalize could not cleanly unwind, and it failed too.

---

**Sources cited in this chapter:**

- [Using NVSHMEM (NVSHMEM 1.0 archive)](https://docs.nvidia.com/nvshmem/archives/nvshmem-101/api/docs/gen/overview.html) — the real, verified definitions this chapter quotes: OpenSHMEM's own PGAS description ("a community standard, one-sided communication API that provides a partitioned global address space (PGAS) parallel programming model"), the Symmetric Heap definition, and the device-side API description ("Device-side APIs can be called by CUDA kernel threads to efficiently access locations in symmetric memory through one-sided read (get), write (put), and atomic update API calls").
- [Demystifying NVSHMEM: A System-Level Analysis on Symmetric Memory and Device-Initiated Operations in GPU Communication](https://arxiv.org/html/2606.05951v1) — the real, verified contrast this chapter's intro quotes between NCCL's host-driven model ("NCCL was host-driven: the CPU enqueues collective operations, and the library selects algorithms and schedules") and NVSHMEM's device-initiated model ("communication and synchronization are launched directly from GPU code rather than through a host-managed control path"), and the real symmetric-memory definition used in Section 22.1's own Background.
- `nvidia-nvshmem-cu12` (PyPI package, version 3.7.2) — genuinely installed and used for every code file in this chapter; this exact environment's own installed headers (`nvshmem.h`, `nvshmemx.h`, `driver_types.h`-adjacent NVSHMEM headers under `include/host/`, `include/device/`, and `include/non_abi/`) were grepped directly for real API names, flag values (`NVSHMEMX_INIT_WITH_MPI_COMM`), and the real `nvshmemx_status` error enum (`NVSHMEMX_ERROR_INTERNAL = 7`), per Chapter 21's own established practice of checking installed headers directly rather than trusting a web summary.
- [NVSHMEM Install Guide: Compiling and Linking with nvcc](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html) — the real compile-command shape this chapter's own `nvcc -rdc=true -ccbin ... -gencode=... -lnvshmem_host -lnvshmem_device` commands follow.
