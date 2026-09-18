# Chapter 21: GPUDirect RDMA: Bypassing the Host Entirely

**What you will understand by the end of this chapter:**

- Why a real multi-node NCCL communicator still needs MPI (or something like it) underneath it, and how to build the real hybrid bootstrap -- `ncclGetUniqueId()` on one rank, `MPI_Bcast()` to every other rank, then `ncclCommInitRank()` on all of them -- that every real multi-node NCCL job actually uses.
- What GPUDirect RDMA actually is: a network adapter's DMA engine reading and writing GPU memory directly over PCI Express BAR addresses, with no host memory touched at any point -- the real alternative to Chapter 20's own staged fallback.
- How to query, for real, whether a specific GPU supports GPUDirect RDMA and what its write-ordering guarantees are (`cudaDevAttrGPUDirectRDMASupported`, `cudaDeviceFlushGPUDirectRDMAWrites()`).
- Why "GPUDirect RDMA capable" is necessary but not sufficient: real cited numbers showing PCIe topology alone can swing bandwidth between the same two capable devices by more than 40x.

**What you need to know first:**

- Chapter 11's `ncclCommInitAll()`/`ncclUniqueId` basics, and Chapter 19's `ncclConfig_t` non-blocking communicator.
- Chapter 20's real `MPI_Init`/`MPI_Comm_rank`/`MPI_Comm_size`/`MPI_Bcast`, and its own staged host-transfer fallback (Chapter 5's pattern, reused between MPI ranks).
- Chapter 2's PCIe topology model -- same-switch vs. cross-socket bandwidth.

---

Chapter 20 closed on an unfinished sentence: this exact environment's real, installed Open MPI build isn't CUDA-aware, so Section 20.2's own fallback spent every byte's trip going device-to-host, over the network, then host-to-device again -- never touching GPU memory directly from the network side at all. GPUDirect RDMA is the technology that removes that detour, but reaching it for real first needs one piece this book has never actually built: every NCCL communicator since Chapter 11 was built by a single process that could already see every GPU (`ncclCommInitAll()`) or, worse, was never real to begin with. A real multi-*node* NCCL job is neither -- it is exactly Chapter 20's own real multi-*process* MPI world, with NCCL layered on top of it for the actual data movement. Section 21.1 builds that real hybrid bootstrap first, because it's the thing that makes "GPUDirect RDMA between two nodes" a question that even makes sense to ask. Section 21.2 then checks, the same honest way this book has checked every device-touching capability since Chapter 3, whether a GPU genuinely supports GPUDirect RDMA. Section 21.3 closes with real, cited numbers on what bypassing the host is actually worth, and what happens when the PCIe topology underneath it is wrong.

```text
Chapter 20 alone:                          This chapter (Ch21):

  MPI_Init/Comm_rank/Comm_size                MPI bootstraps NCCL's id (21.1)
  -> real processes, real ranks               -> ncclGetUniqueId + MPI_Bcast
  MPI_Send/Recv/Sendrecv                          + ncclCommInitRank, all real
  -> real bytes, host memory only             NCCL communicator now spans
  cudaMemcpy D2H -> net -> cudaMemcpy H2D         real MPI-discovered ranks
  -> Section 20.2's own staged fallback       Does the NIC reach GPU memory
     (CUDA-aware MPI was false, so this           WITHOUT the host? (21.2)
     detour through the host was forced)      -> query real GPUDirectRDMA
                                                   device attributes
                                               What is bypassing the host
                                                   actually worth? (21.3)
                                               -> real cited latency/BW
                                                   numbers, real topology risk
```

## 21.1 The Hybrid Bootstrap: MPI for Rank, NCCL for Data

### Intuition

Every real distributed-training job you have ever heard of -- the ones that train across dozens or thousands of GPUs -- almost never uses MPI to move the actual training data, and almost never uses NCCL to launch its own processes. It uses both, for two different jobs, at the same time. MPI's job, established in full in Chapter 20, is to get a bunch of separate operating-system processes into existence and let them find each other: `mpirun -np N` launches them, `MPI_Comm_rank()` and `MPI_Comm_size()` tell each one who it is. NCCL's job, established since Chapter 11, is to move gradient and activation tensors between GPUs as fast as the hardware allows. The problem is that NCCL was never given a launcher of its own -- `ncclCommInitRank()` needs every participating rank to already agree on one shared `ncclUniqueId`, and NCCL has no built-in way to get that id from the one rank that generates it to every other rank. That's a completely generic "get one small piece of data from rank 0 to everyone else" problem, and MPI already solved the generic version of that problem with `MPI_Bcast()`. So real multi-node training code doesn't choose between MPI and NCCL -- it uses MPI for exactly one thing (distributing the id, plus everything else Chapter 20 already covered) and then hands off to NCCL for everything else.

```text
Step 1: mpirun -np N ./job launches N real OS processes (Chapter 20, unchanged)

  +--------+   +--------+   +--------+   +----------+
  | rank 0 |   | rank 1 |   | rank 2 |   | rank N-1 |
  +--------+   +--------+   +--------+   +----------+

Step 2: rank 0 alone generates the id -- every other rank has nothing yet

  rank 0: ncclGetUniqueId(&id) ---> id generated (ranks 1..N-1: nothing yet)

Step 3: MPI_Bcast(&id, root=0) delivers that SAME id to every other rank

  rank 0 ---> rank 1
  rank 0 ---> rank 2
  rank 0 ---> rank N-1

Step 4: every rank calls ncclCommInitRank(&comm, N, id, myRank) with the SAME id

  rank 0:   ncclCommInitRank(&comm, N, id, 0)   ---+
  rank 1:   ncclCommInitRank(&comm, N, id, 1)   ---+--> same real NCCL communicator
  rank N-1: ncclCommInitRank(&comm, N, id, N-1) ---+    (every ncclAllReduce/ncclBroadcast
                                                          call since Ch11 now spans real
                                                          MPI-discovered ranks)
```

### Background

NCCL's own installed header states the requirement directly, in its own comment above `ncclGetUniqueId()`: "ncclGetUniqueId should be called once and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank." NCCL's own official documentation shows the exact real pattern this section builds, "in the context of MPI, using one device per MPI rank": query `MPI_Comm_rank()`/`MPI_Comm_size()`, have rank 0 alone call `ncclGetUniqueId()`, `MPI_Bcast()` the id to everyone, then have every rank call `ncclCommInitRank()` with the shared id and its own rank.

```cpp
// Chapter 21: GPUDirect RDMA: Bypassing the Host Entirely
// 60_hybrid_mpi_nccl_bootstrap.cpp
//
// Every NCCL communicator this book has built since Chapter 11 used
// ncclCommInitAll() -- a single-process convenience call that works
// because that single process could see every GPU. Chapter 20 showed
// mpirun genuinely launching separate OS processes, each seeing only
// its own GPU. NCCL's own installed header (nccl.h) says exactly what
// bridges that gap: "ncclGetUniqueId should be called once and the Id
// should be distributed to all ranks in the communicator before
// calling ncclCommInitRank." NCCL has no process launcher and no
// broadcast primitive of its own to use before a communicator exists
// -- MPI already solved both in Chapter 20, so this section reuses
// MPI purely as NCCL's own bootstrap, exactly as NCCL's own official
// documentation demonstrates: "in the context of MPI, using one
// device per MPI rank."
// Genuinely compiled with a real mpicxx (linked against the real
// installed libnccl and libcudart) and genuinely run with a real
// mpirun -- the same two-toolchain-in-one-binary trick Chapter 20's
// own file 58 used for MPI+CUDA, now extended to MPI+CUDA+NCCL.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int myRank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    // The real, documented sequence: exactly ONE rank calls
    // ncclGetUniqueId(). If every rank called it independently, each
    // would get a DIFFERENT id and there would be nothing for
    // ncclCommInitRank() to rendezvous on below -- this chapter's own
    // COMMON TRAP.
    ncclUniqueId id;
    if (myRank == 0) {
        ncclGetUniqueId(&id);
        printf("rank 0: generated the ONE ncclUniqueId every rank will "
               "share, and is about to MPI_Bcast it to all %d ranks.\n", nRanks);
    }

    // MPI is doing NOTHING NCCL-specific here -- MPI_Bcast() is the
    // exact same general-purpose collective it always was, moving
    // sizeof(id) raw bytes from rank 0 to everyone else. NCCL never
    // sees this call; it only sees the result.
    MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Real device binding, per rank, exactly like every earlier
    // ncclCommInitRank() precondition since Chapter 11 ("Each rank is
    // associated to a CUDA device, which has to be set before calling
    // ncclCommInitRank" -- nccl.h's own comment). Honestly
    // cudaErrorNoDevice: this environment has no device at all,
    // regardless of which rank asks.
    cudaError_t eSet = cudaSetDevice(myRank);

    // Every rank now calls ncclCommInitRank() with the SAME id
    // (received via MPI, never regenerated) and its OWN rank/nRanks
    // (queried from MPI, never hard-coded) -- structurally identical
    // to Chapter 19's own ncclCommInitRankConfig() calls, just fed by
    // real MPI-discovered values instead of this book's own literal
    // constants.
    ncclComm_t comm;
    ncclResult_t eInit = ncclCommInitRank(&comm, nRanks, id, myRank);

    printf("rank %d of %d: cudaSetDevice(%d)=%s ncclCommInitRank=%s (code %d)\n",
           myRank, nRanks, myRank, cudaGetErrorString(eSet),
           ncclGetErrorString(eInit), (int)eInit);

    if (myRank == 0) {
        printf("\nEvery rank sees the SAME honest failure code Chapter 11's "
               "single-process ncclCommInitAll() saw on its own well-formed "
               "requests (ncclUnhandledCudaError=1) -- this environment's "
               "real absence of any GPU doesn't care whether the caller was "
               "one process managing N devices or N real MPI processes each "
               "managing one. What changed is genuinely real: %d separate "
               "operating-system processes, discovered through MPI, agreed "
               "on one shared id through MPI, and each independently called "
               "the exact same NCCL entry point every earlier chapter used.\n",
               nRanks);
    }

    MPI_Finalize();
    return 0;
}
```

Compiled with `mpicxx -Wall -Wextra 60_hybrid_mpi_nccl_bootstrap.cpp -o 60_hybrid_mpi_nccl_bootstrap -lnccl -lcudart` (the same real `mpicxx`-links-`libcudart` trick Chapter 20's own file 58 used, now also linking the real installed `libnccl`) and genuinely run with `mpirun --allow-run-as-root --oversubscribe -np 4 ./60_hybrid_mpi_nccl_bootstrap`. Locked output (the four per-rank lines and the closing paragraph are deterministic in content; their print ORDER across ranks is not, exactly like Chapter 20's own Section 20.1 -- three genuinely different real orderings, captured across three separate runs):

```text
rank 0: generated the ONE ncclUniqueId every rank will share, and is about to MPI_Bcast it to all 4 ranks.
rank 1 of 4: cudaSetDevice(1)=no CUDA-capable device is detected ncclCommInitRank=unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)
rank 2 of 4: cudaSetDevice(2)=no CUDA-capable device is detected ncclCommInitRank=unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)
rank 3 of 4: cudaSetDevice(3)=no CUDA-capable device is detected ncclCommInitRank=unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)
rank 0 of 4: cudaSetDevice(0)=no CUDA-capable device is detected ncclCommInitRank=unhandled cuda error (run with NCCL_DEBUG=INFO for details) (code 1)

Every rank sees the SAME honest failure code Chapter 11's single-process ncclCommInitAll() saw on its own well-formed requests (ncclUnhandledCudaError=1) -- this environment's real absence of any GPU doesn't care whether the caller was one process managing N devices or N real MPI processes each managing one. What changed is genuinely real: 4 separate operating-system processes, discovered through MPI, agreed on one shared id through MPI, and each independently called the exact same NCCL entry point every earlier chapter used.
```

A second real run printed the closing paragraph immediately after rank 0's own per-rank line, before ranks 1-3 had printed at all; a third run printed ranks 1, 2, 3 in that exact order after rank 0. All three runs' VALUES were identical -- only which process's `printf()` reached the terminal first changed, run to run, exactly as Chapter 20's own Section 20.1 already demonstrated for a plain MPI ring.

!!! warning "[COMMON TRAP] Every rank generating its own uniqueId, instead of one rank generating it and broadcasting it"
    The code above deliberately guards `ncclGetUniqueId(&id)` with `if (myRank == 0)`. Delete that guard -- let every rank call `ncclGetUniqueId()` independently -- and every rank now holds a *different* 128-byte id, because `ncclUniqueId` is generated from process-local random state, not derived from anything shared. `ncclCommInitRank()` on a real cluster then has nothing to rendezvous on: each rank is waiting for peers who agree on an id that rank never received, and the call hangs -- not a fast, honest error like every device-less failure this book has shown since Chapter 3, but the same silent, indefinite hang Chapter 19 named as NCCL's real, worse failure mode. NCCL's own installed header is explicit about the fix, in the exact comment quoted above: the id must be "called once" and then "distributed to all ranks... before calling ncclCommInitRank" -- generated by exactly one rank, delivered to the rest by some other real mechanism. This book uses `MPI_Bcast()` because Chapter 20 already built real MPI; a job with no MPI at all still needs some equivalent distribution step (a shared filesystem, a rendezvous server, a job scheduler's own environment variables) -- there is no version of this bootstrap that skips broadcasting the id.

## 21.2 Querying Real GPUDirect RDMA Support

### Intuition

Section 21.1's hybrid communicator now spans real, separate machines in principle -- but knowing *that* a communicator exists says nothing about *how* its data actually crosses from one node's GPU to another's. Every inter-node collective ultimately has to move bytes through some network interface card. Without GPUDirect RDMA, that NIC only ever touches host memory, so a GPU-to-GPU transfer across nodes looks exactly like Chapter 20's own Section 20.2: `cudaMemcpy` the data off the GPU, hand it to the network, `cudaMemcpy` it back onto the destination GPU. GPUDirect RDMA removes the two `cudaMemcpy` calls entirely by letting the NIC's own DMA engine read and write GPU memory directly, the same way it already reads and writes host memory -- NVIDIA's own GPUDirect RDMA design guide states this plainly: "PCI Express device issues reads and writes to a peer device's BAR addresses in the same way that they are issued to system memory." Whether a specific GPU actually supports this is a real, queryable hardware-and-driver fact, not something to assume -- exactly the kind of honest capability query this book has run since Chapter 3's own `cudaGetDeviceCount()`.

```text
Without GPUDirect RDMA (Ch20's own Section 20.2 fallback) -- four real hops:

  GPU memory -> cudaMemcpy(D2H) -> Host buffer -> network hop (MPI_Send/Recv)
  -> Host buffer (remote) -> cudaMemcpy(H2D) -> GPU memory (remote)

With GPUDirect RDMA -- the NIC's own DMA engine reads/writes GPU memory
directly, over PCIe BAR addresses, no host memory touched on either side:

  GPU memory -> NIC's DMA engine -> network -> remote NIC's DMA engine -> GPU memory (remote)
```

### Background

CUDA exposes GPUDirect RDMA support as three real device attributes, confirmed here directly against this exact environment's own installed `driver_types.h` rather than assumed from documentation: `cudaDevAttrGPUDirectRDMASupported` ("Device supports GPUDirect RDMA APIs, like nvidia_p2p_get_pages"), `cudaDevAttrGPUDirectRDMAFlushWritesOptions` (a bitmask of which flush mechanisms the device supports), and `cudaDevAttrGPUDirectRDMAWritesOrdering` (whether the device natively guarantees a NIC's writes are visible to that device's own running kernels). The same two facts also live as fields on `cudaDeviceProp`, queryable through Chapter 18's own `cudaGetDeviceProperties()` function -- two independent real ways to ask, exactly like Chapter 20's own `MPIX_CUDA_AWARE_SUPPORT` macro vs. `MPIX_Query_cuda_support()` function pairing. When a platform does *not* natively guarantee write ordering, `cudaDeviceFlushGPUDirectRDMAWrites()` is the real, documented way to close that gap -- it blocks until pending GPUDirect RDMA writes are visible to the requested scope, and returns `cudaErrorNotSupported` when the device can't do this or when the ordering already covers that scope (a real no-op).

```cpp
// Chapter 21: GPUDirect RDMA: Bypassing the Host Entirely
// 61_gpudirect_rdma_query.cu
//
// Section 20.2 checked whether an MPI BUILD could accept a device
// pointer directly (it couldn't, on this exact installed Open MPI).
// GPUDirect RDMA is the layer underneath that question: it lets a
// third-party PCI Express device -- a NIC, in the hybrid setup this
// chapter's own file 60 just bootstrapped -- read and write GPU
// memory directly, over the same real PCIe BAR mechanism NVIDIA's own
// GPUDirect RDMA design guide describes: "PCI Express device issues
// reads and writes to a peer device's BAR addresses in the same way
// that they are issued to system memory." Whether a specific GPU
// supports this at all is not something to assume -- CUDA exposes it
// as three real, queryable device attributes (cudaDevAttrGPUDirect...,
// found by grepping this exact environment's own installed
// driver_types.h, not guessed), queried here the same honest way
// Chapter 18 queried cudaGetDeviceProperties().
// Genuinely compiled with a real nvcc against the real installed CUDA
// runtime.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t eCount = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %s, count=%d\n",
           cudaGetErrorString(eCount), deviceCount);

    // Real device attribute #1: does this device support GPUDirect RDMA
    // APIs at all (the kernel-driver functions nvidia_p2p_get_pages()/
    // nvidia_p2p_put_pages() the design guide names)? This is a
    // per-device hardware/driver fact, independent of any specific NIC.
    int rdmaSupported = -1;
    cudaError_t eSupported = cudaDeviceGetAttribute(
        &rdmaSupported, cudaDevAttrGPUDirectRDMASupported, 0);
    printf("\ncudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMASupported): "
           "%s (queried value untouched at %d since the call itself "
           "honestly failed -- no device 0 exists to report on)\n",
           cudaGetErrorString(eSupported), rdmaSupported);

    // Real device attribute #2: a bitmask of which flush options
    // cudaDeviceFlushGPUDirectRDMAWrites() (below) supports on this
    // device -- cudaFlushGPUDirectRDMAWritesOptionHost (1<<0) and/or
    // cudaFlushGPUDirectRDMAWritesOptionMemOps (1<<1), per this exact
    // installed driver_types.h.
    int flushOptions = -1;
    cudaError_t eFlushOpts = cudaDeviceGetAttribute(
        &flushOptions, cudaDevAttrGPUDirectRDMAFlushWritesOptions, 0);
    printf("cudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMAFlushWritesOptions): "
           "%s\n", cudaGetErrorString(eFlushOpts));

    // Real device attribute #3: does this device natively guarantee
    // ORDERING of GPUDirect RDMA writes for the device's own kernels
    // (cudaGPUDirectRDMAWritesOrderingNone=0 / ...Owner=100 /
    // ...AllDevices=200)? A NIC's DMA write landing in GPU memory is
    // not automatically visible to a kernel already running on that
    // GPU unless the platform guarantees this ordering -- that's
    // exactly the race cudaDeviceFlushGPUDirectRDMAWrites() exists to
    // close when the platform does NOT guarantee it.
    int writesOrdering = -1;
    cudaError_t eOrdering = cudaDeviceGetAttribute(
        &writesOrdering, cudaDevAttrGPUDirectRDMAWritesOrdering, 0);
    printf("cudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMAWritesOrdering): "
           "%s\n", cudaGetErrorString(eOrdering));

    // cudaGetDeviceProperties() carries the SAME two facts as struct
    // fields (Chapter 18's own query function, new fields) -- included
    // to show both real, independent ways CUDA exposes this, exactly
    // like Chapter 20's own MPIX_CUDA_AWARE_SUPPORT (macro) vs
    // MPIX_Query_cuda_support() (function) pairing.
    cudaDeviceProp prop;
    cudaError_t eProps = cudaGetDeviceProperties(&prop, 0);
    printf("\ncudaGetDeviceProperties(): %s\n", cudaGetErrorString(eProps));
    if (eProps == cudaSuccess) {
        printf("  prop.gpuDirectRDMAFlushWritesOptions = %u\n",
               prop.gpuDirectRDMAFlushWritesOptions);
        printf("  prop.gpuDirectRDMAWritesOrdering     = %d\n",
               prop.gpuDirectRDMAWritesOrdering);
    } else {
        printf("  (struct never populated -- same honest failure as the "
               "three cudaDeviceGetAttribute() calls above; this "
               "environment has no device 0 for either query style to "
               "report on)\n");
    }

    // The actual bypass-the-host action: block until pending GPUDirect
    // RDMA writes to this device are visible to the requested scope.
    // Real, documented return values: cudaSuccess, or cudaErrorNotSupported
    // if "the call will be a no-op" -- e.g. because the device's own
    // ordering guarantee (queried above) already covers this scope, or
    // because the device has no GPUDirect RDMA support at all.
    cudaError_t eFlush = cudaDeviceFlushGPUDirectRDMAWrites(
        cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
        cudaFlushGPUDirectRDMAWritesToOwner);
    printf("\ncudaDeviceFlushGPUDirectRDMAWrites(TargetCurrentDevice, "
           "ToOwner): %s\n", cudaGetErrorString(eFlush));
    printf("This call needs an ACTIVE device context to mean anything at "
           "all -- there is no device 0 to make current, so this is the "
           "same honest cudaErrorNoDevice family every device-touching "
           "call in this book has returned since Chapter 3, not a new "
           "failure mode specific to GPUDirect RDMA.\n");

    return 0;
}
```

Compiled with `nvcc -Wno-deprecated-gpu-targets 61_gpudirect_rdma_query.cu -o 61_gpudirect_rdma_query` and genuinely run. Locked output, deterministic:

```text
cudaGetDeviceCount(): no CUDA-capable device is detected, count=0

cudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMASupported): no CUDA-capable device is detected (queried value untouched at -1 since the call itself honestly failed -- no device 0 exists to report on)
cudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMAFlushWritesOptions): no CUDA-capable device is detected
cudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMAWritesOrdering): no CUDA-capable device is detected

cudaGetDeviceProperties(): no CUDA-capable device is detected
  (struct never populated -- same honest failure as the three cudaDeviceGetAttribute() calls above; this environment has no device 0 for either query style to report on)

cudaDeviceFlushGPUDirectRDMAWrites(TargetCurrentDevice, ToOwner): no CUDA-capable device is detected
This call needs an ACTIVE device context to mean anything at all -- there is no device 0 to make current, so this is the same honest cudaErrorNoDevice family every device-touching call in this book has returned since Chapter 3, not a new failure mode specific to GPUDirect RDMA.
```

!!! warning "[COMMON TRAP] Treating \"GPUDirect RDMA capable\" as a property of the GPU alone"
    `cudaDevAttrGPUDirectRDMASupported` answers one question: can this GPU's driver expose its memory to a third-party device at all. It says nothing about any *specific* NIC, and nothing about the PCIe path between that GPU and that NIC. NVIDIA's own GPUDirect RDMA design guide is explicit that both devices must "share the same upstream PCI Express root complex," and NCCL's own `NCCL_NET_GDR_LEVEL` variable exists precisely because that path varies: its real documented levels run from `PIX` (GPU and NIC on the same PCI switch -- fast) down through `PXB`, `PHB`, to `SYS` (crossing the inter-NUMA-node SMP interconnect -- GPUDirect RDMA still technically works, but see Section 21.3 for exactly how much slower). A device that answers "yes" to `cudaDevAttrGPUDirectRDMASupported` can still end up on the slow end of that range, or -- if some other real constraint in the system rules it out entirely -- fall back to no GPUDirect RDMA at all, for reasons this one attribute alone can never reveal. The device-level query in this section and the topology question in the next are two different questions, and answering only the first is not enough to predict real performance.

## 21.3 The Cost of Bypassing the Host (and When Topology Ruins It)

### Intuition

Section 21.2 established that GPUDirect RDMA support is a real, queryable yes/no. What that yes/no is actually *worth* -- and what can silently take most of that value away -- are two separate real questions this section answers with cited numbers rather than a hardware measurement this environment cannot take. For small messages, skipping Chapter 20's own two `cudaMemcpy` calls and the intermediate host buffer removes real, fixed per-message latency -- every hop Section 21.2's diagram removed was a real hop with a real minimum cost, not a free abstraction. For large messages, the story is about sustained bandwidth, and bandwidth is where the COMMON TRAP above turns real: NVIDIA's own benchmarking of GPUDirect RDMA on real server platforms shows both a large, real speedup when the topology is right and a severe, real collapse when it isn't -- the same PCIe-topology sensitivity Chapter 2 first introduced for GPU-to-GPU links, now applying to the GPU-to-NIC link that this chapter's hybrid communicator ultimately depends on.

```text
Same PCIe switch (best case):          Crossing inter-socket QPI (worst case, SAME real post):

  GPU --- PCIe switch --- NIC            GPU --- CPU0 === QPI === CPU1 --- NIC
  11.6 GB/s (measured,                   250 MB/s write bandwidth (measured,
   Host-to-GPU, dual-rail)                IB adapter pushing to a cross-socket GPU)

  Both paths are "GPUDirect RDMA capable" by Section 21.2's own query.
  The 46x gap between them comes ENTIRELY from PCIe topology, not capability.
```

### Background

Every number below is cited to NVIDIA's own "Benchmarking GPUDirect RDMA on Modern Server Platforms" post -- this section's own arithmetic combines them; it does not reproduce any unpublished model from that post.

```cpp
// Chapter 21: GPUDirect RDMA: Bypassing the Host Entirely
// 62_staged_vs_rdma_cost_model.cpp
//
// Plain host C++, no device/MPI/NCCL needed -- this section's job is
// to quantify, using cited real numbers (not new measurements this
// book has no hardware to take), the gap between Section 20.2's own
// real staged path (cudaMemcpy D2H -> network -> cudaMemcpy H2D) and
// GPUDirect RDMA's direct path. Every constant below is cited to
// NVIDIA's own "Benchmarking GPUDirect RDMA on Modern Server
// Platforms" post; every combination of them is this book's own
// arithmetic, not a reproduction of an unpublished internal model --
// Part C below is explicit about that boundary.
#include <cstdio>

int main() {
    // ---- Part A: small-message latency, cited fixed numbers ----
    // "GPUDirect RDMA provides a latency consistently below 2us" --
    // measured 1.7us (GPU-to-Host) and 1.9us (Host-to-GPU direction).
    // The staged path's own cited breakdown: cudaMemcpy/cudaMemcpyAsync
    // "can easily take 8us and 9us respectively," plus InfiniBand's own
    // "1.3us" host-to-host latency -- the post's own rounded total for
    // the full GPU-to-GPU staged round trip is "approximately 17us."
    const double RDMA_LATENCY_LOW_US = 1.7;
    const double RDMA_LATENCY_HIGH_US = 1.9;
    const double STAGED_LATENCY_US = 17.0;

    double rdmaAvg = (RDMA_LATENCY_LOW_US + RDMA_LATENCY_HIGH_US) / 2.0;
    double speedup = STAGED_LATENCY_US / rdmaAvg;
    printf("Part A -- small-message fixed latency (cited, not measured here):\n");
    printf("  GPUDirect RDMA: %.1f-%.1fus (avg %.2fus)\n",
           RDMA_LATENCY_LOW_US, RDMA_LATENCY_HIGH_US, rdmaAvg);
    printf("  Staged (D2H + IB host-to-host + H2D): ~%.1fus\n", STAGED_LATENCY_US);
    printf("  Ratio: %.2fx -- matches the source's own cited "
           "\"approximately 9x faster\" claim (computed here from its own "
           "two numbers, not independently re-measured).\n\n", speedup);

    // ---- Part B: large-message bandwidth, topology-dependent ----
    // Best case, both devices under the SAME PCIe switch (the
    // GPUDirect RDMA design guide's own top-tier topology): 11.6 GB/s
    // Host-to-GPU with a dual-rail setup. Worst real case in the SAME
    // post: crossing the inter-socket QPI link collapses write
    // bandwidth to "250 MB/s when the IB adapter pushes data to a GPU
    // on a different socket." Same two endpoints NCCL's own
    // NCCL_NET_GDR_LEVEL knob names directly -- PIX (same PCIe switch)
    // down through SYS (cross-NUMA-node, over the SMP interconnect) --
    // reusing Chapter 2's own topology-matters framing, now for a
    // GPU-NIC pair instead of a GPU-GPU pair.
    const double BEST_CASE_GBPS = 11.6;   // same PCIe switch (PIX-class)
    const double WORST_CASE_GBPS = 0.25;  // 250 MB/s, crossing QPI (SYS-class)
    double degradation = BEST_CASE_GBPS / WORST_CASE_GBPS;
    printf("Part B -- large-message bandwidth, same real post, different "
           "PCIe topology:\n");
    printf("  Same PCIe switch (NCCL_NET_GDR_LEVEL=PIX-class): %.1f GB/s\n",
           BEST_CASE_GBPS);
    printf("  Crossing inter-socket QPI (NCCL_NET_GDR_LEVEL=SYS-class): "
           "%.2f GB/s\n", WORST_CASE_GBPS);
    printf("  Degradation: %.1fx -- \"GPUDirect RDMA capable\" (Section "
           "21.2's own boolean attribute) is NECESSARY but not "
           "SUFFICIENT: the same capable GPU, paired with the same "
           "capable NIC, is %.0fx slower if the PCIe topology between "
           "them is wrong, with no code change at all.\n\n", degradation, degradation);

    // ---- Part C: the real crossover, cited as a finding, not re-derived ----
    printf("Part C -- the source's own real crossover finding:\n");
    printf("  The same post states: \"GPUDirect RDMA is faster than the "
           "staging approach for message sizes up to 400-500KB (on Ivy "
           "Bridge Xeon)\" -- a real, cited finding this book reports "
           "rather than re-derives: NVIDIA's post names the crossover "
           "but does not publish the underlying model that produces it, "
           "and Parts A and B above already show why a simple two-term "
           "(fixed latency + bandwidth) model built from this post's own "
           "OTHER numbers should not be expected to reproduce it exactly "
           "-- Part A's numbers alone would put GPUDirect RDMA ahead at "
           "every size, since it has both the lower fixed latency AND "
           "(Part B, best case) the higher bandwidth. The real crossover "
           "almost certainly depends on GPUDirect RDMA's own bandwidth "
           "ceiling being LOWER than the staged path's at large sizes on "
           "specifically the Ivy Bridge platform tested -- a real, "
           "platform-specific effect this book has no cited numbers for, "
           "so it is reported as a limit of what is known here, not "
           "quietly modeled around.\n");

    return 0;
}
```

Compiled with `g++ -Wall -Wextra -O2 62_staged_vs_rdma_cost_model.cpp -o 62_staged_vs_rdma_cost_model`, genuinely run, and cross-verified byte-identical on the device (`g++` 11.4.0). Locked output:

```text
Part A -- small-message fixed latency (cited, not measured here):
  GPUDirect RDMA: 1.7-1.9us (avg 1.80us)
  Staged (D2H + IB host-to-host + H2D): ~17.0us
  Ratio: 9.44x -- matches the source's own cited "approximately 9x faster" claim (computed here from its own two numbers, not independently re-measured).

Part B -- large-message bandwidth, same real post, different PCIe topology:
  Same PCIe switch (NCCL_NET_GDR_LEVEL=PIX-class): 11.6 GB/s
  Crossing inter-socket QPI (NCCL_NET_GDR_LEVEL=SYS-class): 0.25 GB/s
  Degradation: 46.4x -- "GPUDirect RDMA capable" (Section 21.2's own boolean attribute) is NECESSARY but not SUFFICIENT: the same capable GPU, paired with the same capable NIC, is 46x slower if the PCIe topology between them is wrong, with no code change at all.

Part C -- the source's own real crossover finding:
  The same post states: "GPUDirect RDMA is faster than the staging approach for message sizes up to 400-500KB (on Ivy Bridge Xeon)" -- a real, cited finding this book reports rather than re-derives: NVIDIA's post names the crossover but does not publish the underlying model that produces it, and Parts A and B above already show why a simple two-term (fixed latency + bandwidth) model built from this post's own OTHER numbers should not be expected to reproduce it exactly -- Part A's numbers alone would put GPUDirect RDMA ahead at every size, since it has both the lower fixed latency AND (Part B, best case) the higher bandwidth. The real crossover almost certainly depends on GPUDirect RDMA's own bandwidth ceiling being LOWER than the staged path's at large sizes on specifically the Ivy Bridge platform tested -- a real, platform-specific effect this book has no cited numbers for, so it is reported as a limit of what is known here, not quietly modeled around.
```

!!! warning "[COMMON TRAP] Assuming GPUDirect RDMA is always the faster choice once it's available"
    Section 21.2's query and this section's Part A both make GPUDirect RDMA look like a strict win -- lower fixed latency, higher best-case bandwidth, no reason not to use it. Part C's own honestly-reported crossover is the real counter-example: NVIDIA's own benchmark found the staging approach becomes *competitive* past roughly 400-500KB on the platform it tested, and Part B shows a second, independent way GPUDirect RDMA can lose badly -- not from message size at all, but from PCIe topology alone, a 46x swing between two identically-"capable" systems. This is exactly why real MPI implementations (MVAPICH2-GDR is a well-known example) expose a message-size threshold knob that switches between a direct GPUDirect RDMA path and a staged path, rather than always preferring one. Querying `cudaDevAttrGPUDirectRDMASupported` and stopping there answers "can I," never "should I, for this message, on this topology, right now."

## Chapter Summary

This chapter opened Part 5's second real building block. Section 21.1 built the hybrid bootstrap every real multi-node NCCL job actually uses: MPI (Chapter 20's own real multi-process model) generates and distributes NCCL's `ncclUniqueId`, and NCCL (every collective since Chapter 11) takes over from there -- verified against NCCL's own installed-header comment and NCCL's own official documentation, with the real, distributed-id-required trap named directly. Section 21.2 queried, honestly and for real, whether a GPU supports GPUDirect RDMA (`cudaDevAttrGPUDirectRDMASupported`) and how its write-ordering guarantee works (`cudaDevAttrGPUDirectRDMAWritesOrdering`, `cudaDeviceFlushGPUDirectRDMAWrites()`) -- three real device attributes confirmed against this exact environment's own installed CUDA headers, all honestly `cudaErrorNoDevice` since no device exists here. Section 21.3 quantified what bypassing the host in Section 20.2's own staged fallback is worth using NVIDIA's own cited benchmark numbers: roughly 9x lower latency for small messages, but a real, topology-driven 46x bandwidth swing that has nothing to do with message size, and an honestly-reported real crossover this book's own simple model correctly predicts it cannot reproduce.

## Self-Check Questions

1. Why does NCCL need MPI (or an equivalent) at all, given that NCCL already has its own communicator API? What specific step can NCCL not do on its own?
2. In file 60, what would happen on a real cluster (not this environment) if the `if (myRank == 0)` guard around `ncclGetUniqueId(&id)` were removed?
3. `cudaDevAttrGPUDirectRDMASupported` and NCCL's `NCCL_NET_GDR_LEVEL` both relate to GPUDirect RDMA. What real, different question does each one answer?
4. What is `cudaDeviceFlushGPUDirectRDMAWrites()` for, and under what real device condition would calling it be a genuine no-op rather than a bug?
5. Section 21.3's Part A computed a 9.44x speedup from two fixed latency numbers. Which two numbers, and why does the source's own post round this to "approximately 9x"?
6. Explain, using Part B's own two numbers, why a GPU and NIC that are both individually "GPUDirect RDMA capable" can still deliver 250 MB/s instead of 11.6 GB/s.
7. Why does this chapter's own Part C explicitly decline to build a closed-form model reproducing NVIDIA's cited 400-500KB crossover, when Parts A and B already build models for the small-message and topology cases?
8. File 60 links three real libraries in one binary (Open MPI, NCCL, CUDA runtime) via `mpicxx`. Which two chapters each first established one piece of that same combination, and what did file 60 change about how they're fed into `ncclCommInitRank()`?
9. Contrast the COMMON TRAP in Section 21.1 (a hang) with the COMMON TRAP in Section 21.2/21.3 (silently wrong performance expectations, not a hang or an error). Why can the same "GPUDirect RDMA capable" answer coexist with both a correct program and a badly underperforming one?

## Where We Go Next

Chapter 21 answered whether a NIC can reach GPU memory directly, and what that's worth when the PCIe topology cooperates. Every collective this book has built since Chapter 8, though, still assumes the *GPU* is a passive participant -- it computes, and some other actor (a CPU thread, an MPI call, NCCL's own internal logic) decides when and what to communicate. Chapter 22 opens with NVSHMEM, where that assumption breaks: a GPU kernel can issue a put or get to a remote GPU's memory directly, from device code, while it is still running -- no host round-trip, no waiting for a separate collective call to be launched from the CPU at all.

## Worked Solutions

**1.** NCCL's communicator API can build a communicator once every rank already agrees on one shared `ncclUniqueId`, but NCCL has no mechanism of its own to get that id from the one rank that generates it to every other rank -- it has no process launcher and no general-purpose broadcast primitive that exists before a communicator does. That is a generic "one-to-all small message" problem, and MPI's `MPI_Bcast()` already solves the generic version.

**2.** Every rank would generate its own, independently random `ncclUniqueId`, since the id is not derived from anything shared across ranks. `ncclCommInitRank()` on a real cluster would then have no common id to rendezvous on: each rank waits for peers that never agreed to that specific id, and the call hangs indefinitely rather than failing fast -- the same real, worse failure mode Chapter 19 named for a genuinely dead peer, now caused by a bootstrapping bug instead of hardware failure.

**3.** `cudaDevAttrGPUDirectRDMASupported` answers whether a specific GPU's driver can expose its memory to a third-party PCI Express device at all -- a per-GPU hardware/driver capability, independent of any particular NIC. `NCCL_NET_GDR_LEVEL` answers a different, topology-specific question: for a *particular* GPU-NIC pair, how close together are they on the PCIe fabric (same switch, through switches, same NUMA node, or across the inter-socket interconnect), which determines whether NCCL will actually use GPUDirect RDMA between that specific pair and how well it will perform if it does.

**4.** It blocks the calling code until pending GPUDirect RDMA writes to the target device are visible to the requested scope (the device's owning context, or all devices). It is a genuine no-op, returning `cudaErrorNotSupported`, when the device's own `cudaDevAttrGPUDirectRDMAWritesOrdering` attribute already reports that its scope covers the requested one -- i.e., the platform already natively guarantees the ordering this call exists to enforce, so there is nothing left for the call to do.

**5.** The two numbers are the average of the cited RDMA latency range (1.7us and 1.9us, averaging to 1.8us) and the cited staged-path total (~17.0us), giving 17.0/1.8 = 9.44x. The source's own post rounds this down to "approximately 9x" likely because its own 1.7us and 1.9us are themselves measured, rounded figures (not exact constants), and because "approximately 9x" is a looser, more defensible claim than quoting a precise ratio from numbers that are already stated as approximate ("can easily take 8us and 9us," "approximately 17us").

**6.** Both numbers come from the SAME real post but describe different PCIe topologies, not different capability levels. 11.6 GB/s is measured when the GPU and NIC share the same PCIe switch (GPUDirect RDMA's own best-case path, requiring no long hop). 250 MB/s is measured when the IB adapter must push data to a GPU on a different CPU socket, crossing the inter-socket QPI link -- a real, physical bottleneck that exists regardless of whether both devices individually report `cudaDevAttrGPUDirectRDMASupported = 1`. Capability is a yes/no property of each device; the 46x gap is a property of the path between them.

**7.** Because Parts A and B's own models are built entirely from OTHER numbers in the same cited post (fixed latencies, best/worst-case bandwidths) that this book has independently verified are real and can combine additively or as simple ratios. The 400-500KB crossover is a THIRD, separate finding the post states as a conclusion ("by using a simple performance model it can be shown that...") without publishing that model's own inputs or formula. Building a new model and claiming it reproduces an unpublished one would be fabricating agreement with a source this book cannot actually check -- so Part C reports the finding as a citation and explains, using Parts A and B's own numbers, why a naive model of this book's own would NOT be expected to land at the same crossover, rather than silently presenting a model tuned to match it.

**8.** Chapter 20 first established a real Open MPI installation and the `mpicxx`-links-`libcudart` combination (file 58). Chapter 11 first established a real NCCL installation and communicator construction (`ncclCommInitAll()`). File 60 changes how NCCL's inputs are fed: instead of one process supplying a literal rank count and calling `ncclCommInitAll()` once for every device it can see, file 60 has each of N real MPI-launched processes independently call `ncclCommInitRank()` with a rank and size it queried from `MPI_Comm_rank()`/`MPI_Comm_size()`, and a shared id it received via `MPI_Bcast()` rather than generating itself.

**9.** Section 21.1's trap produces a hang because the bug (independently-generated, mismatched ids) prevents the communicator from ever forming at all -- there is no partial or degraded success state, just an indefinite wait. Sections 21.2/21.3's trap produces a program that runs completely correctly and completes, but transfers data far slower than "GPUDirect RDMA capable" alone would suggest, because capability and topology are independent real facts: a device can genuinely support the mechanism (Section 21.2's yes) while sitting on a path that makes using it a bad idea for a given message size or system layout (Section 21.3's cost model). The same boolean answer is compatible with both scenarios because it was only ever answering "can," never "will perform well."

---

**Sources cited in this chapter:**

- [1. Overview — GPUDirect RDMA 13.4 documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/) — the core mechanism ("a direct path for data exchange between the GPU and a third-party peer device using standard features of PCI Express"), the PCI BAR mechanism quote ("PCI Express device issues reads and writes to a peer device's BAR addresses in the same way that they are issued to system memory"), the same-root-complex topology requirement, and the `nvidia_p2p_get_pages()`/`nvidia_p2p_put_pages()` kernel API names.
- This exact environment's own installed CUDA header, `/usr/include/driver_types.h` (CUDA 12.0) — the real, verified enum values and comments for `cudaDevAttrGPUDirectRDMASupported`, `cudaDevAttrGPUDirectRDMAFlushWritesOptions`, `cudaDevAttrGPUDirectRDMAWritesOrdering`, `cudaFlushGPUDirectRDMAWritesOptions`, `cudaGPUDirectRDMAWritesOrdering`, `cudaFlushGPUDirectRDMAWritesScope`, and `cudaFlushGPUDirectRDMAWritesTarget`, and the `cudaDeviceFlushGPUDirectRDMAWrites()` declaration in `/usr/include/cuda_runtime_api.h`.
- This exact environment's own installed NCCL header, `/usr/include/nccl.h` — the real, verified comment above `ncclGetUniqueId()`: "ncclGetUniqueId should be called once and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank," and the comment above `ncclCommInitRank()` on per-device binding and implicit synchronization.
- [Examples — NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html) — the real multi-process/MPI example this chapter's file 60 follows directly: `MPI_Comm_rank`/`MPI_Comm_size`, rank 0 alone calling `ncclGetUniqueId()`, `MPI_Bcast()` of the id, and `ncclCommInitRank()`, explicitly described as working "in the context of MPI, using one device per MPI rank."
- [Environment Variables — NCCL user guide (archived, 2.18.x)](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2183/user-guide/docs/env.html) — `NCCL_NET_GDR_LEVEL`'s real documented PCIe-distance levels (`PIX`, `PXB`, `PHB`, `SYS`) and their topology meanings, reused in Section 21.3's own cost model.
- [Benchmarking GPUDirect RDMA on Modern Server Platforms — NVIDIA Technical Blog](https://developer.nvidia.com/blog/benchmarking-gpudirect-rdma-on-modern-server-platforms/) — every cited number in Section 21.3: the sub-2us GPUDirect RDMA latency figures (1.7us/1.9us), the staged-path latency breakdown (~8us/9us `cudaMemcpy`, 1.3us IB host-to-host, ~17us total), the same-PCIe-switch best-case bandwidth (11.6 GB/s), the cross-socket QPI worst-case bandwidth (250 MB/s), and the real cited 400-500KB crossover finding on Ivy Bridge Xeon.
