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
