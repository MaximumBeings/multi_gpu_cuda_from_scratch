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
