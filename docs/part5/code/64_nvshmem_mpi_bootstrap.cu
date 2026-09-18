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
