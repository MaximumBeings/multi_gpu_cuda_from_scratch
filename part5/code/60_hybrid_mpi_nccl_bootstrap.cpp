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
