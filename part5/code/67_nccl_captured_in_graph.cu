// Chapter 23: CUDA Graphs Across Multiple GPUs and Multiple Nodes
// 67_nccl_captured_in_graph.cu
//
// File 66 captured plain CUDA work (kernels, a peer copy) across two
// devices in one graph. NCCL's own user guide states a real, separate
// capability this section builds next: "Starting with NCCL 2.9, NCCL
// operations can be captured by CUDA Graphs. This support requires a
// minimum CUDA version of 11.3." No new NCCL API is needed for this --
// a collective call issued between cudaStreamBeginCapture() and
// cudaStreamEndCapture() is captured exactly like any other kernel
// launch, because ncclAllReduce() is, underneath, ordinary kernel
// launches enqueued on the communicator's stream. This reuses Chapter
// 21's own real hybrid MPI+NCCL bootstrap (ncclGetUniqueId() on rank 0,
// MPI_Bcast() to every rank, then ncclCommInitRank() on all ranks).
//
// This file adds ONE real, disciplined check that no earlier chapter's
// collective code strictly needed: it verifies ncclCommInitRank()
// actually returned ncclSuccess BEFORE doing anything else with the
// resulting handle -- including beginning a graph capture. This
// section's own COMMON TRAP explains, with a real reproduced crash, why
// that check is not optional once graph capture enters the picture.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    ncclUniqueId id;
    if (rank == 0) {
        ncclResult_t idErr = ncclGetUniqueId(&id);
        printf("Rank %d: ncclGetUniqueId() -> %s\n", rank, ncclGetErrorString(idErr));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    printf("Rank %d: MPI_Bcast(id) done -- every rank now holds the SAME id "
           "(Chapter 21's own bootstrap, reused unchanged)\n", rank);

    cudaError_t setDevErr = cudaSetDevice(rank);
    printf("Rank %d: cudaSetDevice(%d) -> %s\n", rank, rank, cudaGetErrorString(setDevErr));

    ncclComm_t comm;
    ncclResult_t commErr = ncclCommInitRank(&comm, worldSize, id, rank);
    printf("Rank %d: ncclCommInitRank() -> %s\n", rank, ncclGetErrorString(commErr));

    // THE GUARD. Every earlier chapter's collective code (Ch11, Ch17,
    // Ch21) called the next real API regardless of this return value,
    // because in every one of those cases the next call still returned
    // its OWN honest error code rather than corrupting anything. That is
    // NOT true here -- see this section's COMMON TRAP for the real,
    // reproduced crash that skipping this check causes the moment a
    // collective on this handle is captured (or even just called) into
    // a stream. This program stops here rather than reproduce it.
    if (commErr != ncclSuccess) {
        printf("Rank %d: STOPPING before touching this communicator again -- "
               "ncclCommInitRank() did not return ncclSuccess, so nothing else "
               "on `comm` is safe to call, capture included.\n", rank);
        MPI_Finalize();
        return 0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *sendBuf = nullptr, *recvBuf = nullptr;
    cudaMalloc(&sendBuf, sizeof(float) * 1024);
    cudaMalloc(&recvBuf, sizeof(float) * 1024);

    printf("Rank %d: --- beginning capture (the collective goes INSIDE the graph) ---\n", rank);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    ncclAllReduce(sendBuf, recvBuf, 1024, ncclFloat, ncclSum, comm, stream);
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, 0);

    printf("Rank %d: --- launching the captured graph 3 times (no re-issuing "
           "ncclAllReduce by hand each time) ---\n", rank);
    for (int iter = 0; iter < 3; iter++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);

    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
