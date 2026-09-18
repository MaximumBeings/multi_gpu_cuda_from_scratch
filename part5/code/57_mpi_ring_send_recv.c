// Chapter 20: MPI From Scratch, Then CUDA-Aware MPI: Scaling Beyond One Node
// 57_mpi_ring_send_recv.c
//
// Every device-touching call this book has made since Chapter 3 needed
// EITHER a real GPU (and got an honest cudaErrorNoDevice/ncclInvalidArgument
// right back, since this environment has never had one) OR a host-side
// simulation standing in for N real devices (Chapter 4 onward). MPI needs
// neither. MPI_Init() doesn't ask for a GPU at all -- it's a multi-PROCESS
// model, and mpirun genuinely launches N separate operating-system
// processes right now, on this machine's real CPU cores. MPI_Comm_rank()
// below returns a REAL rank, not a simulated one and not an honest error.
// This section builds the real basics -- Init/Comm_rank/Comm_size, plus a
// real point-to-point ring exchange, structurally the same shape as
// Chapter 9's own ring, but with every buffer now a real, separate
// process's own real memory, and every exchange a genuine MPI_Send/
// MPI_Recv pair instead of a simulated array copy.
//
// Takes two OPTIONAL arguments so this one file can also drive this
// section's own COMMON TRAP without a second program: argv[1] is how
// many ints each rank sends (default 1, the safe case Background uses);
// argv[2], if given as "sendrecv", switches from a naive MPI_Send-then-
// MPI_Recv ring to a single real MPI_Sendrecv() call instead -- the
// real, standard fix for the naive version's own real deadlock risk at
// large argv[1] values, demonstrated honestly in the COMMON TRAP below
// (running the naive mode at 1,000,000 ints genuinely deadlocks and
// must be run under a bounded `timeout`, since real blocking MPI_Send
// implementation-defined buffering runs out at that size).
// Genuinely compiled with a real mpicc against the real installed
// Open MPI, genuinely run with a real mpirun.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elems = (argc > 1) ? atoi(argv[1]) : 1;
    int useSendrecv = (argc > 2) && (strcmp(argv[2], "sendrecv") == 0);

    // Chapter 9's own ring topology, real this time: rank i sends to
    // (i+1)%size and receives from (i-1+size)%size -- no wraparound
    // special-casing needed, the modular arithmetic handles it exactly
    // like Chapter 9's own ring did.
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    int* sendBuf = (int*)malloc((size_t)elems * sizeof(int));
    int* recvBuf = (int*)malloc((size_t)elems * sizeof(int));
    for (int i = 0; i < elems; ++i) sendBuf[i] = rank * 100;

    if (useSendrecv) {
        // The real, standard fix: MPI_Sendrecv() posts both the send
        // and the matching receive as one call, so there is no window
        // where every rank is blocked inside MPI_Send() waiting for a
        // receive that no rank has posted yet.
        MPI_Sendrecv(sendBuf, elems, MPI_INT, next, 0,
                     recvBuf, elems, MPI_INT, prev, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // The naive shape: every rank calls MPI_Send() BEFORE any rank
        // has called MPI_Recv(). Small messages usually survive this
        // (Open MPI's own eager-send buffering just copies the data
        // into a temporary system buffer and returns) -- large ones
        // don't, which is this section's own COMMON TRAP.
        fprintf(stderr, "rank %d: about to MPI_Send %zu bytes to rank %d\n",
                rank, (size_t)elems * sizeof(int), next);
        MPI_Send(sendBuf, elems, MPI_INT, next, 0, MPI_COMM_WORLD);
        fprintf(stderr, "rank %d: MPI_Send returned; about to MPI_Recv from rank %d\n",
                rank, prev);
        MPI_Recv(recvBuf, elems, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    printf("rank %d of %d: sent %d, received %d (elems=%d, mode=%s)\n",
           rank, size, sendBuf[0], recvBuf[0], elems, useSendrecv ? "sendrecv" : "naive");

    free(sendBuf);
    free(recvBuf);
    MPI_Finalize();
    return 0;
}
