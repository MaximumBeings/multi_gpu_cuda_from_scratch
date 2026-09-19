// Appendix G: Common Failure Modes -- Deadlocks, Silent Corruption From
// Missed Synchronization, and Topology Mismatches
// 138_collective_order_deadlock.c
//
// Appendix G.1 -- Chapter 20's own File 57 already genuinely reproduced
// one real MPI deadlock: a naive point-to-point ring where every rank
// calls MPI_Send() before any rank calls MPI_Recv() -- but that
// deadlock only manifests once the message crosses Open MPI's own
// eager-send buffering threshold (Chapter 20 needed 1,000,000 ints; a
// single int silently survives). This file builds the OTHER, absolute
// version of the same real hazard: every rank calls MPI_Recv() BEFORE
// any rank calls MPI_Send(). MPI_Recv() genuinely blocks until a
// matching message arrives, with no buffering escape hatch at any
// message size -- if every rank in a ring is blocked inside its own
// MPI_Recv(), no rank can ever reach the MPI_Send() call the rank
// ahead of it in the ring is waiting for, and the program hangs
// forever, confirmed here the same way Chapter 20 confirmed its own
// deadlock: run under a bounded `timeout`, which kills the genuinely-
// hung job and reports exit code 124. The real, standard fix -- half
// the ranks send first, half receive first, so the blocking calls
// interleave instead of all waiting on each other -- is built and
// verified alongside it.
//
// argv[1] == "fixed"    -> even ranks Send-then-Recv, odd ranks Recv-then-Send (completes)
// argv[1] == "deadlock" (or omitted) -> every rank Recv-then-Send (hangs at ANY message size)
//
// Compile: mpicc -std=c11 -Wall -Wextra -O2 138_collective_order_deadlock.c -o 138_collective_order_deadlock
// Run:     timeout 6 mpirun --allow-run-as-root --oversubscribe -np 4 ./138_collective_order_deadlock fixed
//          timeout 6 mpirun --allow-run-as-root --oversubscribe -np 4 ./138_collective_order_deadlock deadlock
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int fixed = (argc >= 2) && (strcmp(argv[1], "fixed") == 0);
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;
    int sendVal = rank, recvVal = -1;

    if (fixed && (rank % 2 == 0)) {
        // Even ranks: Send first, Recv second -- interleaves correctly
        // with the odd ranks below, which do the opposite.
        fprintf(stderr, "rank %d (even, fixed order): about to MPI_Send to %d\n", rank, next);
        MPI_Send(&sendVal, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        fprintf(stderr, "rank %d: MPI_Send returned; about to MPI_Recv from %d\n", rank, prev);
        MPI_Recv(&recvVal, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stderr, "rank %d: MPI_Recv returned\n", rank);
    } else {
        // Odd ranks in "fixed" mode, and EVERY rank in "deadlock" mode:
        // Recv first, Send second.
        fprintf(stderr, "rank %d: about to MPI_Recv from %d (recv-first order)\n", rank, prev);
        MPI_Recv(&recvVal, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fprintf(stderr, "rank %d: MPI_Recv returned; about to MPI_Send to %d\n", rank, next);
        MPI_Send(&sendVal, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        fprintf(stderr, "rank %d: MPI_Send returned\n", rank);
    }

    printf("rank %d done, recvVal=%d (expected %d)\n", rank, recvVal, prev);
    MPI_Finalize();
    return 0;
}
