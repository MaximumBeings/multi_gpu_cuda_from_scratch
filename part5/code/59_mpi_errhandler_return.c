// Chapter 20: MPI From Scratch, Then CUDA-Aware MPI: Scaling Beyond One Node
// 59_mpi_errhandler_return.c
//
// Chapter 19 quoted MPI's own real ULFM extension (MPIX_Comm_revoke(),
// MPIX_Comm_shrink()) as a genuinely richer alternative to NCCL's
// abort-and-rebuild cycle, but never called either function -- Section
// 20.3's own Background text explains why, verified the same real way
// Chapter 19 verified NCCL's own API surface: this exact environment's
// installed Open MPI (apt's libopenmpi-dev, 4.1.6) predates ULFM's real
// merge into Open MPI's mainline (2020, stable from 5.0), confirmed by
// `ompi_info`'s own real "MPI extensions:" list (affinity, cuda,
// pcollreq -- no ftmpi) and by grepping the installed mpi-ext.h for
// MPIX_Comm_revoke/MPIX_Comm_shrink (absent). What standard MPI DOES
// offer, in every version back to MPI-1.1, is a weaker, real
// alternative this file builds: MPI_ERRORS_RETURN, installed via
// MPI_Comm_set_errhandler(), which lets a rank get an error CODE back
// instead of the whole job aborting -- but, per the MPI standard's own
// explicit wording, does NOT guarantee the state of MPI is still usable
// afterward, which is exactly the gap ULFM's own revoke/shrink exists
// to close for real.
// Genuinely compiled with a real mpicc against the real installed
// Open MPI, genuinely run with a real mpirun.
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // The real, standard alternative to the default MPI_ERRORS_ARE_FATAL
    // handler every earlier call in this chapter has run under, silently
    // (every MPI_Send/MPI_Recv in 57 and 58 above would have aborted the
    // whole job on a real error, exactly like this handler's own name
    // says -- they simply never hit one).
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    if (rank == 0) {
        printf("rank 0: about to MPI_Send to invalid destination rank 999 "
               "(real world size %d), now under MPI_ERRORS_RETURN\n", size);
        int val = 42;
        int rc = MPI_Send(&val, 1, MPI_INT, 999, 0, MPI_COMM_WORLD);
        char errStr[MPI_MAX_ERROR_STRING];
        int errLen = 0;
        MPI_Error_string(rc, errStr, &errLen);
        printf("rank 0: MPI_Send RETURNED rc=%d (%s) -- this process did NOT abort\n", rc, errStr);
        printf("rank 0: NOTE -- the MPI standard's own words on this: 'the state of "
               "MPI is undefined' after an error is detected, even under "
               "MPI_ERRORS_RETURN. Catching rc above is real, but it is NOT the same "
               "guarantee as ULFM's own MPIX_Comm_revoke()/MPIX_Comm_shrink(), which "
               "define exactly what a communicator's surviving ranks may do next.\n");
    }

    printf("rank %d of %d: reached MPI_Finalize\n", rank, size);
    MPI_Finalize();
    return 0;
}
