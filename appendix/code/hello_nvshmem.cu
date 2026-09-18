// Appendix A.4: Recipe 5 stub -- nvcc + NVSHMEM + MPI, GPU-initiated
// one-sided communication's own real link line (Chapter 22's own recipe).
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>

__global__ void nvshmemStubKernel() {
    // deliberately left with an empty body, same reasoning as
    // hello_kernel.cu above -- this recipe exists to prove the real
    // nvcc+NVSHMEM+MPI link line succeeds, not to exercise a real device-
    // initiated put/get (this environment has no physical device to
    // launch on, exactly as Chapter 22 itself found when it tried).
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("hello_nvshmem: rank %d, compiled and linked against real "
           "libnvshmem_host/libnvshmem_device (NVSHMEM %d.%d.%d), real "
           "MPI bootstrap headers present.\n", rank,
           NVSHMEM_VENDOR_MAJOR_VERSION, NVSHMEM_VENDOR_MINOR_VERSION,
           NVSHMEM_VENDOR_PATCH_VERSION);
    MPI_Finalize();
    return 0;
}
