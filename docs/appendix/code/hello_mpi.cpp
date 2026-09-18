// Appendix A.4: Recipe 4 stub -- mpicxx, CUDA-aware host-side link line.
#include <cstdio>
#include <mpi.h>
#include <cuda_runtime.h>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rtVersion = 0;
    cudaRuntimeGetVersion(&rtVersion);
    printf("hello_mpi: rank %d, linked against real libcudart (CUDA "
           "runtime %d.%d)\n", rank, rtVersion / 1000, (rtVersion % 1000) / 10);
    MPI_Finalize();
    return 0;
}
