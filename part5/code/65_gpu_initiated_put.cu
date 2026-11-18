// 65_gpu_initiated_put.cu
// Checked two-PE example. Compile and execution evidence are captured separately.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

static void bind_device(int rank) {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, count;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    CUDA_OK(cudaGetDeviceCount(&count));
    if (localRank >= count) {
        fprintf(stderr, "Insufficient GPUs: local rank %d, visible devices %d\n", localRank, count);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_OK(cudaSetDevice(localRank));
}

__global__ void putKernel(int *destination, int value, int peer) {
    nvshmem_int_p(destination, value, peer);
    nvshmem_quiet(); // complete this issuing thread's remote operation
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bind_device(rank); // one local PE per GPU, before NVSHMEM initialization
    MPI_Comm world = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_set_attr_mpi_comm_args(&world, &attr);
    int rc = nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    if (rc != 0) MPI_Abort(MPI_COMM_WORLD, 3);
    int pe = nvshmem_my_pe(), nPes = nvshmem_n_pes();
    int errors = 0;
    int *target = static_cast<int *>(nvshmem_malloc(sizeof(int)));
    if (!target) MPI_Abort(MPI_COMM_WORLD, 4);
    CUDA_OK(cudaMemset(target, 0xff, sizeof(int)));
    nvshmem_barrier_all(); // all destination allocations are initialized
    int value = 777 + pe, peer = (pe + 1) % nPes;
    void *args[] = {&target, &value, &peer};
    rc = nvshmemx_collective_launch(reinterpret_cast<const void *>(putKernel),
                                  dim3(1), dim3(1), args, 0, 0);
    if (rc != 0) MPI_Abort(MPI_COMM_WORLD, 5);
    CUDA_OK(cudaDeviceSynchronize());
    nvshmem_barrier_all(); // every issuing PE has completed its put and quiet
    int received = -1;
    CUDA_OK(cudaMemcpy(&received, target, sizeof(int), cudaMemcpyDeviceToHost));
    int expected = 777 + (pe + nPes - 1) % nPes;
    if (received != expected) ++errors;
    printf("PE %d ring put: received=%d expected=%d\n", pe, received, expected);
    nvshmem_free(target);
    int totalErrors = 0;
    MPI_Allreduce(&errors, &totalErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (pe == 0) printf("VALIDATION %s: %d PEs, errors=%d\n",
                        totalErrors ? "FAIL" : "PASS", nPes, totalErrors);
    nvshmem_finalize();
    MPI_Finalize();
    return totalErrors ? 1 : 0;
}
