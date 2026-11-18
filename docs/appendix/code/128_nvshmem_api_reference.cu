// 128_nvshmem_api_reference.cu
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

__global__ void getAndAdd(int *value,int *fetched,int *counter,int peer) {
    nvshmem_int_get(fetched,value,1,peer);
    nvshmem_int_atomic_add(counter,1,0);
    nvshmem_quiet();
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
    int *value=static_cast<int *>(nvshmem_malloc(sizeof(int)));
    int *fetched=static_cast<int *>(nvshmem_malloc(sizeof(int)));
    int *counter=static_cast<int *>(nvshmem_malloc(sizeof(int)));
    if(!value || !fetched || !counter) MPI_Abort(MPI_COMM_WORLD,4);
    int own=10*(pe+1);
    CUDA_OK(cudaMemcpy(value,&own,sizeof(int),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(fetched,0xff,sizeof(int)));
    CUDA_OK(cudaMemset(counter,0,sizeof(int)));
    nvshmem_barrier_all();
    int peer=(pe+1)%nPes;
    void *args[]={&value,&fetched,&counter,&peer};
    rc=nvshmemx_collective_launch(reinterpret_cast<const void *>(getAndAdd),dim3(1),dim3(1),args,0,0);
    if(rc!=0) MPI_Abort(MPI_COMM_WORLD,5);
    CUDA_OK(cudaDeviceSynchronize()); nvshmem_barrier_all();
    int got=-1,count=-1;
    CUDA_OK(cudaMemcpy(&got,fetched,sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&count,counter,sizeof(int),cudaMemcpyDeviceToHost));
    if(got!=10*(peer+1) || count!=(pe==0?nPes:0)) ++errors;
    if(nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD)!=pe) ++errors;
    printf("PE %d: neighbor=%d received=%d counter=%d\n",pe,peer,got,count);
    nvshmem_free(counter); nvshmem_free(fetched); nvshmem_free(value);
    int totalErrors = 0;
    MPI_Allreduce(&errors, &totalErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (pe == 0) printf("VALIDATION %s: %d PEs, errors=%d\n",
                        totalErrors ? "FAIL" : "PASS", nPes, totalErrors);
    nvshmem_finalize();
    MPI_Finalize();
    return totalErrors ? 1 : 0;
}
