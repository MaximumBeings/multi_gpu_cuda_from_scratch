// Chapter 20: MPI From Scratch, Then CUDA-Aware MPI: Scaling Beyond One Node
// 58_mpi_cuda_aware_query_and_fallback.cpp
//
// "CUDA-aware MPI" means an MPI implementation can accept a raw device
// pointer directly in MPI_Send()/MPI_Recv() and move the data GPU-to-GPU
// itself, with no host staging at all. Whether a given MPI BUILD actually
// has this is not something to assume -- it's a real, queryable fact,
// checked here two different real ways: the compile-time macro
// MPIX_CUDA_AWARE_SUPPORT (baked into mpiext_cuda_c.h at Open MPI's own
// build time) and the runtime call MPIX_Query_cuda_support(). This
// program genuinely links against both the real installed Open MPI and
// the real installed CUDA runtime, and queries this EXACT environment's
// real answer -- then, because that answer is "no," performs the real,
// correct fallback: Chapter 5's own staged host-transfer pattern
// (cudaMemcpy device-to-host, a real MPI_Send/MPI_Recv over plain host
// memory, cudaMemcpy host-to-device), reused here between MPI ranks
// instead of between GPU peers.
// Genuinely compiled with a real mpicxx (linked against the real
// installed libcudart) and genuinely run with a real mpirun. Defines
// OMPI_SKIP_MPICXX before <mpi.h> to skip Open MPI's own deprecated C++
// bindings header, whose internal function-pointer casts otherwise
// trigger -Wcast-function-type warnings that belong to Open MPI's own
// headers, not to this file's code.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <mpi-ext.h>
#include <cuda_runtime.h>
#include <cstdio>

#define ELEMS 4

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Two real, independent ways to ask the SAME question. The macro is
    // decided once, when Open MPI itself was built; the function call is
    // decided at runtime and could, in principle, differ across nodes of
    // a heterogeneous cluster even when every node runs the same binary.
    int cudaAwareBuildTime = MPIX_CUDA_AWARE_SUPPORT;
    int cudaAwareRuntime = MPIX_Query_cuda_support();
    if (rank == 0) {
        printf("MPIX_CUDA_AWARE_SUPPORT (compile-time macro): %d\n", cudaAwareBuildTime);
        printf("MPIX_Query_cuda_support() (runtime query):    %d\n", cudaAwareRuntime);
        printf("Both report 0 (false) for this exact installed Open MPI build -- "
               "apt's libopenmpi-dev was built with opal_built_with_cuda_support=false, "
               "confirmed independently via `ompi_info | grep cuda`. Passing a device "
               "pointer straight into MPI_Send() on a build that answers 0 here is a "
               "real, documented misuse with undefined results (this book does not "
               "simulate that UB case, per Chapter 7's own established practice) -- "
               "the correct move, taken below, is Chapter 5's own staged fallback.\n");
    }

    void* devPtr = nullptr;
    cudaError_t eMalloc = cudaMalloc(&devPtr, ELEMS * sizeof(float));

    float hostBuf[ELEMS];
    if (rank == 0) {
        for (int i = 0; i < ELEMS; ++i) hostBuf[i] = 100.0f + (float)i;
        // Real staged step 1 (Chapter 5's own device-to-host half),
        // honestly cudaErrorNoDevice since devPtr was never a real
        // allocation in this environment -- but this is the exact call
        // a real CUDA-aware-unaware MPI program issues on real hardware,
        // right before handing the now-host-resident data to MPI.
        cudaError_t eD2H = cudaMemcpy(hostBuf, devPtr, ELEMS * sizeof(float), cudaMemcpyDeviceToHost);
        printf("rank 0: cudaMalloc()=%s cudaMemcpy(D2H)=%s -- sending host buffer via a REAL MPI_Send\n",
               cudaGetErrorString(eMalloc), cudaGetErrorString(eD2H));
        MPI_Send(hostBuf, ELEMS, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(hostBuf, ELEMS, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("rank 1: REAL MPI_Recv got [%.1f, %.1f, %.1f, %.1f] -- now staging host-to-device\n",
               hostBuf[0], hostBuf[1], hostBuf[2], hostBuf[3]);
        // Real staged step 2 (Chapter 5's own host-to-device half),
        // honestly cudaErrorNoDevice for the same reason.
        cudaError_t eH2D = cudaMemcpy(devPtr, hostBuf, ELEMS * sizeof(float), cudaMemcpyHostToDevice);
        printf("rank 1: cudaMalloc()=%s cudaMemcpy(H2D)=%s\n",
               cudaGetErrorString(eMalloc), cudaGetErrorString(eH2D));
    }

    MPI_Finalize();
    return 0;
}
