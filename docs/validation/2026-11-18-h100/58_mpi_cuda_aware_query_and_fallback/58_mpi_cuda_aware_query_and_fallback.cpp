// Chapter 20: checked host-staged MPI transport between GPU-backed ranks.
// CUDA-aware MPI capability is queried, but this example always uses host staging.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <mpi-ext.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static void mpiCheck(int error, const char* call) {
    if (error == MPI_SUCCESS) return;
    char text[MPI_MAX_ERROR_STRING]; int length = 0;
    MPI_Error_string(error, text, &length);
    std::fprintf(stderr, "%s: %.*s\n", call, length, text);
    MPI_Abort(MPI_COMM_WORLD, 1); std::exit(1);
}
static void cudaCheck(cudaError_t error, const char* call) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", call, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1); std::exit(1);
}
#define MPI_CHECK(call) mpiCheck((call), #call)
#define CUDA_CHECK(call) cudaCheck((call), #call)

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (ranks < 2) {
        std::fprintf(stderr, "Needs at least 2 MPI ranks\n");
        MPI_Abort(MPI_COMM_WORLD, 2); return 2;
    }
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                 MPI_INFO_NULL, &local));
    int localRank = 0, devices = 0;
    MPI_CHECK(MPI_Comm_rank(local, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices < 1) { MPI_Abort(MPI_COMM_WORLD, 2); return 2; }
    // Explicit round-robin binding also supports more local ranks than GPUs.
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    if (rank == 0)
        std::printf("CUDA-aware MPI: build=%d runtime=%d; using host staging\n",
                    MPIX_CUDA_AWARE_SUPPORT, MPIX_Query_cuda_support());

    constexpr int elements = 4096;
    float* device = nullptr;
    CUDA_CHECK(cudaMalloc(&device, elements * sizeof(float)));
    float host[elements];
    // Rank-specific, exactly representable values make corruption observable.
    for (int i = 0; i < elements; ++i)
        host[i] = float((i * 17 + rank * 13) % 1021) * 0.25f - 100.0f;
    CUDA_CHECK(cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost));
    // Every rank sends its initialized payload to its successor.
    const int previous = (rank + ranks - 1) % ranks;
    float received[elements];
    MPI_CHECK(MPI_Sendrecv(host, elements, MPI_FLOAT, (rank + 1) % ranks, 0,
                           received, elements, MPI_FLOAT, previous, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    CUDA_CHECK(cudaMemcpy(device, received, sizeof(received), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(received, device, sizeof(received), cudaMemcpyDeviceToHost));
    int errors = 0;
    for (int i = 0; i < elements; ++i) {
        const float expected = float((i * 17 + previous * 13) % 1021) * 0.25f - 100.0f;
        if (!std::isfinite(received[i]) || received[i] != expected) ++errors;
    }
    int totalErrors = 0;
    MPI_CHECK(MPI_Allreduce(&errors, &totalErrors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFree(device));
    MPI_CHECK(MPI_Comm_free(&local));
    if (rank == 0)
        std::printf("VALIDATION %s: host-staged MPI, %d ranks, 4096 values/rank, errors=%d\n",
                    totalErrors ? "FAIL" : "PASS", ranks, totalErrors);
    MPI_CHECK(MPI_Finalize());
    return totalErrors ? 1 : 0;
}
