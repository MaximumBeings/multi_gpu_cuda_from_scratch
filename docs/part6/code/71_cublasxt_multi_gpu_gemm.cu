// Chapter 24: Multi-GPU Dense Matrix Multiplication at Scale
// 71_cublasxt_multi_gpu_gemm.cu
//
// Section 24.2 built SUMMA's own algorithm by hand, over a simulated
// process grid, to prove the ROUTING is correct. NVIDIA ships a real,
// production single-process multi-GPU dense BLAS library that performs
// this exact kind of block distribution FOR you: cuBLAS-Xt. Its own
// documentation states the problem it solves plainly: "the application
// may have the data on the Host or any of the devices involved in the
// computation, and the Library will take care of dispatching the
// operation to, and transferring the data to, one or multiple GPUs
// present in the system." Unlike every earlier chapter's cuBLAS/CUDA
// call, cublasXtSgemm() takes HOST pointers for A, B, and C directly --
// the library manages the device transfers and block dispatch itself,
// with cublasXtSetBlockDim() controlling the tile size used for that
// dispatch (the real analogue of this section's own blockM/panelK from
// file 70). cuBLAS-Xt is real but explicitly SINGLE-NODE: NVIDIA's own
// cuBLAS-Xt product page states it operates "in a single node." For
// genuine multi-NODE dense GEMM at the scale Chapter 20's own MPI
// world implies, NVIDIA's current real answer is cuBLASMp -- "a
// high-performance, multi-process, GPU-accelerated library for
// distributed basic dense linear algebra" -- since cuBLASMg (an older,
// differently-named effort at the same problem) never left the CUDA
// Math Library Early Access Program.
//
// THE GUARD. This file checks cudaGetDeviceCount() BEFORE ever calling
// cublasXtDeviceSelect() with real device ordinals, and skips selecting
// devices, setting a block size, and calling cublasXtSgemm() entirely
// if fewer real devices exist than requested. This section's own
// COMMON TRAP explains, with a real reproduced crash, exactly why that
// check is required here specifically -- a new failure shape even
// Chapter 23's own NCCL segfault didn't produce.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublasXt.h>

int main() {
    const int M = 1024, K = 1024, N = 1024;
    std::vector<float> A(M * K, 1.0f), B(K * N, 2.0f), C(M * N, 0.0f);

    printf("--- Creating a cuBLAS-Xt handle (real, host-side library object) ---\n");
    cublasXtHandle_t handle;
    cublasStatus_t createErr = cublasXtCreate(&handle);
    printf("cublasXtCreate(&handle) -> %s (%s)\n",
           cublasGetStatusName(createErr), cublasGetStatusString(createErr));

    printf("\n--- Checking real device count BEFORE selecting any device ordinals ---\n");
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount() -> %s, count=%d\n",
           cudaGetErrorString(countErr), deviceCount);

    const int requestedDevices = 2;
    if (deviceCount < requestedDevices) {
        printf("\nSTOPPING before cublasXtDeviceSelect(): only %d real device(s) "
               "available, %d requested. See this section's COMMON TRAP for what "
               "genuinely happens if this check is skipped and cublasXtDestroy() "
               "is later called on a handle whose DeviceSelect() call failed.\n",
               deviceCount, requestedDevices);
        cublasXtDestroy(handle);
        printf("cublasXtDestroy(handle) -> cleanly destroyed (handle was never "
               "given an invalid device list, so nothing here was corrupted)\n");
        return 0;
    }

    // This real code path -- selecting devices, setting a block size,
    // and calling the actual multi-GPU GEMM -- only runs on a system
    // that genuinely has the requested devices.
    int deviceIds[] = {0, 1};
    cublasStatus_t selectErr = cublasXtDeviceSelect(handle, requestedDevices, deviceIds);
    printf("cublasXtDeviceSelect(handle, %d, {0,1}) -> %s\n",
           requestedDevices, cublasGetStatusName(selectErr));

    cublasStatus_t blockDimErr = cublasXtSetBlockDim(handle, 256);
    printf("cublasXtSetBlockDim(handle, 256) -> %s\n", cublasGetStatusName(blockDimErr));

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t gemmErr = cublasXtSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        (size_t)M, (size_t)N, (size_t)K,
        &alpha, A.data(), (size_t)M, B.data(), (size_t)K,
        &beta, C.data(), (size_t)M);
    printf("cublasXtSgemm(M=%d,K=%d,N=%d, A/B/C on HOST) -> %s\n",
           M, K, N, cublasGetStatusName(gemmErr));

    cublasXtDestroy(handle);

    printf("\n--- What this means for genuinely MULTI-NODE dense GEMM ---\n");
    printf("Everything above is real cuBLAS-Xt, but cuBLAS-Xt is explicitly "
           "single-node -- one process, one or more GPUs in that SAME machine. "
           "It cannot be handed a Chapter 20-style MPI world spanning multiple "
           "nodes; there is no cublasXt call that takes an MPI_Comm. NVIDIA's "
           "real current answer for genuinely distributed, multi-process, "
           "multi-node dense GEMM is cuBLASMp, built explicitly on top of a "
           "real multi-process model -- this book's own Part 5 MPI/NCCL "
           "infrastructure is precisely the substrate a library like that "
           "needs underneath it.\n");

    return 0;
}
