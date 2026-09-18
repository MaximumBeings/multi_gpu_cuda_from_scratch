// Chapter 7: CUDA Inter-Process Communication -- Sharing Memory and
// Events Across Processes
// 17_ipc_mem_handle.cu
//
// The real IPC memory-sharing sequence: export a device allocation
// with cudaIpcGetMemHandle(), open the resulting opaque handle with
// cudaIpcOpenMemHandle() (as another process would), and release it
// with cudaIpcCloseMemHandle(). Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    void* devPtr = nullptr;
    cudaError_t eAlloc = cudaMalloc(&devPtr, 64);
    printf("cudaMalloc(exported allocation): %s (code %d)\n", cudaGetErrorString(eAlloc), (int)eAlloc);

    cudaIpcMemHandle_t handle;
    cudaError_t eGet = cudaIpcGetMemHandle(&handle, devPtr);
    printf("cudaIpcGetMemHandle: %s (code %d)\n", cudaGetErrorString(eGet), (int)eGet);

    void* importedPtr = nullptr;
    cudaError_t eOpen = cudaIpcOpenMemHandle(&importedPtr, handle, cudaIpcMemLazyEnablePeerAccess);
    printf("cudaIpcOpenMemHandle: %s (code %d)\n", cudaGetErrorString(eOpen), (int)eOpen);

    cudaError_t eClose = cudaIpcCloseMemHandle(importedPtr);
    printf("cudaIpcCloseMemHandle: %s (code %d)\n", cudaGetErrorString(eClose), (int)eClose);

    if (eAlloc == cudaSuccess) cudaFree(devPtr);
    return 0;
}
