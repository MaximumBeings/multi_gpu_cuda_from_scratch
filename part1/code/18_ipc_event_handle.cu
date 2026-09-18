// Chapter 7: CUDA Inter-Process Communication -- Sharing Memory and
// Events Across Processes
// 18_ipc_event_handle.cu
//
// The real IPC event-sharing sequence: create an event with the two
// flags interprocess sharing requires (cudaEventDisableTiming |
// cudaEventInterprocess), export it with cudaIpcGetEventHandle(), and
// open the resulting handle with cudaIpcOpenEventHandle(). Genuinely
// compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaEvent_t event;
    cudaError_t eCreate = cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess);
    printf("cudaEventCreateWithFlags(DisableTiming|Interprocess): %s (code %d)\n",
           cudaGetErrorString(eCreate), (int)eCreate);

    cudaIpcEventHandle_t handle;
    cudaError_t eGet = cudaIpcGetEventHandle(&handle, event);
    printf("cudaIpcGetEventHandle: %s (code %d)\n", cudaGetErrorString(eGet), (int)eGet);

    cudaEvent_t importedEvent;
    cudaError_t eOpen = cudaIpcOpenEventHandle(&importedEvent, handle);
    printf("cudaIpcOpenEventHandle: %s (code %d)\n", cudaGetErrorString(eOpen), (int)eOpen);

    if (eCreate == cudaSuccess) cudaEventDestroy(event);
    if (eOpen == cudaSuccess) cudaEventDestroy(importedEvent);
    return 0;
}
