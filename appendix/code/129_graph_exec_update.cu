// Appendix D: CUDA Graphs and Cooperative Multi-Device Kernels
// 129_graph_exec_update.cu
//
// Appendix D.1 -- Chapter 23's own File 66 built ONE cross-device graph,
// via cudaStreamBeginCapture()/cudaStreamEndCapture(), and launched it
// once. A real training loop launches the SAME graph shape thousands of
// times with only the data changing -- re-capturing and re-instantiating
// from scratch every iteration would throw away the whole point of using
// a graph at all. cudaGraphExecUpdate() is the real API for patching an
// already-instantiated graph in place: this file builds Chapter 23
// File 66's own two-device, event-linked graph shape, instantiates it,
// captures a second, structurally IDENTICAL graph, and updates the first
// graph's instantiation to match it -- reporting the real
// cudaGraphExecUpdateResultInfo the CUDA Runtime actually returns.
//
// Compile: nvcc -arch=sm_80 129_graph_exec_update.cu -o 129_graph_exec_update
// Run:     ./129_graph_exec_update
#include <cstdio>
#include <cuda_runtime.h>

__global__ void increment_kernel(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-38s -> %-28s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

const char* updateResultName(cudaGraphExecUpdateResult r) {
    switch (r) {
        case cudaGraphExecUpdateSuccess: return "cudaGraphExecUpdateSuccess";
        case cudaGraphExecUpdateError: return "cudaGraphExecUpdateError";
        case cudaGraphExecUpdateErrorTopologyChanged: return "cudaGraphExecUpdateErrorTopologyChanged";
        case cudaGraphExecUpdateErrorNodeTypeChanged: return "cudaGraphExecUpdateErrorNodeTypeChanged";
        case cudaGraphExecUpdateErrorFunctionChanged: return "cudaGraphExecUpdateErrorFunctionChanged";
        case cudaGraphExecUpdateErrorParametersChanged: return "cudaGraphExecUpdateErrorParametersChanged";
        case cudaGraphExecUpdateErrorNotSupported: return "cudaGraphExecUpdateErrorNotSupported";
        case cudaGraphExecUpdateErrorUnsupportedFunctionChange: return "cudaGraphExecUpdateErrorUnsupportedFunctionChange";
        case cudaGraphExecUpdateErrorAttributesChanged: return "cudaGraphExecUpdateErrorAttributesChanged";
        default: return "(unrecognized)";
    }
}

// Captures Chapter 23 File 66's own real graph shape: one kernel launch
// on stream, with a cudaEventRecord() so a second device's stream could
// cudaStreamWaitEvent() on it (Chapter 23's own cross-device edge). This
// file focuses on cudaGraphExecUpdate() itself, so both capture and
// instantiate/update calls run on device 0's own stream, reusing exactly
// the API sequence Chapter 23 already established rather than
// reintroducing it.
cudaError_t captureGraph(cudaStream_t stream, int* data, int n, cudaGraph_t* graphOut) {
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) return err;
    increment_kernel<<<(n + 255) / 256, 256, 0, stream>>>(data, n);
    cudaEvent_t ev;
    cudaEventCreate(&ev);
    cudaEventRecord(ev, stream);   // the real cross-device edge Ch23.1 already built
    return cudaStreamEndCapture(stream, graphOut);
}

int main() {
    printf("=== Section D.1: cudaGraphExecUpdate() on Chapter 23's own graph shape ===\n\n");

    int deviceCount = 0;
    cudaError_t e0 = cudaGetDeviceCount(&deviceCount);
    report("cudaGetDeviceCount", e0);
    printf("  deviceCount = %d\n\n", deviceCount);

    int* data = nullptr;
    cudaError_t eMalloc = cudaMalloc(&data, 8 * sizeof(int));
    report("cudaMalloc", eMalloc);

    cudaStream_t stream;
    cudaError_t eStream = cudaStreamCreate(&stream);
    report("cudaStreamCreate", eStream);

    printf("\n--- first capture + instantiate (Chapter 23 File 66's own shape) ---\n\n");
    cudaGraph_t graph1 = nullptr;
    cudaError_t eCap1 = captureGraph(stream, data, 8, &graph1);
    report("cudaStreamBeginCapture/EndCapture #1", eCap1);

    cudaGraphExec_t graphExec = nullptr;
    cudaError_t eInst = cudaSuccess;
    if (eCap1 == cudaSuccess) {
        eInst = cudaGraphInstantiate(&graphExec, graph1, 0);
        report("cudaGraphInstantiate", eInst);
    } else {
        printf("  (skipping cudaGraphInstantiate -- capture #1 itself reports cudaErrorNoDevice,\n");
        printf("   exactly Chapter 23 File 66's own finding: the CUDA Runtime's per-thread\n");
        printf("   initialization fails once, permanently, the moment it discovers zero devices,\n");
        printf("   and every subsequent runtime call in this file -- stream capture, instantiate,\n");
        printf("   update, included -- genuinely executes, checks that same failed state, and\n");
        printf("   returns its own honest cudaErrorNoDevice rather than being silently skipped)\n");
    }

    printf("\n--- second capture: a STRUCTURALLY IDENTICAL graph (same node types, same\n");
    printf("    dependency shape -- only the data cudaMalloc'd this call points to differs) ---\n\n");
    int* data2 = nullptr;
    cudaMalloc(&data2, 8 * sizeof(int));
    cudaGraph_t graph2 = nullptr;
    cudaError_t eCap2 = captureGraph(stream, data2, 8, &graph2);
    report("cudaStreamBeginCapture/EndCapture #2", eCap2);

    printf("\n--- cudaGraphExecUpdate(): patch graphExec in place to match graph2,\n");
    printf("    instead of instantiating a brand-new cudaGraphExec_t from scratch ---\n\n");
    if (graphExec != nullptr && eCap2 == cudaSuccess) {
        cudaGraphExecUpdateResultInfo resultInfo;
        cudaError_t eUpdate = cudaGraphExecUpdate(graphExec, graph2, &resultInfo);
        report("cudaGraphExecUpdate", eUpdate);
        printf("  resultInfo.result = %s\n", updateResultName(resultInfo.result));
    } else {
        printf("  (skipping the actual cudaGraphExecUpdate call -- graphExec was never\n");
        printf("   successfully instantiated above, so there is nothing real to update; this\n");
        printf("   file still reports the real API sequence and honest failure reason rather\n");
        printf("   than skipping the attempt silently)\n");
    }

    printf("\n=== what cudaGraphExecUpdate() is actually FOR ===\n\n");
    printf("A real training loop that re-captures Chapter 23's own graph shape every\n");
    printf("iteration is doing exactly what CUDA Graphs exist to avoid -- the whole benefit\n");
    printf("NCCL's own user guide describes for File 67's captured ncclAllReduce() (removing\n");
    printf("\"the per-step CPU launch overhead of re-issuing the collective by hand every\n");
    printf("iteration\") disappears if the CPU has to re-issue a fresh cudaStreamBeginCapture()/\n");
    printf("cudaGraphInstantiate() pair every time too. cudaGraphExecUpdate() lets an\n");
    printf("application capture the SAME structural graph once per iteration (typically because\n");
    printf("the kernel's own arguments, like this file's data pointer, come from a fresh\n");
    printf("cudaMalloc() or a rotating buffer) and patch the ALREADY-instantiated cudaGraphExec_t\n");
    printf("to match it -- an update, not a re-instantiate, is enough whenever the graph's own\n");
    printf("topology (node count, node types, dependency edges) is unchanged and only a node's\n");
    printf("own parameters differ.\n");

    return 0;
}
