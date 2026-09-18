// Chapter 23: CUDA Graphs Across Multiple GPUs and Multiple Nodes
// 66_cross_device_graph_capture.cu
//
// Every earlier chapter's device work was either one call at a time (Ch3
// onward) or one collective at a time (Ch8 onward) -- the CPU issues each
// operation, waits or moves on, then issues the next one. This section
// builds something structurally different: ONE captured CUDA graph that
// contains real work issued on TWO different devices' streams, joined by
// a real cross-stream dependency. The CUDA Programming Guide states this
// plainly: "Stream capture can handle cross-stream dependencies expressed
// with cudaEventRecord() and cudaStreamWaitEvent(), provided the event
// being waited upon was recorded into the same capture graph." NVIDIA's
// own "Getting Started with CUDA Graphs" post is more direct still:
// "graphs may also span multiple GPUs."
#include <cstdio>
#include <cuda_runtime.h>

__global__ void incrementKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1;
}

int main() {
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount() -> %s, count=%d\n",
           cudaGetErrorString(countErr), deviceCount);

    // The origin stream: cudaStreamBeginCapture() is called here, and per
    // the Programming Guide's own rule, cudaStreamEndCapture() must be
    // called on THIS SAME stream, even though device 1's own stream will
    // also be capturing work as part of the same graph.
    cudaStream_t streamDev0, streamDev1;
    cudaEvent_t dev0Done;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    printf("\n--- Building device 0's stream and event ---\n");
    cudaError_t e1 = cudaSetDevice(0);
    printf("cudaSetDevice(0) -> %s\n", cudaGetErrorString(e1));
    cudaError_t e2 = cudaStreamCreate(&streamDev0);
    printf("cudaStreamCreate(&streamDev0) -> %s\n", cudaGetErrorString(e2));
    cudaError_t e3 = cudaEventCreate(&dev0Done);
    printf("cudaEventCreate(&dev0Done) -> %s\n", cudaGetErrorString(e3));

    printf("\n--- Building device 1's stream ---\n");
    cudaError_t e4 = cudaSetDevice(1);
    printf("cudaSetDevice(1) -> %s\n", cudaGetErrorString(e4));
    cudaError_t e5 = cudaStreamCreate(&streamDev1);
    printf("cudaStreamCreate(&streamDev1) -> %s\n", cudaGetErrorString(e5));

    printf("\n--- Beginning capture on device 0's stream (the origin stream) ---\n");
    cudaError_t e6 = cudaSetDevice(0);
    printf("cudaSetDevice(0) -> %s\n", cudaGetErrorString(e6));
    cudaError_t e7 = cudaStreamBeginCapture(streamDev0, cudaStreamCaptureModeGlobal);
    printf("cudaStreamBeginCapture(streamDev0, Global) -> %s\n", cudaGetErrorString(e7));

    int *devBuf0 = nullptr;
    cudaError_t e8 = cudaMallocAsync(&devBuf0, sizeof(int) * 256, streamDev0);
    printf("cudaMallocAsync(devBuf0, streamDev0) -> %s\n", cudaGetErrorString(e8));
    incrementKernel<<<1, 256, 0, streamDev0>>>(devBuf0, 256);
    printf("incrementKernel<<<...,streamDev0>>>() enqueued (device 0's own work)\n");

    // The cross-stream/cross-device edge: device 0's stream records an
    // event; device 1's stream waits on it. Per the Programming Guide,
    // this event MUST be recorded inside the same capture sequence for
    // the dependency to become a real graph edge rather than a runtime
    // error.
    cudaError_t e9 = cudaEventRecord(dev0Done, streamDev0);
    printf("cudaEventRecord(dev0Done, streamDev0) -> %s\n", cudaGetErrorString(e9));

    cudaError_t e10 = cudaSetDevice(1);
    printf("cudaSetDevice(1) -> %s\n", cudaGetErrorString(e10));
    cudaError_t e11 = cudaStreamWaitEvent(streamDev1, dev0Done, 0);
    printf("cudaStreamWaitEvent(streamDev1, dev0Done) -> %s "
           "(this is the actual cross-device graph edge)\n",
           cudaGetErrorString(e11));

    int *devBuf1 = nullptr;
    cudaError_t e12 = cudaMallocAsync(&devBuf1, sizeof(int) * 256, streamDev1);
    printf("cudaMallocAsync(devBuf1, streamDev1) -> %s\n", cudaGetErrorString(e12));
    // A real peer-to-peer style transfer: device 1 waits for device 0's
    // buffer, then would copy it across -- reusing Chapter 5's own
    // cudaMemcpyPeerAsync(), now issued INSIDE a captured graph rather
    // than as a standalone call.
    cudaError_t e13 = cudaMemcpyPeerAsync(devBuf1, 1, devBuf0, 0,
                                          sizeof(int) * 256, streamDev1);
    printf("cudaMemcpyPeerAsync(devBuf1<-devBuf0, streamDev1) -> %s\n",
           cudaGetErrorString(e13));
    incrementKernel<<<1, 256, 0, streamDev1>>>(devBuf1, 256);
    printf("incrementKernel<<<...,streamDev1>>>() enqueued (device 1's own work)\n");

    printf("\n--- Ending capture on the origin stream (streamDev0) ---\n");
    cudaError_t e14 = cudaSetDevice(0);
    cudaError_t e15 = cudaStreamEndCapture(streamDev0, &graph);
    printf("cudaSetDevice(0) -> %s\n", cudaGetErrorString(e14));
    printf("cudaStreamEndCapture(streamDev0, &graph) -> %s\n", cudaGetErrorString(e15));

    printf("\n--- Instantiating and launching the graph ---\n");
    cudaError_t e16 = cudaGraphInstantiate(&graphExec, graph, 0);
    printf("cudaGraphInstantiate(&graphExec, graph) -> %s\n", cudaGetErrorString(e16));
    cudaError_t e17 = cudaGraphLaunch(graphExec, streamDev0);
    printf("cudaGraphLaunch(graphExec, streamDev0) -> %s\n", cudaGetErrorString(e17));
    cudaError_t e18 = cudaStreamSynchronize(streamDev0);
    printf("cudaStreamSynchronize(streamDev0) -> %s\n", cudaGetErrorString(e18));

    printf("\n--- Relaunching the SAME instantiated graph a second time ---\n");
    cudaError_t e19 = cudaGraphLaunch(graphExec, streamDev0);
    printf("cudaGraphLaunch(graphExec, streamDev0) [2nd launch, no re-capture] -> %s\n",
           cudaGetErrorString(e19));

    return 0;
}
