// Chapter 23: CUDA Graphs Across Multiple GPUs and Multiple Nodes
// 68_host_node_and_multi_process_graphs.cu
//
// Files 66-67 captured real device/collective work into a graph via
// stream capture. This section asks the multi-NODE version of the same
// question directly: can a single CUDA graph object span more than one
// process, the way file 66's graph spanned two devices in ONE process?
// The CUDA Graphs API itself answers this by omission -- there is no
// API to hand a cudaGraph_t handle to a different process, no
// "cudaGraphSend"/"cudaGraphRecv." A graph is a host-side object built
// and instantiated inside one process's address space; nothing in the
// Runtime API surface lets that object, or its instantiated executable
// form, cross a process boundary at all.
//
// A CUDA GRAPH HOST NODE (cudaGraphAddHostNode) looks, at first glance,
// like it might be the way around this: a node in the graph that runs
// arbitrary CPU code (an MPI_Send/MPI_Recv call, say) when the graph
// executes. This section builds one for real and shows exactly how far
// that idea gets. The CUDA Runtime API's own restriction on the host
// function is explicit: per the closely related cudaStreamAddCallback
// (whose restriction cudaGraphAddHostNode's own documentation points
// back to), "no CUDA function may be called from callback," and
// separately, for the modern cudaLaunchHostFunc that shares the exact
// same cudaHostFn_t signature as a host node's own callback, "The host
// function must not make any CUDA API calls." A host node can run
// ordinary CPU logic (including a real MPI call) -- it can NOT be used
// to smuggle a second GPU's or a second process's CUDA work into this
// process's own graph.
#include <cstdio>
#include <cuda_runtime.h>

int hostNodeCallCount = 0;

void CUDART_CB hostNodeFn(void *userData) {
    hostNodeCallCount++;
    printf("  [host node fired -- this is where a real MPI_Send()/MPI_Recv() "
           "call could go, since this is ordinary CPU code, not a CUDA call]\n");
}

int main() {
    printf("--- Building a graph object and a host node (no device touched yet) ---\n");
    cudaGraph_t graph;
    cudaError_t createErr = cudaGraphCreate(&graph, 0);
    printf("cudaGraphCreate(&graph, 0) -> %s\n", cudaGetErrorString(createErr));

    cudaHostNodeParams hostParams;
    hostParams.fn = hostNodeFn;
    hostParams.userData = nullptr;

    cudaGraphNode_t hostNode;
    cudaError_t hostNodeErr = cudaGraphAddHostNode(&hostNode, graph, nullptr, 0, &hostParams);
    printf("cudaGraphAddHostNode(&hostNode, graph, ...) -> %s\n",
           cudaGetErrorString(hostNodeErr));
    printf("Both calls above look like pure host-side bookkeeping -- a graph "
           "object and a CPU-only callback node, nothing GPU-shaped at all -- "
           "but they honestly fail the SAME way file 66's device-touching calls "
           "did. The CUDA Runtime API implicitly initializes its driver context "
           "on the first real runtime call from a thread, exactly as it has "
           "since Chapter 3, and with zero devices present that implicit "
           "initialization fails before cudaGraphCreate() ever gets to do its "
           "own host-side work.\n");

    printf("\n--- Instantiating and launching: this DOES need a device ---\n");
    cudaGraphExec_t graphExec;
    cudaError_t instErr = cudaGraphInstantiate(&graphExec, graph, 0);
    printf("cudaGraphInstantiate(&graphExec, graph) -> %s\n", cudaGetErrorString(instErr));

    cudaStream_t stream;
    cudaError_t streamErr = cudaStreamCreate(&stream);
    printf("cudaStreamCreate(&stream) -> %s\n", cudaGetErrorString(streamErr));

    cudaError_t launchErr = cudaGraphLaunch(graphExec, stream);
    printf("cudaGraphLaunch(graphExec, stream) -> %s\n", cudaGetErrorString(launchErr));
    printf("hostNodeCallCount = %d (the host node never actually fired -- launch "
           "itself never reached a real device to run the graph on)\n",
           hostNodeCallCount);

    printf("\n--- What this means for MULTI-NODE scaling with CUDA graphs ---\n");
    printf("A host node could run a real MPI call when a graph reaches it, but "
           "every real per-rank graph in a genuine multi-node job (Chapter 20's "
           "own MPI ranks, Chapter 21's own hybrid MPI+NCCL bootstrap) is still "
           "built, instantiated, and launched entirely LOCALLY, inside that one "
           "rank's own process -- the graph object itself never crosses a rank "
           "boundary. What DOES cross ranks, exactly as files 66-67 showed for "
           "devices, is coordination through already-existing primitives: an "
           "NCCL collective captured INSIDE each rank's own local graph (file "
           "67), whose communicator was bootstrapped across ranks by real MPI "
           "(Chapter 20/21) before any graph existed at all.\n");

    return 0;
}
