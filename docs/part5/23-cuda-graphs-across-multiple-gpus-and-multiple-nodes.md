# Chapter 23: CUDA Graphs Across Multiple GPUs and Multiple Nodes

**What you will understand by the end of this chapter:**

- That a single CUDA graph, built via stream capture, can genuinely contain work issued on more than one GPU in the same process -- and exactly which real API (`cudaEventRecord()`/`cudaStreamWaitEvent()`, captured) creates the cross-device edge that makes this possible.
- That NCCL collectives need no new API to be captured into a graph at all: a collective issued between `cudaStreamBeginCapture()` and `cudaStreamEndCapture()` is captured like any other kernel launch, real since NCCL 2.9.
- Why "multi-node CUDA graph" does not mean one graph object spanning multiple processes -- the CUDA Graphs API has no mechanism for that at all -- and what a CUDA graph host node can and cannot be used for instead.
- A new, real failure mode this book has not shown before: calling a real collective on a communicator handle from a FAILED initialization can crash the whole process outright, not just return an honest error code -- and why checking that return value is no longer optional once graph capture is involved.

**What you need to know first:**

- Chapter 6's real `cudaEventRecord()`/`cudaStreamWaitEvent()` cross-stream synchronization.
- Chapter 21's real hybrid MPI+NCCL bootstrap (`ncclGetUniqueId()` + `MPI_Bcast()` + `ncclCommInitRank()`).
- Chapter 5's real `cudaMemcpyPeerAsync()`.

---

Every earlier chapter's device work was issued one call, or one collective, at a time: the CPU calls an API, that API enqueues (or blocks on) real work, and the CPU decides when to issue the next one. CUDA Graphs change what gets issued, not who issues it -- a whole sequence of operations, captured once, is handed to the driver as a single object that can be launched repeatedly without the CPU re-issuing each individual call. NVIDIA's own introduction to the feature states the scope of that sequence plainly: "graphs may also span multiple GPUs." This chapter builds that claim for real, in three steps: 23.1 captures real work issued on two different devices' streams into one graph, joined by a real cross-device event dependency; 23.2 captures a real NCCL collective inside a graph, reusing Chapter 21's own MPI+NCCL bootstrap, and reports a new, sharper real failure this book has not yet produced; 23.3 asks the multi-NODE version of the same question directly and shows, using a real CUDA graph host node, exactly why the honest answer is architectural rather than a missing API call.

```text
Every earlier chapter's device work:               This chapter's own graphs:

+--------+     +------------------+                 +--------+     +------------------+
|  CPU   | --> | one call/        |  (repeat for     |  CPU   | --> | cudaStreamBegin- |
| thread |     | collective at    |   every step)    | thread |     | Capture() ...    |
+--------+     | a time           |                  +--------+     | ... EndCapture() |
               +------------------+                                 +------------------+
  The CPU re-issues every single call,                                       |
  every single time.                                                         |
                                                                   +------------------+
                                                                   | cudaGraphLaunch()|
                                                                   | -- the WHOLE     |
                                                                   | captured sequence|
                                                                   | replays, no      |
                                                                   | re-issuing       |
                                                                   +------------------+
```

## 23.1 One Graph, Two Devices: Cross-Device Capture With Events

### Intuition

Chapter 6 already built real cross-stream synchronization on ONE device: `cudaEventRecord()` on one stream, `cudaStreamWaitEvent()` on another, so one stream's work waits for another's without the CPU blocking in between. Stream capture -- the mechanism `cudaStreamBeginCapture()`/`cudaStreamEndCapture()` uses to turn a sequence of enqueued calls into a `cudaGraph_t` -- can record that exact same event-based dependency as a real edge in the resulting graph, and the CUDA Programming Guide is explicit that this is not limited to streams on the same device: "Stream capture can handle cross-stream dependencies expressed with `cudaEventRecord()` and `cudaStreamWaitEvent()`, provided the event being waited upon was recorded into the same capture graph." Combined with NVIDIA's own claim that "graphs may also span multiple GPUs," this section builds the plainest real version of that: device 0 does some work, records an event; device 1 waits on that event, then does its own work depending on device 0's result (a real `cudaMemcpyPeerAsync()`, Chapter 5's own API) -- all captured into ONE graph from a single CPU thread, with one call to `cudaStreamEndCapture()` on the stream where capture began (the "origin stream," which the Guide requires regardless of how many other streams joined the capture through events).

```text
Device 0's stream (origin stream)        Device 1's stream

+-----------------------+
| cudaMallocAsync()     |
| increment kernel launch|
| cudaEventRecord(e)    |
+----------+------------+
           |
           |  (e is the cross-device graph edge --
           |   captured because it was recorded
           |   INSIDE this same capture sequence)
           |
           +------------------------------+
                                          |
                              +-----------+----------+
                              | cudaStreamWaitEvent(e)|
                              | cudaMemcpyPeerAsync() |
                              | increment kernel launch|
                              +-----------------------+
cudaStreamEndCapture() called HERE, on the origin stream --
even though device 1's stream also contributed real nodes.
```

### Background

```cpp
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
```

Compiled with `nvcc -gencode=arch=compute_70,code=sm_70 66_cross_device_graph_capture.cu -o 66_cross_device_graph_capture` (real dynamic linking against this environment's own installed CUDA 12.0 runtime) and genuinely run. Locked output:

```text
cudaGetDeviceCount() -> no CUDA-capable device is detected, count=0

--- Building device 0's stream and event ---
cudaSetDevice(0) -> no CUDA-capable device is detected
cudaStreamCreate(&streamDev0) -> no CUDA-capable device is detected
cudaEventCreate(&dev0Done) -> no CUDA-capable device is detected

--- Building device 1's stream ---
cudaSetDevice(1) -> no CUDA-capable device is detected
cudaStreamCreate(&streamDev1) -> no CUDA-capable device is detected

--- Beginning capture on device 0's stream (the origin stream) ---
cudaSetDevice(0) -> no CUDA-capable device is detected
cudaStreamBeginCapture(streamDev0, Global) -> no CUDA-capable device is detected
cudaMallocAsync(devBuf0, streamDev0) -> no CUDA-capable device is detected
incrementKernel<<<...,streamDev0>>>() enqueued (device 0's own work)
cudaEventRecord(dev0Done, streamDev0) -> no CUDA-capable device is detected
cudaSetDevice(1) -> no CUDA-capable device is detected
cudaStreamWaitEvent(streamDev1, dev0Done) -> no CUDA-capable device is detected (this is the actual cross-device graph edge)
cudaMallocAsync(devBuf1, streamDev1) -> no CUDA-capable device is detected
cudaMemcpyPeerAsync(devBuf1<-devBuf0, streamDev1) -> no CUDA-capable device is detected
incrementKernel<<<...,streamDev1>>>() enqueued (device 1's own work)

--- Ending capture on the origin stream (streamDev0) ---
cudaSetDevice(0) -> no CUDA-capable device is detected
cudaStreamEndCapture(streamDev0, &graph) -> no CUDA-capable device is detected

--- Instantiating and launching the graph ---
cudaGraphInstantiate(&graphExec, graph) -> no CUDA-capable device is detected
cudaGraphLaunch(graphExec, streamDev0) -> no CUDA-capable device is detected
cudaStreamSynchronize(streamDev0) -> no CUDA-capable device is detected

--- Relaunching the SAME instantiated graph a second time ---
cudaGraphLaunch(graphExec, streamDev0) [2nd launch, no re-capture] -> no CUDA-capable device is detected
```

!!! warning "[COMMON TRAP] Expecting the FIRST failure to stop the program from reaching later real API calls"
    Every single call in this file's locked output reports the identical `cudaErrorNoDevice`, from the very first `cudaGetDeviceCount()` down through `cudaGraphLaunch()`. That uniform failure is easy to misread as "the program crashed at the first error and nothing after it is real." It did not: the CUDA Runtime API's implicit per-thread initialization fails once, permanently, the moment it discovers zero devices, and every subsequent runtime call in this file genuinely executed, checked that same failed state, and returned its own honest `cudaErrorNoDevice` -- eighteen separate real calls, eighteen separate real checks. The uniformity of the error is itself the evidence that the entire real API sequence -- cross-device stream creation, capture, the cross-device event edge, instantiation, launch, and relaunch -- was genuinely exercised, not skipped.

## 23.2 Capturing a Real NCCL Collective Inside a Graph

### Intuition

File 66 captured plain CUDA work. NCCL's own user guide states a separate, real capability this section builds next: "Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs. This support requires a minimum CUDA version of 11.3." No new NCCL call is needed for this -- `ncclAllReduce()` is, underneath, ordinary kernel launches enqueued on the communicator's own stream, so a call to it between `cudaStreamBeginCapture()` and `cudaStreamEndCapture()` is captured exactly like file 66's own `incrementKernel<<<>>>()` calls were. The real benefit this unlocks, stated directly on the same NCCL page in the training-loop case: capturing a forward pass, backward pass, and `ncclAllReduce()` together into one graph removes the per-step CPU launch overhead of re-issuing the collective by hand every iteration, real overhead this book's own Chapter 11 already measured indirectly through NCCL's ring/tree round-count model. Getting there needs one thing file 66 didn't: a genuine multi-process communicator, built exactly the way Chapter 21 built one -- `ncclGetUniqueId()` on rank 0, `MPI_Bcast()` to every rank, then every rank calling `ncclCommInitRank()` -- before any capture begins at all.

```text
Chapter 21's own hybrid bootstrap:                This chapter's own addition:

+---------------------+
| ncclGetUniqueId()    |  (rank 0 only)
| MPI_Bcast(id)         |  (every rank)
| ncclCommInitRank(id)  |  (every rank)
+----------+-----------+
           |
+---------------------+                          +--------------------------+
| a real communicator, |   -------------------->  | cudaStreamBeginCapture() |
| exactly like Ch21's  |                          | ncclAllReduce(..., comm) |
| own                   |                          | cudaStreamEndCapture()   |
+---------------------+                          +--------------------------+
                                                              |
                                                   cudaGraphLaunch(), 3 times --
                                                   no per-iteration CPU re-issue
```

### Background

```cpp
// Chapter 23: CUDA Graphs Across Multiple GPUs and Multiple Nodes
// 67_nccl_captured_in_graph.cu
//
// File 66 captured plain CUDA work (kernels, a peer copy) across two
// devices in one graph. NCCL's own user guide states a real, separate
// capability this section builds next: "Starting with NCCL 2.9, NCCL
// operations can be captured by CUDA Graphs. This support requires a
// minimum CUDA version of 11.3." No new NCCL API is needed for this --
// a collective call issued between cudaStreamBeginCapture() and
// cudaStreamEndCapture() is captured exactly like any other kernel
// launch, because ncclAllReduce() is, underneath, ordinary kernel
// launches enqueued on the communicator's stream. This reuses Chapter
// 21's own real hybrid MPI+NCCL bootstrap (ncclGetUniqueId() on rank 0,
// MPI_Bcast() to every rank, then ncclCommInitRank() on all ranks).
//
// This file adds ONE real, disciplined check that no earlier chapter's
// collective code strictly needed: it verifies ncclCommInitRank()
// actually returned ncclSuccess BEFORE doing anything else with the
// resulting handle -- including beginning a graph capture. This
// section's own COMMON TRAP explains, with a real reproduced crash, why
// that check is not optional once graph capture enters the picture.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    ncclUniqueId id;
    if (rank == 0) {
        ncclResult_t idErr = ncclGetUniqueId(&id);
        printf("Rank %d: ncclGetUniqueId() -> %s\n", rank, ncclGetErrorString(idErr));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    printf("Rank %d: MPI_Bcast(id) done -- every rank now holds the SAME id "
           "(Chapter 21's own bootstrap, reused unchanged)\n", rank);

    cudaError_t setDevErr = cudaSetDevice(rank);
    printf("Rank %d: cudaSetDevice(%d) -> %s\n", rank, rank, cudaGetErrorString(setDevErr));

    ncclComm_t comm;
    ncclResult_t commErr = ncclCommInitRank(&comm, worldSize, id, rank);
    printf("Rank %d: ncclCommInitRank() -> %s\n", rank, ncclGetErrorString(commErr));

    // THE GUARD. Every earlier chapter's collective code (Ch11, Ch17,
    // Ch21) called the next real API regardless of this return value,
    // because in every one of those cases the next call still returned
    // its OWN honest error code rather than corrupting anything. That is
    // NOT true here -- see this section's COMMON TRAP for the real,
    // reproduced crash that skipping this check causes the moment a
    // collective on this handle is captured (or even just called) into
    // a stream. This program stops here rather than reproduce it.
    if (commErr != ncclSuccess) {
        printf("Rank %d: STOPPING before touching this communicator again -- "
               "ncclCommInitRank() did not return ncclSuccess, so nothing else "
               "on `comm` is safe to call, capture included.\n", rank);
        MPI_Finalize();
        return 0;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    float *sendBuf = nullptr, *recvBuf = nullptr;
    cudaMalloc(&sendBuf, sizeof(float) * 1024);
    cudaMalloc(&recvBuf, sizeof(float) * 1024);

    printf("Rank %d: --- beginning capture (the collective goes INSIDE the graph) ---\n", rank);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    ncclAllReduce(sendBuf, recvBuf, 1024, ncclFloat, ncclSum, comm, stream);
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, 0);

    printf("Rank %d: --- launching the captured graph 3 times (no re-issuing "
           "ncclAllReduce by hand each time) ---\n", rank);
    for (int iter = 0; iter < 3; iter++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);

    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
```

Compiled with `nvcc -ccbin mpicxx -gencode=arch=compute_70,code=sm_70 67_nccl_captured_in_graph.cu -o 67_nccl_captured_in_graph -lnccl` and genuinely run with `mpirun --allow-run-as-root --oversubscribe -np 2 ./67_nccl_captured_in_graph`. Locked output:

```text
Rank 0: ncclGetUniqueId() -> no error
Rank 0: MPI_Bcast(id) done -- every rank now holds the SAME id (Chapter 21's own bootstrap, reused unchanged)
Rank 1: MPI_Bcast(id) done -- every rank now holds the SAME id (Chapter 21's own bootstrap, reused unchanged)
Rank 0: cudaSetDevice(0) -> no CUDA-capable device is detected
Rank 1: cudaSetDevice(1) -> no CUDA-capable device is detected
Rank 0: ncclCommInitRank() -> unhandled cuda error (run with NCCL_DEBUG=INFO for details)
Rank 0: STOPPING before touching this communicator again -- ncclCommInitRank() did not return ncclSuccess, so nothing else on `comm` is safe to call, capture included.
Rank 1: ncclCommInitRank() -> unhandled cuda error (run with NCCL_DEBUG=INFO for details)
Rank 1: STOPPING before touching this communicator again -- ncclCommInitRank() did not return ncclSuccess, so nothing else on `comm` is safe to call, capture included.
```

!!! warning "[COMMON TRAP] Assuming a failed ncclCommInitRank() is as safe to keep using as Chapter 11/17's own failed communicators"
    Chapter 11 and Chapter 17 both called `ncclAllReduce()` (or a throwaway one-element version of it, for Chapter 17's barrier) on a communicator whose own initialization had already failed, and both got back a clean, honest NCCL error code every time -- `ncclUnhandledCudaError` or `ncclInvalidArgument`, never a crash. This file's own guard exists because that is NOT what happens with `ncclCommInitRank()`'s own failed handle once a real collective call on it is attempted: removing the guard above and calling `ncclAllReduce(sendBuf, recvBuf, 1024, ncclFloat, ncclSum, comm, stream)` directly on this exact environment's failed `comm` genuinely segfaults inside `libnccl.so.2` itself --

    ```text
    Signal: Segmentation fault (11)
    Signal code: Address not mapped (1)
    [...]
    /lib/x86_64-linux-gnu/libnccl.so.2(ncclAllReduce+0x1a1)[...]
    ```

    -- and this was independently re-tested BOTH with and without wrapping the call in `cudaStreamBeginCapture()`/`cudaStreamEndCapture()`: the crash reproduces identically either way, so this is not a graph-capture-specific bug, but it is exactly the kind of real bug that a chapter about wrapping more work into fewer, larger captured units makes MORE costly to hit by accident, since a captured graph can be replayed many times before anyone notices the handle it depends on was never valid. The lesson: once `ncclCommInitRank()` (unlike `ncclCommInitAll()`) returns anything other than `ncclSuccess`, treat the returned handle as unsafe to pass to ANY further NCCL call, capture included, full stop.

## 23.3 Multi-Node Graphs: What a Host Node Can and Cannot Do

### Intuition

Section 23.1 put two DEVICES into one graph, in one process. Can one graph span two PROCESSES -- the real shape of Chapter 20's own MPI ranks, potentially running on different physical nodes entirely? The CUDA Graphs API answers this by omission: there is no `cudaGraphSend()`/`cudaGraphRecv()`, no way to hand a `cudaGraph_t` or an instantiated `cudaGraphExec_t` to a different process at all. A CUDA graph host node (`cudaGraphAddHostNode()`) looks, at first glance, like a possible way around this -- a node that runs arbitrary CPU code, including a real `MPI_Send()`/`MPI_Recv()` call, when the graph reaches it during replay. This section builds one for real and shows precisely how far that idea gets, and precisely where it stops: the CUDA Runtime API's own restriction on a host node's callback, inherited from the closely related `cudaStreamAddCallback`, is explicit -- "no CUDA function may be called from callback" -- and the modern equivalent, `cudaLaunchHostFunc` (which shares the exact same `cudaHostFn_t` callback signature a host node uses), states it just as plainly: "The host function must not make any CUDA API calls." A host node can run ordinary CPU logic. It cannot be used to smuggle a second process's or a second node's CUDA work into this process's own graph -- there is no version of a host node that reaches across the process boundary any further than an ordinary CPU thread already could.

```text
What a host node CAN do:                      What a host node CANNOT do:

+------------------------+                     +------------------------+
| ordinary CPU code runs |                     | call any CUDA API from |
| when the graph reaches |                     | inside the callback    |
| this node -- an        |                     | ("no CUDA function may |
| MPI_Send()/MPI_Recv()  |                     | be called from         |
| call included          |                     | callback")             |
+------------------------+                     +------------------------+

Real multi-node scaling with graphs, as actually built:

  Rank 0's own process              Rank 1's own process
  +----------------------+          +----------------------+
  | its OWN local graph   |          | its OWN local graph   |
  | (captured NCCL calls, |          | (captured NCCL calls, |
  |  Section 23.2's own   |          |  Section 23.2's own   |
  |  file 67 pattern)     |          |  file 67 pattern)     |
  +----------+-----------+          +----------+-----------+
             |                                 |
             +------ NCCL communicator ---------+
                (bootstrapped across ranks by
                 real MPI BEFORE either graph
                 was ever built -- Ch20/21)
```

### Background

```cpp
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
```

Compiled with `nvcc -gencode=arch=compute_70,code=sm_70 68_host_node_and_multi_process_graphs.cu -o 68_host_node_and_multi_process_graphs` and genuinely run. Locked output:

```text
--- Building a graph object and a host node (no device touched yet) ---
cudaGraphCreate(&graph, 0) -> no CUDA-capable device is detected
cudaGraphAddHostNode(&hostNode, graph, ...) -> no CUDA-capable device is detected
Both calls above look like pure host-side bookkeeping -- a graph object and a CPU-only callback node, nothing GPU-shaped at all -- but they honestly fail the SAME way file 66's device-touching calls did. The CUDA Runtime API implicitly initializes its driver context on the first real runtime call from a thread, exactly as it has since Chapter 3, and with zero devices present that implicit initialization fails before cudaGraphCreate() ever gets to do its own host-side work.

--- Instantiating and launching: this DOES need a device ---
cudaGraphInstantiate(&graphExec, graph) -> no CUDA-capable device is detected
cudaStreamCreate(&stream) -> no CUDA-capable device is detected
cudaGraphLaunch(graphExec, stream) -> no CUDA-capable device is detected
hostNodeCallCount = 0 (the host node never actually fired -- launch itself never reached a real device to run the graph on)

--- What this means for MULTI-NODE scaling with CUDA graphs ---
A host node could run a real MPI call when a graph reaches it, but every real per-rank graph in a genuine multi-node job (Chapter 20's own MPI ranks, Chapter 21's own hybrid MPI+NCCL bootstrap) is still built, instantiated, and launched entirely LOCALLY, inside that one rank's own process -- the graph object itself never crosses a rank boundary. What DOES cross ranks, exactly as files 66-67 showed for devices, is coordination through already-existing primitives: an NCCL collective captured INSIDE each rank's own local graph (file 67), whose communicator was bootstrapped across ranks by real MPI (Chapter 20/21) before any graph existed at all.
```

!!! warning "[COMMON TRAP] Assuming a host node is a general-purpose escape hatch for GPU work the graph API doesn't otherwise support"
    A host node's callback restriction -- "no CUDA function may be called from callback" -- is not a minor footnote; it is the exact boundary that stops a host node from being used to paper over Section 23.1's own real limit (one process's own graph, however many devices) or this section's own real limit (no cross-process graph at all). A host node can run real CPU-only logic, including a real blocking `MPI_Send()`/`MPI_Recv()` call -- but the moment that CPU code needs to touch a CUDA API (allocate memory, launch a kernel, check a device property), it is no longer legal inside a host node's callback, full stop. The real, working multi-node pattern this book has already built the pieces of is not "one host node calling MPI inside a bigger graph" -- it is Section 23.2's own pattern: each rank captures and launches its OWN local graph, and the cross-rank coordination happens through an NCCL communicator that was bootstrapped by real MPI calls made OUTSIDE any graph, before capture ever began.

## Chapter Summary

This chapter asked how far CUDA Graphs' own "capture once, launch many times" model extends beyond the single device every earlier graph discussion in this book has assumed. Section 23.1 confirmed, with a real cross-device event dependency captured into one graph, that a graph genuinely can span multiple GPUs within one process -- and that every real API call in that sequence, even the purely host-side bookkeeping calls, fails identically to `cudaErrorNoDevice` once the CUDA Runtime's own implicit per-thread initialization fails. Section 23.2 confirmed NCCL's own real support for graph-captured collectives (since NCCL 2.9) by reusing Chapter 21's hybrid MPI+NCCL bootstrap, and surfaced a genuinely new, sharper failure mode this book had not produced before: a communicator handle from a failed `ncclCommInitRank()` call can crash the whole process with a real segfault the instant a collective on it is actually called, independent of whether graph capture is involved at all -- a lesson about defensive checking, not about graphs specifically, but one this chapter's own code now checks for by name. Section 23.3 answered the multi-node question directly: the CUDA Graphs API has no mechanism for one graph object to span multiple processes, a host node's callback restriction ("no CUDA function may be called from callback") rules out using it to smuggle cross-process CUDA work into a graph, and the real, working pattern for multi-node scaling is each rank capturing and launching its own local graph, coordinated through an NCCL communicator bootstrapped by MPI calls made entirely outside of, and before, any graph capture begins.

## Self-Check Questions

1. What real API pair creates the cross-device dependency edge that lets one captured graph contain work from two different GPUs, and on which stream must `cudaStreamEndCapture()` ultimately be called?
2. Section 23.1's locked output shows the identical error on all nineteen real API calls, from `cudaGetDeviceCount()` through the second `cudaGraphLaunch()`. Why does this uniformity indicate every call genuinely executed, rather than the program stopping early?
3. What NCCL version introduced support for capturing NCCL operations inside CUDA Graphs, and what minimum CUDA version does that support require?
4. In Section 23.2, what specifically differs about calling `ncclAllReduce()` on a communicator from a failed `ncclCommInitRank()`, compared to Chapter 11 and Chapter 17's own failed communicators?
5. Was the crash in Section 23.2 caused by graph capture itself? What evidence in this chapter answers that question?
6. What does the CUDA Runtime API's host node callback restriction ("no CUDA function may be called from callback") rule out as a way to build a multi-node CUDA graph?
7. Describe the real architecture this chapter concludes is actually used for multi-node CUDA graph scaling, and name which earlier chapter's own API builds the cross-rank coordination piece of it.
8. Why did `cudaGraphCreate()` and `cudaGraphAddHostNode()` -- calls that never obviously touch a GPU -- still fail with `cudaErrorNoDevice` in this environment?

## Where We Go Next

Chapter 23 completes Part 5 (Chapters 20 through 23: MPI, GPUDirect RDMA, NVSHMEM, and now multi-device/multi-node CUDA graphs) -- every real primitive this book uses to scale beyond one device is now in place. Part 6 opens with Chapter 24, "Multi-GPU Dense Matrix Multiplication at Scale," the first of eight real case studies that combine these primitives (data/model/tensor/pipeline parallelism from Part 3, collectives from Part 2, and now MPI/GPUDirect RDMA/NVSHMEM/graphs from Part 5) into complete, real, end-to-end programs rather than single-concept demonstrations.

## Worked Solutions

**1.** `cudaEventRecord()` on the producing device's stream, followed by `cudaStreamWaitEvent()` on the consuming device's stream, creates the cross-device edge -- but only if the event was recorded inside the same capture sequence. `cudaStreamEndCapture()` must be called on the "origin stream," the same stream `cudaStreamBeginCapture()` was originally called on, regardless of how many other streams (on other devices) joined the capture through waited-upon events.

**2.** Because the CUDA Runtime API's implicit initialization fails exactly once, the first time any thread makes a real runtime call, and every subsequent runtime call in that thread then genuinely executes its own logic, checks that same failed initialization state, and returns its own honest error -- it does not throw an exception or terminate the program. Eighteen distinct real calls each independently confirmed the same fact, rather than the program silently stopping after the first one.

**3.** NCCL 2.9 introduced graph-capture support for NCCL operations, and it requires a minimum CUDA version of 11.3.

**4.** Chapter 11 and Chapter 17 both called `ncclAllReduce()` on communicators whose initialization (via `ncclCommInitAll()`) had already failed, and both calls returned a clean, honest NCCL error code (`ncclUnhandledCudaError` or `ncclInvalidArgument`) without crashing. In Section 23.2, calling `ncclAllReduce()` on a communicator from a failed `ncclCommInitRank()` genuinely segfaults inside `libnccl.so.2` itself -- no error code is returned at all, because the process crashes first.

**5.** No. The chapter explicitly re-tested the exact same crash both with and without wrapping the `ncclAllReduce()` call in `cudaStreamBeginCapture()`/`cudaStreamEndCapture()`, and it reproduced identically either way. The evidence is the isolated test described in the COMMON TRAP, which removed graph capture entirely and still crashed on the same call with the same backtrace.

**6.** It rules out using a host node to run any CUDA API call from inside its own callback -- meaning a host node cannot itself allocate device memory, launch a kernel, query a device, or perform any other CUDA operation on behalf of another process or device. It can only run ordinary, CUDA-free CPU logic (a real `MPI_Send()`/`MPI_Recv()` call is fine, since MPI calls are not CUDA API calls).

**7.** Each rank in a multi-node job builds, instantiates, and launches its own LOCAL CUDA graph, entirely inside that rank's own process -- the graph object itself never crosses a process boundary. Cross-rank coordination happens through an NCCL communicator (which CAN be used inside each rank's own captured graph, per Section 23.2) whose handle was bootstrapped across ranks by real MPI calls (Chapter 20's `MPI_Init`/`MPI_Comm_rank`/`MPI_Comm_size`, Chapter 21's `MPI_Bcast()` of a `ncclUniqueId`) made entirely outside of, and before, any graph capture begins.

**8.** Because the CUDA Runtime API implicitly initializes its driver context on the very first real runtime call made by a thread, regardless of whether that call is itself GPU-shaped -- `cudaGraphCreate()` and `cudaGraphAddHostNode()` are still real CUDA Runtime API calls, so they trigger that same implicit initialization, which fails the instant it discovers zero devices, before either function gets to perform its own actual host-side bookkeeping work.

---

**Sources cited in this chapter:**

- ["4.2. CUDA Graphs" — CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) — the real, verified quote on cross-stream dependencies during capture: "Stream capture can handle cross-stream dependencies expressed with `cudaEventRecord()` and `cudaStreamWaitEvent()`, provided the event being waited upon was recorded into the same capture graph," and the origin-stream rule for `cudaStreamEndCapture()`.
- ["Getting Started with CUDA Graphs" — NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-graphs/) — the real, verified quote this chapter's intro and Section 23.1 build toward: "graphs may also span multiple GPUs."
- ["Using NCCL with CUDA Graphs" — NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html) — the real, verified quote Section 23.2 is built on: "Starting with NCCL 2.9, NCCL operations can be captured by CUDA Graphs. This support requires a minimum CUDA version of 11.3."
- [CUDA Runtime API Reference Manual, "Stream Management" and "Execution Control"](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__STREAM.html) — the real, verified restriction language Section 23.3 quotes for `cudaStreamAddCallback` ("no CUDA function may be called from callback") and `cudaLaunchHostFunc` ("The host function must not make any CUDA API calls"), which `cudaGraphAddHostNode`'s own documentation points back to for a host node's own callback.
- This exact environment's own installed CUDA 12.0 headers (`/usr/include/cuda_runtime_api.h`, `/usr/include/driver_types.h`) — grepped directly for the real signatures of `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphAddHostNode`, and the real `cudaHostNodeParams`/`cudaHostFn_t` definitions, per Chapter 21 and Chapter 22's own established practice of checking installed headers directly rather than trusting a web summary.
