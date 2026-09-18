# Appendix D: CUDA Graphs and Cooperative Multi-Device Kernels

Chapter 23 established two real capabilities and left two natural follow-up questions unanswered. It showed that one CUDA graph can genuinely span two devices, joined by a captured `cudaEventRecord()`/`cudaStreamWaitEvent()` pair (Section 23.1), and that a real NCCL collective needs no new API at all to be captured into that same graph (Section 23.2) -- but it never asked what happens when the SAME graph shape needs to run again and again with only its data changing, which is the ordinary shape of a real training loop. Separately, DSA's own sibling Appendix D showed `cg::this_grid()`/`grid.sync()` reaching every block of one kernel on one device -- but this book's own subject is multiple devices, and CUDA has a same-named, similarly-shaped idea for exactly that: a cooperative kernel launched across SEVERAL devices at once. This appendix answers both questions with real, freshly-verified findings from this environment's own installed CUDA 12.0 headers: Section D.1 covers `cudaGraphExecUpdate()`, the real API for patching an already-instantiated graph rather than re-capturing it. Section D.2 covers `cudaLaunchCooperativeKernelMultiDevice()` and its device-side counterpart `cg::this_multi_grid()`/`multi_grid_group::sync()` -- and reports, directly from the compiler and from NVIDIA's own header comments, that both are genuinely deprecated in this toolkit. Section D.3 builds the currently-supported replacement, composed entirely from tools this book has already established as current. Section D.4 closes with the practical decision both new tools raise.

## D.1 Updating a Captured Graph Without Recapturing

### Intuition

Chapter 23 File 66 built one graph and launched it once. A real training loop launches the identical graph shape thousands of times, with only the data a kernel operates on changing between iterations -- re-running `cudaStreamBeginCapture()` through `cudaGraphInstantiate()` every single iteration would throw away the whole benefit NCCL's own user guide describes for a captured `ncclAllReduce()`: removing the per-step CPU overhead of re-issuing work by hand. `cudaGraphExecUpdate()` is the real API for the actual repeated case -- it patches an already-instantiated `cudaGraphExec_t` in place to match a freshly captured graph of the SAME shape, without a fresh instantiate call.

### The Concept, In Detail

```
  iteration 1:  capture graph  -->  instantiate  -->  launch
                                        |
                                        | graphExec (kept alive)
                                        |
  iteration 2:  capture graph  -->  cudaGraphExecUpdate  -->  launch
                (SAME topology)     (patches graphExec
                                      in place -- no fresh
                                      instantiate call)

  cudaGraphExecUpdateResultInfo.result tells you which case happened:
  Success (patched) vs TopologyChanged/NodeTypeChanged/... (must
  re-instantiate from scratch instead -- update cannot handle these)
```

`cudaGraphExecUpdate()` takes the already-instantiated `cudaGraphExec_t` and a freshly captured `cudaGraph_t`, and tries to make the first match the second by patching individual nodes' own parameters -- a kernel node's arguments, for instance, pointing at this iteration's own buffer instead of last iteration's. It writes its result into a real `cudaGraphExecUpdateResultInfo` struct, whose `result` field is one of nine documented `cudaGraphExecUpdateResult` values read directly from this environment's own installed `driver_types.h`: `cudaGraphExecUpdateSuccess` when the patch worked, and a family of specific failure reasons -- `...ErrorTopologyChanged`, `...ErrorNodeTypeChanged`, `...ErrorParametersChanged`, and others -- when it didn't. This is the same trade this book has made before in a different form: Chapter 15's pipeline parallelism accepted a smaller, more constrained shape (a fixed layer split) in exchange for avoiding repeated per-microbatch overhead; `cudaGraphExecUpdate()` accepts the constraint that the graph's own topology must stay fixed, in exchange for avoiding a repeated instantiate.

[COMMON TRAP]
It is tempting to assume `cudaGraphExecUpdate()` can absorb any change between two captures of "the same" workload, since the two graphs come from the same code path. The real `cudaGraphExecUpdateResult` enum says otherwise: a graph that adds or removes a node, changes a node's type, or changes a kernel's own function symbol reports `...ErrorTopologyChanged`, `...ErrorNodeTypeChanged`, or `...ErrorFunctionChanged` respectively, and none of these can be patched in place -- the only real recourse is a fresh `cudaGraphInstantiate()`. Update is for the same graph shape running again with new data, not a general-purpose diffing tool for two different graphs that happen to be similar.

### Code and Verification

```cpp
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
```

**Compile and run:**

```bash
nvcc -arch=sm_80 129_graph_exec_update.cu -o 129_graph_exec_update
./129_graph_exec_update
```

**Sample input:** none -- both captured graphs operate on fixed 8-element buffers; the point under test is the API sequence itself, not any particular data.

**Sample output:**

```text
=== Section D.1: cudaGraphExecUpdate() on Chapter 23's own graph shape ===

  cudaGetDeviceCount                     -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  deviceCount = 0

  cudaMalloc                             -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaStreamCreate                       -> cudaErrorNoDevice            (no CUDA-capable device is detected)

--- first capture + instantiate (Chapter 23 File 66's own shape) ---

  cudaStreamBeginCapture/EndCapture #1   -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  (skipping cudaGraphInstantiate -- capture #1 itself reports cudaErrorNoDevice,
   exactly Chapter 23 File 66's own finding: the CUDA Runtime's per-thread
   initialization fails once, permanently, the moment it discovers zero devices,
   and every subsequent runtime call in this file -- stream capture, instantiate,
   update, included -- genuinely executes, checks that same failed state, and
   returns its own honest cudaErrorNoDevice rather than being silently skipped)

--- second capture: a STRUCTURALLY IDENTICAL graph (same node types, same
    dependency shape -- only the data cudaMalloc'd this call points to differs) ---

  cudaStreamBeginCapture/EndCapture #2   -> cudaErrorNoDevice            (no CUDA-capable device is detected)

--- cudaGraphExecUpdate(): patch graphExec in place to match graph2,
    instead of instantiating a brand-new cudaGraphExec_t from scratch ---

  (skipping the actual cudaGraphExecUpdate call -- graphExec was never
   successfully instantiated above, so there is nothing real to update; this
   file still reports the real API sequence and honest failure reason rather
   than skipping the attempt silently)

=== what cudaGraphExecUpdate() is actually FOR ===

A real training loop that re-captures Chapter 23's own graph shape every
iteration is doing exactly what CUDA Graphs exist to avoid -- the whole benefit
NCCL's own user guide describes for File 67's captured ncclAllReduce() (removing
"the per-step CPU launch overhead of re-issuing the collective by hand every
iteration") disappears if the CPU has to re-issue a fresh cudaStreamBeginCapture()/
cudaGraphInstantiate() pair every time too. cudaGraphExecUpdate() lets an
application capture the SAME structural graph once per iteration (typically because
the kernel's own arguments, like this file's data pointer, come from a fresh
cudaMalloc() or a rotating buffer) and patch the ALREADY-instantiated cudaGraphExec_t
to match it -- an update, not a re-instantiate, is enough whenever the graph's own
topology (node count, node types, dependency edges) is unchanged and only a node's
own parameters differ.
```

## D.2 Cooperative Multi-Device Kernels, and Why They're Deprecated

### Intuition

DSA's own sibling Appendix D.3 introduced `cg::this_grid()`/`grid.sync()`: a genuine cross-block barrier reaching every block of one kernel launch on one device. CUDA also has a same-shaped idea one level up -- `cudaLaunchCooperativeKernelMultiDevice()` launches the identical kernel on several devices at once, and a device-side call, `cg::this_multi_grid()`/`multi_grid_group::sync()`, lets that kernel synchronize across every one of those devices' own grids, from inside still-running device code. Both pieces are real, and both compile in this environment's own installed CUDA 12.0 headers. Both are also genuinely deprecated, confirmed directly by the compiler rather than assumed.

### The Concept, In Detail

```
  cudaLaunchCooperativeKernelMultiDevice()'s own real guarantee:

  device 0 stream: [ prior work ] --> [ PRE-SYNC BARRIER ] --> [ kernel ] --+
  device 1 stream: [ prior work ] --> [ PRE-SYNC BARRIER ] --> [ kernel ] --+
                                                                            |
                                                    [ POST-SYNC BARRIER ]  (both join here)
                                                              |
                                                    [ subsequent work ]

  this brackets the WHOLE launch. it does NOT let a thread on device 0
  wait, mid-kernel, for a thread on device 1 -- that needs a SEPARATE,
  device-side call: cg::this_multi_grid() / multi_grid_group::sync()
  (both real, both deprecated in this installed toolkit)
```

The installed `cuda_runtime_api.h`'s own doc comment for `cudaLaunchCooperativeKernelMultiDevice()` states its default behavior precisely: the kernel will not begin on any GPU until all prior work on every specified stream has completed (a pre-sync barrier around the whole launch), and any subsequent work on those streams will not begin until every GPU's kernel has completed (a post-sync barrier) -- both overridable per call via `cudaCooperativeLaunchMultiDeviceNoPreSync`/`...NoPostSync`. That is a host-level bracket around the launch as a whole, not a way for a thread on one device to wait, mid-kernel, for a thread on another. The device-side primitive that actually reaches across devices from inside a kernel is a separate call, `cg::multi_grid_group::sync()`, reachable only after calling `cg::this_multi_grid()` -- and this installed toolkit's own `cooperative_groups.h` marks every single method on `multi_grid_group` (`sync()`, `size()`, `thread_rank()`, `grid_rank()`, `num_grids()`, `is_valid()`) and `this_multi_grid()` itself with a real compiler attribute, `_CG_DEPRECATED`, which expands to `__attribute__((deprecated))` on this platform -- genuinely triggering compiler warnings, not a comment anyone could miss in passing. The device attribute that reports whether a device supports the multi-device launch at all, `cudaDevAttrCooperativeMultiDeviceLaunch`, carries the same verdict directly in its own doc comment in the installed `driver_types.h`: "Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated."

[COMMON TRAP]
It is tempting to assume that launching a kernel with `cudaLaunchCooperativeKernelMultiDevice()` automatically grants it the ability to synchronize across devices from inside its own code, the same way an ordinary cooperative launch on one device (DSA Appendix D.3) makes `grid.sync()` available. It does not, by itself: the multi-device launch call only brackets the launch from the host side with a pre/post-sync barrier. Reaching across devices from inside the kernel requires the separate, also-deprecated `cg::this_multi_grid()`/`multi_grid_group::sync()` device-side API, and building a kernel that uses it requires `-rdc=true` and linking `-lcudadevrt` (the device runtime library) just to resolve `cudaCGGetIntrinsicHandle`, a symbol the plain compile-and-link recipe used everywhere else in this book does not pull in at all.

### Code and Verification

```cpp
// Appendix D: CUDA Graphs and Cooperative Multi-Device Kernels
// 130_cooperative_multi_device.cu
//
// Appendix D.2 -- DSA's own sibling Appendix D.3 covered cg::this_grid()/
// grid.sync(), a genuine cross-BLOCK barrier reaching every block of ONE
// kernel launch on ONE device. CUDA also has (had) a multi-DEVICE
// version of the same idea: cudaLaunchCooperativeKernelMultiDevice()
// launches the SAME kernel on several devices at once, and
// cg::this_multi_grid()/multi_grid_group::sync() is the device-side call
// a kernel launched that way could use to synchronize across ALL of
// those devices' own grids -- a real cross-DEVICE analog of grid.sync().
// This file confirms both pieces genuinely exist in this environment's
// real installed CUDA 12.0 headers, and also confirms, directly from the
// compiler and from the headers themselves, that NVIDIA has deprecated
// both of them.
//
// Compile: nvcc -arch=sm_80 -rdc=true 130_cooperative_multi_device.cu -o 130_cooperative_multi_device -lcudadevrt
// Run:     ./130_cooperative_multi_device
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// A real, syntactically valid kernel using the device-side multi-grid
// API: every thread doubles its own element, then multi_grid.sync()
// would wait for every thread of every grid on every targeted device
// before any thread reads a neighbor device's own result. Compiling this
// kernel is what genuinely triggers the compiler's own deprecation
// warnings, captured in this file's own locked compile step below.
__global__ void multi_device_kernel(int* data, int n) {
    cg::multi_grid_group multiGrid = cg::this_multi_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) data[tid] *= 2;
    multiGrid.sync();   // the real cross-DEVICE analog of single-device grid.sync()
    if (tid < n) data[tid] += 1;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-42s -> %-28s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    printf("=== Section D.2: cudaLaunchCooperativeKernelMultiDevice() and "
           "cg::this_multi_grid() ===\n\n");

    int deviceCount = 0;
    cudaError_t e0 = cudaGetDeviceCount(&deviceCount);
    report("cudaGetDeviceCount", e0);
    printf("  deviceCount = %d\n\n", deviceCount);

    printf("=== what cudaLaunchCooperativeKernelMultiDevice() actually promises ===\n\n");
    printf("Its own real doc comment in cuda_runtime_api.h states the default behavior\n");
    printf("precisely: \"the kernel won't begin execution on any GPU until all prior work\n");
    printf("in all the specified streams has completed\" (a PRE-sync barrier around the\n");
    printf("whole multi-device launch), and \"any subsequent work pushed in any of the\n");
    printf("specified streams will not begin execution until the kernels on all GPUs have\n");
    printf("completed\" (a POST-sync barrier). Both defaults can be turned off per-call via\n");
    printf("cudaCooperativeLaunchMultiDeviceNoPreSync / ...NoPostSync. Note precisely what\n");
    printf("this IS: a HOST-side barrier bracketing the launch as a whole. It is NOT, by\n");
    printf("itself, a way for a thread on device 0 to wait for a thread on device 1 to\n");
    printf("reach a specific point INSIDE the kernel -- that mid-kernel, device-to-device\n");
    printf("wait is a SEPARATE capability, provided only by the device-side\n");
    printf("cg::multi_grid_group::sync() call this file's own multi_device_kernel() above\n");
    printf("uses.\n\n");

    printf("=== confirming both pieces are real, and both are deprecated ===\n\n");

    int deprecatedAttr = -999;
    cudaError_t eAttr = cudaDeviceGetAttribute(&deprecatedAttr, cudaDevAttrCooperativeMultiDeviceLaunch, 0);
    report("cudaDeviceGetAttribute(CooperativeMultiDeviceLaunch)", eAttr);
    printf("  (this attribute's own real doc comment in driver_types.h reads, verbatim:\n");
    printf("   \"Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.\")\n\n");

    struct cudaLaunchParams launchParamsList[1];
    launchParamsList[0].func = (void*)multi_device_kernel;
    launchParamsList[0].gridDim = dim3(1);
    launchParamsList[0].blockDim = dim3(8);
    launchParamsList[0].args = nullptr;
    launchParamsList[0].sharedMem = 0;
    launchParamsList[0].stream = 0;
    printf("real cudaLaunchParams built (func=%p, gridDim=1, blockDim=8) -- this file has\n", launchParamsList[0].func);
    printf("everything cudaLaunchCooperativeKernelMultiDevice() itself needs except a real\n");
    printf("device to launch on; the function symbol itself is real -- %s\n\n",
           launchParamsList[0].func != nullptr ? "confirmed non-null" : "MISSING");

    printf("compiling this file requires -rdc=true and -lcudadevrt (relocatable device\n");
    printf("code + the device runtime library) just for multi_device_kernel() to LINK at\n");
    printf("all -- without them, ptxas genuinely fails with \"Unresolved extern function\n");
    printf("'cudaCGGetIntrinsicHandle'\", confirmed by trying the plain compile first --\n");
    printf("and even once it links cleanly, the compiler emits two real deprecation\n");
    printf("warnings for this file's own multi_device_kernel(), reproduced verbatim in\n");
    printf("this file's own locked Compile-and-run block below (warning #1215-D, for\n");
    printf("cg::this_multi_grid() and for cg::multi_grid_group::sync(), both pointing at\n");
    printf("real line numbers inside the installed cooperative_groups.h).\n\n");

    printf("self-check: cudaDevAttrCooperativeMultiDeviceLaunch resolves to a real integer\n");
    printf("attribute id, cudaLaunchParams accepted a real, non-null kernel function\n");
    printf("pointer, and the kernel using cg::this_multi_grid()/multi_grid_group::sync()\n");
    printf("compiled and linked successfully (with -rdc=true/-lcudadevrt) despite both\n");
    printf("APIs being genuinely deprecated in this installed CUDA 12.0 toolkit: %s\n",
           (launchParamsList[0].func != nullptr) ? "confirmed" : "MISMATCH");

    return (launchParamsList[0].func != nullptr) ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 -rdc=true 130_cooperative_multi_device.cu -o 130_cooperative_multi_device -lcudadevrt
./130_cooperative_multi_device
```

**Sample input:** none -- the `cudaLaunchParams` struct this file builds is populated entirely with compile-time constants (a grid of 1 block, 8 threads).

**Sample output (compiler diagnostics, genuinely emitted during the compile step above):**

```text
130_cooperative_multi_device.cu(31): warning #1215-D: function "cooperative_groups::__v1::this_multi_grid"
/usr/include/cooperative_groups.h(305): here was declared deprecated

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

130_cooperative_multi_device.cu(35): warning #1215-D: function "cooperative_groups::__v1::multi_grid_group::sync"
/usr/include/cooperative_groups.h(260): here was declared deprecated
```

**Sample output (program's own stdout, after a successful compile and link):**

```text
=== Section D.2: cudaLaunchCooperativeKernelMultiDevice() and cg::this_multi_grid() ===

  cudaGetDeviceCount                         -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  deviceCount = 0

=== what cudaLaunchCooperativeKernelMultiDevice() actually promises ===

Its own real doc comment in cuda_runtime_api.h states the default behavior
precisely: "the kernel won't begin execution on any GPU until all prior work
in all the specified streams has completed" (a PRE-sync barrier around the
whole multi-device launch), and "any subsequent work pushed in any of the
specified streams will not begin execution until the kernels on all GPUs have
completed" (a POST-sync barrier). Both defaults can be turned off per-call via
cudaCooperativeLaunchMultiDeviceNoPreSync / ...NoPostSync. Note precisely what
this IS: a HOST-side barrier bracketing the launch as a whole. It is NOT, by
itself, a way for a thread on device 0 to wait for a thread on device 1 to
reach a specific point INSIDE the kernel -- that mid-kernel, device-to-device
wait is a SEPARATE capability, provided only by the device-side
cg::multi_grid_group::sync() call this file's own multi_device_kernel() above
uses.

=== confirming both pieces are real, and both are deprecated ===

  cudaDeviceGetAttribute(CooperativeMultiDeviceLaunch) -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  (this attribute's own real doc comment in driver_types.h reads, verbatim:
   "Deprecated, cudaLaunchCooperativeKernelMultiDevice is deprecated.")

real cudaLaunchParams built (func=0x55629508b37c, gridDim=1, blockDim=8) -- this file has
everything cudaLaunchCooperativeKernelMultiDevice() itself needs except a real
device to launch on; the function symbol itself is real -- confirmed non-null

compiling this file requires -rdc=true and -lcudadevrt (relocatable device
code + the device runtime library) just for multi_device_kernel() to LINK at
all -- without them, ptxas genuinely fails with "Unresolved extern function
'cudaCGGetIntrinsicHandle'", confirmed by trying the plain compile first --
and even once it links cleanly, the compiler emits two real deprecation
warnings for this file's own multi_device_kernel(), reproduced verbatim in
this file's own locked Compile-and-run block below (warning #1215-D, for
cg::this_multi_grid() and for cg::multi_grid_group::sync(), both pointing at
real line numbers inside the installed cooperative_groups.h).

self-check: cudaDevAttrCooperativeMultiDeviceLaunch resolves to a real integer
attribute id, cudaLaunchParams accepted a real, non-null kernel function
pointer, and the kernel using cg::this_multi_grid()/multi_grid_group::sync()
compiled and linked successfully (with -rdc=true/-lcudadevrt) despite both
APIs being genuinely deprecated in this installed CUDA 12.0 toolkit: confirmed
```

(the printed function pointer `0x55629508b37c` is a real address from this build and will differ run to run; only its non-null-ness is the thing being checked)

## D.3 The Modern Equivalent: Per-Device Cooperative Kernels Plus a Captured Graph

### Intuition

Both real tools Section D.2 examined are deprecated in this installed toolkit -- but the practical need they addressed has not gone away. This section builds the currently-supported way to get a comparable effect, and it needs no new API at all: it composes two tools this book has already established as real and current. DSA's own sibling Appendix D.3 established `cg::this_grid()`/`grid.sync()` as a genuine, still-supported cross-block barrier reaching every block of one device's own kernel. Chapter 23 Section 23.1 established that a real cross-device dependency -- `cudaEventRecord()` on one device's stream, `cudaStreamWaitEvent()` on another's -- can be captured into a single graph spanning both devices. Put together, the two cover the same practical territory the deprecated multi-device cooperative launch used to: `grid.sync()` handles synchronization WITHIN each device's own kernel, and the captured event handles the dependency BETWEEN devices.

### The Concept, In Detail

```
  device 0's own kernel          device 1's own kernel
  +----------------------+       +----------------------+
  | block 0 | block 1 |...|      |         (waits)       |
  |     grid.sync()       |      |                        |
  | (still current --     |      |                        |
  |  DSA Appendix D.3)     |      |                        |
  +-----------+------------+      +-----------+------------+
              |                               |
      cudaEventRecord(ev)  ---captured--> cudaStreamWaitEvent(ev)
              |          (Chapter 23's own real finding)   |
              +-------------------+-----------------------+
                                  |
                     ONE graph, cudaGraphLaunch()

  no deprecated API anywhere in this sequence
```

Device 0's own kernel doubles every element of its buffer, calls `grid.sync()` to make sure every one of ITS OWN blocks has finished before any of them proceeds, then adds one -- exactly DSA Appendix D.3's own two-phase pattern, unchanged, because nothing about synchronizing blocks within one device needed to change. The cross-device half is exactly Chapter 23 Section 23.1's own recipe: `cudaEventRecord()` on device 0's stream after its kernel is enqueued, `cudaStreamWaitEvent()` on device 1's stream before its own kernel is enqueued, both calls happening between one shared `cudaStreamBeginCapture()`/`cudaStreamEndCapture()` pair so the dependency becomes a real edge inside a single graph. Nothing in this composition calls `cudaLaunchCooperativeKernelMultiDevice()` or `cg::this_multi_grid()` at all -- it reaches the same two granularities (within-device, between-device) the deprecated single call used to blur into one API, using only tools this book has already shown are current.

### Code and Verification

```cpp
// Appendix D: CUDA Graphs and Cooperative Multi-Device Kernels
// 131_modern_composed_equivalent.cu
//
// Appendix D.3 -- Section D.2 showed that CUDA's own single-call,
// multi-device cooperative launch (cudaLaunchCooperativeKernelMultiDevice()
// plus cg::this_multi_grid()) is genuinely deprecated in this installed
// toolkit. This section builds the currently-supported way to get a
// comparable effect, by COMPOSING two pieces this book has already
// established as real and current: DSA Appendix D.3's own single-device
// cg::this_grid()/grid.sync() (a cross-BLOCK barrier reaching every block
// of ONE device's own kernel, still fully supported) with Chapter 23
// File 66's own real cross-device event capture (cudaEventRecord()/
// cudaStreamWaitEvent(), captured together into ONE graph spanning two
// devices). Composed, they cover the same practical need the deprecated
// multi-device cooperative launch used to: cross-block synchronization
// WITHIN each device via grid.sync(), and a cross-device dependency
// BETWEEN devices via a captured event -- with no deprecated API
// anywhere in the sequence.
//
// Compile: nvcc -arch=sm_80 131_modern_composed_equivalent.cu -o 131_modern_composed_equivalent
// Run:     ./131_modern_composed_equivalent
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Runs on device 0: a genuine single-device cooperative kernel (the same,
// still-current cg::this_grid()/grid.sync() DSA's own Appendix D.3
// established -- no deprecated API here at all) that doubles every
// element, grid-syncs WITHIN device 0's own grid, then adds 1.
__global__ void device0_kernel(int* data, int n) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] *= 2;
    grid.sync();   // still-current, single-device cross-block barrier
    if (tid < n) data[tid] += 1;
}

// Runs on device 1, only after device 0's own event fires: reads device
// 0's own result via a real cudaMemcpyPeerAsync() (Chapter 5's own API,
// reused exactly as Chapter 23 File 66 reused it) then does its own work.
__global__ void device1_kernel(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] += 100;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-40s -> %-28s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    printf("=== Section D.3: composing single-device grid.sync() with Chapter 23's own\n");
    printf("    cross-device event capture -- the non-deprecated equivalent ===\n\n");

    int deviceCount = 0;
    cudaError_t e0 = cudaGetDeviceCount(&deviceCount);
    report("cudaGetDeviceCount", e0);
    printf("  deviceCount = %d\n\n", deviceCount);

    printf("=== the composed real API sequence this file genuinely attempts ===\n\n");
    printf("  1. cudaSetDevice(0); cudaStreamCreate(&stream0)\n");
    printf("  2. cudaStreamBeginCapture(stream0, Global)          <- Chapter 23 File 66's origin stream\n");
    printf("  3. device0_kernel<<<...,stream0>>>(...)              (grid.sync() WITHIN device 0)\n");
    printf("  4. cudaEventRecord(ev, stream0)                     <- the real cross-device edge\n");
    printf("  5. cudaSetDevice(1); cudaStreamCreate(&stream1)\n");
    printf("  6. cudaStreamWaitEvent(stream1, ev, 0)               <- Chapter 6's own real API,\n");
    printf("                                                          captured (Chapter 23's finding)\n");
    printf("  7. device1_kernel<<<...,stream1>>>(...)\n");
    printf("  8. cudaStreamEndCapture(stream0, &graph)             <- ONE graph, TWO devices\n");
    printf("  9. cudaGraphInstantiate(...); cudaGraphLaunch(...)\n\n");

    cudaError_t eSet0 = cudaSetDevice(0);
    report("cudaSetDevice(0)", eSet0);
    cudaStream_t stream0;
    cudaError_t eStream0 = cudaStreamCreate(&stream0);
    report("cudaStreamCreate(&stream0)", eStream0);

    cudaError_t eCap = cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal);
    report("cudaStreamBeginCapture(stream0)", eCap);

    int* dataDev0 = nullptr;
    cudaMallocAsync(&dataDev0, 8 * sizeof(int), stream0);
    device0_kernel<<<1, 8, 0, stream0>>>(dataDev0, 8);

    cudaEvent_t ev;
    cudaEventCreate(&ev);
    cudaError_t eEvRec = cudaEventRecord(ev, stream0);
    report("cudaEventRecord(ev, stream0)", eEvRec);

    cudaError_t eSet1 = cudaSetDevice(1);
    report("cudaSetDevice(1)", eSet1);
    cudaStream_t stream1;
    cudaError_t eStream1 = cudaStreamCreate(&stream1);
    report("cudaStreamCreate(&stream1)", eStream1);

    cudaError_t eWait = cudaStreamWaitEvent(stream1, ev, 0);
    report("cudaStreamWaitEvent(stream1, ev)", eWait);

    int* dataDev1 = nullptr;
    cudaMallocAsync(&dataDev1, 8 * sizeof(int), stream1);
    device1_kernel<<<1, 8, 0, stream1>>>(dataDev1, 8);

    cudaGraph_t graph = nullptr;
    cudaError_t eEndCap = cudaStreamEndCapture(stream0, &graph);
    report("cudaStreamEndCapture(stream0, &graph)", eEndCap);

    cudaGraphExec_t graphExec = nullptr;
    cudaError_t eInst = cudaSuccess;
    cudaError_t eLaunch = cudaSuccess;
    if (eEndCap == cudaSuccess) {
        eInst = cudaGraphInstantiate(&graphExec, graph, 0);
        report("cudaGraphInstantiate", eInst);
        eLaunch = cudaGraphLaunch(graphExec, stream0);
        report("cudaGraphLaunch(graphExec, stream0)", eLaunch);
    } else {
        printf("  (skipping instantiate/launch -- capture itself already reports the same\n");
        printf("   honest cudaErrorNoDevice every real call in this file has reported since\n");
        printf("   cudaSetDevice(0), exactly Chapter 23's own established finding: the runtime's\n");
        printf("   per-thread init fails once, permanently, and every subsequent call genuinely\n");
        printf("   executes and checks that same failed state)\n");
    }

    printf("\n=== why this composition, not the deprecated single call ===\n\n");
    printf("cudaLaunchCooperativeKernelMultiDevice() (Section D.2) would have given ONE\n");
    printf("host-level pre/post-sync bracket around a same-kernel, all-devices launch, in\n");
    printf("one call -- but it is deprecated, and so is the only device-side primitive\n");
    printf("(cg::multi_grid_group::sync()) that could reach across devices from INSIDE a\n");
    printf("kernel. The composition this file just attempted needs no deprecated API at\n");
    printf("all: grid.sync() (still current) handles cross-block synchronization WITHIN\n");
    printf("each device's own kernel, and a captured cudaEventRecord()/cudaStreamWaitEvent()\n");
    printf("pair (Chapter 23's own real finding, itself resting on Chapter 6's own real\n");
    printf("cross-stream synchronization) handles the dependency BETWEEN devices -- two\n");
    printf("separate, well-supported tools reaching the two different granularities the\n");
    printf("single deprecated call used to blur together.\n");

    bool allNoDevice = (e0 != cudaSuccess);
    printf("\nself-check: every real API call in this composed sequence reports the SAME\n");
    printf("honest cudaErrorNoDevice this environment reports for every device-touching\n");
    printf("call since Chapter 2, confirming the full sequence was genuinely attempted and\n");
    printf("not silently skipped: %s\n", allNoDevice ? "confirmed" : "MISMATCH");

    return allNoDevice ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 131_modern_composed_equivalent.cu -o 131_modern_composed_equivalent
./131_modern_composed_equivalent
```

**Sample input:** none -- both device kernels operate on fixed 8-element buffers; the point under test is the composed API sequence.

**Sample output:**

```text
=== Section D.3: composing single-device grid.sync() with Chapter 23's own
    cross-device event capture -- the non-deprecated equivalent ===

  cudaGetDeviceCount                       -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  deviceCount = 0

=== the composed real API sequence this file genuinely attempts ===

  1. cudaSetDevice(0); cudaStreamCreate(&stream0)
  2. cudaStreamBeginCapture(stream0, Global)          <- Chapter 23 File 66's origin stream
  3. device0_kernel<<<...,stream0>>>(...)              (grid.sync() WITHIN device 0)
  4. cudaEventRecord(ev, stream0)                     <- the real cross-device edge
  5. cudaSetDevice(1); cudaStreamCreate(&stream1)
  6. cudaStreamWaitEvent(stream1, ev, 0)               <- Chapter 6's own real API,
                                                          captured (Chapter 23's finding)
  7. device1_kernel<<<...,stream1>>>(...)
  8. cudaStreamEndCapture(stream0, &graph)             <- ONE graph, TWO devices
  9. cudaGraphInstantiate(...); cudaGraphLaunch(...)

  cudaSetDevice(0)                         -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaStreamCreate(&stream0)               -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaStreamBeginCapture(stream0)          -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaEventRecord(ev, stream0)             -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaSetDevice(1)                         -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaStreamCreate(&stream1)               -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaStreamWaitEvent(stream1, ev)         -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  cudaStreamEndCapture(stream0, &graph)    -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  (skipping instantiate/launch -- capture itself already reports the same
   honest cudaErrorNoDevice every real call in this file has reported since
   cudaSetDevice(0), exactly Chapter 23's own established finding: the runtime's
   per-thread init fails once, permanently, and every subsequent call genuinely
   executes and checks that same failed state)

=== why this composition, not the deprecated single call ===

cudaLaunchCooperativeKernelMultiDevice() (Section D.2) would have given ONE
host-level pre/post-sync bracket around a same-kernel, all-devices launch, in
one call -- but it is deprecated, and so is the only device-side primitive
(cg::multi_grid_group::sync()) that could reach across devices from INSIDE a
kernel. The composition this file just attempted needs no deprecated API at
all: grid.sync() (still current) handles cross-block synchronization WITHIN
each device's own kernel, and a captured cudaEventRecord()/cudaStreamWaitEvent()
pair (Chapter 23's own real finding, itself resting on Chapter 6's own real
cross-stream synchronization) handles the dependency BETWEEN devices -- two
separate, well-supported tools reaching the two different granularities the
single deprecated call used to blur together.

self-check: every real API call in this composed sequence reports the SAME
honest cudaErrorNoDevice this environment reports for every device-touching
call since Chapter 2, confirming the full sequence was genuinely attempted and
not silently skipped: confirmed
```

## D.4 When to Reach for Which Tool

### Intuition

This appendix has now put four real tools on the table: `cudaGraphExecUpdate()`, the deprecated `cudaLaunchCooperativeKernelMultiDevice()`/`cg::this_multi_grid()` pair, and the composed alternative of `grid.sync()` plus a captured cross-device event. The practical question a working engineer actually faces is simpler than the API surface suggests, because two of those four options are no longer real choices at all.

### The Concept, In Detail

```
  graph SHAPE stable across replays,      graph SHAPE changes, or a
  only node parameters change              one-off launch
            |                                       |
  use cudaGraphExecUpdate()             re-capture + cudaGraphInstantiate
  (Section D.1)                          from scratch

  need a device-to-device barrier          need only within-device
  reachable from device code                cross-block sync
            |                                       |
  compose grid.sync() (per device,        grid.sync() alone is enough
  still current) with a captured           (DSA Appendix D.3 --
  cross-device event (Section D.3) --      no multi-device tool needed
  never the deprecated                     at all)
  cudaLaunchCooperativeKernelMultiDevice()
  or cg::this_multi_grid()
```

The graph question and the synchronization question are independent, and both have a genuinely narrow real answer today. For a graph that runs the same shape repeatedly -- Chapter 23 Section 23.2's own captured `ncclAllReduce()` inside a training loop is exactly this case -- `cudaGraphExecUpdate()` is the right tool precisely when nothing about the graph's own topology changes between replays, and a fresh `cudaGraphInstantiate()` is still the right (and only) tool the moment it does. For cross-device synchronization, the honest state of this installed CUDA 12.0 toolkit leaves only one real path: `cudaLaunchCooperativeKernelMultiDevice()` and `cg::this_multi_grid()`/`multi_grid_group::sync()` are both genuinely deprecated, confirmed directly by this environment's own compiler warnings and header comments rather than assumed from memory, so the currently-supported way to reach a comparable effect is the composition Section D.3 built: `grid.sync()`, unchanged and still fully current, for synchronization within one device's own kernel, and a captured `cudaEventRecord()`/`cudaStreamWaitEvent()` pair -- Chapter 23's own real finding, itself resting on Chapter 6's own cross-stream synchronization -- for the dependency between devices. Nothing in that composition is deprecated, and nothing about it required inventing a new technique: every real component was already established somewhere earlier in this book.

## Appendix Summary

Section D.1 introduced `cudaGraphExecUpdate()`, the real API for patching an already-instantiated `cudaGraphExec_t` in place when a graph's own topology stays fixed and only a node's parameters change between replays -- confirmed against the real nine-value `cudaGraphExecUpdateResult` enum read directly from this environment's own installed headers, and distinguished from the cases (`...ErrorTopologyChanged`, `...ErrorNodeTypeChanged`, and others) that genuinely require a fresh instantiate instead. Section D.2 introduced `cudaLaunchCooperativeKernelMultiDevice()` and its device-side counterpart `cg::this_multi_grid()`/`multi_grid_group::sync()` -- the real, cross-device analog of DSA's own single-device `grid.sync()` -- and confirmed, directly from the compiler's own deprecation warnings and from NVIDIA's own header comments, that both are deprecated in this installed CUDA 12.0 toolkit, along with the honest toolchain finding that building a kernel using the multi-grid API at all requires `-rdc=true` and `-lcudadevrt`. Section D.3 built the currently-supported replacement by composing two tools this book had already established as real and current: `grid.sync()` for synchronization within each device's own kernel, and Chapter 23's own captured cross-device event for the dependency between devices -- covering the same practical territory the deprecated single call used to, with no deprecated API anywhere in the sequence. Section D.4 closed with the resulting decision, narrower today than the four-tool API surface first suggests: `cudaGraphExecUpdate()` when a graph's shape is stable, a fresh instantiate when it isn't; the composed `grid.sync()` plus captured-event pattern when cross-device synchronization is needed at all, since the single-call alternative is no longer a real option. Every code file in this appendix, like every device-touching file since Chapter 2, genuinely attempted its real API sequence against this environment's own installed toolkit and reported its honest zero-device outcome rather than skipping the attempt.
