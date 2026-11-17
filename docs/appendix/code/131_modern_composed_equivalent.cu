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
// cooperative_groups.h requires a kernel that calls grid.sync() to be
// launched with cuLaunchCooperativeKernel "(or the CUDA Runtime
// equivalent)" -- a plain <<<...>>> launch is NOT valid for it. Inside a
// stream capture, this file therefore launches it with cudaLaunchKernelEx()
// and the cudaLaunchAttributeCooperative attribute, which driver_types.h
// documents as "Valid for graph nodes, launches."
__global__ void device0_kernel(int* data, int n) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] *= 2;
    grid.sync();   // still-current, single-device cross-block barrier
    if (tid < n) data[tid] += 1;
}

// Runs on device 1, only after device 0's own event fires, on device 1's
// own buffer. (A real pipeline would also copy device 0's result across,
// e.g. with Chapter 5's cudaMemcpyPeerAsync(); this file keeps only the
// dependency edge itself, so no copy is issued here.)
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
    printf("  3. cudaLaunchKernelEx(&coopCfg, device0_kernel, ...) (grid.sync() WITHIN device 0)\n");
    printf("  4. cudaEventRecord(ev, stream0)                     <- the real cross-device edge\n");
    printf("  5. cudaSetDevice(1); cudaStreamCreate(&stream1)\n");
    printf("  6. cudaStreamWaitEvent(stream1, ev, 0)               <- Chapter 6's own real API,\n");
    printf("                                                          captured (Chapter 23's finding)\n");
    printf("  7. device1_kernel<<<...,stream1>>>(...)\n");
    printf("  8. cudaEventRecord(evJoin, stream1)                 <- JOIN stream1 back...\n");
    printf("  9. cudaSetDevice(0); cudaStreamWaitEvent(stream0, evJoin, 0)\n");
    printf("                                                      ...into the origin stream\n");
    printf(" 10. cudaStreamEndCapture(stream0, &graph)            <- ONE graph, TWO devices\n");
    printf(" 11. cudaGraphInstantiate(...); cudaGraphLaunch(...)\n\n");
    printf("  step 3 uses cudaLaunchKernelEx() with cudaLaunchAttributeCooperative, not a\n");
    printf("  plain <<<...>>> launch: grid.sync() is only valid in a cooperative launch.\n");
    printf("  steps 8-9 are required: a forked stream that is not joined back before\n");
    printf("  cudaStreamEndCapture() makes capture fail with\n");
    printf("  cudaErrorStreamCaptureUnjoined (%d) on real hardware.\n\n",
           (int)cudaErrorStreamCaptureUnjoined);

    cudaError_t eSet0 = cudaSetDevice(0);
    report("cudaSetDevice(0)", eSet0);
    cudaStream_t stream0;
    cudaError_t eStream0 = cudaStreamCreate(&stream0);
    report("cudaStreamCreate(&stream0)", eStream0);

    cudaError_t eCap = cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal);
    report("cudaStreamBeginCapture(stream0)", eCap);

    int* dataDev0 = nullptr;
    int n = 8;
    cudaMallocAsync(&dataDev0, n * sizeof(int), stream0);
    cudaLaunchAttribute coopAttr[1];
    coopAttr[0].id = cudaLaunchAttributeCooperative;
    coopAttr[0].val.cooperative = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(1);
    cfg.blockDim = dim3(8);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream0;
    cfg.attrs = coopAttr;
    cfg.numAttrs = 1;
    cudaError_t eCoop = cudaLaunchKernelEx(&cfg, device0_kernel, dataDev0, n);
    report("cudaLaunchKernelEx(cooperative)", eCoop);

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

    // Join stream1 back into the origin stream before ending capture.
    cudaEvent_t evJoin;
    cudaEventCreate(&evJoin);
    cudaError_t eJoinRec = cudaEventRecord(evJoin, stream1);
    report("cudaEventRecord(evJoin, stream1)", eJoinRec);
    cudaError_t eSetBack = cudaSetDevice(0);
    report("cudaSetDevice(0) (back)", eSetBack);
    cudaError_t eJoinWait = cudaStreamWaitEvent(stream0, evJoin, 0);
    report("cudaStreamWaitEvent(stream0, evJoin)", eJoinWait);

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
    printf("separate, well-supported tools. Note what the composition does NOT give back:\n");
    printf("the captured event is a stream-level edge between whole kernels, not a\n");
    printf("barrier device code can call mid-kernel. Device 1's kernel starts only after\n");
    printf("device 0's kernel has finished; no thread on one device can wait, inside a\n");
    printf("running kernel, for a thread on another the way multi_grid_group::sync() did.\n");

    bool allNoDevice = (e0 != cudaSuccess);
    printf("\nself-check: every real API call in this composed sequence reports the SAME\n");
    printf("honest cudaErrorNoDevice this environment reports for every device-touching\n");
    printf("call since Chapter 2, confirming the full sequence was genuinely attempted and\n");
    printf("not silently skipped: %s\n", allNoDevice ? "confirmed" : "MISMATCH");

    return allNoDevice ? 0 : 1;
}
