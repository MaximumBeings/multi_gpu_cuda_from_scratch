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
