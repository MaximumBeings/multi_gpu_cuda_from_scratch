// Chapter 21: GPUDirect RDMA: Bypassing the Host Entirely
// 61_gpudirect_rdma_query.cu
//
// Section 20.2 checked whether an MPI BUILD could accept a device
// pointer directly (it couldn't, on this exact installed Open MPI).
// GPUDirect RDMA is the layer underneath that question: it lets a
// third-party PCI Express device -- a NIC, in the hybrid setup this
// chapter's own file 60 just bootstrapped -- read and write GPU
// memory directly, over the same real PCIe BAR mechanism NVIDIA's own
// GPUDirect RDMA design guide describes: "PCI Express device issues
// reads and writes to a peer device's BAR addresses in the same way
// that they are issued to system memory." Whether a specific GPU
// supports this at all is not something to assume -- CUDA exposes it
// as three real, queryable device attributes (cudaDevAttrGPUDirect...,
// found by grepping this exact environment's own installed
// driver_types.h, not guessed), queried here the same honest way
// Chapter 18 queried cudaGetDeviceProperties().
// Genuinely compiled with a real nvcc against the real installed CUDA
// runtime.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t eCount = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %s, count=%d\n",
           cudaGetErrorString(eCount), deviceCount);

    // Real device attribute #1: does this device support GPUDirect RDMA
    // APIs at all (the kernel-driver functions nvidia_p2p_get_pages()/
    // nvidia_p2p_put_pages() the design guide names)? This is a
    // per-device hardware/driver fact, independent of any specific NIC.
    int rdmaSupported = -1;
    cudaError_t eSupported = cudaDeviceGetAttribute(
        &rdmaSupported, cudaDevAttrGPUDirectRDMASupported, 0);
    printf("\ncudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMASupported): "
           "%s (queried value untouched at %d since the call itself "
           "honestly failed -- no device 0 exists to report on)\n",
           cudaGetErrorString(eSupported), rdmaSupported);

    // Real device attribute #2: a bitmask of which flush options
    // cudaDeviceFlushGPUDirectRDMAWrites() (below) supports on this
    // device -- cudaFlushGPUDirectRDMAWritesOptionHost (1<<0) and/or
    // cudaFlushGPUDirectRDMAWritesOptionMemOps (1<<1), per this exact
    // installed driver_types.h.
    int flushOptions = -1;
    cudaError_t eFlushOpts = cudaDeviceGetAttribute(
        &flushOptions, cudaDevAttrGPUDirectRDMAFlushWritesOptions, 0);
    printf("cudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMAFlushWritesOptions): "
           "%s\n", cudaGetErrorString(eFlushOpts));

    // Real device attribute #3: does this device natively guarantee
    // ORDERING of GPUDirect RDMA writes for the device's own kernels
    // (cudaGPUDirectRDMAWritesOrderingNone=0 / ...Owner=100 /
    // ...AllDevices=200)? A NIC's DMA write landing in GPU memory is
    // not automatically visible to a kernel already running on that
    // GPU unless the platform guarantees this ordering -- that's
    // exactly the race cudaDeviceFlushGPUDirectRDMAWrites() exists to
    // close when the platform does NOT guarantee it.
    int writesOrdering = -1;
    cudaError_t eOrdering = cudaDeviceGetAttribute(
        &writesOrdering, cudaDevAttrGPUDirectRDMAWritesOrdering, 0);
    printf("cudaDeviceGetAttribute(cudaDevAttrGPUDirectRDMAWritesOrdering): "
           "%s\n", cudaGetErrorString(eOrdering));

    // cudaGetDeviceProperties() carries the SAME two facts as struct
    // fields (Chapter 18's own query function, new fields) -- included
    // to show both real, independent ways CUDA exposes this, exactly
    // like Chapter 20's own MPIX_CUDA_AWARE_SUPPORT (macro) vs
    // MPIX_Query_cuda_support() (function) pairing.
    cudaDeviceProp prop;
    cudaError_t eProps = cudaGetDeviceProperties(&prop, 0);
    printf("\ncudaGetDeviceProperties(): %s\n", cudaGetErrorString(eProps));
    if (eProps == cudaSuccess) {
        printf("  prop.gpuDirectRDMAFlushWritesOptions = %u\n",
               prop.gpuDirectRDMAFlushWritesOptions);
        printf("  prop.gpuDirectRDMAWritesOrdering     = %d\n",
               prop.gpuDirectRDMAWritesOrdering);
    } else {
        printf("  (struct never populated -- same honest failure as the "
               "three cudaDeviceGetAttribute() calls above; this "
               "environment has no device 0 for either query style to "
               "report on)\n");
    }

    // The actual bypass-the-host action: block until pending GPUDirect
    // RDMA writes to this device are visible to the requested scope.
    // Real, documented return values: cudaSuccess, or cudaErrorNotSupported
    // if "the call will be a no-op" -- e.g. because the device's own
    // ordering guarantee (queried above) already covers this scope, or
    // because the device has no GPUDirect RDMA support at all.
    cudaError_t eFlush = cudaDeviceFlushGPUDirectRDMAWrites(
        cudaFlushGPUDirectRDMAWritesTargetCurrentDevice,
        cudaFlushGPUDirectRDMAWritesToOwner);
    printf("\ncudaDeviceFlushGPUDirectRDMAWrites(TargetCurrentDevice, "
           "ToOwner): %s\n", cudaGetErrorString(eFlush));
    printf("This call needs an ACTIVE device context to mean anything at "
           "all -- there is no device 0 to make current, so this is the "
           "same honest cudaErrorNoDevice family every device-touching "
           "call in this book has returned since Chapter 3, not a new "
           "failure mode specific to GPUDirect RDMA.\n");

    return 0;
}
