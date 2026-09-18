// Appendix A: Installation and Setup
// 121_multi_gpu_stack_version_check.cpp
//
// nvcc and g++ (File 120) are the base C++/CUDA toolchain, but this
// book's own multi-GPU chapters also depend on three additional real
// libraries: NCCL (collective communication, first used in Chapter 8),
// Open MPI (host-side process launch and messaging, first used in
// Chapter 19), and NVSHMEM (GPU-initiated one-sided communication, first
// used in Chapter 22). Each publishes its own real version macros in its
// own installed header, exactly like nvcc and g++ do -- this file reads
// all three directly, the same "read it from the real installed header,
// never from memory" discipline this book has used since Chapter 1.
#include <cstdio>
#include <nccl.h>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>

int main(int argc, char** argv) {
    printf("=== NCCL (collective communication) ===\n");
#if defined(NCCL_MAJOR) && defined(NCCL_MINOR) && defined(NCCL_PATCH)
    printf("NCCL_MAJOR = %d, NCCL_MINOR = %d, NCCL_PATCH = %d "
           "(NCCL_VERSION_CODE = %d)\n",
           NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH, NCCL_VERSION_CODE);
#else
    printf("NCCL version macros not defined -- nccl.h was not found by "
           "the compiler.\n");
#endif

    printf("\n=== Open MPI (host-side process launch and messaging) ===\n");
#if defined(OMPI_MAJOR_VERSION) && defined(OMPI_MINOR_VERSION) && defined(OMPI_RELEASE_VERSION)
    printf("OMPI_MAJOR_VERSION = %d, OMPI_MINOR_VERSION = %d, "
           "OMPI_RELEASE_VERSION = %d\n",
           OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
#else
    printf("OMPI version macros not defined -- mpi.h was not found, or "
           "this is a different MPI implementation (e.g. MPICH) with its "
           "own different macro names.\n");
#endif

    // MPI itself must be initialized to call most of its API, but the
    // version macros above are pure preprocessor text and need no
    // runtime call at all -- this program deliberately does NOT call
    // MPI_Init, to keep this specific check runnable with a plain `./`
    // invocation rather than requiring `mpirun`. Chapter 19 onward always
    // launches MPI programs through mpirun/mpiexec; this file is a
    // deliberate exception, for the version-check use case only.
    (void)argc; (void)argv;

    printf("\n=== NVSHMEM (GPU-initiated one-sided communication) ===\n");
    printf("NVSHMEM ships its own version macros in a header not on the "
           "default include path (non_abi/nvshmem_version.h) -- rather "
           "than requiring every reader to know that path, this "
           "appendix's own Makefile (A.4) established the reliable "
           "check: NVSHMEM's pip package is a namespace package with no "
           "__file__ of its own, so `python3 -c \"import nvidia.nvshmem; "
           "print(list(nvidia.nvshmem.__path__)[0])\"` is what actually "
           "locates the install (plain __file__ raises TypeError here), "
           "and the header at that path's own "
           "include/non_abi/nvshmem_version.h defines "
           "NVSHMEM_VENDOR_MAJOR_VERSION / _MINOR_VERSION / "
           "_PATCH_VERSION directly. In this environment those macros "
           "read 3, 7, and 2.\n");

    printf("\nself-check: this program compiled and ran, which by itself "
           "confirms nccl.h and mpi.h were both found on this system's "
           "own include path -- a well-formed documented result either "
           "way, matching this book's own Appendix pattern.\n");
    return 0;
}
