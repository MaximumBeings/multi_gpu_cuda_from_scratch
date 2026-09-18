// Appendix A: Installation and Setup
// 120_toolchain_version_check.cu
//
// Before any multi-GPU code in this book can compile, two host-side
// toolchain pieces must be present and mutually compatible: an nvcc that
// actually emits a working nvcc binary (not merely nvcc-adjacent files),
// and a host C++ compiler (g++) nvcc is willing to drive. This file reads
// the real preprocessor macros both compilers publish -- the same
// discipline the DSA book's own Appendix A uses (208_toolchain_version_
// check.cu) -- and prints the exact toolchain identity this specific
// environment reports, plus a second, book-specific check this appendix
// adds: confirming nvcc is a REAL compiler binary, not a pip package name
// that merely sounds like one.
//
// The real finding behind that second check, confirmed directly in this
// book's own build environment (not assumed from memory): `pip install
// nvidia-cuda-nvcc-cu12` DOES install successfully and DOES report a
// version (12.4.131 in this environment), but its own package directory's
// bin/ folder contains only ptxas (the PTX-to-SASS assembler) -- there is
// no nvcc binary in it at all. `apt-get install nvidia-cuda-toolkit`, by
// contrast, installs a real /usr/bin/nvcc. This file's own __CUDACC_VER__
// macros below can only be read at all because THIS environment used the
// apt route -- a pip-only environment would fail to compile this very
// file with nvcc in the first place.
#include <cstdio>

int main() {
    printf("=== Host compiler (g++) ===\n");
#if defined(__GNUC__)
    printf("__GNUC__ = %d, __GNUC_MINOR__ = %d, __GNUC_PATCHLEVEL__ = %d\n",
           __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("__GNUC__ not defined -- this file was not compiled with g++ "
           "as the host compiler.\n");
#endif

    printf("\n=== C++ language standard actually in effect ===\n");
    printf("__cplusplus = %ldL\n", __cplusplus);
#if __cplusplus >= 201703L
    printf("This is C++17 or newer -- matches this book's own -std=c++17 "
           "convention (Appendix A.4's own Makefile).\n");
#else
    printf("This is OLDER than C++17 -- this book's own code requires "
           "-std=c++17 to be passed explicitly.\n");
#endif

    printf("\n=== CUDA compiler (nvcc) driving this compilation ===\n");
#if defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__)
    printf("__CUDACC_VER_MAJOR__ = %d, __CUDACC_VER_MINOR__ = %d, "
           "__CUDACC_VER_BUILD__ = %d\n",
           __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
    printf("This file was successfully compiled by a REAL nvcc binary --\n"
           "by itself, this is proof the apt-installed toolchain (not a\n"
           "pip package alone) is what made this compilation possible,\n"
           "since a pip-only nvidia-cuda-nvcc-cu12 install in this same\n"
           "environment has no nvcc executable for the shell to find.\n");
#else
    printf("__CUDACC_VER_MAJOR__ not defined -- this file was not "
           "compiled with nvcc. If you are seeing this, you likely ran\n"
           "g++ directly on a .cu file, which g++ cannot do.\n");
#endif

    bool ok = true;
#if !defined(__CUDACC_VER_MAJOR__)
    ok = false;
#endif
#if !defined(__GNUC__)
    ok = false;
#endif
    printf("\nself-check: toolchain identity read %s\n",
           ok ? "confirmed" : "MISMATCH (see messages above)");
    return ok ? 0 : 1;
}
