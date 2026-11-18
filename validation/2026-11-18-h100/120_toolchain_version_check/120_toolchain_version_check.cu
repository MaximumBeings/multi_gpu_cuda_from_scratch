// 120_toolchain_version_check.cu
// Compiler identity; version macros do not establish the package installation method.
#include <cstdio>
int main(){
    printf("C++ language level: %ld\n",(long)__cplusplus);
#ifdef __GNUC__
    printf("GNU-compatible host compiler macros: %d.%d.%d\n",__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__);
#endif
#ifdef __CUDACC_VER_MAJOR__
    printf("CUDA compiler: %d.%d.%d\n",__CUDACC_VER_MAJOR__,__CUDACC_VER_MINOR__,__CUDACC_VER_BUILD__);
    bool ok=__cplusplus>=201703L;
    printf("Compiler identity: %s (installation source is not inferred)\n",ok?"PASS":"FAIL");return ok?0:1;
#else
    fprintf(stderr,"Compile this .cu translation unit with nvcc and -std=c++17.\n");return 1;
#endif
}
