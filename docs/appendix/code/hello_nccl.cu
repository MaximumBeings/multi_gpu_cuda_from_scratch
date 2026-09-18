// Appendix A.4: Recipe 3 stub -- nvcc + NCCL headers/link line.
#include <cstdio>
#include <nccl.h>
int main() {
    printf("hello_nccl: compiled and linked against real libnccl "
           "(NCCL %d.%d.%d).\n", NCCL_MAJOR, NCCL_MINOR, NCCL_PATCH);
    return 0;
}
