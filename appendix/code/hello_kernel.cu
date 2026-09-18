// Appendix A.4: Recipe 2 stub -- plain nvcc, no additional libraries.
#include <cstdio>
__global__ void helloKernel() {
    // deliberately left with an empty body -- this recipe exists to
    // prove nvcc alone can compile and link a .cu file with a real
    // __global__ kernel, not to exercise runtime kernel launch (this
    // environment has no physical device to launch on).
}
int main() {
    printf("hello_kernel: compiled with plain nvcc, one real (unlaunched) "
           "__global__ kernel present.\n");
    return 0;
}
