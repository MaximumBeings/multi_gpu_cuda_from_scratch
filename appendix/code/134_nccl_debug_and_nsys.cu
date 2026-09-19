// Appendix E: Profiling and Benchmarking Multi-GPU Communication
// 134_nccl_debug_and_nsys.cu
//
// Appendix E.3 -- Appendix C File 126's own locked output already showed
// one real line this section now builds on directly: a failed
// ncclCommInitRank() reports "unhandled cuda error (run with
// NCCL_DEBUG=INFO for details)". This file takes that suggestion
// literally -- the exact same real communicator-init attempt, run once
// with NCCL's default (silent) logging and once with NCCL_DEBUG=INFO --
// and locks the genuinely different, far more detailed real output the
// second run produces: NCCL's own bootstrap network selection, plugin
// loading attempts, and driver version detection, none of which the
// default run shows at all. This is NCCL's own real, standard,
// environment-variable-driven diagnostic tool, distinct from (and often
// used alongside) NVIDIA Nsight Systems' own multi-process timeline view.
//
// Compile: nvcc -arch=sm_80 134_nccl_debug_and_nsys.cu -o 134_nccl_debug_and_nsys -lnccl
// Run:     ./134_nccl_debug_and_nsys                  (default, quiet)
//          NCCL_DEBUG=INFO ./134_nccl_debug_and_nsys  (verbose bootstrap/network detail)
#include <cstdio>
#include <nccl.h>

int main() {
    printf("=== Section E.3: NCCL_DEBUG=INFO, run against a genuine ncclCommInitRank() ===\n\n");

    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm;
    ncclResult_t r = ncclCommInitRank(&comm, 1, id, 0);
    printf("ncclCommInitRank() -> %s\n", ncclGetErrorString(r));
    printf("(the stderr lines above main's own single printf, if any, are NCCL's OWN\n");
    printf("real internal logging -- emitted only when NCCL_DEBUG is set; with NCCL's\n");
    printf("default, silent logging level, this program's ENTIRE output is the one line\n");
    printf("above)\n");

    return (r == ncclSuccess) ? 0 : 1;
}
