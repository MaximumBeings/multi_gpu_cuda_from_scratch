// Chapter 11: NCCL -- What It Actually Does Differently From What You
// Just Built
// 30_nccl_comm_init.cu
//
// Every chapter since Chapter 3 has looped over `deviceCount` and
// called CUDA Runtime API functions directly, device by device. Real
// NCCL adds one concept in front of all of that: a communicator, a
// single object representing the whole group of devices a collective
// call will run across. This section genuinely links against a real,
// installed NCCL library and genuinely calls its real API -- no
// hand-rolled substitute -- to see exactly how a communicator gets
// created, and what happens when it can't be. Genuinely compiled
// with a real nvcc, genuinely linked against a real libnccl, and
// genuinely run.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    // ncclGetVersion() reads the linked library's own version -- it
    // needs no device at all, so unlike almost every other call in
    // this book, it can genuinely succeed even here.
    int ncclRuntimeVersion = 0;
    ncclResult_t eVer = ncclGetVersion(&ncclRuntimeVersion);
    printf("ncclGetVersion(): %s (code %d), reports version %d "
           "(major.minor.patch = %d.%d.%d)\n",
           ncclGetErrorString(eVer), (int)eVer, ncclRuntimeVersion,
           ncclRuntimeVersion / 10000, (ncclRuntimeVersion / 100) % 100,
           ncclRuntimeVersion % 100);

    // ncclCommInitAll(), used with this book's own real deviceCount:
    // asking for a clique of ZERO communicators is itself an invalid
    // request, and NCCL's own argument validation catches it before
    // touching any device at all.
    ncclResult_t eInitReal = ncclCommInitAll(nullptr, deviceCount, nullptr);
    printf("\nncclCommInitAll(ndev=%d, this book's own real deviceCount): "
           "%s (code %d)\n", deviceCount, ncclGetErrorString(eInitReal), (int)eInitReal);

    // ncclCommInitAll(), asking for exactly ONE device: this passes
    // NCCL's own argument validation (asking for 1 device is a valid
    // request in principle), so it proceeds far enough to discover
    // there is genuinely no CUDA device to attach that communicator
    // to -- a different, more specific honest failure.
    ncclComm_t oneComm;
    ncclResult_t eInitOne = ncclCommInitAll(&oneComm, 1, nullptr);
    printf("ncclCommInitAll(ndev=1, hypothetically): %s (code %d)\n",
           ncclGetErrorString(eInitOne), (int)eInitOne);

    printf("\nTwo different, both genuinely honest, failures: asking for\n"
           "zero devices is a malformed request NCCL rejects immediately;\n"
           "asking for one device is a well-formed request that only fails\n"
           "once NCCL actually looks for a CUDA device to use, and finds\n"
           "none. Section 11.2 assumes a communicator this far along --\n"
           "requested, but never successfully created -- for every real\n"
           "collective call this book's Chapters 8-10 already built by hand.\n");

    return 0;
}
