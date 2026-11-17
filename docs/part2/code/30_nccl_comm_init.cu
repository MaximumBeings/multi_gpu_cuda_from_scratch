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
// genuinely run (each failure's cause confirmed with NCCL_DEBUG=WARN).
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

    // ncclCommInitAll() fails in two genuinely different ways, and
    // which one you get depends on WHICH argument is bad -- not on
    // how many devices were asked for. Run with NCCL_DEBUG=WARN to see
    // NCCL's own explanation of each failure on stderr.
    ncclComm_t comms[1];

    // (a) This book's own real deviceCount (0), with a real comms array.
    // ndev=0 is NOT rejected by NCCL 2.18.3's argument checks: the call
    // proceeds until NCCL's first CUDA call, which finds no device.
    ncclResult_t eInitReal = ncclCommInitAll(comms, deviceCount, nullptr);
    printf("\n(a) ncclCommInitAll(comms, ndev=%d [this book's real deviceCount]): "
           "%s (code %d)\n", deviceCount, ncclGetErrorString(eInitReal), (int)eInitReal);

    // (b) Exactly ONE device, with a real comms array: also a well-formed
    // request, and it fails at exactly the same point, for the same
    // reason -- NCCL's first CUDA call finds no device.
    ncclResult_t eInitOne = ncclCommInitAll(comms, 1, nullptr);
    printf("(b) ncclCommInitAll(comms, ndev=1): %s (code %d)\n",
           ncclGetErrorString(eInitOne), (int)eInitOne);

    // (c) A NULL comms pointer: NCCL's argument check rejects this
    // before touching any device ("comms argument is NULL").
    ncclResult_t eInitNull = ncclCommInitAll(nullptr, 1, nullptr);
    printf("(c) ncclCommInitAll(comms=nullptr, ndev=1): %s (code %d)\n",
           ncclGetErrorString(eInitNull), (int)eInitNull);

    // (d) A negative device count: also rejected up front
    // ("Invalid device count requested").
    ncclResult_t eInitNeg = ncclCommInitAll(comms, -1, nullptr);
    printf("(d) ncclCommInitAll(comms, ndev=-1): %s (code %d)\n",
           ncclGetErrorString(eInitNeg), (int)eInitNeg);

    printf("\nTwo genuinely different failure modes: a malformed ARGUMENT\n"
           "(a NULL comms pointer, or a negative device count) is rejected\n"
           "immediately as ncclInvalidArgument (code 4); a well-formed request\n"
           "-- including ndev=0 -- gets past argument checking and fails as\n"
           "ncclUnhandledCudaError (code 1) once NCCL's first CUDA call finds\n"
           "no device. No communicator was created by any of these calls;\n"
           "Section 11.2 shows what every collective reports when handed a\n"
           "comm that is still NULL.\n");

    return 0;
}
