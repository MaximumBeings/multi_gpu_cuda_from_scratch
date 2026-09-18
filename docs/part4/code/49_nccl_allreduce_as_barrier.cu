// Chapter 17: Barriers and Global Synchronization Across Devices
// 49_nccl_allreduce_as_barrier.cu
//
// NCCL's own Collective Operations documentation lists AllReduce,
// Broadcast, Reduce, AllGather, ReduceScatter, AllToAll, Gather, and
// Scatter -- no Barrier, the same kind of gap Chapter 11 found for
// all-to-all before NCCL added a dedicated call for it. MPI (Part 5)
// has a real, dedicated MPI_Barrier() -- Open MPI's own docs describe
// it plainly: "synchronization between MPI processes in a group,"
// completing "after all group members have entered the barrier."
// NCCL has no equivalent, so real distributed training code fakes one
// the same way a real, publicly filed NCCL GitHub issue (#808,
// "Understanding Barriers") describes discovering by observation: a
// barrier action, under the hood, "show[s] up as a ringreduce kernel"
// -- i.e. a throwaway AllReduce call, issued purely for the property
// that it cannot complete on any rank until every rank has called it.
// Genuinely compiled with a real nvcc against the real installed
// libnccl.
#include <cstdio>
#include <cuda_runtime.h>
#include <nccl.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int WORLD_SIZE = 4;
    ncclComm_t comms[WORLD_SIZE];
    ncclResult_t eInit = ncclCommInitAll(comms, WORLD_SIZE, nullptr);
    printf("\nncclCommInitAll(worldSize=%d): %s (code %d)\n",
           WORLD_SIZE, ncclGetErrorString(eInit), (int)eInit);

    // ncclCommInitAll() reported failure above WITHOUT populating
    // comms[] -- exactly like every real communicator this book has
    // created since Chapter 11, it was never successfully built. The
    // rest of this section uses an explicit nullptr communicator, the
    // same safe convention every chapter since Chapter 12 has used,
    // rather than dereferencing whatever comms[0] happens to still
    // hold after a failed init.
    ncclComm_t comm = nullptr;
    cudaStream_t stream = nullptr;

    // The "barrier" payload: a single throwaway element, never a real
    // tensor. What matters isn't what value comes back -- it's that
    // ncclAllReduce() cannot report completion on ANY rank until every
    // rank in the communicator has issued this exact call.
    float dummy = 0.0f;
    ncclResult_t eBarrier = ncclAllReduce(&dummy, &dummy, 1, ncclFloat, ncclSum, comm, stream);
    printf("ncclAllReduce(1-element dummy buffer, ncclSum) as a barrier: "
           "%s (code %d)\n", ncclGetErrorString(eBarrier), (int)eBarrier);

    // This is the trap this section's own COMMON TRAP names: the NCCL
    // call above only ENQUEUES work onto a stream -- exactly like
    // every kernel launch since Chapter 3 -- and returns to the CPU
    // immediately, whether or not any other rank has reached it yet.
    // Actually BLOCKING this CPU thread until the enqueued collective
    // has genuinely finished needs the same real primitive Chapter 6
    // introduced for exactly this reason.
    cudaError_t eSync = cudaStreamSynchronize(stream);
    printf("cudaStreamSynchronize(): %s (code %d)  <-- THIS is what "
           "actually makes the CPU thread wait; the NCCL call alone "
           "does not.\n", cudaGetErrorString(eSync), (int)eSync);

    printf("\nEvery call above reports an honest failure -- "
           "ncclCommInitAll() on a communicator that was never "
           "successfully created, and no CUDA stream was ever real -- "
           "but the sequence itself is the real pattern this book's own "
           "cited GitHub issue observed: a dummy AllReduce for the "
           "synchronization side effect, followed by a real stream/"
           "device sync to actually block the host. Section 17.3 "
           "checks what a barrier used this way is actually FOR: "
           "keeping every rank's view of shared state consistent.\n");

    return 0;
}
