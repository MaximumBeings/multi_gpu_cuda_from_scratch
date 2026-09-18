// Chapter 17: Barriers and Global Synchronization Across Devices
// 48_per_device_sync_is_not_a_barrier.cu
//
// Chapter 6 established cudaDeviceSynchronize(): it blocks the calling
// CPU thread until ONE device's own queued work has finished. Every
// parallelization strategy Part 3 built -- data (Ch12), model (Ch13),
// tensor (Ch14), pipeline (Ch15), domain decomposition (Ch16) -- needs
// something stronger at certain points: a guarantee that EVERY device
// has reached the same point before ANY of them is allowed to
// continue past it. This section shows, directly, that looping
// cudaDeviceSynchronize() over every device does not provide that
// guarantee, even though it looks like it should.
// Genuinely compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    const int HYPOTHETICAL_WORLD_SIZE = 4;
    printf("\nSynchronizing %d devices the way every device loop since "
           "Chapter 3 has -- one at a time, on this one CPU thread:\n",
           HYPOTHETICAL_WORLD_SIZE);
    for (int rank = 0; rank < HYPOTHETICAL_WORLD_SIZE; ++rank) {
        cudaError_t eSetDevice = cudaSetDevice(rank);
        cudaError_t eSync = cudaDeviceSynchronize();
        printf("  rank %d: cudaSetDevice(): %s (code %d)   "
               "cudaDeviceSynchronize(): %s (code %d)\n",
               rank, cudaGetErrorString(eSetDevice), (int)eSetDevice,
               cudaGetErrorString(eSync), (int)eSync);
    }

    printf("\nNotice what this loop actually guarantees, and what it does\n"
           "NOT: by the time it returns, this CPU thread knows every\n"
           "device's queue, AS OF THE MOMENT EACH ONE WAS CHECKED, was\n"
           "empty. It says nothing about whether rank 0 and rank 3 were\n"
           "AT THE SAME POINT in their own work relative to each other.\n"
           "Rank 0 could have already raced ten iterations ahead of rank\n"
           "3 by the time rank 3's own kernels are even launched -- this\n"
           "loop only checks each device's queue in turn, sequentially,\n"
           "on one CPU thread; it never makes any device wait for any\n"
           "OTHER device. A real global barrier needs every rank to wait\n"
           "for every OTHER rank, not just for its own queue to drain.\n");

    return 0;
}
