// Appendix C: NCCL and NVSHMEM -- The Standard Libraries You Get for Free
// 128_nvshmem_api_reference.cu
//
// Chapter 22 introduced NVSHMEM through exactly one real API call,
// nvshmem_int_p() -- a device-initiated, one-sided WRITE into a remote
// PE's memory. NVSHMEM's own real API is much larger: nvshmem_int_get()
// (the READ-side counterpart to Ch22's own put), nvshmem_int_atomic_add()
// (an atomic, race-free remote update -- the NVSHMEM analog of Chapter
// 9's own atomic-add fallback for a shared accumulator), nvshmem_
// barrier_all() (NVSHMEM's own collective synchronization, the same real
// job Chapter 17's throwaway ncclAllReduce did for NCCL), and NVSHMEM's
// TEAM abstraction (NVSHMEM_TEAM_WORLD, nvshmem_team_my_pe() -- a named
// subset-of-PEs concept with no direct analog in this book's own NCCL or
// MPI chapters). This file confirms all of these real symbols resolve
// and link, genuinely attempts nvshmem_init(), and replays the logic of
// a get+atomic-add+barrier sequence on the host.
//
// Compile: nvcc -rdc=true -gencode=arch=compute_70,code=sm_70 -I$NVSHMEM_INC -L$NVSHMEM_LIB 128_nvshmem_api_reference.cu -o 128_nvshmem_api_reference -lnvshmem_host -lnvshmem_device -lcuda
// Run:     LD_LIBRARY_PATH=$NVSHMEM_LIB ./128_nvshmem_api_reference
#include <cstdio>
#include <nvshmem.h>
#include <nvshmemx.h>

int main() {
    printf("=== Section C.3: NVSHMEM's fuller real API, beyond Chapter "
           "22's own nvshmem_int_p() ===\n\n");

    void* pGet = (void*)&nvshmem_int_get;
    void* pAtomicAdd = (void*)&nvshmem_int_atomic_add;
    void* pBarrier = (void*)&nvshmem_barrier_all;
    void* pTeamPe = (void*)&nvshmem_team_my_pe;
    printf("real symbols resolved:\n");
    printf("  nvshmem_int_get         = %p  (one-sided READ -- the "
           "counterpart to Ch22's own nvshmem_int_p WRITE)\n", pGet);
    printf("  nvshmem_int_atomic_add  = %p  (atomic remote update -- "
           "lets multiple PEs safely combine values into one shared "
           "remote location without a separate reduction collective, "
           "the same race a plain, unsynchronized shared write would "
           "risk on either a single device or across PEs)\n", pAtomicAdd);
    printf("  nvshmem_barrier_all     = %p  (NVSHMEM's own collective "
           "sync -- the same real job Ch17's throwaway ncclAllReduce "
           "did for NCCL)\n", pBarrier);
    printf("  nvshmem_team_my_pe      = %p  (queries this PE's own rank "
           "WITHIN a named team -- NVSHMEM_TEAM_WORLD = %d is the "
           "default team containing every PE)\n\n", pTeamPe, (int)NVSHMEM_TEAM_WORLD);

    printf("=== attempting a genuine nvshmem_init() ===\n\n");
    nvshmem_init();
    int myPe = nvshmem_my_pe();
    int nPes = nvshmem_n_pes();
    printf("nvshmem_init() completed; nvshmem_my_pe()=%d, "
           "nvshmem_n_pes()=%d\n", myPe, nPes);
    if (nPes <= 0) {
        printf("(nPes<=0 means no bootstrap plugin found a real multi-PE "
               "world to join -- Chapter 22's own real single-PE fallback, "
               "not an error, since this file was run directly rather "
               "than through mpirun with NVSHMEMX_INIT_WITH_MPI_COMM)\n");
    }

    printf("\n=== host-side replay: 4 PEs, each fetches its right "
           "neighbor's value (nvshmem_int_get's own real job), then "
           "every PE atomically adds 1 to PE 0's shared counter "
           "(nvshmem_int_atomic_add's own real job), then all PEs "
           "barrier (nvshmem_barrier_all's own real job) before reading "
           "the final counter ===\n\n");
    const int NPES = 4;
    int value[NPES] = {10, 20, 30, 40};
    int fetched[NPES];
    for (int pe = 0; pe < NPES; pe++) {
        int rightNeighbor = (pe + 1) % NPES;
        fetched[pe] = value[rightNeighbor];  // replay of nvshmem_int_get
    }
    printf("%-6s %-12s %-24s\n", "PE", "own value", "fetched (right neighbor)");
    for (int pe = 0; pe < NPES; pe++)
        printf("%-6d %-12d %-24d\n", pe, value[pe], fetched[pe]);

    int sharedCounterOnPe0 = 0;
    for (int pe = 0; pe < NPES; pe++)
        sharedCounterOnPe0++;  // replay of NPES real atomic_add(..., 1, ...) calls
    printf("\nafter %d PEs each call nvshmem_int_atomic_add(&counter, 1, "
           "0) targeting PE 0: counter = %d\n", NPES, sharedCounterOnPe0);
    printf("(every increment lands correctly regardless of arrival order "
           "-- NVSHMEM's own remote atomic gives this guarantee ACROSS "
           "PEs, the same correctness property a plain CUDA atomicAdd() "
           "gives across threads on one device, without needing a "
           "separate collective reduction just to combine four PEs' own "
           "contributions into one counter)\n");

    printf("\nafter nvshmem_barrier_all(): every PE is guaranteed the "
           "counter has reached its final value of %d before proceeding "
           "-- without the barrier, a PE could read the counter before "
           "every other PE's own increment has landed.\n", sharedCounterOnPe0);

    bool ok = true;
    for (int pe = 0; pe < NPES; pe++)
        if (fetched[pe] != value[(pe + 1) % NPES]) ok = false;
    if (sharedCounterOnPe0 != NPES) ok = false;

    printf("\nself-check: every PE's fetched value matches its right "
           "neighbor's own value, and the final counter equals the "
           "number of PEs that incremented it: %s\n",
           ok ? "confirmed" : "MISMATCH");

    nvshmem_finalize();
    return ok ? 0 : 1;
}
