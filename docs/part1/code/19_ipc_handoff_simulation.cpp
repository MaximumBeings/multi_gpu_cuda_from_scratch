// Chapter 7: CUDA Inter-Process Communication -- Sharing Memory and
// Events Across Processes
// 19_ipc_handoff_simulation.cpp
//
// Simulated "device memory table" -- stands in for the driver-level
// bookkeeping a real cudaIpcMemHandle_t addresses. This book's
// simulations exchange the exact real MESSAGE PATTERN real code uses
// (export an opaque handle, open it to get a usable reference), not
// the real opaque bytes -- the table index plays that role here.
#include <cstdio>
#include <vector>

struct SimAllocation {
    std::vector<int> data;
};

std::vector<SimAllocation> g_table;

// Export side: the simulated equivalent of cudaIpcGetMemHandle(). Per
// the real documentation, this "is a lightweight operation" -- it does
// not copy or move the underlying data, only hands back a reference to
// it, exactly like this function does.
int simExportHandle(int allocationIndex) {
    return allocationIndex; // the "opaque handle"
}

// Import side: the simulated equivalent of cudaIpcOpenMemHandle().
// Returns a reference to the SAME underlying storage the exporter
// owns -- no copy at all, which is the entire point of IPC memory
// sharing versus Chapter 5's cudaMemcpy-based transfers.
std::vector<int>& simOpenMemHandle(int handle) {
    return g_table[handle].data;
}

int main() {
    g_table.push_back(SimAllocation{ {100, 200, 300, 400} });
    int handle = simExportHandle(0);
    printf("Exporter allocated data and exported handle %d.\n", handle);

    std::vector<int>& imported = simOpenMemHandle(handle);
    printf("Importer opened handle %d.\n", handle);

    bool initialMatch = (imported == g_table[0].data);
    printf("Immediately after opening, importer's view matches exporter's data: %s\n",
           initialMatch ? "PASS" : "FAIL");

    // The defining property of real IPC memory sharing: it is the SAME
    // physical memory, not a copy. A mutation the exporter makes AFTER
    // the handle was opened must be visible to the importer immediately,
    // with no further transfer call of any kind.
    g_table[0].data[2] = 9999;
    bool liveMatch = (imported[2] == 9999);
    printf("After exporter mutates index 2 with NO further transfer call,\n"
           "importer's view reflects it immediately: %s\n", liveMatch ? "PASS" : "FAIL");

    printf("\n[Documented, NOT executed here -- this would be real undefined\n"
           "behavior on real hardware, so this simulation only prints the\n"
           "warning rather than reproducing it]: calling cudaFree() on the\n"
           "exporter's allocation before the importer calls\n"
           "cudaIpcCloseMemHandle() is undefined behavior, per the CUDA\n"
           "Runtime API documentation.\n");

    return (initialMatch && liveMatch) ? 0 : 1;
}
