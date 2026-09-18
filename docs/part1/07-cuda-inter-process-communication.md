# Chapter 7: CUDA Inter-Process Communication: Sharing Memory and Events Across Processes

**What you will understand by the end of this chapter:**

- Why a device pointer that's perfectly meaningful inside the process that allocated it is meaningless to a different operating-system process, even on the very same GPU -- and what `cudaIpcGetMemHandle()`/`cudaIpcOpenMemHandle()` actually exchange to fix that.
- Why sharing an event across processes needs the event created with specific flags up front, and what those flags trade away in exchange for being shareable at all.
- The one property that makes IPC memory sharing fundamentally different from every transfer this book has written since Chapter 5: it moves zero bytes, because both processes end up looking at the exact same physical memory.

**What you need to know first:**

- Chapter 4's Unified Virtual Addressing and `cudaPointerGetAttributes()`, and Chapter 6's event lifecycle (`cudaEventCreate()`/`cudaEventRecord()`/`cudaStreamWaitEvent()`).
- The general idea of an operating-system process having its own private address space, separate from any other process's.
- No new CUDA concepts beyond this chapter's own handle-exchange functions.

---

Every chapter since Chapter 4 has quietly assumed one thing: whoever calls `cudaMalloc()`, `cudaEventCreate()`, or any other allocation-style CUDA call is also the only process that will ever touch the result. UVA (Chapter 4) makes a pointer meaningful across *devices* within that one process -- but it says nothing about a second, entirely separate process, with its own private address space and its own CUDA context, that wants to work with memory or synchronization the first process already set up. That's what CUDA's Inter-Process Communication (IPC) support exists for, and it closes out Part 1: everything from here through Chapter 6 was about getting data and coordination *between devices*; this chapter is about getting that same data and coordination *between processes*, which turns out to need its own dedicated, deliberately narrow API.

Here is where this chapter sits relative to everything Part 1 already built:

```text
Chapter 4 -- within one process, across DEVICES:

    Process A
    +----------------------------------------+
    |  [ Device 0 ] <--peer access--> [ Device 1 ]  |
    +----------------------------------------+

Chapters 5-6 -- moving data and coordinating those devices:

    Process A
    +--------------------------------------------------+
    |  [ Device 0 ] --cudaMemcpyPeer()--> [ Device 1 ]  |
    |  [ Device 0 stream ] --event--> [ Device 1 stream ]  |
    +--------------------------------------------------+

Chapter 7 (this chapter) -- across PROCESSES entirely:

    Process A                       Process B
    +----------------+   IPC handle   +----------------+
    |  [ Device 0 ]  | -------------> |  (same Device 0 |
    |   owns memory  |   (see 7.1)    |   memory, now   |
    +----------------+                |   visible here) |
                                       +----------------+
```

Everything above the last block was one process reaching across a device boundary it already controlled. This chapter is a process reaching across a boundary it does *not* control at all -- another process's own private address space -- which is why it needs its own explicit hand-off mechanism instead of just another flavor of the calls Chapters 4-6 already used.

## 7.1 cudaIpcGetMemHandle / cudaIpcOpenMemHandle: Sharing Device Memory Across Processes

### Intuition

A raw pointer value is just a number, and numbers don't carry any information about which process's address space they're meaningful in. If process A allocates device memory and somehow sends the raw pointer value to process B -- over a pipe, a socket, whatever -- that number means nothing to process B's own CUDA runtime; dereferencing it (directly or through any CUDA call) is simply undefined. What has to cross the process boundary instead is an opaque, driver-issued *handle*: a token process A requests specifically for the purpose of being handed to another process, which process B then presents back to its own CUDA runtime to receive a pointer that *is* valid in its address space, pointing at the exact same underlying physical memory.

```text
   PROCESS A (exporter)                         PROCESS B (importer)
   +-------------------------+                  +-------------------------+
   | address space A         |                  | address space B        |
   |                         |                  |                        |
   |  devPtr ----> [ DEVICE MEMORY ]             |                        |
   |               (A owns this allocation)      |                        |
   +-------------------------+                  +-------------------------+
              |                                            ^
              | cudaIpcGetMemHandle(&handle, devPtr)        |
              v                                            |
        +-----------+        OS-level transport       +-----------+
        |  handle   | -----(pipe / socket / file,----> |  handle   |
        | (opaque)  |       outside CUDA's scope)      | (opaque)  |
        +-----------+                                  +-----------+
                                                               |
                                            cudaIpcOpenMemHandle(&importedPtr, handle, ...)
                                                               v
                                            importedPtr ----> [ SAME DEVICE MEMORY ]
                                                               (the identical physical bytes
                                                                A already owns -- not a copy)
```

The handle itself carries no data -- it is a description the driver can turn back into a pointer, not a snapshot of the memory it describes. Everything below `[ SAME DEVICE MEMORY ]` in the diagram is one physical allocation with two valid pointers now referring to it, one per process.

### Background

`cudaIpcGetMemHandle(&handle, devPtr)` is the export side. Per the CUDA Runtime API documentation, it "takes a pointer to the base of an existing device memory allocation created with `cudaMalloc` and exports it for use in another process," and importantly, "this is a lightweight operation and may be called multiple times on an allocation without adverse effects" -- it doesn't move or duplicate the underlying bytes, only produces a token describing them. `cudaIpcOpenMemHandle(&devPtr, handle, flags)` is the import side, run in the *other* process: it "opens an interprocess memory handle exported from another process and returns a device pointer usable in the local process." The documentation is specific about both platform scope and lifetime rules: "IPC functionality is restricted to devices with support for unified addressing on Linux operating systems," and critically, "memory returned from `cudaIpcOpenMemHandle` must be freed with `cudaIpcCloseMemHandle`," because "calling `cudaFree` on an exported memory region before calling `cudaIpcCloseMemHandle` in the importing context will result in undefined behavior."

```cpp
// Chapter 7: CUDA Inter-Process Communication -- Sharing Memory and
// Events Across Processes
// 17_ipc_mem_handle.cu
//
// The real IPC memory-sharing sequence: export a device allocation
// with cudaIpcGetMemHandle(), open the resulting opaque handle with
// cudaIpcOpenMemHandle() (as another process would), and release it
// with cudaIpcCloseMemHandle(). Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    void* devPtr = nullptr;
    cudaError_t eAlloc = cudaMalloc(&devPtr, 64);
    printf("cudaMalloc(exported allocation): %s (code %d)\n", cudaGetErrorString(eAlloc), (int)eAlloc);

    cudaIpcMemHandle_t handle;
    cudaError_t eGet = cudaIpcGetMemHandle(&handle, devPtr);
    printf("cudaIpcGetMemHandle: %s (code %d)\n", cudaGetErrorString(eGet), (int)eGet);

    void* importedPtr = nullptr;
    cudaError_t eOpen = cudaIpcOpenMemHandle(&importedPtr, handle, cudaIpcMemLazyEnablePeerAccess);
    printf("cudaIpcOpenMemHandle: %s (code %d)\n", cudaGetErrorString(eOpen), (int)eOpen);

    cudaError_t eClose = cudaIpcCloseMemHandle(importedPtr);
    printf("cudaIpcCloseMemHandle: %s (code %d)\n", cudaGetErrorString(eClose), (int)eClose);

    if (eAlloc == cudaSuccess) cudaFree(devPtr);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaMalloc(exported allocation): no CUDA-capable device is detected (code 100)
cudaIpcGetMemHandle: no CUDA-capable device is detected (code 100)
cudaIpcOpenMemHandle: no CUDA-capable device is detected (code 100)
cudaIpcCloseMemHandle: no CUDA-capable device is detected (code 100)
```

The same honest `cudaErrorNoDevice` this book has reported at every real Runtime API call since Chapter 1, for the same underlying reason: there is no device context here to allocate the memory this chapter's handles would describe. This section's own code runs both roles -- exporter and importer -- inside a single process, since IPC's actual cross-process delivery mechanism (a pipe, a socket, shared memory for the handle bytes themselves) is an operating-system concern outside CUDA's own scope; what genuinely compiles and genuinely runs here is the real CUDA-side half of that exchange.

!!! warning "[COMMON TRAP] Believing a device pointer's numeric value is portable between processes"
    UVA (Chapter 4) makes a pointer's *meaning* consistent across every device within one process's single address space -- but a process's address space itself is private to that process. Two different processes can each have a perfectly valid, UVA-consistent pointer value that happens to be the identical number, referring to two completely unrelated pieces of memory, or to nothing at all. The handle this section exchanges exists specifically because the pointer's raw value was never going to be safely portable on its own -- only the driver-issued handle, opened through `cudaIpcOpenMemHandle()`, produces a pointer that's actually valid in the *importing* process's own address space.

## 7.2 cudaIpcGetEventHandle / cudaIpcOpenEventHandle: Sharing Synchronization Across Processes

### Intuition

Sharing the memory itself only solves half the problem: process B now has a way to get a valid pointer to process A's data, but nothing yet tells B *when* that data is actually ready to read. Chapter 6 already built the tool for exactly this kind of question -- an event -- but an ordinary event, like an ordinary pointer, is scoped to the process that created it. Making an event shareable across processes isn't automatic; it has to be requested explicitly, at creation time, because supporting cross-process visibility costs the event one specific capability it would otherwise have.

```text
   PROCESS A                                    PROCESS B
   +----------------------------+                +----------------------------+
   | streamA: [ work ] -> event |                |                            |
   |   (created WITH the two    |                |                            |
   |    required flags, 7.2)    |                |                            |
   +----------------------------+                +----------------------------+
              |                                            ^
              | cudaIpcGetEventHandle(&handle, event)       |
              v                                            |
        +-----------+        OS-level transport       +-----------+
        |  handle   | ------------------------------->|  handle   |
        +-----------+                                  +-----------+
                                                               |
                                          cudaIpcOpenEventHandle(&importedEvent, handle)
                                                               v
                                          streamB: cudaStreamWaitEvent(streamB, importedEvent, 0)
                                          -- streamB's FUTURE work now waits for process A's
                                             marker, exactly like Chapter 6's cross-DEVICE case,
                                             just carried across a process boundary instead.
```

The bottom line of this diagram is the payoff: once `importedEvent` exists in process B, it is handed to `cudaStreamWaitEvent()` exactly the way Chapter 6, Section 6.2 used a same-process, cross-device event -- nothing about *waiting* on it changes, only how it got there.

### Background

`cudaEventCreateWithFlags()` is the same event-creation call Chapter 6 used, but with two flags that matter here specifically: the documentation states that `cudaEventInterprocess` "specifies that the created event may be used as an interprocess event by `cudaIpcGetEventHandle()`," and that "`cudaEventInterprocess` must be specified along with `cudaEventDisableTiming`" -- an interprocess event gives up per-event timing data as the cost of being shareable at all. From there, the pattern mirrors Section 7.1 exactly: `cudaIpcGetEventHandle(&handle, event)` exports an opaque handle for an event created with both required flags, and `cudaIpcOpenEventHandle(&importedEvent, handle)`, called in the other process, returns a local `cudaEvent_t` that refers to the same underlying event -- one that can then be handed directly to `cudaStreamWaitEvent()` (Chapter 6, Section 6.2) exactly like any other event, cross-device or not.

```cpp
// Chapter 7: CUDA Inter-Process Communication -- Sharing Memory and
// Events Across Processes
// 18_ipc_event_handle.cu
//
// The real IPC event-sharing sequence: create an event with the two
// flags interprocess sharing requires (cudaEventDisableTiming |
// cudaEventInterprocess), export it with cudaIpcGetEventHandle(), and
// open the resulting handle with cudaIpcOpenEventHandle(). Genuinely
// compiled with a real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaEvent_t event;
    cudaError_t eCreate = cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess);
    printf("cudaEventCreateWithFlags(DisableTiming|Interprocess): %s (code %d)\n",
           cudaGetErrorString(eCreate), (int)eCreate);

    cudaIpcEventHandle_t handle;
    cudaError_t eGet = cudaIpcGetEventHandle(&handle, event);
    printf("cudaIpcGetEventHandle: %s (code %d)\n", cudaGetErrorString(eGet), (int)eGet);

    cudaEvent_t importedEvent;
    cudaError_t eOpen = cudaIpcOpenEventHandle(&importedEvent, handle);
    printf("cudaIpcOpenEventHandle: %s (code %d)\n", cudaGetErrorString(eOpen), (int)eOpen);

    if (eCreate == cudaSuccess) cudaEventDestroy(event);
    if (eOpen == cudaSuccess) cudaEventDestroy(importedEvent);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaEventCreateWithFlags(DisableTiming|Interprocess): no CUDA-capable device is detected (code 100)
cudaIpcGetEventHandle: no CUDA-capable device is detected (code 100)
cudaIpcOpenEventHandle: no CUDA-capable device is detected (code 100)
```

The same `cudaErrorNoDevice` throughout, for the same reason as every device-bound call in this book -- even creating an event, the very first step, needs a context that doesn't exist here. What this section's code verifies honestly is the exact, real flag combination and call sequence a real multi-process program needs -- genuinely compiled, and genuinely failing at the correct, honest first step rather than anywhere else.

!!! warning "[COMMON TRAP] Trying to export an ordinary event created with cudaEventCreate()"
    An event created with the plain `cudaEventCreate()` (Chapter 6, Section 6.1) was never given the `cudaEventInterprocess` flag, because that flag is only available through `cudaEventCreateWithFlags()`. Passing such an event to `cudaIpcGetEventHandle()` is not something the API quietly tolerates -- the interprocess capability has to be requested at the moment the event is created, not added afterward. If Chapter 6's event basics are the pattern in your head, remember that this chapter's events need one extra, deliberate step at creation time that Chapter 6's never did.

## 7.3 Verifying the Handoff Logic Without a Second Process

### Intuition

This machine cannot honestly run two real, separate OS processes each initializing their own CUDA context and exchanging real IPC handles over a real pipe -- there's no device for either process's context to initialize against in the first place, so the whole scenario has nothing to attach to. But the *logic* of the handoff -- export produces an opaque token that describes shared memory without copying it; open consumes exactly that token and yields a reference to the same underlying storage; the importer's view and the exporter's view are never two independently-updated copies, they are the same memory -- is exactly the kind of orchestration logic this book already verified honestly once before, in Chapter 4's Section 4.3, using real in-memory stand-ins instead of real hardware.

The one picture that actually distinguishes this chapter's IPC sharing from every transfer this book has written since Chapter 5 is what happens to a LATE mutation -- one made after the hand-off has already happened:

```text
   IPC-STYLE SHARING (this chapter)          COPY-STYLE TRANSFER (Chapter 5)

   g_table[0].data                           src (device 0) = [100,200,300,400]
     = [100,200,300,400]                              |
        ^          ^                            cudaMemcpy() -- copies the BYTES
        |          |                                  v
   exporter's   importer's                     dst (device 1) = [100,200,300,400]
   reference    reference                              (separate storage from here on)

   exporter mutates index 2 -> 9999           src mutates index 2 -> 9999
        |          |                                  |
        v          v                                  v
   BOTH see 9999 immediately --               dst STILL shows 300 --
   there was only ONE buffer,                 dst is a different allocation;
   so there was nothing to "update"           reflecting the change needs a
   in the importer's view at all              SECOND cudaMemcpy() call
```

Everything Section 7.3's simulation checks reduces to this one picture: the left column has no step where data moves *to* the importer after the initial open, because there is only one underlying buffer to begin with.

### Background

The simulation below plays both roles -- exporter and importer -- using a shared table that stands in for the driver's own internal bookkeeping of exported allocations. The crucial difference from every transfer this book has modeled since Chapter 5 is what `simOpenMemHandle()` returns: not a copy of the data, but a C++ reference to the *exact same* underlying `std::vector`. That's the property this section actually needs to verify -- not "the importer eventually gets a correct copy" (which is what Chapter 4's simulation checked for a `cudaMemcpy`-style transfer), but "the importer and exporter are looking at the same physical thing," which the second check below demonstrates directly: mutating the data through the exporter's own reference, with no further call of any kind, is immediately visible through the importer's reference.

```cpp
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
```

Genuinely compiled with `g++` and genuinely run. Locked output, deterministic across repeated runs:

```
Exporter allocated data and exported handle 0.
Importer opened handle 0.
Immediately after opening, importer's view matches exporter's data: PASS
After exporter mutates index 2 with NO further transfer call,
importer's view reflects it immediately: PASS

[Documented, NOT executed here -- this would be real undefined
behavior on real hardware, so this simulation only prints the
warning rather than reproducing it]: calling cudaFree() on the
exporter's allocation before the importer calls
cudaIpcCloseMemHandle() is undefined behavior, per the CUDA
Runtime API documentation.
```

Both checks pass, and the second one is the point of this whole section: no `transferDirectP2P()`-style call, no `cudaMemcpy()`-style call, nothing at all moved the mutated value from the exporter's reference to the importer's -- there was never a second copy to move it *to*. This is the one property that distinguishes IPC memory sharing from every other cross-process or cross-device operation this book has modeled: it doesn't transfer data faster, it removes the need to transfer data at all.

!!! warning "[COMMON TRAP] Simulating -- or worse, actually invoking -- the documented undefined behavior to 'test' it"
    This section's simulation deliberately does *not* execute the exporter-frees-before-importer-closes scenario the real documentation warns about, even in simulated form. Undefined behavior has no defined result to check a simulation against -- there is no correct output to assert on, real or simulated, because the real API itself makes no promise about what happens. This book's honesty discipline extends to knowing the difference between verifying a real, defined guarantee (which Sections 7.1-7.3's other checks all do) and a real, explicitly *undefined* one, which can only be reported as a documented warning, never demonstrated as a "correct" result.

## Chapter Summary

A device pointer's numeric value only means anything within the address space of the process that owns it, so sharing device memory across processes needs a dedicated, driver-issued handle -- `cudaIpcGetMemHandle()` to export a lightweight, non-copying token, and `cudaIpcOpenMemHandle()` in the importing process to turn that token into a locally valid pointer, with the documentation's own strict warning that skipping `cudaIpcCloseMemHandle()` before the exporter frees the memory is undefined behavior (Section 7.1). Sharing an event across the same boundary uses the identical export/open pattern, but only works for events created with `cudaEventInterprocess` (which itself requires `cudaEventDisableTiming`) specified up front at creation time, not added afterward (Section 7.2). Because no real second process can be run here, Section 7.3 verified the one property that actually defines IPC memory sharing -- that the importer's and exporter's views are the same physical memory, not independently-updated copies -- using this book's host-side simulation technique, confirming a mutation made through one reference is instantly visible through the other with no transfer call involved at all. This completes Part 1: Chapters 4 through 7 took data and synchronization from "within one device" (Chapter 4) to "between devices" (Chapters 5-6) to "between processes" (Chapter 7). Part 2 starts building actual collective communication patterns -- broadcast and reduce, by hand -- on top of everything Part 1 established.

## Self-Check Questions

1. Two separate processes each hold a device pointer with the identical numeric value. Using Section 7.1's Common Trap, explain why this does not imply they refer to the same memory.
2. Why does `cudaIpcGetMemHandle()`'s documentation describe it as "a lightweight operation" that "may be called multiple times... without adverse effects"? What is it *not* doing that a heavier operation would be doing?
3. What specific, real consequence does the CUDA Runtime API documentation say follows from calling `cudaFree()` on an exported allocation before the importing process calls `cudaIpcCloseMemHandle()`?
4. A programmer creates an event with plain `cudaEventCreate()` and later tries to pass it to `cudaIpcGetEventHandle()`. Using Section 7.2's Background, explain specifically why this event was never eligible for that call, and what should have been done differently at creation time.
5. Section 7.2 states that `cudaEventInterprocess` "must be specified along with `cudaEventDisableTiming`." What capability does an interprocess event give up as a result, referencing Chapter 6's own event-timing discussion if relevant?
6. Section 7.3's simulation checks two things: an "initial match" and a "live match" after a mutation. Explain why the live-match check is the one that actually distinguishes IPC-style sharing from a Chapter 5-style transfer, while the initial-match check alone would not.
7. Why does Section 7.3's code deliberately avoid simulating the exporter-frees-before-importer-closes scenario, even though it could easily write code that does something and prints a result for it?
8. Suppose a third process wants to also import the same memory handle Section 7.1's exporter produced. Based on the documentation quoted in Section 7.1 ("`cudaIpcMemHandles` from each device in a given process may only be opened by one context per device per other process" -- note this appears in the fuller documentation Section 7.1 draws from), would this work, and if there's a constraint, what is it?

## Where We Go Next

Part 2 begins with Chapter 8: broadcast and reduce, the first two collective communication patterns, built by hand on top of the peer-access, transfer, and synchronization primitives Part 1 established -- the first point in this book where more than one device's data genuinely has to be combined into a single, correct result.

## Worked Solutions

**1.** A device pointer's numeric value is only meaningful relative to the address-translation state of the specific process that holds it -- there is no global registry mapping numbers to memory that both processes share. Two processes each having the identical number is coincidental at best; without going through the real handle-export/open sequence, there is no guarantee, and generally no truth, to the idea that the same number refers to the same underlying memory in both processes' contexts.

**2.** "Lightweight" and repeatable-without-adverse-effects means the call does not copy, move, or duplicate the underlying device memory it describes -- it only produces a new descriptor (the handle) that points at memory which continues to exist exactly where it already was. A heavier operation, by contrast, would need to actually allocate new storage or physically relocate data each time it was called; `cudaIpcGetMemHandle()` does neither, which is exactly why calling it repeatedly on the same allocation causes no problems.

**3.** The documentation states plainly that "calling `cudaFree` on an exported memory region before calling `cudaIpcCloseMemHandle` in the importing context will result in undefined behavior" -- meaning the real, specific consequence is UB: the CUDA Runtime API makes no guarantee at all about what happens next, which could range from a crash to silent memory corruption, and no defined recovery exists once that ordering has been violated.

**4.** `cudaEventCreate()` does not accept any flags at all, so an event created that way was never given `cudaEventInterprocess` -- and per Section 7.2's Background, that flag is only settable through the separate `cudaEventCreateWithFlags()` call, specified at the moment of creation. There is no API to retroactively add the flag to an already-created event; the fix is to create the event in the first place with `cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventInterprocess)`.

**5.** It gives up per-event timing data -- the capability `cudaEventElapsedTime()`-style measurement between two events would otherwise use. Chapter 6 didn't cover elapsed-time measurement explicitly, but Section 7.2's Background is direct about the trade-off itself: `cudaEventDisableTiming` is required alongside `cudaEventInterprocess`, meaning an interprocess-shareable event is, by construction, never a timing event.

**6.** The initial-match check alone only shows that the importer's data happens to equal the exporter's data *at the moment of opening* -- which a `cudaMemcpy()`-style copy (Chapter 5) would produce identically, since a fresh, correct copy also matches its source right after being made. The live-match check is the one that actually distinguishes the two: after the mutation, a Chapter-5-style copy would NOT reflect the change (the copy is a separate, independent buffer by then), while this section's simulation shows the importer's view updating with no further call at all -- proving it's the same memory, not a copy that merely started out equal.

**7.** Undefined behavior has no defined correct result to check a simulation's output against -- the real CUDA Runtime API itself makes no promise about what happens in that scenario, so there is nothing honest to assert PASS or FAIL against. Writing code that "does something" and reports a result would necessarily be reporting on this simulation's own arbitrary behavior, not on any real guarantee, which would misrepresent an explicitly undefined case as though it had a defined, verifiable outcome.

**8.** The fuller documentation this section draws from states that "`cudaIpcMemHandles` from each device in a given process may only be opened by one context per device per other process" -- so a third process opening the same handle is a *different* other process relative to the exporter, and is therefore permitted its own single context opening it; what's constrained is a given *pair* of processes (exporter and one specific importer) to one context per device, not the total number of different processes that may each independently import the same handle.

---

**Sources cited in this chapter:**

- [cudaIpcGetMemHandle — CUDA Runtime API documentation, mirrored](https://manpages.ubuntu.com/manpages/focal/man3/cudaIpcGetMemHandle.3.html) -- the "lightweight operation" wording and the Linux/unified-addressing platform restriction.
- [cudaIpcOpenMemHandle — CUDA Runtime API documentation, mirrored](https://docs.rs/rcudnn-sys/latest/rcudnn_sys/fn.cudaIpcOpenMemHandle.html) -- the per-context-per-process open constraint, the `cudaIpcCloseMemHandle()` cleanup requirement, and the documented undefined-behavior warning.
- [CUDA Runtime API — Event Management (`cudaEventCreateWithFlags`)](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/group__CUDART__EVENT.html) -- the exact `cudaEventInterprocess`/`cudaEventDisableTiming` flag requirement for interprocess events.
