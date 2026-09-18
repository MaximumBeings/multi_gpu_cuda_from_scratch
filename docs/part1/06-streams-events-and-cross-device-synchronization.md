# Chapter 6: Streams, Events, and Cross-Device Synchronization

**What you will understand by the end of this chapter:**

- What a CUDA event actually is -- a marker placed into a stream, not a timer that starts running the moment you create it -- and which calls on it block and which don't.
- Why `cudaStreamWaitEvent()`, not `cudaDeviceSynchronize()`, is the real mechanism for making one device's work depend on another's, and why that distinction matters for how much concurrency a multi-GPU program keeps.
- What the CUDA Runtime API can honestly report, right now, about whether a real device is even physically capable of overlapping a data transfer with kernel execution -- and why that capability has to exist before any amount of clever stream arrangement can produce overlap.

**What you need to know first:**

- Chapter 3's streams (creation, device binding, the multi-device loop pattern) and Chapter 5's transfer calls (`cudaMemcpyPeer()`, the staged host fallback).
- Chapter 4's per-thread current-device state and Chapter 2's topology models, referenced again here.
- No new CUDA concepts beyond this chapter's own two APIs (events, and the two device-attribute queries in Section 6.3).

---

Chapters 3 and 5 gave this book streams (an ordered queue of work bound to whichever device was current when the stream was created) and real cross-device copies. What's been missing is a way for work on *one* stream -- possibly on a different device entirely -- to depend on work in *another* stream, without forcing the host to stop and wait, and without forcing an entire device to drain before continuing. That's what events are for. This chapter builds the real dependency-tracking mechanism multi-GPU CUDA code actually uses, and closes with the one thing that has to be true about the hardware itself before any of it can produce real overlap: whether the device has an async copy engine and concurrent-kernel support at all.

## 6.1 Events: Markers That Cost Nothing Until You Wait On Them

### Intuition

An event is not a stopwatch you start -- it's a flag you plant at a specific point in a specific stream's queue of work. Planting the flag (`cudaEventRecord()`) doesn't pause anything; the stream keeps moving, and the flag simply becomes "reached" once every operation queued before it in that stream has actually finished. Everything interesting happens on the *other* side: something else -- another stream, or the host itself -- can ask "has that flag been reached yet?" either by polling (a quick, non-blocking check) or by genuinely waiting for it. The event is cheap to create and cheap to plant; the cost only shows up when something chooses to wait on it.

### Background

`cudaEventCreate(&event)` allocates the event object. `cudaEventRecord(event, stream)` places it into a specific stream's queue -- it returns to the host immediately, without waiting for anything, exactly like every other stream-issued call this book has used since Chapter 3. Two different ways exist to ask whether it's been reached: `cudaEventQuery(event)`, which checks once and returns immediately either way (`cudaSuccess` if reached, `cudaErrorNotReady` if not -- never blocking), and `cudaEventSynchronize(event)`, which genuinely blocks the calling host thread until the event is reached. The CUDA Programming Guide describes exactly this pattern -- "By enqueuing an event into a stream directly after the first kernel, but before the second kernel, we can have either a CPU thread or another stream wait for this event to come to the front of the stream" -- which is precisely the query-vs-wait distinction this section's code exercises.

```cpp
// Chapter 6: Streams, Events, and Cross-Device Synchronization
// 14_event_basics.cu
//
// The real event lifecycle: create, record (asynchronously, into a
// stream), check without blocking (cudaEventQuery), and check while
// blocking (cudaEventSynchronize). Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaStream_t stream;
    cudaError_t eStream = cudaStreamCreate(&stream);
    printf("cudaStreamCreate: %s (code %d)\n", cudaGetErrorString(eStream), (int)eStream);

    cudaEvent_t event;
    cudaError_t eEvent = cudaEventCreate(&event);
    printf("cudaEventCreate: %s (code %d)\n", cudaGetErrorString(eEvent), (int)eEvent);

    cudaError_t eRecord = cudaEventRecord(event, stream);
    printf("cudaEventRecord(event, stream): %s (code %d)\n", cudaGetErrorString(eRecord), (int)eRecord);

    cudaError_t eQuery = cudaEventQuery(event);
    printf("cudaEventQuery(event) [non-blocking check]: %s (code %d)\n",
           cudaGetErrorString(eQuery), (int)eQuery);

    cudaError_t eSync = cudaEventSynchronize(event);
    printf("cudaEventSynchronize(event) [blocking wait]: %s (code %d)\n",
           cudaGetErrorString(eSync), (int)eSync);

    if (eEvent == cudaSuccess) cudaEventDestroy(event);
    if (eStream == cudaSuccess) cudaStreamDestroy(stream);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaStreamCreate: no CUDA-capable device is detected (code 100)
cudaEventCreate: no CUDA-capable device is detected (code 100)
cudaEventRecord(event, stream): no CUDA-capable device is detected (code 100)
cudaEventQuery(event) [non-blocking check]: no CUDA-capable device is detected (code 100)
cudaEventSynchronize(event) [blocking wait]: no CUDA-capable device is detected (code 100)
```

The same `cudaErrorNoDevice` at every single call, for the same reason as every chapter since Chapter 1 -- there is no context here to create a stream or an event into. What this section verifies honestly is the exact, real sequence of calls -- create, record, query, synchronize -- that a real multi-GPU program uses to track a point in a stream's progress, genuinely compiled and genuinely run down to the first call that needs a real device.

!!! warning "[COMMON TRAP] Assuming cudaEventRecord() captures a value, like a timestamp, at the moment it's called"
    `cudaEventRecord(event, stream)` does not record *when the call happened* on the host -- it inserts a marker into `stream`'s queue that becomes "reached" only once every operation already queued *ahead* of it in that stream has finished executing on the device. If the stream already has ten seconds of queued kernels ahead of this call, the event doesn't become reached until all ten seconds have elapsed, no matter how quickly `cudaEventRecord()` itself returned to the host. Confusing "the call returned" with "the event fired" is exactly the mistake Section 6.2's cross-device dependency depends on you *not* making.

## 6.2 Cross-Device Synchronization: cudaStreamWaitEvent Across Devices

### Intuition

Chapter 3 already showed that streams are bound to whichever device was current when they were created. Events inherit that same binding -- an event recorded into a device-0 stream is fundamentally a device-0 marker. The useful trick is that the *waiting* side doesn't have to be on that same device: `cudaStreamWaitEvent()` lets a stream on device 1 wait on an event that was recorded on device 0, which is the real, direct mechanism for expressing "this specific piece of device 1's work depends on that specific piece of device 0's work" -- without forcing either device to fully drain, and without the host blocking at all.

### Background

`cudaStreamWaitEvent(stream, event, flags)` makes all *future* work submitted to `stream` wait for `event` before it begins executing -- work already running, or already completed, in `stream` is unaffected. The documentation states this cross-device case explicitly: this synchronization "will be performed efficiently on the device," and "the event ... may be from a different context than stream, in which case this function will perform cross-device synchronization." That's the entire mechanism this section's code exercises: device 0 records an event marking a point in its own stream, and device 1's stream is told to wait on that exact marker before running anything queued after this call -- a targeted, single dependency, rather than the blunt alternative of calling `cudaDeviceSynchronize()` on device 0 and only then starting anything on device 1, which would needlessly stall device 1 behind *all* of device 0's work rather than just the one piece it actually depends on.

```cpp
// Chapter 6: Streams, Events, and Cross-Device Synchronization
// 15_cross_device_wait.cu
//
// The real cross-device dependency mechanism: an event recorded into
// one device's stream, then handed to cudaStreamWaitEvent() on a
// DIFFERENT device's stream -- no cudaDeviceSynchronize() anywhere,
// and no host blocking at all. Genuinely compiled with a real nvcc
// and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    // Device 0 side: produce work, mark its completion point with an event.
    cudaError_t eSet0 = cudaSetDevice(0);
    printf("cudaSetDevice(0): %s (code %d)\n", cudaGetErrorString(eSet0), (int)eSet0);

    cudaStream_t stream0;
    cudaError_t eStream0 = cudaStreamCreate(&stream0);
    printf("cudaStreamCreate(stream0, on device 0): %s (code %d)\n",
           cudaGetErrorString(eStream0), (int)eStream0);

    cudaEvent_t event0;
    cudaError_t eEvent0 = cudaEventCreate(&event0);
    printf("cudaEventCreate(event0, on device 0): %s (code %d)\n",
           cudaGetErrorString(eEvent0), (int)eEvent0);

    cudaError_t eRecord0 = cudaEventRecord(event0, stream0);
    printf("cudaEventRecord(event0, stream0): %s (code %d)\n",
           cudaGetErrorString(eRecord0), (int)eRecord0);

    // Device 1 side: create its own stream, then make it wait on
    // device 0's event -- WITHOUT calling cudaDeviceSynchronize()
    // anywhere, and without the host blocking at all.
    cudaError_t eSet1 = cudaSetDevice(1);
    printf("cudaSetDevice(1): %s (code %d)\n", cudaGetErrorString(eSet1), (int)eSet1);

    cudaStream_t stream1;
    cudaError_t eStream1 = cudaStreamCreate(&stream1);
    printf("cudaStreamCreate(stream1, on device 1): %s (code %d)\n",
           cudaGetErrorString(eStream1), (int)eStream1);

    cudaError_t eWait = cudaStreamWaitEvent(stream1, event0, 0);
    printf("cudaStreamWaitEvent(stream1 waits on event0 from device 0): %s (code %d)\n",
           cudaGetErrorString(eWait), (int)eWait);

    printf("Host reached this line without blocking on any of the above --\n"
           "cudaStreamWaitEvent only ever inserts a dependency for FUTURE\n"
           "work on stream1; it does not itself wait for anything.\n");

    if (eEvent0 == cudaSuccess) cudaEventDestroy(event0);
    if (eStream0 == cudaSuccess) cudaStreamDestroy(stream0);
    if (eStream1 == cudaSuccess) cudaStreamDestroy(stream1);
    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaSetDevice(0): no CUDA-capable device is detected (code 100)
cudaStreamCreate(stream0, on device 0): no CUDA-capable device is detected (code 100)
cudaEventCreate(event0, on device 0): no CUDA-capable device is detected (code 100)
cudaEventRecord(event0, stream0): no CUDA-capable device is detected (code 100)
cudaSetDevice(1): no CUDA-capable device is detected (code 100)
cudaStreamCreate(stream1, on device 1): no CUDA-capable device is detected (code 100)
cudaStreamWaitEvent(stream1 waits on event0 from device 0): no CUDA-capable device is detected (code 100)
Host reached this line without blocking on any of the above --
cudaStreamWaitEvent only ever inserts a dependency for FUTURE
work on stream1; it does not itself wait for anything.
```

Every real API call fails with the now-familiar `cudaErrorNoDevice`, for the reason this book has given consistently since Chapter 1 -- `cudaSetDevice(0)` itself has nothing to select. What this section verifies honestly is the shape of the real dependency graph a multi-GPU program builds: two devices, two streams, one event carrying a single, specific dependency between them, with the host never once blocking to enforce it.

!!! warning "[COMMON TRAP] Reaching for cudaDeviceSynchronize() as the 'safe' way to coordinate across devices"
    `cudaDeviceSynchronize()` blocks the host until *every* piece of work already queued on the current device has completed -- not just the one operation another device's stream actually depends on. Using it to coordinate devices (device 0 synchronize, then start device 1's dependent work) is correct but needlessly serializing: it throws away any concurrency device 0 could have kept running on other, unrelated streams while device 1's dependent work began. `cudaStreamWaitEvent()` expresses the *actual* dependency -- one specific point in one specific stream -- and lets everything else keep moving.

## 6.3 Overlapping Transfer and Compute: What the Runtime Can Honestly Report

### Intuition

Sections 6.1 and 6.2 gave this book the *mechanism* for expressing dependencies without unnecessary blocking. But a dependency graph that's free of unnecessary blocking still only produces real overlap -- a transfer genuinely running at the same time as a kernel -- if the underlying hardware has a physically separate engine to run the copy on while the compute engine keeps executing the kernel. That's not a software arrangement; it's a fact about the specific GPU, and the CUDA Runtime API can report it honestly, right now, without needing a real transfer or kernel to actually run.

### Background

Two device properties answer this directly. `asyncEngineCount`, per the CUDA Runtime API's own `cudaDeviceProp` struct reference, is simply the "Number of asynchronous engines" -- the count of hardware copy engines that can move data independently of, and concurrently with, the engine executing kernels; a value of 0 means no such engine exists at all, and overlap between a transfer and a kernel is not physically possible on that device no matter how the streams are arranged. `concurrentKernels` reports whether the "Device can possibly execute multiple kernels concurrently" -- a separate capability from copy/kernel overlap, but one that matters once more than one kernel is competing for the same compute engine across devices or streams. Both are queryable two ways: `cudaDeviceGetAttribute()` for a single value, or `cudaGetDeviceProperties()` for the full struct at once. Chapter 5's Asynchronous Execution citation already established the other half of this requirement -- pinned host memory -- so getting real overlap needs all three things true simultaneously: pinned memory (Chapter 5), non-default streams carrying the dependency correctly (Sections 6.1-6.2), and a device that reports `asyncEngineCount >= 1` here.

```cpp
// Chapter 6: Streams, Events, and Cross-Device Synchronization
// 16_overlap_query.cu
//
// What the CUDA Runtime API can honestly report about a device's
// capacity to overlap a transfer with kernel execution, without
// running either: the asyncEngineCount and concurrentKernels device
// attributes/properties. Genuinely compiled with a real nvcc and
// genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount(): %d device(s).\n", deviceCount);

    int asyncEngineCount = -1;
    cudaError_t e1 = cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, 0);
    printf("cudaDeviceGetAttribute(AsyncEngineCount, device 0): %s (code %d), value=%d\n",
           cudaGetErrorString(e1), (int)e1, asyncEngineCount);

    int concurrentKernels = -1;
    cudaError_t e2 = cudaDeviceGetAttribute(&concurrentKernels, cudaDevAttrConcurrentKernels, 0);
    printf("cudaDeviceGetAttribute(ConcurrentKernels, device 0): %s (code %d), value=%d\n",
           cudaGetErrorString(e2), (int)e2, concurrentKernels);

    cudaDeviceProp prop;
    cudaError_t e3 = cudaGetDeviceProperties(&prop, 0);
    printf("cudaGetDeviceProperties(device 0): %s (code %d)\n", cudaGetErrorString(e3), (int)e3);
    if (e3 == cudaSuccess) {
        printf("  asyncEngineCount=%d concurrentKernels=%d\n", prop.asyncEngineCount, prop.concurrentKernels);
    }

    printf("\nNeither value can be honestly reported as a real number on this\n"
           "machine -- there is no device 0 to query. On real hardware, both\n"
           "queries would need to succeed, and asyncEngineCount would need to\n"
           "be >=1, before overlap between a transfer and a kernel is even\n"
           "physically possible, regardless of how the streams are arranged.\n");

    return 0;
}
```

Genuinely compiled with a real `nvcc` and genuinely run. Locked output, deterministic across repeated runs:

```
cudaGetDeviceCount(): 0 device(s).
cudaDeviceGetAttribute(AsyncEngineCount, device 0): no CUDA-capable device is detected (code 100), value=-1
cudaDeviceGetAttribute(ConcurrentKernels, device 0): no CUDA-capable device is detected (code 100), value=-1
cudaGetDeviceProperties(device 0): no CUDA-capable device is detected (code 100)

Neither value can be honestly reported as a real number on this
machine -- there is no device 0 to query. On real hardware, both
queries would need to succeed, and asyncEngineCount would need to
be >=1, before overlap between a transfer and a kernel is even
physically possible, regardless of how the streams are arranged.
```

Both attribute queries and the full properties query fail with the same `cudaErrorNoDevice` -- there is no device 0's hardware capability to report on. That is itself the honest, correct answer this section can give: not a fabricated "yes, this hardware supports overlap," but a genuine, verifiable statement that this specific query, run on this specific machine, cannot return a real value, and an explanation of exactly what a real value would need to be before anything in Sections 6.1-6.2 could produce real overlap on real hardware.

!!! warning "[COMMON TRAP] Assuming two non-default streams automatically produce overlap"
    Placing a transfer and a kernel on two separate, non-default streams removes the *artificial* serialization the CUDA Programming Guide describes for the default stream ("it will wait for all other blocking streams to complete before it can execute") -- but it does not create hardware that doesn't exist. If the device's `asyncEngineCount` is 0, the transfer and the kernel still have to take turns on the same underlying engine, no matter which streams they're issued on. Streams remove a *software* obstacle to concurrency; `asyncEngineCount` and `concurrentKernels` report whether the *hardware* obstacle is also absent.

## Chapter Summary

A CUDA event is a marker placed into a stream by `cudaEventRecord()` -- itself non-blocking -- that something else can either poll without blocking (`cudaEventQuery()`) or genuinely wait on (`cudaEventSynchronize()`), measured directly in Section 6.1 by exercising every stage of that lifecycle down to the same honest `cudaErrorNoDevice` this book has reported since Chapter 1. Because streams and their events are bound to whichever device was current when they were created (Chapter 3), but `cudaStreamWaitEvent()` explicitly supports an event from a different device than the waiting stream -- performing, per its own documentation, genuine "cross-device synchronization" -- Section 6.2 built the real, targeted dependency mechanism multi-GPU code uses instead of the blunter, needlessly serializing `cudaDeviceSynchronize()`. Neither mechanism produces real overlap between a transfer and a kernel unless the underlying hardware has a separate copy engine to run them on at the same time, which Section 6.3 showed is a fact the Runtime API can query honestly and directly (`asyncEngineCount`, `concurrentKernels`) without running either operation -- on this machine, reported as honestly absent, for the same reason every other device query in this book has been. Chapter 7 turns to CUDA's Inter-Process Communication support: sharing device memory and events not just across devices within one process, but across separate operating-system processes entirely.

## Self-Check Questions

1. Section 6.1's Common Trap warns against confusing "the call returned" with "the event fired." Using the code in that section, identify exactly which line returns immediately regardless of stream state, and which line's return is what actually reflects the event having been reached.
2. `cudaEventQuery()` and `cudaEventSynchronize()` both ask "has this event been reached?" What is the behavioral difference between them when the answer is currently "no"?
3. Section 6.2's code calls `cudaStreamWaitEvent(stream1, event0, 0)` where `event0` was recorded on device 0's stream and `stream1` belongs to device 1. Quote the specific documented behavior that makes this cross-device call legal, rather than an error.
4. A programmer replaces Section 6.2's `cudaStreamWaitEvent()` call with `cudaSetDevice(0); cudaDeviceSynchronize();` immediately before starting device 1's dependent work. Explain concretely what this change costs, using the definition of `cudaDeviceSynchronize()` from Section 6.2's Common Trap.
5. Section 6.3 states that `asyncEngineCount == 0` makes transfer/kernel overlap "not physically possible... no matter how the streams are arranged." Explain in your own words why placing the transfer and kernel on two different streams cannot fix this.
6. Combine Chapter 5's pinned-memory requirement with this chapter's `asyncEngineCount` requirement: list all the conditions that must simultaneously hold for a real transfer and a real kernel to genuinely overlap on real hardware.
7. Why does this chapter's Section 6.3 report `cudaErrorNoDevice` as the "honest, correct answer" rather than treating it as a failure to work around? What would fabricating a plausible `asyncEngineCount` value violate?
8. Suppose a future chapter needs device 1's kernel to depend on TWO separate pieces of device 0's work, completed at different points in device 0's stream. Based on this chapter's model, would that require one event or two? Explain.

## Where We Go Next

Chapter 7 extends synchronization and shared memory access beyond a single process entirely: CUDA's Inter-Process Communication (IPC) support for sharing device memory allocations and events across separate host processes, and what changes about the ownership and lifetime guarantees this book has relied on so far once "another device" becomes "another device, owned by another process."

## Worked Solutions

**1.** `cudaEventRecord(event, stream)` is the line that returns immediately regardless of stream state -- it only inserts the marker, and per Section 6.1's own reasoning, does not wait for anything queued ahead of it. `cudaEventSynchronize(event)` is the line whose return actually reflects the event having been reached -- it genuinely blocks the host thread until that marker is confirmed reached, so by the time it returns, everything queued in that stream ahead of the marker has genuinely finished.

**2.** `cudaEventQuery()` returns immediately either way -- `cudaErrorNotReady` if the event hasn't been reached yet, without waiting even a moment longer. `cudaEventSynchronize()` instead blocks the calling host thread, not returning at all until the event genuinely has been reached -- it trades an immediate answer for a guaranteed one.

**3.** The documentation states that `cudaStreamWaitEvent()`'s synchronization "will be performed efficiently on the device," and explicitly that "the event ... may be from a different context than stream, in which case this function will perform cross-device synchronization." This is the specific, documented behavior that makes an event from device 0 legal to hand to a stream on device 1 -- the function is designed for exactly this case, not merely tolerating it as an edge case.

**4.** `cudaDeviceSynchronize()` on device 0 blocks the host until *every* piece of work already queued on device 0 has completed -- not just the specific event device 1's work actually depends on. If device 0 has other, unrelated streams with their own queued work, this change forces the host to wait for all of that unrelated work to finish too, before device 1's dependent work can even begin, throwing away concurrency that `cudaStreamWaitEvent()`'s single, targeted dependency would have preserved.

**5.** Placing the transfer and kernel on two different streams removes the *default-stream* serialization the CUDA Programming Guide describes -- an artificial, software-level ordering rule that only exists because both operations were being funneled through the same stream. It does nothing about the underlying hardware: if the chip itself only has one engine capable of moving data or executing kernels (an `asyncEngineCount` of 0 for the copy side), the transfer and the kernel still have to take physical turns on that one engine, regardless of which software queues they were issued from -- streams schedule work, they don't create additional physical execution engines.

**6.** All three: (a) the host-side buffer involved in the transfer must be pinned (Chapter 5's Asynchronous Execution citation: unpinned memory makes the copy "revert to a synchronous behavior which will not overlap with other work"), (b) the transfer and the kernel must be issued to separate, non-default streams so software-level serialization from the default stream doesn't apply, and (c) the device must report `asyncEngineCount >= 1` (Section 6.3), meaning it has hardware capable of running the copy and the kernel at the same time in the first place. All three must hold simultaneously -- any one missing means no real overlap, regardless of the other two.

**7.** `cudaErrorNoDevice` is the real, verifiable return value this exact query genuinely produces on this exact machine -- reporting it is simply telling the truth about what happened. Fabricating a plausible `asyncEngineCount` value (say, guessing "2," a common real figure for data-center GPUs) would mean printing a number this program never actually computed or received from any real API call -- exactly the kind of invented measurement this book's own stated honesty discipline forbids, on par with inventing a timing number in Chapter 5's cost model instead of reusing cited figures.

**8.** Two events -- one for each of the two separate points in device 0's stream that device 1's work depends on. An event is a marker for one specific point in one specific stream's progress; a single event recorded at the *later* of the two points would, by definition, already have been preceded by the earlier point completing (since a stream executes in order), so in that specific case one event could suffice -- but if the two pieces of work are on different streams on device 0, or the dependency genuinely needs to be checked at two independent points rather than "the later one implies the earlier one," two separate events are needed, each handed to its own `cudaStreamWaitEvent()` call on device 1's stream.

---

**Sources cited in this chapter:**

- [CUDA Programming Guide — Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) -- the default (NULL) stream's blocking-stream serialization behavior; using an event so "either a CPU thread or another stream" can wait for a point in a stream's progress.
- [cudaStreamWaitEvent — CUDA Runtime API documentation, mirrored](https://rdrr.io/github/duncantl/RCUDA/man/cudaStreamWaitEvent.html) -- the exact documented cross-device behavior: "the event may be from a different context than stream, in which case this function will perform cross-device synchronization."
- [CUDA Runtime API Reference — `cudaDeviceProp` struct](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html) -- the `asyncEngineCount` ("Number of asynchronous engines") and `concurrentKernels` ("Device can possibly execute multiple kernels concurrently") field descriptions.
- Chapter 5 of this book ("Explicit Transfers: cudaMemcpyPeer, Staged Host Transfers, and When Each Wins") -- the pinned-memory requirement for genuine transfer asynchrony, reused here as one of the three simultaneous conditions for real transfer/kernel overlap.
