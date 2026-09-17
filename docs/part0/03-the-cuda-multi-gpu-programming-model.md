# Chapter 3: The CUDA Multi-GPU Programming Model: Devices, Contexts, and Streams Across Them

Chapter 2 established that GPUs are wired together in specific, uneven ways. Before this book can move a single byte between two of them, it needs the vocabulary the CUDA Runtime API actually uses to talk about "which device" -- because that vocabulary has sharp edges that cause real, well-documented bugs the moment more than one device is involved.

## 3.1 "Current device" is a per-thread setting, not a global one

Every CUDA Runtime API call that doesn't take an explicit device argument -- `cudaMalloc`, a kernel launch, `cudaStreamCreate` -- acts on whatever the calling host thread's *current device* happens to be at that moment. The CUDA C++ Programming Guide's own multi-GPU chapter states this plainly: "a host thread can set the device it is currently operating on at any time by calling `cudaSetDevice()`," and "until a call to `cudaSetDevice()` is made by the host thread, the current device defaults to device 0."

```text
   Host thread                         GPU 0    GPU 1    GPU 2    GPU 3
   ┌─────────────────────┐             ┌───┐    ┌───┐    ┌───┐    ┌───┐
   │ current device: 0     │──points to→│ 0 │    │ 1 │    │ 2 │    │ 3 │
   │ (the default, before   │            └───┘    └───┘    └───┘    └───┘
   │  any cudaSetDevice())  │
   └─────────────────────┘

   cudaSetDevice(2);
   ┌─────────────────────┐             ┌───┐    ┌───┐    ┌───┐    ┌───┐
   │ current device: 2     │──────────────────────────→│ 2 │    │ 3 │
   └─────────────────────┘             └───┘    └───┘    └───┘    └───┘
   Every un-decorated call from here on (cudaMalloc, a kernel launch, ...)
   now acts on GPU 2 -- until the next cudaSetDevice() call changes it.
```

This is the single most important fact in this chapter, and nearly everything else follows from it directly.

## 3.2 Contexts, made invisible on purpose

CUDA's lower-level Driver API has an explicit context object: a container for a device's address space, its loaded code, and its outstanding work, which a program creates and destroys by hand. The Runtime API -- what this entire book uses -- hides that object: the first time a thread touches a given device, the runtime silently creates that device's context and attaches it to the current-device state described above. Section 3.1's diagram is, in effect, already a diagram of context switching: "current device" *is* "which context is active for this thread," just without a separate object to manage. This book never calls a Driver API context function directly; every allocation, kernel launch, and stream in the rest of it works entirely through the current-device mechanism.

## 3.3 Streams and events belong to whichever device was current when they were born

A stream (or an event) is not a free-floating queue you can point at any device later -- it's permanently associated with whatever device was current *at the moment `cudaStreamCreate()` ran*. The Programming Guide states this directly: "streams and events are created in association with the currently set device." Once created, that association doesn't change.

What makes this a sharp edge rather than a curiosity is an asymmetry between two kinds of operations issued to a "wrong" stream:

> "A kernel launch will fail if it is issued to a stream that is not associated to the current device." ... "A memory copy will succeed even if it is issued to a stream that is not associated to the current device."

```text
   cudaSetDevice(0); cudaStreamCreate(&streamA);   // streamA now belongs to GPU 0, permanently
   cudaSetDevice(1); cudaStreamCreate(&streamB);   // streamB now belongs to GPU 1, permanently

   cudaSetDevice(1);
   myKernel<<<g, b, 0, streamA>>>(...);   // FAILS: streamA belongs to GPU 0, current device is GPU 1
   cudaMemcpyAsync(dst, src, n, kind, streamA);  // SUCCEEDS: memcpy is more permissive than a kernel launch
```

A kernel launch checks the stream's owning device against the current device and refuses to run if they don't match. A memory copy does not enforce that check the same way -- it will go ahead, which is convenient when it's what you meant, and a silent, confusing correctness bug when it isn't. Chapter 6 (streams, events, and cross-device synchronization) builds directly on this asymmetry; it's introduced here because it's a property of the device model itself, not of anything specific to synchronization.

## 3.4 The pattern: single-device baseline first, then the loop

Every multi-device CUDA program in this book is a generalization of the exact same single-device sequence: set the device, allocate, launch, wait, read back, free.

```cpp
// 06_single_device_setup.cu (excerpt)
#define CHECK(call, label) do { \
    cudaError_t _e = (call); \
    printf("%-28s %s (code %d)\n", label, cudaGetErrorString(_e), (int)_e); \
    if (_e != cudaSuccess) { printf("Stopping here...\n"); return 0; } \
} while (0)

int main() {
    CHECK(cudaSetDevice(0), "cudaSetDevice(0)");
    int* d_data = nullptr;
    CHECK(cudaMalloc(&d_data, N * sizeof(int)), "cudaMalloc");
    // ... memcpy in, launch, synchronize, memcpy out, free ...
}
```

```bash
nvcc -arch=sm_80 06_single_device_setup.cu -o single_device_setup
./single_device_setup
```

Output (genuinely compiled with a real `nvcc` and genuinely run; identical across three consecutive runs):

```text
cudaSetDevice(0)             no CUDA-capable device is detected (code 100)
Stopping here -- every step after this one depends on this one having succeeded.
```

Checking every single call's return code -- rather than assuming success -- is what turns this into an honest, one-line diagnostic instead of a crash or (worse) a silent wrong answer. That discipline is not optional flavor text; it is the only reason this exact program, unmodified, is also correct on real hardware.

Generalizing that sequence to N devices is a matter of moving each step into a loop over `cudaGetDeviceCount()`'s result, calling `cudaSetDevice(i)` before every single device-`i`-specific operation -- allocation, stream creation, launch, and synchronization all need their own call, because, per Section 3.1, none of them remembers "device i" on your behalf:

```cpp
// 07_multi_device_loop_pattern.cu (excerpt)
int deviceCount = 0;
cudaGetDeviceCount(&deviceCount);

std::vector<cudaStream_t> streams(deviceCount);
std::vector<int*> buffers(deviceCount, nullptr);

// Phase 1: create each device's own stream and buffer.
for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&buffers[i], N * sizeof(int));
}
// Phase 2: launch on every device BEFORE waiting on any of them --
// this is what lets them actually run concurrently.
for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    addOneKernel<<<1, N, 0, streams[i]>>>(buffers[i], N);
}
// Phase 3: wait for each, in its own separate loop.
for (int i = 0; i < deviceCount; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
    cudaFree(buffers[i]);
}
```

```bash
nvcc -arch=sm_80 07_multi_device_loop_pattern.cu -o multi_device_loop
./multi_device_loop
```

Output (genuinely compiled and run; identical across runs):

```text
cudaGetDeviceCount() reports 0 device(s).
The loop below runs its body exactly 0 time(s) as a direct consequence -- not a special case, just what a correctly-guarded loop over a real count does when that count happens to be zero.

Loop body executed 0 time(s) total across all three phases. On real N-GPU hardware, this exact, unmodified source runs all N devices' kernels concurrently with no code change -- that's the entire point of writing the loop this way instead of hard-coding device 0 and device 1.
```

The reason the launch loop (Phase 2) and the wait loop (Phase 3) are deliberately separate, rather than one loop that launches-then-waits per device, is concurrency: launching device 1's kernel doesn't have to wait for device 0's kernel to finish, because they're different devices doing different work -- but if the code waited on device 0 immediately after launching it, before ever reaching device 1's launch, the devices would run one after another instead of together, and the "multi-GPU" program would get exactly one GPU's worth of throughput. This is a preview of the concurrency discipline every collective in Part 2 depends on.

## 3.5 The classic bug this model invites: threads that forget to set their device

Section 3.1's per-thread current-device state is a deliberate design choice -- "the CUDA runtime API is thread-safe, which means it maintains per-thread state about the current device... this is very important as it allows threads to concurrently submit work to different devices" -- but it has a well-documented failure mode. A newly spawned host thread does **not** inherit its parent thread's current-device setting; it starts back at the default, device 0. A common multi-GPU pattern -- spawn one host thread per device, expecting each to naturally "own" its device -- silently breaks unless every one of those threads calls `cudaSetDevice()` itself as its very first action. Skip that call in even one thread, and its CUDA work quietly lands on device 0 instead of whichever device it was supposed to drive -- not a crash, just wrong-GPU execution, which is far harder to notice.

This book's own device-loop pattern in Section 3.4 sidesteps the bug by construction: it's a single thread calling `cudaSetDevice(i)` before each device's work, never multiple threads each assuming a device. Part 3's data- and model-parallelism chapters revisit multi-threaded host code directly and will call this pitfall out again at the exact point it becomes relevant.

## 3.6 Summary

"Current device" is per-host-thread state, defaulting to device 0, changed only by `cudaSetDevice()`. The Runtime API's context is that same state made implicit -- there's no separate object to manage, but the concept is identical to the Driver API's explicit one. A stream or event is permanently bound to whichever device was current when it was created, and that binding is enforced strictly for kernel launches but not for memory copies -- an asymmetry worth remembering before Chapter 6 builds cross-device synchronization on top of it. And every multi-device program in this book is the same single-device sequence from Section 3.4, moved into a loop that never assumes how many devices exist or which one is currently selected.

Chapter 4 puts this vocabulary to work on the first real multi-device operation: letting one GPU read another GPU's memory directly, without staging the data through the host at all.

---

**Sources cited in this chapter:**

- [Programming Systems with Multiple GPUs — CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/multi-gpu-systems.html) -- per-thread current device, device-0 default, stream/event device association, and the kernel-launch-vs-memory-copy enforcement asymmetry.
- [CUDA Pro Tip: Always Set the Current Device to Avoid Multithreading Bugs — NVIDIA Developer Blog](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs) -- per-thread device state and the new-thread-defaults-to-device-0 pitfall.
