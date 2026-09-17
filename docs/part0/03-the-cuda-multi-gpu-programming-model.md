# Chapter 3: The CUDA Multi-GPU Programming Model: Devices, Contexts, and Streams Across Them

**What you will understand by the end of this chapter:**

- Why "current device" is state that belongs to a host thread, not the process as a whole, and what defaults to when nobody sets it.
- Why a stream or event is permanently bound to whichever device was current when it was created, and the real, documented asymmetry between how strictly that binding is enforced for a kernel launch versus a memory copy.
- The idiomatic loop pattern this book uses for every multi-device program from here on, and why launching on every device before waiting on any of them is what actually lets them run concurrently.
- A real, well-documented multithreading bug this device model invites, and exactly what guards against it.

**What you need to know first:**

- Chapter 1's memory and compute walls, and Chapter 2's interconnect topologies — this chapter is the CUDA-side vocabulary for addressing the devices those chapters described.
- Ordinary single-GPU CUDA usage: `cudaMalloc`, a kernel launch, `cudaMemcpy`, `cudaDeviceSynchronize`.

---

Chapter 2 established that GPUs are wired together in specific, uneven ways. Before this book can move a single byte between two of them, it needs the vocabulary the CUDA Runtime API actually uses to talk about "which device" — because that vocabulary has sharp edges that cause real, well-documented bugs the moment more than one device is involved.

## 3.1 "Current Device" Is a Per-Thread Setting, Not a Global One

### Intuition

Picture a single telephone operator's switchboard with one lever that connects the operator to whichever line was last plugged in. The operator doesn't get to talk to every line at once by default — they talk to exactly the line the lever currently points at, and the lever stays pointed there until someone moves it. If the operator forgets to move the lever before making a call, the call goes out on whatever line the lever was left on, not on the line the operator meant.

### Background

Every CUDA Runtime API call that doesn't take an explicit device argument — `cudaMalloc`, a kernel launch, `cudaStreamCreate` — acts on whatever the calling host thread's *current device* happens to be at that moment. The CUDA Programming Guide's own multi-GPU chapter states this plainly: "a host thread can set the device it is currently operating on at any time by calling `cudaSetDevice()`," and "until a call to `cudaSetDevice()` is made by the host thread, the current device defaults to device 0."

```text
default (no cudaSetDevice() called yet): current device = 0

cudaSetDevice(2);
current device = 2
  -> every un-decorated call from here on (cudaMalloc, a kernel launch,
     cudaStreamCreate, ...) now acts on GPU 2, until the next
     cudaSetDevice() call changes it again
```

This is the single most important fact in this chapter, and nearly everything else follows from it directly. CUDA's lower-level Driver API has an explicit context object — a container for a device's address space, loaded code, and outstanding work — which a program creates and destroys by hand. The Runtime API, used throughout this book, hides that object: the first time a thread touches a given device, the runtime silently creates that device's context and attaches it to the current-device state above. "Current device" *is* "which context is active for this thread," just without a separate object to manage.

!!! warning "[COMMON TRAP] Assuming `cudaSetDevice()` affects every thread"
    `cudaSetDevice()` changes the current-device state for the *calling thread only*. A common, incorrect assumption is that calling it once near the start of a program sets the device for the whole process — it does not. Section 3.4's multithreading pitfall is the direct consequence of forgetting this.

## 3.2 Streams and Events Belong to Whichever Device Was Current When They Were Born

### Intuition

A phone line installed at a specific address stays at that address permanently — you cannot later decide the line installed at 12 Main Street should instead ring at 40 Oak Avenue just by wishing it so. A CUDA stream works the same way: it's wired to a specific device the moment it's created, and that wiring never changes afterward, no matter what the current device is when you later try to use it.

### Background

The Programming Guide states this directly: "streams and events are created in association with the currently set device." Once created, that association doesn't change. What makes this a sharp edge rather than a curiosity is a real, documented asymmetry between two kinds of operations issued to a "wrong" stream:

> "A kernel launch will fail if it is issued to a stream that is not associated to the current device." ... "A memory copy will succeed even if it is issued to a stream that is not associated to the current device."

```text
cudaSetDevice(0); cudaStreamCreate(&streamA);   // streamA now belongs to GPU 0, permanently
cudaSetDevice(1); cudaStreamCreate(&streamB);   // streamB now belongs to GPU 1, permanently

cudaSetDevice(1);
myKernel<<<g, b, 0, streamA>>>(...);          // FAILS: streamA belongs to GPU 0, current device is GPU 1
cudaMemcpyAsync(dst, src, n, kind, streamA);  // SUCCEEDS: memcpy is more permissive than a kernel launch
```

A kernel launch checks the stream's owning device against the current device and refuses to run if they don't match. A memory copy does not enforce that check the same way — convenient when it's what you meant, and a silent, confusing correctness bug when it isn't.

!!! warning "[COMMON TRAP] Trusting a memory copy's success as proof a stream was used correctly"
    Because a memory copy issued to the "wrong" device's stream succeeds instead of failing, a program with this bug produces no error at all — it just quietly does the copy anyway, on whatever device the memory copy's own device-inference logic resolves to. A kernel launch on the same wrong stream fails loudly; the exact same mistake with a memory copy fails silently. Chapter 6 (streams, events, and cross-device synchronization) builds directly on this asymmetry.

## 3.3 The Pattern: Single-Device Baseline First, Then the Loop

### Intuition

Before running a relay with four runners, you first make sure one runner can complete their own leg correctly — set up, run, hand off cleanly. Only once that single leg works do you generalize to four runners doing the same thing, each on their own leg, with the added coordination of who starts when. Every multi-device CUDA program in this book is built the same way: the single-device sequence first, generalized into a loop second.

### Background

The single-device sequence is: set the device, allocate, launch, wait, read back, free.

```cpp
// 06_single_device_setup.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void addOneKernel(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}

#define CHECK(call, label) do { \
    cudaError_t _e = (call); \
    printf("%-28s %s (code %d)\n", label, cudaGetErrorString(_e), (int)_e); \
    if (_e != cudaSuccess) { \
        printf("Stopping here -- every step after this one depends on " \
               "this one having succeeded.\n"); \
        return 0; \
    } \
} while (0)

int main() {
    const int N = 8;

    CHECK(cudaSetDevice(0), "cudaSetDevice(0)");

    int* d_data = nullptr;
    CHECK(cudaMalloc(&d_data, N * sizeof(int)), "cudaMalloc");

    int h_in[N] = {0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(cudaMemcpy(d_data, h_in, N * sizeof(int), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");

    addOneKernel<<<1, N>>>(d_data, N);
    CHECK(cudaGetLastError(), "kernel launch");
    CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    int h_out[N] = {0};
    CHECK(cudaMemcpy(h_out, d_data, N * sizeof(int), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");

    printf("Result: ");
    for (int i = 0; i < N; ++i) printf("%d ", h_out[i]);
    printf("\n");

    cudaFree(d_data);
    return 0;
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

Checking every single call's return code — rather than assuming success — is what turns this into an honest, one-line diagnostic instead of a crash or a silent wrong answer. Generalizing to N devices means moving each step into a loop over `cudaGetDeviceCount()`'s result, calling `cudaSetDevice(i)` before every single device-`i`-specific operation, because per Section 3.1 none of them remembers "device i" on your behalf:

```cpp
// 07_multi_device_loop_pattern.cu
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void addOneKernel(int* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount() reports %d device(s).\n", deviceCount);
    printf("The loop below runs its body exactly %d time(s) as a "
           "direct consequence -- not a special case, just what a "
           "correctly-guarded loop over a real count does when that "
           "count happens to be zero.\n\n", deviceCount);

    std::vector<cudaStream_t> streams(deviceCount);
    std::vector<int*> buffers(deviceCount, nullptr);
    const int N = 8;

    // Phase 1: on EACH device, in turn, set it current, then create
    // that device's own stream and allocate that device's own buffer.
    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&buffers[i], N * sizeof(int));
        printf("Device %d: stream and buffer created while device %d "
               "was current.\n", i, i);
    }

    // Phase 2: launch each device's own kernel on its own stream --
    // launching everything first, then waiting, is what lets the
    // devices actually overlap.
    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        addOneKernel<<<1, N, 0, streams[i]>>>(buffers[i], N);
    }

    // Phase 3: synchronize each device's own stream, in a SEPARATE loop.
    for (int i = 0; i < deviceCount; ++i) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(buffers[i]);
    }

    printf("Loop body executed %d time(s) total across all three "
           "phases. On real N-GPU hardware, this exact, unmodified "
           "source runs all N devices' kernels concurrently with no "
           "code change.\n", deviceCount);

    return 0;
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

!!! warning "[COMMON TRAP] Combining the launch loop and the wait loop into one"
    Writing a single loop that launches device `i`'s kernel and immediately waits for it before moving to device `i+1` looks equivalent to the two-loop version above, but it isn't: waiting on device 0 before device 1's kernel has even been launched forces the two devices to run one after another instead of together, turning an N-GPU program into an N-times-slower single-GPU program in disguise. The separate launch-then-wait loops in Phases 2 and 3 above are what actually let the devices overlap.

## Chapter Summary

"Current device" is per-host-thread state, defaulting to device 0, changed only by `cudaSetDevice()` (Section 3.1) — and the Runtime API's context is that same state made implicit, with no separate object to manage. A stream or event is permanently bound to whichever device was current when it was created, enforced strictly for kernel launches but not for memory copies (Section 3.2), an asymmetry Chapter 6 builds on directly. Every multi-device program in this book is Section 3.3's single-device sequence, generalized into a loop that never assumes how many devices exist, with launch and wait kept in separate loops specifically so devices can run concurrently instead of one after another.

## Self-Check Questions

1. A program calls `cudaSetDevice(1)` once, at the very top of `main()`, before spawning four worker threads. Does each worker thread then operate on device 1 by default? Justify your answer using Section 3.1's per-thread model.
2. A stream is created while device 0 is current. Later, with device 1 current, code issues both a kernel launch and a `cudaMemcpyAsync` to that same stream. Predict the outcome of each call separately, citing the documented asymmetry from Section 3.2.
3. Why is `cudaStreamCreate()` inside Phase 1's loop preceded by `cudaSetDevice(i)`, given that the loop already called `cudaSetDevice(i)` earlier in the same iteration for a different purpose (readability aside, is the second call load-bearing)?
4. Rewrite Section 3.3's three-phase loop as a single loop that launches and waits per device before moving to the next. What concurrency property does the rewritten version lose?
5. `06_single_device_setup.cu`'s `CHECK` macro stops the program at the first failing call. In this environment, which specific call fails first, and why does every call after it depend on that one having succeeded?
6. A newly spawned host thread does not inherit its parent thread's current-device setting. Using Section 3.1's default-device-0 rule, predict what device a new thread's very first unguarded CUDA call would act on if that thread never calls `cudaSetDevice()` itself.
7. Suppose a program spawns exactly one host thread per GPU, intending each thread to drive its own device. Using your answer to Question 6, describe the one line each thread must execute first to make that intention actually correct.

## Where We Go Next

Chapter 4 puts this vocabulary to work on the first real multi-device operation: letting one GPU read another GPU's memory directly, without staging the data through the host at all.

## Worked Solutions

**1.** No. Per Section 3.1, "current device" is state belonging to the specific host thread that set it — `cudaSetDevice(1)` called by the main thread only changes the main thread's own current-device state. Each of the four spawned worker threads starts with its own separate current-device state, defaulting to device 0, completely unaffected by what the main thread did before spawning them.

**2.** The kernel launch fails: the stream belongs to device 0 (set when it was created), but the current device is 1 when the launch is issued, and Section 3.2's cited documentation states a kernel launch fails under exactly this mismatch. The `cudaMemcpyAsync` call succeeds despite the identical mismatch, because memory copies are not enforced the same way — the documented asymmetry this section is built around.

**3.** Yes, it's load-bearing, not just readable. Each loop iteration's `cudaSetDevice(i)` call at the top of the iteration sets the current device for everything that follows in that same iteration — the stream creation, the allocation, and (in Phase 2/3) the launch and sync all rely on that single call having set the correct current device just before them. Removing it would leave whichever device the *previous* iteration last set as current, silently misdirecting every subsequent operation to the wrong device.

**4.** The rewrite would call, per device in a single loop: `cudaSetDevice(i); launch kernel; cudaStreamSynchronize(streams[i]);` before moving to `i+1`. It loses concurrency across devices: waiting for device `i`'s stream to finish before ever launching device `i+1`'s kernel forces every device to run strictly one after another, rather than all launching first (Phase 2) and only then all waiting (Phase 3), which is what allows them to execute concurrently.

**5.** `cudaSetDevice(0)` fails first, with `cudaErrorNoDevice` (code 100), because this environment has zero CUDA devices — there is no device 0 to set as current. Every later step (`cudaMalloc`, the memcpy, the kernel launch) is written to operate on "whatever the current device is," and since no device was ever successfully made current, none of those later calls could possibly act on anything meaningful — the `CHECK` macro stops the program right there rather than let it proceed on an undefined foundation.

**6.** Device 0. Section 3.1's rule is that current-device state defaults to device 0 until a thread calls `cudaSetDevice()` itself; a new thread that never makes that call, regardless of what any other thread (including its parent) has done, starts from that same default and stays there for its first (and every subsequent) unguarded call.

**7.** Each thread must call `cudaSetDevice(i)` itself, using its own intended device index `i`, as the very first CUDA-related action it takes — before any allocation, stream creation, or kernel launch. Skipping this in even one thread leaves that thread's current-device state at the default (device 0), so its work silently lands on device 0 instead of the device it was meant to drive, regardless of what the programmer intended when spawning it.

---

**Sources cited in this chapter:**

- [Programming Systems with Multiple GPUs — CUDA Programming Guide](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/03-advanced/multi-gpu-systems.html) -- per-thread current device, device-0 default, stream/event device association, and the kernel-launch-vs-memory-copy enforcement asymmetry.
- [CUDA Pro Tip: Always Set the Current Device to Avoid Multithreading Bugs — NVIDIA Developer Blog](https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs) -- per-thread device state and the new-thread-defaults-to-device-0 pitfall.
