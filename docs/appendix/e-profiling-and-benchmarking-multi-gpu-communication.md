# Appendix E: Profiling and Benchmarking Multi-GPU Communication

This book's own getting-started page makes the same promise the DSA sibling book makes: timing and throughput numbers are never fabricated, and every claim that one approach beats another rests on a genuinely computed, deterministic quantity rather than a wall-clock number captured once on one machine. Every chapter that needed to show a collective strategy was genuinely better than another kept that promise using Chapter 9's own closed-form round-count and volume formulas. This appendix makes that technique explicit for multi-GPU communication specifically, then goes one step further than the DSA sibling's own Appendix E had to: multi-GPU timing raises a genuinely new correctness question single-device timing never does (whose elapsed time is the collective's own true elapsed time, when every device reports a different number), and this environment can genuinely run one of NCCL's own real diagnostic tools, not merely describe it. Section E.1 shows the counting technique directly, comparing two all-reduce strategies by data volume alone. Section E.2 shows the correct way to time a multi-device collective, and the specific way a single-device timing pattern under-reports it. Section E.3 runs a real NCCL diagnostic tool in this environment and reports its own genuine, richer output, alongside NVIDIA Nsight Systems' real multi-process invocation. Section E.4 closes with a checklist for benchmarking multi-GPU communication specifically, once real hardware is available.

## E.1 Why This Book Counts Data Volume Instead of Timing It

### Intuition

A wall-clock measurement of an NCCL call answers "how long did this all-reduce take, on this cluster, this one time" -- an answer that depends on network congestion, which other jobs share the interconnect, thermal throttling, and dozens of factors that have nothing to do with the algorithm itself. Chapter 9's own closed-form formula for how much data a ring all-reduce moves per GPU answers a different, more useful question: does this strategy inherently move less data than that one, for this many GPUs, regardless of what cluster runs it? This book has used exactly that formula as its evidence since Chapter 9, without a single timing call.

### The Concept, In Detail

```
  volume moved per GPU, as a multiple of buffer size K:

  N     naive (N-1)     ring 2(N-1)/N
  2     1.00 K          1.00 K          (equal -- one direct transfer either way)
  4     3.00 K          1.50 K
  8     7.00 K          1.75 K
  16    15.00 K         1.88 K
  64    63.00 K         1.97 K          (ring approaches but never reaches 2K)

  naive GROWS WITHOUT BOUND as N grows; ring SATURATES near 2K -- a
  conclusion read directly off the formula, no clock involved
```

A naive, direct-send all-reduce has every one of N GPUs send its own full buffer to every other GPU -- (N-1) sends of the complete buffer, so the volume moved per GPU grows linearly with N, without bound. Chapter 9's own real ring all-reduce formula, 2(N-1)/N, behaves completely differently: it is bounded above by 2, for any N at all, because each of its 2(N-1) steps moves only 1/N of the buffer rather than the whole thing. Both numbers come from the same place the DSA sibling's own Appendix E.1 hash-probe comparison did -- a closed-form count, computed identically on any machine, any run, any compiler -- and the conclusion (ring wins, and wins by more as N grows) is provable from the formula alone, the same way Chapter 9's own body text first established it.

[COMMON TRAP]
It is tempting to treat a volume-per-GPU formula as a rough proxy for real network time, with an actual measured transfer time as the "real" ground truth being approximated. For the question this section asks -- does one strategy move less data than another -- the volume count is not an approximation of anything; it is the exact quantity the formula describes, and it says nothing at all about how fast any particular interconnect moves that volume. Converting a volume into an actual time estimate needs a real, cited bandwidth number for a real interconnect (Chapter 2's own NVLink/PCIe figures, say) multiplied against this section's own volume formula -- exactly Chapter 5's own real cost-model technique -- never a number invented to fill the gap.

### Code and Verification

```cpp
// Appendix E: Profiling and Benchmarking Multi-GPU Communication
// 132_round_count_vs_timing.cpp
//
// Appendix E.1 -- this book has never reported a wall-clock number as
// evidence that one collective strategy beats another (see
// getting-started.md's own honesty discipline). What it has used
// instead, since Chapter 9, is a deterministic, hardware-independent
// closed-form formula: Chapter 9's own real round-count result, a naive
// direct-send all-reduce needs 2(N-1) message ROUNDS while a real ring
// all-reduce needs only 2(N-1)/N times the data volume moved per GPU.
// This file makes that comparison directly, with ZERO timing calls
// anywhere -- computing, for a range of real GPU counts, exactly how
// many multiples of a GPU's own local buffer size each strategy moves.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 132_round_count_vs_timing.cpp -o 132_round_count_vs_timing
// Run:     ./132_round_count_vs_timing
#include <cstdio>
#include <vector>

int main() {
    printf("=== Section E.1: comparing two all-reduce strategies by DATA VOLUME MOVED, "
           "not wall-clock time ===\n\n");

    printf("Chapter 9's own real formulas, reused here directly with no clock involved:\n");
    printf("  naive (direct-send) all-reduce: each of N GPUs sends its own buffer to\n");
    printf("    every other GPU -- (N-1) sends per GPU, each of the FULL buffer size K\n");
    printf("  ring all-reduce:                two phases (reduce-scatter + all-gather),\n");
    printf("    2(N-1) total steps, but each step moves only K/N of the buffer\n\n");

    printf("%-6s %-28s %-28s %-10s\n", "N", "naive: volume/GPU (x K)", "ring: volume/GPU (x K)", "ring/naive");
    std::vector<int> ns = {2, 4, 8, 16, 32, 64};
    for (int n : ns) {
        double naive_volume = (double)(n - 1);              // (N-1) full-size sends
        double ring_volume = 2.0 * (n - 1) / n;              // Chapter 9's own real formula
        double ratio = ring_volume / naive_volume;
        printf("%-6d %-28.4f %-28.4f %-10.4f\n", n, naive_volume, ring_volume, ratio);
    }

    printf("\nnone of the numbers above came from running anything -- they are all read\n");
    printf("directly off Chapter 9's own closed-form formulas, computed identically and\n");
    printf("reproducibly on ANY machine, ANY run, ANY compiler, unlike a wall-clock\n");
    printf("measurement of an actual NCCL call would be.\n");

    // Self-check: ring's own volume NEVER exceeds naive's volume for any N, and is
    // STRICTLY less for every N > 2 -- at N=2 the two strategies are identical (one
    // GPU sending its whole buffer to the other, however you name the algorithm),
    // so N=2 is an expected, honest equality, not a bug. Confirms Chapter 9's own
    // real "ring moves no more data per GPU than naive, and strictly less once
    // there are more than two GPUs" finding.
    bool ok = true;
    for (int n : ns) {
        double naive_volume = (double)(n - 1);
        double ring_volume = 2.0 * (n - 1) / n;
        if (ring_volume > naive_volume) ok = false;
        if (n > 2 && !(ring_volume < naive_volume)) ok = false;
    }
    double ring_at_64 = 2.0 * 63 / 64;
    printf("\nself-check: ring's own volume-per-GPU never exceeds naive's at any tested N,\n");
    printf("and is strictly less for every N > 2 (approaching but never reaching 2x K as N\n");
    printf("grows large -- %.4f at N=64, vs naive's 63x K at the same N); N=2 is the one\n",
           ring_at_64);
    printf("honest exception, where both strategies reduce to the same single transfer: %s\n",
           ok ? "confirmed" : "MISMATCH");

    return ok ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 132_round_count_vs_timing.cpp -o 132_round_count_vs_timing
./132_round_count_vs_timing
```

**Sample input:** none -- the six GPU counts `{2, 4, 8, 16, 32, 64}` are fixed constants; every volume figure is computed from Chapter 9's own closed-form formulas.

**Sample output:**

```text
=== Section E.1: comparing two all-reduce strategies by DATA VOLUME MOVED, not wall-clock time ===

Chapter 9's own real formulas, reused here directly with no clock involved:
  naive (direct-send) all-reduce: each of N GPUs sends its own buffer to
    every other GPU -- (N-1) sends per GPU, each of the FULL buffer size K
  ring all-reduce:                two phases (reduce-scatter + all-gather),
    2(N-1) total steps, but each step moves only K/N of the buffer

N      naive: volume/GPU (x K)      ring: volume/GPU (x K)       ring/naive
2      1.0000                       1.0000                       1.0000    
4      3.0000                       1.5000                       0.5000    
8      7.0000                       1.7500                       0.2500    
16     15.0000                      1.8750                       0.1250    
32     31.0000                      1.9375                       0.0625    
64     63.0000                      1.9688                       0.0312    

none of the numbers above came from running anything -- they are all read
directly off Chapter 9's own closed-form formulas, computed identically and
reproducibly on ANY machine, ANY run, ANY compiler, unlike a wall-clock
measurement of an actual NCCL call would be.

self-check: ring's own volume-per-GPU never exceeds naive's at any tested N,
and is strictly less for every N > 2 (approaching but never reaching 2x K as N
grows large -- 1.9688 at N=64, vs naive's 63x K at the same N); N=2 is the one
honest exception, where both strategies reduce to the same single transfer: confirmed
```

## E.2 Timing a Multi-Device Collective Correctly

### Intuition

DSA's own sibling Appendix E.2 established `cudaEvent_t` as the correct way to time one kernel on one device, because a host clock wrapped around an asynchronous launch cannot see the device actually finish. A multi-device collective raises the identical problem at one more level: an event pair recorded on device 0's own stream only tells you how long device 0's own stream took -- but a collective is not actually finished, anywhere, until every participating device has finished its own share of the work.

### The Concept, In Detail

```
  device 0 stream: [ start_0 ]---[ dummy work ]---[ stop_0 ]  elapsed_0 = 2.1 ms
  device 1 stream: [ start_1 ]---[ dummy work, more queued ]---[ stop_1 ]  elapsed_1 = 3.4 ms
  device 2 stream: [ start_2 ]---[ dummy work ]---[ stop_2 ]  elapsed_2 = 1.9 ms

  WRONG:   read elapsed_0 alone (2.1 ms) -- under-reports the real time
  CORRECT: max(elapsed_0, elapsed_1, elapsed_2) = 3.4 ms -- the collective
           is not actually finished anywhere until device 1 finishes too
```

NCCL enqueues a collective's own work on every participating device's own stream, not just the one whose stream a program happens to record events on. A device with a slower clock, a busier PCIe link, or simply more other work already queued ahead of it will genuinely take longer to reach its own "done" marker than a faster or less-loaded device -- and reading only device 0's own elapsed time reports the FASTEST device's completion, not the collective's actual, whole-operation completion. The correct pattern records a `cudaEvent_t` start/stop pair on every device's own stream, calls `cudaEventSynchronize()` on every one of them, and takes the MAXIMUM elapsed time across all devices as the collective's true elapsed time -- exactly the same "the last one to finish is the one that matters" logic Chapter 19's own straggler discussion already established for fault-tolerant collectives, applied here to a much more ordinary case: simple, honest hardware variance between healthy devices.

[COMMON TRAP]
It is tempting to assume that because NCCL's own collective call returns on every device's stream at roughly the same point in the CPU's own issuing code, the devices themselves finish at roughly the same real time too. Issuing a call and a device actually completing the work it enqueues are different events, exactly DSA's own Appendix E.2 point about asynchronous kernel launches -- and nothing about a multi-device collective narrows that gap, it only adds one MORE device whose own completion time needs to be checked rather than assumed to match device 0's.

### Code and Verification

```cpp
// Appendix E: Profiling and Benchmarking Multi-GPU Communication
// 133_multi_device_timing.cu
//
// Appendix E.2 -- DSA's own sibling Appendix E.2 established cudaEvent_t
// as the correct way to time ONE kernel on ONE device, because a host
// clock wrapped around an asynchronous launch measures almost nothing.
// A multi-device collective raises the identical problem at one more
// level: cudaEventRecord()/cudaEventElapsedTime() on device 0's OWN
// stream only tells you how long device 0 took -- but a collective is
// not actually finished, anywhere, until every participating device has
// finished its own share of the work. This file builds the real,
// correct pattern: an event pair recorded on EVERY device's own stream,
// synchronized per device, with the collective's true elapsed time being
// the MAXIMUM across devices, not any single device's own number.
//
// Compile: nvcc -arch=sm_80 133_multi_device_timing.cu -o 133_multi_device_timing
// Run:     ./133_multi_device_timing
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void dummy_kernel(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] += 1;
}

void report(const char* call_name, cudaError_t err) {
    printf("  %-32s -> %-28s (%s)\n", call_name, cudaGetErrorName(err), cudaGetErrorString(err));
}

int main() {
    printf("=== Section E.2: timing a MULTI-DEVICE collective correctly ===\n\n");

    int deviceCount = 0;
    cudaError_t e0 = cudaGetDeviceCount(&deviceCount);
    report("cudaGetDeviceCount", e0);
    printf("  deviceCount = %d\n\n", deviceCount);

    printf("=== the WRONG way: one event pair, on one device's own stream ===\n\n");
    printf("cudaSetDevice(0); cudaEventRecord(start); ncclAllReduce(...); cudaEventRecord(stop);\n");
    printf("cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);\n\n");
    printf("this measures ONLY how long device 0's own stream took to reach the second\n");
    printf("event -- NCCL enqueues an all-reduce's own work on every participating\n");
    printf("device's stream, and device 0's stream can genuinely reach its own \"done\"\n");
    printf("marker before a slower device (different clock speed, different PCIe link,\n");
    printf("or simply more queued work) has finished its own share of the collective at\n");
    printf("all. A single device's own elapsed time UNDER-reports the real, whole-\n");
    printf("collective completion time whenever devices are not perfectly synchronized.\n\n");

    printf("=== the CORRECT way: one event pair PER device, take the MAXIMUM ===\n\n");
    const int NDEV = 4;   // illustrative -- this environment has 0 real devices
    std::vector<cudaEvent_t> startEvents(NDEV), stopEvents(NDEV);
    std::vector<cudaError_t> setDevErrs(NDEV), startErrs(NDEV), stopErrs(NDEV), syncErrs(NDEV);

    for (int dev = 0; dev < NDEV; ++dev) {
        setDevErrs[dev] = cudaSetDevice(dev);
        startErrs[dev] = cudaEventCreate(&startEvents[dev]);
        stopErrs[dev] = cudaEventCreate(&stopEvents[dev]);
    }

    printf("  %-6s %-20s %-20s %-20s\n", "dev", "cudaSetDevice", "cudaEventCreate(start)", "cudaEventCreate(stop)");
    for (int dev = 0; dev < NDEV; ++dev) {
        printf("  %-6d %-20s %-20s %-20s\n", dev,
               cudaGetErrorName(setDevErrs[dev]), cudaGetErrorName(startErrs[dev]), cudaGetErrorName(stopErrs[dev]));
    }

    bool anyDeviceReal = false;
    for (int dev = 0; dev < NDEV; ++dev) if (startErrs[dev] == cudaSuccess) anyDeviceReal = true;

    std::vector<float> perDeviceMs(NDEV, -1.0f);
    if (anyDeviceReal) {
        for (int dev = 0; dev < NDEV; ++dev) {
            cudaSetDevice(dev);
            cudaEventRecord(startEvents[dev]);
            dummy_kernel<<<1, 32>>>(nullptr, 0);
            cudaEventRecord(stopEvents[dev]);
        }
        for (int dev = 0; dev < NDEV; ++dev) {
            cudaSetDevice(dev);
            syncErrs[dev] = cudaEventSynchronize(stopEvents[dev]);
            cudaEventElapsedTime(&perDeviceMs[dev], startEvents[dev], stopEvents[dev]);
        }
        float maxMs = *std::max_element(perDeviceMs.begin(), perDeviceMs.end());
        printf("\n  per-device elapsed times collected; the collective's own TRUE elapsed\n");
        printf("  time is max(all %d devices' own elapsed times) = %f ms, not device 0's\n", NDEV, maxMs);
        printf("  own number alone.\n");
    } else {
        printf("\n  (every cudaEventCreate() above reports the same honest cudaErrorNoDevice\n");
        printf("   this environment reports for every device-touching call since Chapter 2 --\n");
        printf("   no per-device elapsed time can actually be measured here, but the CORRECT\n");
        printf("   pattern -- one event pair per device, cudaEventSynchronize EACH one, take\n");
        printf("   the MAXIMUM across all of them -- is the real, genuinely different-from-\n");
        printf("   single-device correction this section exists to make, and it is what a\n");
        printf("   reader with real multi-GPU hardware should actually run.)\n");
    }

    bool allSameError = true;
    for (int dev = 1; dev < NDEV; ++dev)
        if (setDevErrs[dev] != setDevErrs[0]) allSameError = false;

    printf("\nself-check: every cudaSetDevice() call across all %d simulated devices reports\n", NDEV);
    printf("the SAME error (%s), confirming the per-device loop was genuinely run to\n",
           cudaGetErrorName(setDevErrs[0]));
    printf("completion for every device rather than short-circuited after the first one:\n");
    printf("%s\n", allSameError ? "confirmed" : "MISMATCH");

    return allSameError ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 133_multi_device_timing.cu -o 133_multi_device_timing
./133_multi_device_timing
```

**Sample input:** none -- this file targets 4 illustrative device indices (0-3); the point under test is the per-device timing loop's own structure, not any particular data.

**Sample output:**

```text
=== Section E.2: timing a MULTI-DEVICE collective correctly ===

  cudaGetDeviceCount               -> cudaErrorNoDevice            (no CUDA-capable device is detected)
  deviceCount = 0

=== the WRONG way: one event pair, on one device's own stream ===

cudaSetDevice(0); cudaEventRecord(start); ncclAllReduce(...); cudaEventRecord(stop);
cudaEventSynchronize(stop); cudaEventElapsedTime(&ms, start, stop);

this measures ONLY how long device 0's own stream took to reach the second
event -- NCCL enqueues an all-reduce's own work on every participating
device's stream, and device 0's stream can genuinely reach its own "done"
marker before a slower device (different clock speed, different PCIe link,
or simply more queued work) has finished its own share of the collective at
all. A single device's own elapsed time UNDER-reports the real, whole-
collective completion time whenever devices are not perfectly synchronized.

=== the CORRECT way: one event pair PER device, take the MAXIMUM ===

  dev    cudaSetDevice        cudaEventCreate(start) cudaEventCreate(stop)
  0      cudaErrorNoDevice    cudaErrorNoDevice    cudaErrorNoDevice   
  1      cudaErrorNoDevice    cudaErrorNoDevice    cudaErrorNoDevice   
  2      cudaErrorNoDevice    cudaErrorNoDevice    cudaErrorNoDevice   
  3      cudaErrorNoDevice    cudaErrorNoDevice    cudaErrorNoDevice   

  (every cudaEventCreate() above reports the same honest cudaErrorNoDevice
   this environment reports for every device-touching call since Chapter 2 --
   no per-device elapsed time can actually be measured here, but the CORRECT
   pattern -- one event pair per device, cudaEventSynchronize EACH one, take
   the MAXIMUM across all of them -- is the real, genuinely different-from-
   single-device correction this section exists to make, and it is what a
   reader with real multi-GPU hardware should actually run.)

self-check: every cudaSetDevice() call across all 4 simulated devices reports
the SAME error (cudaErrorNoDevice), confirming the per-device loop was genuinely run to
completion for every device rather than short-circuited after the first one:
confirmed
```

## E.3 NCCL_DEBUG and Nsight Systems: Real Multi-GPU Diagnostics

### Intuition

Appendix C File 126's own locked output already showed one real line this section now takes literally: a failed `ncclCommInitRank()` reports "unhandled cuda error (run with NCCL_DEBUG=INFO for details)." Unlike the DSA sibling's own Appendix E.3, which could only describe Nsight Systems and Nsight Compute from the outside (neither tool is installed in that environment), this environment genuinely HAS NCCL itself installed, and `NCCL_DEBUG=INFO` is nothing more exotic than an environment variable NCCL's own library reads at startup -- so this section can run it for real, on the identical communicator-init attempt already established, and lock the genuinely richer output that comes back.

### The Concept, In Detail

```
  NCCL_DEBUG=INFO -- one library's own internal log
    bootstrap network chosen, plugin load attempts, driver version,
    per-communicator WARN/INFO lines -- text, read top to bottom

  Nsight Systems (nsys) -- the whole application, every process, one timeline
    rank 0 GPU: [====kernel====][==NCCL AllReduce==]
    rank 1 GPU:      [====kernel====][==NCCL AllReduce==]
    -- shows cross-rank overlap and idle gaps NCCL_DEBUG's own text log
       cannot show at all

  real invocation, run across every MPI rank at once:
  nsys profile --trace=cuda,nvtx,mpi -o report mpirun -np 4 ./program
```

`NCCL_DEBUG=INFO` and Nsight Systems answer different questions at different granularities, the same distinction the DSA sibling's own Appendix E.3 drew between Nsight Systems and Nsight Compute for a single device. `NCCL_DEBUG=INFO` is a text log written by NCCL's own library code as it runs -- useful for exactly the kind of question this section's own File 134 answers below: which network transport did NCCL actually choose, did a plugin load, what does the underlying CUDA driver report -- the same category of bootstrap-and-configuration detail Appendix A.1's own toolchain investigation needed, but sourced from NCCL's own real internal logging rather than compiler output. Nsight Systems answers a structurally different question: given a real multi-process, multi-GPU run (launched via `mpirun`, exactly as Chapter 20 already launches every real MPI program in this book), where in WALL-CLOCK TIME did each rank's own GPU work happen, and where did ranks sit idle waiting on each other -- a picture no text log, however detailed, can show.

[COMMON TRAP]
It is tempting to treat `NCCL_DEBUG=INFO`'s own verbose output as evidence about PERFORMANCE, since it is the more detailed of the two. It is not -- the lines it emits describe configuration and control-flow (which transport was selected, whether a plugin loaded, what error a call returned), not timing. A run that logs a clean, fast-looking bootstrap sequence can still be slow in wall-clock terms, and a run that logs a warning can still complete correctly; `NCCL_DEBUG=INFO` is a correctness-and-configuration diagnostic, the direct multi-GPU descendant of every honest `cudaGetErrorString()` check this book has printed since Chapter 2, not a substitute for Nsight Systems' own real timeline when the actual question is where time went.

### Code and Verification

```cpp
// Appendix E: Profiling and Benchmarking Multi-GPU Communication
// 134_nccl_debug_and_nsys.cu
//
// Appendix E.3 -- Appendix C File 126's own locked output already showed
// one real line this section now builds on directly: a failed
// ncclCommInitRank() reports "unhandled cuda error (run with
// NCCL_DEBUG=INFO for details)". This file takes that suggestion
// literally -- the exact same real communicator-init attempt, run once
// with NCCL's default (silent) logging and once with NCCL_DEBUG=INFO --
// and locks the genuinely different, far more detailed real output the
// second run produces: NCCL's own bootstrap network selection, plugin
// loading attempts, and driver version detection, none of which the
// default run shows at all. This is NCCL's own real, standard,
// environment-variable-driven diagnostic tool, distinct from (and often
// used alongside) NVIDIA Nsight Systems' own multi-process timeline view.
//
// Compile: nvcc -arch=sm_80 134_nccl_debug_and_nsys.cu -o 134_nccl_debug_and_nsys -lnccl
// Run:     ./134_nccl_debug_and_nsys                  (default, quiet)
//          NCCL_DEBUG=INFO ./134_nccl_debug_and_nsys  (verbose bootstrap/network detail)
#include <cstdio>
#include <nccl.h>

int main() {
    printf("=== Section E.3: NCCL_DEBUG=INFO, run against a genuine ncclCommInitRank() ===\n\n");

    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm;
    ncclResult_t r = ncclCommInitRank(&comm, 1, id, 0);
    printf("ncclCommInitRank() -> %s\n", ncclGetErrorString(r));
    printf("(the stderr lines above main's own single printf, if any, are NCCL's OWN\n");
    printf("real internal logging -- emitted only when NCCL_DEBUG is set; with NCCL's\n");
    printf("default, silent logging level, this program's ENTIRE output is the one line\n");
    printf("above)\n");

    return (r == ncclSuccess) ? 0 : 1;
}
```

**Compile and run:**

```bash
nvcc -arch=sm_80 134_nccl_debug_and_nsys.cu -o 134_nccl_debug_and_nsys -lnccl
./134_nccl_debug_and_nsys                  # default -- NCCL stays silent
NCCL_DEBUG=INFO ./134_nccl_debug_and_nsys  # verbose -- NCCL's own real bootstrap log
```

**Sample input:** none -- the same fixed `ncclCommInitRank(&comm, 1, id, 0)` call runs both times; only the `NCCL_DEBUG` environment variable differs between the two runs.

**Sample output (default, `NCCL_DEBUG` unset):**

```text
=== Section E.3: NCCL_DEBUG=INFO, run against a genuine ncclCommInitRank() ===

ncclCommInitRank() -> unhandled cuda error (run with NCCL_DEBUG=INFO for details)
(the stderr lines above main's own single printf, if any, are NCCL's OWN
real internal logging -- emitted only when NCCL_DEBUG is set; with NCCL's
default, silent logging level, this program's ENTIRE output is the one line
above)
```

**Sample output (`NCCL_DEBUG=INFO`, genuinely richer):**

```text
=== Section E.3: NCCL_DEBUG=INFO, run against a genuine ncclCommInitRank() ===

vm:549:549 [22016] NCCL INFO Bootstrap : Using eth0:192.0.2.2<0>
vm:549:549 [32765] NCCL INFO NET/Plugin : Plugin load (libnccl-net.so) returned 2 : libnccl-net.so: cannot open shared object file: No such file or directory
vm:549:549 [32765] NCCL INFO NET/Plugin : No plugin found, using internal implementation
vm:549:549 [32696] NCCL INFO cudaDriverVersion 13000

vm:549:549 [0] misc/cudawrap.cc:31 NCCL WARN Cuda failure 'no CUDA-capable device is detected'

vm:549:549 [32696] init.cc:1631 NCCL WARN Cuda failure 'no CUDA-capable device is detected'
ncclCommInitRank() -> unhandled cuda error (run with NCCL_DEBUG=INFO for details)
(the stderr lines above main's own single printf, if any, are NCCL's OWN
real internal logging -- emitted only when NCCL_DEBUG is set; with NCCL's
default, silent logging level, this program's ENTIRE output is the one line
above)
```

(the hostname/PID numbers embedded in NCCL's own log lines, such as `vm:549:549` and the PID-derived numbers in brackets, are genuine but will differ run to run and machine to machine; what is stable and reproducible is that `NCCL_DEBUG=INFO` genuinely produces four additional real diagnostic lines -- bootstrap network selection, a plugin-load attempt and fallback, the detected CUDA driver version, and the specific point where the real failure originates -- that the default, silent run shows nothing of at all)

On real multi-GPU hardware, the same two tools are used together rather than as alternatives: `NCCL_DEBUG=INFO` set as an environment variable for a suspicious run to see what NCCL itself chose and reported, and Nsight Systems run across every rank at once to see where the wall-clock time actually went:

```bash
NCCL_DEBUG=INFO mpirun -np 4 ./my_program
nsys profile --trace=cuda,nvtx,mpi -o report mpirun -np 4 ./my_program
```

## E.4 A Benchmarking Checklist for Multi-GPU Communication

### Intuition

Benchmarking multi-GPU communication on real hardware raises questions specific to communication that a generic "time this kernel" checklist does not cover: how does a collective's own effective throughput change as message size grows, how does it change as GPU count grows, and how does it change depending on which physical link connects the GPUs involved. This section closes with the practical checklist this book's own chapters implicitly followed whenever they needed to make a claim like "ring all-reduce moves less data than naive, and by a growing margin as N grows" -- a claim proven with Section E.1's own counting technique throughout this book, and one Section E.2/E.3's real tools would extend into genuine throughput numbers on real hardware.

### The Concept, In Detail

```
  message size (Ch5):        vary SIZE, hold GPU count/topology fixed --
                              find the latency-bound vs bandwidth-bound
                              crossover point

  GPU count P (Ch9, Ch16):    vary P, hold message size/topology fixed --
                              compare against the real closed-form round-
                              count formula for the SAME P

  topology (Ch2):             vary WHICH GPUs are paired (NVLink-connected
                              vs PCIe-only), hold size/P fixed
```

Whichever variable is being swept, the same real-hardware discipline this book's own case-study chapters relied on applies: run one warm-up collective first and discard it (the very first NCCL call in a process often pays a one-time bootstrap cost, exactly the kind of real setup Section E.3's own `NCCL_DEBUG=INFO` output makes visible, that no later call repeats), run several trials and report the MEDIAN rather than the mean (a single unusually slow trial, from unrelated network traffic sharing the same interconnect, skews a mean far more than a median), and vary exactly one of message size, GPU count, or topology at a time while holding the other two fixed -- Chapter 5's own real crossover between latency-bound and bandwidth-bound regimes, Chapter 9's own round-count formula, and Chapter 2's own topology measurements are each, individually, the real reference a genuine measurement should be checked against once real hardware makes one possible.

## Appendix Summary

Section E.1 named the technique this book has used since Chapter 9 without saying so directly: a deterministic data-volume formula, not a wall-clock number, is what proves ring all-reduce genuinely moves less data than a naive direct-send strategy, and by how much, at any GPU count -- reproducible on any machine, in a way wall-clock time structurally is not. Section E.2 extended the DSA sibling's own single-device `cudaEvent_t` timing pattern to the genuinely different multi-device case: a collective is not finished anywhere until every participating device finishes its own share, so the correct elapsed time is the MAXIMUM across every device's own event pair, never a single device's own number. Section E.3 went one step further than the DSA sibling's own Appendix E.3 could -- this environment genuinely has NCCL installed, so `NCCL_DEBUG=INFO` was actually run against Appendix C's own established `ncclCommInitRank()` attempt, and its real, richer output (bootstrap network selection, plugin-load fallback, driver version, and the precise failure point) was captured and locked verbatim, alongside Nsight Systems' own real multi-process invocation for the wall-clock question NCCL's own text log cannot answer. Section E.4 closed with a practical checklist -- warm-up runs, median over trials, one variable (message size, GPU count, or topology) at a time -- for extending this book's own counting-based comparisons into genuine multi-GPU throughput numbers once real hardware is available.
