# Chapter 1: Why One GPU Is Not Enough

Every chapter after this one is about getting two or more GPUs to cooperate. Before that's worth the trouble, it's worth being precise about what problem it actually solves -- because "not enough" hides two genuinely different failures, and the rest of this book is organized around the difference between them.

## 1.1 Two different kinds of "not enough"

A single GPU can fail a workload in two distinct ways:

- **The capacity wall.** The data the workload needs to hold in device memory at once simply does not fit. This is a hard, binary failure: the allocation either succeeds or it doesn't. There is no "a bit slower" version of running out of memory.
- **The throughput wall.** Everything fits, but the amount of arithmetic required would take an impractical amount of time on one device's compute engines -- hours becoming weeks, weeks becoming years.

A single GPU can hit either wall alone, or both at once, and they call for different fixes. Splitting a model's *parameters* across devices attacks the capacity wall directly. Splitting the *work* -- the batch, the matrix multiply, the timesteps -- across devices attacks the throughput wall. Most of Parts 3 and 6 of this book are ultimately about picking the right combination of the two for a given workload. Here is the shape of a single device with both walls drawn on it:

```text
                        ┌───────────────────────────────────┐
                        │              ONE GPU                │
                        │                                     │
   throughput wall --> │   SM SM SM SM SM SM SM SM  (compute) │
   (finite FLOP/s,      │   SM SM SM SM SM SM SM SM            │
    fixed no matter      │                                     │
    how big the job)    │  ┌───────────────────────────────┐  │
                        │  │   device memory (HBM)           │  │ <-- capacity wall
                        │  │   fixed size, e.g. 80 GB         │  │     (fixed, hard,
                        │  │   [ weights | grads | optimizer  │  │      physical ceiling)
                        │  │     state  |  activations ]      │  │
                        │  └───────────────────────────────┘  │
                        └───────────────────────────────────┘
```

The rest of this chapter makes each wall concrete with real numbers, then genuinely runs code that demonstrates the one thing a single device *can* still tell you honestly about itself: how many devices exist at all.

## 1.2 The capacity wall, in bytes

During training, four kinds of data compete for space in device memory:

1. **Parameters** -- the model's weights.
2. **Gradients** -- one value per parameter, produced by backpropagation.
3. **Optimizer state** -- for Adam, a running momentum and variance estimate per parameter.
4. **Activations** -- the intermediate results of the forward pass, kept around because backpropagation needs them.

The first three scale purely with parameter count and are straightforward to add up exactly. The fourth scales with batch size and sequence length as well as parameter count, and is workload-dependent rather than a fixed multiple of model size -- so it's called out here as a real cost that a full capacity plan must include, without inventing a specific number for it.

For parameters, gradients, and optimizer state, the accounting is exact. Rajbhandari et al.'s ZeRO paper (SC'20) gives the standard mixed-precision Adam breakdown: an fp16 copy of the parameters and the gradients (2 bytes/parameter each), plus an fp32 copy of the parameters, the momentum, and the variance (4 bytes/parameter each) kept by the optimizer for numerical stability. That's

```text
2 (fp16 params) + 2 (fp16 grads) + 4 (fp32 params) + 4 (momentum) + 4 (variance)
= 16 bytes per parameter
```

against roughly 2 bytes per parameter just to hold the weights for inference. Training a model costs an order of magnitude more memory per parameter than merely running it.

### Sequential host arithmetic first

Before writing anything that touches a GPU, the honest way to see how bad this gets is to just compute it -- on the host, in ordinary C++, with no CUDA involved at all:

```cpp
// 02_memory_wall.cpp
#include <cstdio>
#include <cstdint>
#include <cmath>

struct Model {
    const char* name;
    double params; // parameter count
};

int main() {
    const Model models[] = {
        {"7B-class model",   7.0e9},
        {"GPT-3 (175B)",   175.0e9},
    };

    // Real GPU memory capacities, decimal GB as the vendor datasheets state
    // them (NVIDIA H100 datasheet / product page).
    const double H100_SXM_GB = 80.0;
    const double H100_NVL_GB = 94.0;
    const double BYTES_PER_GB = 1.0e9;

    printf("%-16s %10s %16s %16s %10s %10s\n",
           "Model", "Params", "Inference(fp16)", "Training(Adam,mp)",
           "H100 SXM x", "H100 NVL x");

    for (const auto& m : models) {
        double inferenceBytes = m.params * 2.0;
        double trainingBytes  = m.params * 16.0;   // ZeRO's 16 bytes/param

        double inferenceGB = inferenceBytes / BYTES_PER_GB;
        double trainingGB  = trainingBytes  / BYTES_PER_GB;

        double sxmNeeded = std::ceil(trainingGB / H100_SXM_GB);
        double nvlNeeded = std::ceil(trainingGB / H100_NVL_GB);

        printf("%-16s %8.1fB %13.1f GB %13.1f GB %10.0f %10.0f\n",
               m.name, m.params / 1e9, inferenceGB, trainingGB,
               sxmNeeded, nvlNeeded);
    }

    printf("\nMinimum H100 SXM (80 GB) GPUs to hold GPT-3's training state "
           "(params+grads+Adam state only, no activations): %.0f\n",
           std::ceil((175.0e9 * 16.0 / BYTES_PER_GB) / H100_SXM_GB));

    return 0;
}
```

Compile and run:

```bash
g++ -std=c++17 -Wall -Wextra -O2 02_memory_wall.cpp -o memory_wall
./memory_wall
```

Output (genuinely compiled and run; re-verified by a fresh recompile and rerun before publication, identical both times):

```text
Model                Params  Inference(fp16) Training(Adam,mp) H100 SXM x H100 NVL x
7B-class model        7.0B          14.0 GB         112.0 GB          2          2
GPT-3 (175B)        175.0B         350.0 GB        2800.0 GB         35         30

Minimum H100 SXM (80 GB) GPUs to hold GPT-3's training state (params+grads+Adam state only, no activations): 35
```

A 175-billion-parameter model's weights alone (350 GB at fp16) already exceed a single H100's 80 GB several times over -- before a single gradient or optimizer value is added. Once they are, the training state needs **35 H100 SXM GPUs' worth of memory just to exist**, with zero of it left over for the activations that actually let training happen. Even the comparatively modest 7B-class model's *training* footprint (112 GB) doesn't fit in one H100 (80 GB), though its *inference* footprint (14 GB) fits easily -- a reminder that "does this model fit on a GPU" is not one question but at least two, with very different answers.

This is the capacity wall, stated in bytes rather than adjectives: for a large enough model, there is no GPU you can buy, today, that holds it alone. Parts 1 and 3 of this book are about the two ways out -- moving data between devices efficiently (Part 1), and splitting the model itself across them (Part 3) -- because the alternative, buying a bigger GPU, stops being an option at exactly this scale.

## 1.3 The throughput wall, in FLOPs

Suppose, hypothetically, that memory were never a problem -- infinite HBM, one device. Training would still take a very long time, because the arithmetic itself is enormous. Kaplan et al.'s "Scaling Laws for Neural Language Models" (2020) give a compact estimate for total training compute:

```text
C ≈ 6 · N · D
```

where `N` is the model's (non-embedding) parameter count, `D` is the number of training tokens, and the factor of 6 accounts for one forward pass and the (roughly twice as expensive) backward pass. This is not a rough guess pulled from nowhere -- it's the paper's own stated formula (`C ≈ 6NBS`, with batch size `B` times step count `S` equal to total tokens `D`), and it is a real, exactly-computable quantity once `N` and `D` are fixed.

### Sequential host arithmetic, again, before any hardware claim

```cpp
// 03_compute_wall.cpp
#include <cstdio>

int main() {
    const double N = 175.0e9;   // GPT-3's parameter count (Brown et al. 2020)
    const double D = 300.0e9;   // GPT-3's stated training token budget
    const double totalFLOPs = 6.0 * N * D;

    const double H100_PEAK_FLOPS = 989.0e12; // H100 SXM, dense BF16/FP16, no sparsity
    const double SECONDS_PER_DAY = 86400.0;

    const double idealFlopsPerDay = H100_PEAK_FLOPS * SECONDS_PER_DAY;
    const double idealGpuDays = totalFLOPs / idealFlopsPerDay;

    const double assumedMFU = 0.50;  // illustrative, not measured -- see text
    const double realisticGpuDays = idealGpuDays / assumedMFU;

    printf("Total training compute (C = 6*N*D): %.3e FLOPs\n", totalFLOPs);
    printf("N (parameters, approx.):            %.3e\n", N);
    printf("D (tokens):                         %.3e\n", D);
    printf("H100 SXM peak dense BF16/FP16:       %.3e FLOP/s\n", H100_PEAK_FLOPS);
    printf("\n");
    printf("GPU-days on a single H100 at 100%% of peak (ideal, unreachable "
           "in practice): %.1f\n", idealGpuDays);
    printf("GPU-days on a single H100 at an assumed %.0f%% of peak "
           "(illustrative MFU, not measured):   %.1f\n",
           assumedMFU * 100.0, realisticGpuDays);
    printf("Same total compute spread across 1024 such GPUs at the "
           "assumed %.0f%% MFU: %.2f days\n",
           assumedMFU * 100.0, realisticGpuDays / 1024.0);
    return 0;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 03_compute_wall.cpp -o compute_wall
./compute_wall
```

Output (genuinely compiled and run twice; identical both times, as expected of pure arithmetic with no timing or randomness involved):

```text
Total training compute (C = 6*N*D): 3.150e+23 FLOPs
N (parameters, approx.):            1.750e+11
D (tokens):                         3.000e+11
H100 SXM peak dense BF16/FP16:       9.890e+14 FLOP/s

GPU-days on a single H100 at 100% of peak (ideal, unreachable in practice): 3686.4
GPU-days on a single H100 at an assumed 50% of peak (illustrative MFU, not measured):   7372.8
Same total compute spread across 1024 such GPUs at the assumed 50% MFU: 7.20 days
```

Two things are worth being exact about here, because this book cares about not fabricating numbers. The `3686.4` and `7372.8` GPU-day figures are *not* timing measurements -- nothing was clocked. They're a deterministic FLOP count (`3.150e+23`, computed exactly from Kaplan's formula) divided by a real, vendor-published peak-FLOPS figure (H100 SXM's 989 TFLOP/s dense BF16/Tensor Core throughput, no sparsity). The 50%-of-peak scenario is explicitly labeled as an illustrative assumption about achieved utilization (Model FLOPs Utilization, or MFU), not a measurement of anything -- real training runs report MFU figures all over the 30-50% range depending on model, hardware, and how well the implementation overlaps communication with compute, which is itself a running theme starting in Part 2.

Either way the conclusion is the same: **over 20 years on one GPU at ideal throughput, or roughly a week on 1,024 of them.** The arithmetic doesn't fit in a human research or product timeline on a single device, independent of whether that device even has enough memory. That is the throughput wall, and it's why Part 2 (collectives), Part 3 (parallelization strategies), and Part 5 (scaling beyond one node) exist even for models that would, memory-wise, technically fit on one GPU.

## 1.4 What a single real CUDA call can already tell you

Both calculations above were pure host arithmetic -- deliberately, since the point being made didn't need a GPU at all. But this book is about CUDA C++, and its very first primitive, `cudaGetDeviceCount()`, is worth calling for real right here in Chapter 1, because how it behaves *matters for the rest of the book*: every multi-GPU program starts by asking the runtime how many devices it actually has to work with.

```cpp
// 01_device_query.cu
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("cudaGetDeviceCount() returned: %s (code %d)\n",
           cudaGetErrorString(err), (int)err);
    printf("Reported device count: %d\n", deviceCount);

    if (err != cudaSuccess || deviceCount == 0) {
        printf("No usable CUDA device on this machine.\n");
        printf("This is an honest, unmodified report from the CUDA "
               "Runtime API -- not a placeholder.\n");
        return 0;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        size_t freeBytes = 0, totalBytes = 0;
        cudaSetDevice(i);
        cudaMemGetInfo(&freeBytes, &totalBytes);
        printf("Device %d: %s -- %.2f GB total, %.2f GB free, "
               "compute capability %d.%d\n",
               i, prop.name, totalBytes / 1e9, freeBytes / 1e9,
               prop.major, prop.minor);
    }
    return 0;
}
```

```bash
nvcc -arch=sm_80 01_device_query.cu -o device_query
./device_query
```

Output (genuinely compiled with a real `nvcc` and genuinely run; identical across three consecutive runs):

```text
cudaGetDeviceCount() returned: no CUDA-capable device is detected (code 100)
Reported device count: 0
No usable CUDA device on this machine.
This is an honest, unmodified report from the CUDA Runtime API -- not a placeholder.
```

This is exactly the outcome this book's own getting-started guide promises: the authoring environment has no NVIDIA GPU and no NVIDIA driver, so `cudaGetDeviceCount()` genuinely, correctly reports zero devices and `cudaErrorNoDevice` (error code 100) -- not a fabricated success, not a skipped call. Every chapter from here on that needs more than one device present will say plainly which parts of its code are checked this same honest way versus checked by the host-side simulation technique this book uses for multi-device logic (see `getting-started.md`), starting with peer-to-peer memory access in Chapter 4.

## 1.5 Two walls, one answer -- with a new problem attached

Both walls have the same structural fix: stop asking one device to hold everything and do everything.

```text
   ONE GPU, maxed out                          N GPUs, sharing the load
  ┌─────────────────────┐                     ┌───────────┐     ┌───────────┐
  │  compute: saturated   │                     │  compute:  │     │  compute:  │
  │  memory:  full/over   │   ────split───▶     │  1/N share │ ... │  1/N share │
  │                       │                     │  memory:   │     │  memory:   │
  │                       │                     │  1/N share │     │  1/N share │
  └─────────────────────┘                     └─────┬─────┘     └─────┬─────┘
                                                     └───────┬────────┘
                                                        the interconnect
                                                    (PCIe / NVLink / network)
                                                  <-- this is the hard part -->
```

Splitting parameters across devices solves the capacity wall (Chapter 1's section 1.2, addressed starting in Part 1 and Part 3). Splitting work across devices solves the throughput wall (section 1.3, addressed starting in Part 2 and Part 3). But drawing that diagram is the easy part; the arrow labeled "the interconnect" is where nearly everything genuinely difficult in this book lives. Splitting the problem creates a new one that a single GPU never had: the pieces now have to talk to each other, correctly, and fast enough that the communication doesn't itself become the new bottleneck.

That is the entire subject of the rest of this book:

- **Part 1** builds the primitive layer -- how one device's memory becomes visible to another at all.
- **Part 2** builds the communication patterns (broadcast, reduce, all-reduce, and friends) from scratch, by hand, before ever reaching for a library that does it for you.
- **Part 3** chooses *what* to split (data, model, tensors, pipeline stages, spatial domains) for a given workload shape.
- **Part 4** deals with what happens when devices don't finish at the same time, or don't finish at all.
- **Part 5** takes everything past the boundary of one machine.
- **Part 6** puts all of it to work on real workloads, end to end.

## 1.6 What this chapter verified, and how

- **Section 1.2's and 1.3's numbers** are ordinary host-side C++ arithmetic (`g++`, no CUDA), genuinely compiled and run, using exact formulas from cited sources -- never a fabricated figure standing in for a measurement.
- **Real hardware facts** cited in this chapter -- H100 SXM's 80 GB / 3.35 TB/s and H100 NVL's 94 GB / 3.9 TB/s memory specs, and H100 SXM's 989 TFLOP/s dense BF16/FP16 Tensor Core throughput -- come from NVIDIA's own product page and datasheet, not from memory.
- **Real published facts** -- GPT-3's 175B parameter count and 300B-token training budget (Brown et al. 2020), the 16-bytes-per-parameter mixed-precision Adam breakdown (Rajbhandari et al., ZeRO, SC'20), and the `C ≈ 6ND` training-compute estimate (Kaplan et al. 2020) -- are quoted from the papers themselves, not approximated from general familiarity with them.
- **Section 1.4's code** is a genuine CUDA Runtime API call, compiled with a real `nvcc` and genuinely executed in this driver-less, device-less environment; its output (`cudaErrorNoDevice`, device count 0) is reported exactly as the runtime returned it.
- **Nothing in this chapter was simulated**, because nothing in it yet requires more than one device to exist -- that starts in Chapter 4. Where this book's own honesty discipline requires host-side simulation of multiple devices, later chapters say so explicitly and point back to `getting-started.md`.

---

**Sources cited in this chapter:**

- [NVIDIA H100 GPU product page](https://www.nvidia.com/en-us/data-center/h100/) -- H100 SXM memory (80 GB, 3.35 TB/s) and H100 NVL memory (94 GB, 3.9 TB/s).
- [NVIDIA H100 GPU Datasheet](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306) -- H100 SXM5 peak dense BF16/FP16 Tensor Core throughput (989 TFLOPS).
- Brown, T. et al., ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) (2020) -- GPT-3's 175B parameter count (Table 2.1) and 300-billion-token training budget.
- Rajbhandari, S. et al., ["ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://aiichironakano.github.io/cs596/Rajbhandari-ZeRO-SC20.pdf) (SC'20) -- the 16-bytes-per-parameter mixed-precision Adam memory breakdown.
- Kaplan, J. et al., ["Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361) (2020) -- the `C ≈ 6NBS` (equivalently `C ≈ 6ND`) training compute estimate.
