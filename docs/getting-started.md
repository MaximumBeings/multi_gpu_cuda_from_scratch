# Getting Started

<!-- execution:overview -->
!!! success "Recorded hardware validation"

    The [November 18 two-H100 rerun](validation-h100-rerun.md) compiled and executed all 145 recorded revisions: 34 numerical/transport checks, 13 API checks and 98 other executions. The earlier [two-H100](validation-h100.md) and [eight-A100](validation-a100.md) records remain available with their original scopes. The A100 pass uses eight MPI processes for multi-rank examples; individual programs retain their documented participant counts. A separate eight-device harness checks all 56 ordered peer-copy pairs, four collective patterns and three halo generations.


<!-- mgpu:opener:start -->
[![Getting Started](images/chapter-openers/getting-started.svg)](images/chapter-openers/getting-started.svg)
<!-- mgpu:opener:end -->

This book targets CUDA C++ compiled with `nvcc` for real NVIDIA GPU architectures -- the appropriate target must be selected per example -- plus MPI (OpenMPI or MPICH) for Part 5's multi-node material. A real multi-GPU machine with a matching driver is what you need to *run* the code in this book end to end; `nvcc` itself only needs a CUDA toolkit and will happily compile for architectures your current machine doesn't have, and host-only MPI programs can run as multiple local ranks without GPUs; CUDA-aware MPI examples still need CUDA devices.

The corrected staged-MPI example (58) passed the November 18 two-H100 numerical rerun and memory check. **All 145 current sources have successful execution records:** 142 match the H100 dataset; the three corrected CPU-only models have [new local records](host-model-corrections.md). See the [additional dated H100 results](validation-h100-rerun.md) for commands, system information and sample outputs.

Select a diagram to open its full-size SVG.

## Connect to the Linux host

Connect using your provider's SSH key and host address, then run the installation steps below on that Linux host. On macOS/Linux, `chmod 400 /path/to/key.pem` protects the local key, and `ssh -i /path/to/key.pem ubuntu@YOUR_INSTANCE_IP` connects to a typical Ubuntu cloud image. Verify the provider's host-key fingerprint on first connection; use the actual image's user name. Windows OpenSSH uses Windows key-file access controls instead of POSIX `chmod`.

## Installing a toolchain

<!-- mgpu:figure-overview:start -->
[![Separate build checks from hardware execution](images/technical/getting-started.svg)](images/technical/getting-started.svg)
<!-- mgpu:figure-overview:end -->


### Choose the installation path

The commands below target **Ubuntu 22.04 or 24.04, x86-64, on a Linux GPU host**, with CUDA **12.8**. Ubuntu 24.04 is the book's recorded host family. Run them in Bash on that host, not on an Apple Silicon laptop. For ARM/Grace, WSL, containers or another distribution, use the corresponding [NVIDIA CUDA installation instructions](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html); the repository and driver steps differ.

| Starting point | Follow this path |
| --- | --- |
| Provider image already runs `nvidia-smi` and `nvcc --version` reports a suitable toolkit | Keep the provider's driver; install only missing libraries/tools and select the intended toolkit. |
| Working driver, missing compiler | Install the CUDA toolkit below; a toolkit-only package does not replace the driver. |
| Fresh Ubuntu GPU host without a driver | Install prerequisites, register the repository, install the driver, reboot, then install CUDA and libraries. |
| Container | Install the driver on the host; use the provider's GPU-enabled container runtime and a CUDA **development** image. Do not install kernel drivers inside the container. |

Record the starting environment:

```bash
cat /etc/os-release
uname -m
uname -r
nvidia-smi
command -v nvcc
nvcc --version
```

On a fresh host, missing `nvidia-smi` or `nvcc` at this stage is the reason to continue installation. On a managed image, first check its documented environment modules/toolkit paths. Conda activation is not a universal CUDA installation step.

### Install Ubuntu prerequisites

On a fresh Ubuntu host:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3 python3-venv python3-pip \
    git wget ca-certificates pciutils
```

`build-essential` supplies GCC/G++, Make and the basic linker toolchain. CUDA also constrains supported host-compiler versions: use the selected toolkit's [host compiler support table](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#host-compiler-support-policy). The captured Ubuntu 24.04 runs used GCC 13.3. A generic Python CUDA wheel is not a replacement for this C++ build environment.

### Register the matching NVIDIA repository

Use this only if the matching NVIDIA APT repository is not already configured. It selects Ubuntu 22.04 versus 24.04 from the actual OS; it stops for other systems. The subshell keeps its temporary variables local.

```bash
(
    set -eu
    . /etc/os-release
    case "$ID:$VERSION_ID:$(uname -m)" in
        ubuntu:22.04:x86_64) cuda_repo=ubuntu2204 ;;
        ubuntu:24.04:x86_64) cuda_repo=ubuntu2404 ;;
        *) echo "Use NVIDIA's instructions for this OS/CPU." >&2; exit 1 ;;
    esac
    keyring_dir=$(mktemp -d)
    trap 'rm -rf "$keyring_dir"' EXIT
    wget -O "$keyring_dir/cuda-keyring.deb" \
        "https://developer.download.nvidia.com/compute/cuda/repos/$cuda_repo/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "$keyring_dir/cuda-keyring.deb"
    sudo apt-get update
)
```

Stop if this block fails. Do not continue with a repository for a different Ubuntu release, or mix a runfile driver installation with unrelated package-manager driver packages. NVIDIA documents the [keyring and repository setup](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/ubuntu.html#network-repository-enablement-amd64).

### Install a driver only when needed

**Preconfigured cloud image:** retain its working, compatible driver. If it needs an upgrade, use the provider's supported procedure. The recorded driver was 580.105.08. For CUDA 12.8 Update 1, this guide selects the release's corresponding **570.124.06 or newer** driver baseline. CUDA 12.x minor-version compatibility permits some applications on older drivers, with feature and PTX/JIT limitations; it is a different compatibility path, not the setup baseline used here. See [NVIDIA's release compatibility tables](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-driver).

**Fresh Ubuntu host you administer:** if Secure Boot requires module-key enrollment, arrange the distribution/provider console access before rebooting. After repository setup, install kernel headers and NVIDIA's open-module driver package. H100 and A100 are supported by the open-module driver family. These commands are for the host, not a container:

```bash
sudo apt-get install -y "linux-headers-$(uname -r)"
sudo apt-get install -y nvidia-open
sudo reboot
```

The reboot disconnects SSH. Reconnect with your original SSH command, then run:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Continue only when the driver sees the intended GPU(s). A custom cloud kernel may need provider-specific headers. With Secure Boot, follow the distribution's module-signing/key-enrollment prompts, which can require console access; do not bypass a failed module load by blindly reinstalling CUDA. The `nvidia-open` package follows the repository's selected driver branch; it does not pin the recorded 580.105.08 patch. See [NVIDIA's driver installation and reboot steps](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/ubuntu.html#driver-installation).

### Install CUDA 12.8 and cuBLAS

Skip installation if this toolkit and its development libraries already work. Otherwise:

```bash
sudo apt-get install -y cuda-toolkit-12-8
```

This versioned **toolkit-only** package supplies `nvcc`, CUDA Runtime development files, the Driver API headers/link stubs, cuBLAS headers/libraries, Compute Sanitizer and CUDA tools. It does not install a kernel driver. NCCL is separate. Avoid the unversioned `cuda` meta-package when you intend to keep the 12.8 toolkit and the provider's driver. The distribution package `nvidia-cuda-toolkit` can be a different CUDA release; it is not this recipe.

APT may install newer patches within the 12.8 line; inspect actual versions rather than expecting an exact historical environment. NVIDIA's [toolkit package definitions](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-installation-guide-linux/index.html#meta-packages) distinguish toolkit, runtime and driver bundles.

### Select the toolkit and library paths

Choose **one** layout; these are alternatives, not consecutive export blocks.

=== "NVIDIA APT toolkit"

    ```bash
    export CUDA_HOME=/usr/local/cuda-12.8
    export CUDA_INCDIR="$CUDA_HOME/include"
    export CUDA_LIBDIR="$CUDA_HOME/lib64"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    hash -r
    command -v nvcc
    nvcc --version
    ```

=== "Provider toolkit under /usr"

    Use this layout when the provider's working `nvcc` is `/usr/bin/nvcc` and its CUDA headers/libraries are under `/usr/include` and `/usr/lib/x86_64-linux-gnu`, as on the recorded Lambda image.

    ```bash
    export CUDA_HOME=/usr
    export CUDA_INCDIR=/usr/include
    export CUDA_LIBDIR=/usr/lib/x86_64-linux-gnu
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    hash -r
    command -v nvcc
    nvcc --version
    ```

For another layout, use the site's module or its real install prefix. Confirm the selected headers and libraries belong to the intended compiler; the multi-GPU collector accepts `CUDA_HOME`, `CUDA_INCDIR` and `CUDA_LIBDIR` explicitly.

APT normally registers its shared libraries with the loader; the explicit toolkit path above also makes the intended 12.8 libraries visible for this shell. For a custom install, substitute its real path. **Never add `$CUDA_HOME/lib64/stubs` to the runtime library path**: `libcuda.so.1` at execution time must come from the installed host driver, not a link-time stub.

To persist this selection in interactive Bash sessions, add these exports once to `~/.bashrc`, then run `source ~/.bashrc` or open a new shell. In scheduler jobs and noninteractive SSH sessions, load the site's module or set these exports explicitly in the job script; those sessions may not read `~/.bashrc`. The conditional expansion avoids an empty trailing library-path entry.

### Install NCCL runtime and development files

NCCL is needed by the collective examples. Install **both** the runtime and development package with the same CUDA-compatible version. For the recorded NCCL 2.26.2/CUDA 12.8 combination, first confirm package availability:

```bash
apt-cache madison libnccl2 libnccl-dev
```

When both list `2.26.2-1+cuda12.8`:

```bash
sudo apt-get install -y \
    libnccl2=2.26.2-1+cuda12.8 \
    libnccl-dev=2.26.2-1+cuda12.8
```

These versions were present in NVIDIA's Ubuntu 22.04 and 24.04 x86-64 repositories when this guide was checked on **November 18, 2026**. If unavailable, select a matching pair built for CUDA 12.x from NVIDIA's repository/archive and record the actual version. Do not silently install the default CUDA 13 build or pass `--allow-downgrades` over a provider-managed stack. An existing compatible NCCL installation can be retained; its headers and runtime must resolve consistently. NVIDIA's [NCCL installation guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#ubuntu) explains explicit package-version selection.

### Check fabric setup on NVSwitch servers

On a **native HGX/DGX NVSwitch server**, the GPU driver and CUDA toolkit alone are not the entire fabric setup. DGX OS normally supplies Fabric Manager; a fresh HGX installation needs the compatible Fabric Manager components for its platform and driver. Use [NVIDIA's bare-metal Fabric Manager procedure](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html#installing-fabric-manager) to choose the packages. Package/version rules differ by NVSwitch generation, and Fabric Manager versioning changed with driver release 620; do not copy a branch number from an unrelated server.

After installing the matching components on the **host that owns the fabric**, enable/start the service and inspect its state:

```bash
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
systemctl is-active nvidia-fabricmanager
nvidia-smi -q -i 0 | grep -i -A 3 Fabric
```

For Hopper NVSwitch systems, check that fabric registration completes successfully before relying on NVLink peer paths. On a managed cloud instance or guest/container, the provider/host administrator may own this service; its absence inside the guest is not an instruction to install or restart it there. PCIe-only and direct-NVLink setups do not universally need Fabric Manager. Do not restart an active fabric service as a routine troubleshooting step.

### Verify files, versions and topology

```bash
nvcc --version
gcc --version
python3 --version
nvidia-smi
nvidia-smi topo -m
ls "$CUDA_INCDIR/cuda_runtime.h" "$CUDA_INCDIR/cuda.h"
ls "$CUDA_INCDIR/cublas_v2.h"
dpkg-query -W libnccl2 libnccl-dev
dpkg -L libnccl-dev | grep '/nccl.h$'
```

The driver supplies GPU access; the toolkit supplies compilation; cuBLAS and NCCL supply libraries. `nvidia-smi`'s “CUDA Version” is driver compatibility, not the installed `nvcc` version. A topology listing describes connectivity, not a measured bandwidth or a successful peer-transfer test. Finish with the book's actual compile, launch and numerical checks below.

## Obtain the sources

After installing Git through the prerequisites above, clone once and enter the repository:

```bash
git clone --branch main https://github.com/MaximumBeings/multi_gpu_cuda_from_scratch.git
cd multi_gpu_cuda_from_scratch
```

Use an existing source checkout if already cloned; check `git branch --show-current`. The `gh-pages` branch contains generated HTML, so switch to `main` before using the source/build commands. The remaining relative paths and `.venv-gpu-deps` environment belong in this repository directory.

## Add MPI and NVSHMEM

### Install or select one Open MPI stack

The book's `mpi-ext.h`/`MPIX_Query_cuda_support` probes use **Open MPI** extensions. On a fresh Ubuntu host, install its launcher and development headers together:

```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev
command -v mpicc mpicxx mpirun
mpicxx --showme:version
mpirun --version
```

On a provider/HPC image, use its existing Open MPI module or installation instead. Select the matching `bin` and library directories together; do not mix a vendor launcher with another MPI's headers or `libmpi`. Do not replace a managed UCX/HPC stack merely to match an example path. The recorded hosts used Open MPI 4.1.x. MPICH is not a drop-in compiler for source files using these Open MPI-specific extensions.

From the repository root, check CPU process launch first:

```bash
mkdir -p build/setup
mpicc -std=c11 -O2 docs/part5/code/57_mpi_ring_send_recv.c -o build/setup/mpi-ring
mpirun --oversubscribe -np 2 ./build/setup/mpi-ring 1024 sendrecv
```

Use `sendrecv` for this startup check: the example's alternate large blocking-send mode deliberately demonstrates a deadlock. Run as your ordinary user. `--oversubscribe` permits a local demonstration on a host with fewer CPU slots; it is not a performance setting. The launcher working with host buffers does not establish CUDA-aware MPI support.

Check the installed Open MPI's reported CUDA capability:

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support
```

The intentional invalid-rank error-handling demonstration (example 59) additionally requires MPI argument checking. Inspect `ompi_info --all --parsable | grep mpi_param_check`. Some vendor performance builds compile checking out: `MPI_ERRORS_RETURN` cannot turn it back on, and `--mca mpi_param_check 1` is then ignored. Use a parameter-checking Open MPI build for that diagnostic, in a separate environment if necessary; leave the provider's transport stack intact. The November 18 rerun records the vendor failure and a successful isolated Ubuntu Open MPI 4.1.6 run.

If device-buffer MPI is needed and support is absent, use the site's GPU-aware MPI/UCX distribution or build a CUDA-enabled stack following [Open MPI's CUDA guidance](https://www.open-mpi.org/faq/?category=runcuda). Installing `libopenmpi-dev` alone does not guarantee CUDA-aware transfers. Host-staged examples can still operate without that capability; only pass a device pointer to MPI after checking the actual implementation and validating the transfer.

### Install the recorded NVSHMEM wheel in isolation

For the numbered NVSHMEM examples, one reproducible route is the **NVSHMEM 3.8.0 CUDA 12 wheel**, installed separately from the system toolkit:

```bash
python3 -m venv .venv-gpu-deps
.venv-gpu-deps/bin/python -m pip install nvidia-nvshmem-cu12==3.8.0
export NVSHMEM_HOME=$(.venv-gpu-deps/bin/python -c 'import nvidia.nvshmem; print(list(nvidia.nvshmem.__path__)[0])')
export NVSHMEM_LIB="$NVSHMEM_HOME/lib"
export NVSHMEM_BOOTSTRAP=MPI
export LD_LIBRARY_PATH="$NVSHMEM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The wheel supplies NVSHMEM headers and libraries; it does not install the NVIDIA kernel driver, a complete `nvcc` toolkit, or Open MPI. Its Python namespace package can have `__file__ == None`; `__path__` locates the real directory. Using the explicit virtual-environment Python also works when the website's separate `.venv` is active.

The 3.8.0 wheel supplies the versioned host library. If the unversioned linker name is missing, create it **only inside this user-owned dependency environment**:

```bash
if [ ! -e "$NVSHMEM_LIB/libnvshmem_host.so" ]; then
    test -f "$NVSHMEM_LIB/libnvshmem_host.so.3" && \
        ln -s libnvshmem_host.so.3 "$NVSHMEM_LIB/libnvshmem_host.so"
fi
ls "$NVSHMEM_HOME/include/nvshmem.h"
ls "$NVSHMEM_LIB/libnvshmem_device.a" "$NVSHMEM_LIB/libnvshmem_host.so.3"
ldd "$NVSHMEM_LIB/nvshmem_bootstrap_mpi.so.3"
```

Resolve every `not found` entry before launching MPI. The bootstrap plugin must resolve the Open MPI library compatible with the selected launcher. Reapply these exports in new shells/jobs. Alternatively, use NVIDIA's [NVSHMEM runtime and development packages](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/nvshmem-install-proc.html) and set `NVSHMEM_HOME`/`NVSHMEM_LIB` to their actual installation paths; do not combine both installations accidentally.

## Clone, compile and run a first GPU check

From the repository root after the earlier checkout:

```bash
mkdir -p build/setup
nvidia-smi --query-gpu=name,uuid --format=csv
nvidia-smi topo -m
```

If already cloned, use that checkout rather than cloning into it again. Select the target for the **actual GPU**: A100 `sm_80`, H100/H200 `sm_90`, B200 `sm_100` with an appropriate toolkit. This book's ordinary CUDA examples differ from the H100 companion's architecture-specific WGMMA `sm_90a` programs. B200 results remain a separate validation target.

For an H100 host with at least two CUDA-visible GPUs, build the existing full-buffer peer-transfer check:

```bash
nvcc -std=c++17 -arch=sm_90 -O2 docs/part1/code/11_memcpy_peer.cu -o build/setup/peer-copy
./build/setup/peer-copy
```

Use `-arch=sm_80` on A100. Check the actual PASS/error count and exit status. A topology label alone does not establish peer access or transfer correctness; use the chapter's staged alternative if direct access is unavailable.

After MPI, NCCL and NVSHMEM are configured, the collector can build selected dependency checks. From the repository root, use a **new output directory for each run**:

```bash
python3 tools/validate_examples.py --arch sm_90 \
    --match 121_multi_gpu_stack_version_check \
    --match 64_nvshmem_mpi_bootstrap \
    --output artifacts/setup-link-check
```

This command is a compile/link check; add `--execute` in a fresh output directory to run the selected checks on a suitable GPU host. For a complete two-H100 pass:

```bash
python3 tools/validate_examples.py --execute --arch sm_90 \
    --output artifacts/local-h100-rerun
```

The default MPI demonstrations use two ranks (single-PE example 63 uses one). For an eight-GPU host, `--mpi-ranks 8` changes the multi-rank MPI examples; it does not rewrite non-MPI examples' documented participant counts. Each local GPU rank must be able to select a distinct device as its source requires; launching extra ranks on one device is not a substitute.

## Multi-node and profiling prerequisites

A same-node MPI/NCCL/NVSHMEM success does not configure a cluster. For multiple nodes, provision matching binaries/libraries and MPI launch access on every node, scheduler allocations, working network interfaces and the site's supported RDMA/UCX/NVSHMEM transport stack. Use [NVIDIA's NVSHMEM prerequisites](https://docs.nvidia.com/nvshmem/release-notes-install-guide/install-guide/abstract.html#software-requirements) for GPUDirect RDMA, GDRCopy and transport-specific requirements; these are not mandatory packages for every same-node NVLink exercise. Do not change network or driver security settings as a generic setup workaround.

Check optional tools with `compute-sanitizer --version` and `ncu --version`. With the CUDA 12.8 repository, `cuda-sanitizer-12-8` and `cuda-nsight-compute-12-8` select matching tool packages. Profiling may require administrator-approved performance-counter access. A sanitizer/profiler run is not a normal bandwidth benchmark.

## Preview this website without a GPU

From the repository root, use a separate documentation environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdocs build --strict
python3 tools/verify_site_links.py
mkdocs serve
```

Open <http://127.0.0.1:8000> on the same machine. For a remote preview, keep the server on localhost and use SSH port forwarding (`ssh -L 8000:127.0.0.1:8000 ...`). The website environment and the NVSHMEM dependency environment serve different purposes.

## The honesty discipline this book follows

The recorded versions of all 145 programs compiled and executed on the two-H100 and eight-A100 hosts; three later host-model corrections have separate local CPU execution records. Current validation distinguishes numerical/transport checks, API checks and host models:

- **Host programs and simulations** are compiled and executed as host programs. Their results check the modeled arithmetic and scheduling, not CUDA concurrency or physical links.
- **GPU numerical checks** initialize data and compare actual device results with an explicit reference.
- **API probes** check the stated capability, lifecycle or expected-error behavior; they are not numerical collective tests.
- **Execution evidence** records source hashes, commands, timestamps, system information and raw logs. The 2026-11-17 two-H100 local NV18 run compiled and executed all 145 programs: 33 numerical checks, 13 API probes and 99 other executions. See [H100 validation](validation-h100.md). Host simulations remain useful supplementary checks; neither they nor this single-node run validate cross-node RDMA, failure recovery or performance on other hardware.

Execution records include the tested source, compile/run commands, UTC dates and captured output. See the [H100 results](validation-h100.md) and [A100 results](validation-a100.md).

Before execution, run `nvidia-smi`, `nvcc --version`, and the chapter’s device-count/topology checks. H100 uses the generic `sm_90` target for these ordinary CUDA examples; B200 uses `sm_100` with a toolkit that supports it. Rebuild and revalidate rather than reusing a Hopper cubin on Blackwell. [NVIDIA’s compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html) explains cubin and PTX compatibility. Run MPI examples with the required number of distinct visible devices; multiple ranks on one GPU are not a substitute for a multi-GPU collective.

From the repository root, `python3 tools/validate_examples.py --output artifacts/local-audit` collects compiler evidence. GPU execution is opt-in with `--execute --arch sm_90` on H100 or `--execute --arch sm_100` on B200. Keep each architecture’s output separate and inspect per-example classifications; API diagnostics and simulations are not numerical GPU validations.

## Compile-line conventions

- Plain C++ host files (`.cpp`): `g++ -std=c++17 -Wall -Wextra -O2 file.cpp -o binary`
- CUDA files with real device code (`.cu`): `nvcc -std=c++17 -arch=sm_90 file.cu -o binary` on H100/H200 (`sm_80` on A100)
- Files that call the CUDA Runtime API from the host add `-lcudart` and the toolchain's include/library paths, shown inline wherever they're used.
- MPI files (`.cpp`, built with the MPI wrapper compiler): `mpic++ -std=c++17 -Wall -Wextra -O2 file.cpp -o binary`, run as `mpirun -np N ./binary`.
- CUDA-aware MPI files combine both: `mpic++ -std=c++17 file.cpp -I"$CUDA_INCDIR" -L"$CUDA_LIBDIR" -lcudart -o binary` after selecting the layout above.

## Prerequisites

This book assumes working knowledge of C++ (structs, templates, pointers, RAII) and prior single-GPU CUDA experience at the level of "you have written and launched a kernel before" -- it does not re-derive the CUDA execution and memory model from nothing the way a first CUDA book would, though Part 0 does rebuild everything specific to *multiple* devices from scratch. No prior MPI experience is assumed; Part 5's own Chapter 20 builds plain MPI from nothing before it ever adds CUDA to the picture.

## Work through the setup before the larger examples

The [worked setup example in Appendix A](appendix/a-installation-and-setup.md) separates host tools, device discovery, rank/device mapping and the first checked result. After setup, follow the numerical walkthrough in each chapter before copying its longer implementation. Preserve the exact compile command and compare complete results with the documented reference; a successful launch alone does not establish the intended multi-GPU computation.
