# Appendix A: Installation and Setup
# A minimal Makefile covering this book's own five real distinct compile
# recipes, extending the DSA book's own two-recipe Appendix A.3 Makefile
# (plain g++, plain nvcc) with the three additional link lines this
# book's own multi-GPU chapters actually need: nvcc+NCCL (Chapter 11
# onward), mpicc/mpicxx (Chapter 20 onward), and nvcc+NVSHMEM+MPI
# (Chapter 22's own real recipe). Each recipe below is the exact real
# flag set this book's own chapters have used and verified since they
# were first introduced -- nothing here is new syntax, only gathered in
# one place for a reader setting up a machine for the first time.

CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2

NVCC := nvcc
NVCC_ARCH := -arch=sm_80

NCCL_INC := /usr/include
NCCL_LIB := /usr/lib/x86_64-linux-gnu

MPICC := mpicc
MPICXX := mpicxx
MPI_INC := /usr/lib/x86_64-linux-gnu/openmpi/include

NVSHMEM_DIR := $(shell python3 -c "import nvidia.nvshmem; print(list(nvidia.nvshmem.__path__)[0])" 2>/dev/null)
NVSHMEM_INC := $(NVSHMEM_DIR)/include
NVSHMEM_LIB := $(NVSHMEM_DIR)/lib

# The NVSHMEM pip wheel ships only the versioned libnvshmem_host.so.3 and
# no unversioned libnvshmem_host.so, so `-lnvshmem_host` alone cannot be
# found by the linker. Rather than writing into the Python package's own
# directory (the `ln -sf` Chapters 22 and 32 use), this Makefile makes a
# local LIBDIR override: a small directory holding the unversioned
# symlink, searched first. At run time the loader still wants the
# versioned .so.3, so run with LD_LIBRARY_PATH=$(NVSHMEM_LIB).
NVSHMEM_SHIM := nvshmem_shim

# Recipe 1: plain host C++, no CUDA at all (this appendix's own File 121
# is host-only aside from the nccl.h/mpi.h headers it merely reads).
host_only: hello_host.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Recipe 2: plain nvcc, no additional libraries (most of Part 1-5's own
# single- and multi-GPU kernels: cudaMemcpy, streams, events).
cuda_only: hello_kernel.cu
	$(NVCC) $(NVCC_ARCH) $< -o $@

# Recipe 3: nvcc + NCCL (Chapter 11 onward: ncclAllReduce and friends).
cuda_nccl: hello_nccl.cu
	$(NVCC) $(NVCC_ARCH) -I$(NCCL_INC) -L$(NCCL_LIB) $< -o $@ -lnccl

# Recipe 4: mpicxx, CUDA-aware host-side process launch -- the exact real
# flag shape Chapter 20's own File 58 established: `mpicxx -Wall -Wextra
# file.cpp -o out -lcudart` (a real mpicxx wrapper driving g++, linking a
# real installed libcudart directly, no -I/-L needed because mpicxx's own
# wrapper already knows its include/lib paths).
mpi_only: hello_mpi.cpp
	$(MPICXX) -Wall -Wextra -DOMPI_SKIP_MPICXX $< -o $@ -lcudart

# Recipe 5: nvcc + NVSHMEM + MPI, real GPU-initiated one-sided put/get
# bootstrapped over MPI -- the exact real flag shape Chapter 22's own
# File 64 established: `nvcc -rdc=true -ccbin mpicxx -gencode=arch=
# compute_70,code=sm_70 -I $NVSHMEM_HOME/include file.cu -o out
# -L $LIBDIR -lnvshmem_host -lnvshmem_device -lcuda -lmpi` (mpicxx as
# nvcc's OWN host compiler, not the other way around), plus the
# unversioned-symlink LIBDIR override described above.
$(NVSHMEM_SHIM)/libnvshmem_host.so:
	mkdir -p $(NVSHMEM_SHIM)
	ln -sf $(NVSHMEM_LIB)/libnvshmem_host.so.3 $@

cuda_nvshmem_mpi: hello_nvshmem.cu $(NVSHMEM_SHIM)/libnvshmem_host.so
	$(NVCC) -rdc=true -ccbin $(MPICXX) -gencode=arch=compute_70,code=sm_70 \
		-I$(NVSHMEM_INC) -L$(NVSHMEM_SHIM) -L$(NVSHMEM_LIB) \
		$< -o $@ -lnvshmem_host -lnvshmem_device -lcuda -lmpi

all: host_only cuda_only cuda_nccl mpi_only cuda_nvshmem_mpi

clean:
	rm -f host_only cuda_only cuda_nccl mpi_only cuda_nvshmem_mpi
	rm -rf $(NVSHMEM_SHIM)

.PHONY: all clean
