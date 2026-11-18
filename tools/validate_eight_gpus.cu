// Real eight-device correctness audit; separate from the book's example sources.
// Build: nvcc -std=c++17 -arch=sm_80 -O2 tools/validate_eight_gpus.cu -lnccl -pthread -o validate_eight_gpus
// Use the executing GPU's supported target for another architecture (e.g. sm_100).
// Run: ./validate_eight_gpus
// No throughput claim: all payloads are checked in full after completion.
#include <cuda_runtime.h>
#include <nccl.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int P = 8;
constexpr size_t PEER_N = 4099, GATHER_N = 517, EXCHANGE_N = 257, HALO_N = 1031;
constexpr size_t CAPACITY = P * PEER_N;

void cuda_check(cudaError_t e, const char* call, int line) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA line ") +
        std::to_string(line) + " " + call + ": " + cudaGetErrorString(e));
}
void nccl_check(ncclResult_t e, const char* call, int line) {
    if (e != ncclSuccess) throw std::runtime_error(std::string("NCCL line ") +
        std::to_string(line) + " " + call + ": " + ncclGetErrorString(e));
}
#define CUDA(call) cuda_check((call), #call, __LINE__)
#define NCCL(call) nccl_check((call), #call, __LINE__)

struct Device {
    int* send = nullptr;
    int* recv = nullptr;
    cudaStream_t stream = nullptr;
    ncclComm_t comm = nullptr;
};
std::array<Device, P> devices;
bool enabled[P][P] = {};
bool capability[P][P] = {};
size_t numerical_errors = 0;

int value(int rank, size_t index) { return rank * 100000 + (int)index * 3 + 7; }
int reduced(size_t index) { return 100000 * (P * (P - 1) / 2) + P * ((int)index * 3 + 7); }
int personalized(int source, int destination, size_t i) {
    return source * 100000 + destination * 1000 + (int)i * 3 + 7;
}
int halo_value(int generation, int rank, size_t interior_index) {
    return generation * 1000000 + rank * 10000 + (int)interior_index * 3 + 7;
}
int halo_poison(int generation, int rank) { return -10000 - generation * 100 - rank; }

void sync_all() {
    for (int rank = 0; rank < P; ++rank) {
        CUDA(cudaSetDevice(rank));
        CUDA(cudaStreamSynchronize(devices[rank].stream));
    }
}

// Host initialization finishes before its temporary host storage is reused.
template<class Expected>
void prepare(size_t count, Expected expected) {
    if (count > CAPACITY) throw std::runtime_error("Internal buffer capacity exceeded");
    std::vector<int> host(count);
    for (int rank = 0; rank < P; ++rank) {
        for (size_t i = 0; i < count; ++i) host[i] = expected(rank, i);
        CUDA(cudaSetDevice(rank));
        CUDA(cudaMemcpy(devices[rank].send, host.data(), count * sizeof(int), cudaMemcpyHostToDevice));
        CUDA(cudaMemsetAsync(devices[rank].recv, 0xa5, CAPACITY * sizeof(int), devices[rank].stream));
    }
}

template<class Expected>
bool check_output(const char* label, size_t count, Expected expected, bool use_send = false) {
    sync_all();
    size_t errors = 0;
    std::vector<int> host(count);
    for (int rank = 0; rank < P; ++rank) {
        CUDA(cudaSetDevice(rank));
        CUDA(cudaMemcpy(host.data(), use_send ? devices[rank].send : devices[rank].recv,
                        count * sizeof(int), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            int want = expected(rank, i);
            if (host[i] != want) {
                if (errors < 8) fprintf(stderr, "%s rank=%d index=%zu got=%d expected=%d\n",
                                       label, rank, i, host[i], want);
                ++errors;
            }
        }
    }
    numerical_errors += errors;
    printf("%s: %s; %zu values checked across 8 ranks; errors=%zu\n",
           label, errors ? "FAIL" : "PASS", P * count, errors);
    return errors == 0;
}

void audit_peers() {
    prepare(PEER_N, [](int rank, size_t i) { return value(rank, i); });
    sync_all();
    std::vector<int> host(PEER_N);
    int tested = 0;
    size_t errors = 0;
    for (int source = 0; source < P; ++source) {
        for (int destination = 0; destination < P; ++destination) {
            if (source == destination) continue;
            CUDA(cudaSetDevice(destination));
            auto& d = devices[destination];
            CUDA(cudaMemsetAsync(d.recv, 0xa5, PEER_N * sizeof(int), d.stream));
            CUDA(cudaMemcpyPeerAsync(d.recv, destination, devices[source].send, source,
                                     PEER_N * sizeof(int), d.stream));
            CUDA(cudaStreamSynchronize(d.stream));
            CUDA(cudaMemcpy(host.data(), d.recv, PEER_N * sizeof(int), cudaMemcpyDeviceToHost));
            size_t pair_errors = 0;
            for (size_t i = 0; i < PEER_N; ++i) {
                if (host[i] != value(source, i)) {
                    if (errors + pair_errors < 8)
                        fprintf(stderr, "peer %d->%d index=%zu got=%d expected=%d\n",
                                source, destination, i, host[i], value(source, i));
                    ++pair_errors;
                }
            }
            errors += pair_errors;
            ++tested;
            printf("PEER_COPY %d->%d: %s; accessor %d can access owner %d: %s; errors=%zu\n",
                   source, destination, pair_errors ? "FAIL" : "PASS", destination, source,
                   capability[destination][source] ? "YES" : "NO", pair_errors);
        }
    }
    numerical_errors += errors;
    if (tested != 56) throw std::runtime_error("Did not execute all 56 ordered peer copies");
    printf("ALL_PEER_PAIRS: %s; pairs=%d; values_checked=%zu; errors=%zu\n",
           errors ? "FAIL" : "PASS", tested, (size_t)tested * PEER_N, errors);
}

void audit_collectives() {
    prepare(PEER_N, [](int rank, size_t i) { return value(rank, i); });
    NCCL(ncclGroupStart());
    for (int rank = 0; rank < P; ++rank) {
        CUDA(cudaSetDevice(rank));
        auto& d = devices[rank];
        NCCL(ncclAllReduce(d.send, d.recv, PEER_N, ncclInt32, ncclSum, d.comm, d.stream));
    }
    NCCL(ncclGroupEnd());
    check_output("ALL_REDUCE", PEER_N, [](int, size_t i) { return reduced(i); });

    prepare(GATHER_N, [](int rank, size_t i) { return value(rank, i); });
    NCCL(ncclGroupStart());
    for (int rank = 0; rank < P; ++rank) {
        CUDA(cudaSetDevice(rank));
        auto& d = devices[rank];
        NCCL(ncclAllGather(d.send, d.recv, GATHER_N, ncclInt32, d.comm, d.stream));
    }
    NCCL(ncclGroupEnd());
    check_output("ALL_GATHER", P * GATHER_N,
                 [](int, size_t i) { return value((int)(i / GATHER_N), i % GATHER_N); });

    prepare(P * GATHER_N, [](int rank, size_t i) { return value(rank, i); });
    NCCL(ncclGroupStart());
    for (int rank = 0; rank < P; ++rank) {
        CUDA(cudaSetDevice(rank));
        auto& d = devices[rank];
        NCCL(ncclReduceScatter(d.send, d.recv, GATHER_N, ncclInt32, ncclSum, d.comm, d.stream));
    }
    NCCL(ncclGroupEnd());
    check_output("REDUCE_SCATTER", GATHER_N,
                 [](int rank, size_t i) { return reduced(rank * GATHER_N + i); });

    prepare(P * EXCHANGE_N, [](int source, size_t i) {
        return personalized(source, (int)(i / EXCHANGE_N), i % EXCHANGE_N);
    });
    NCCL(ncclGroupStart());
    for (int rank = 0; rank < P; ++rank) {
        CUDA(cudaSetDevice(rank));
        auto& d = devices[rank];
        for (int peer = 0; peer < P; ++peer) {
            NCCL(ncclSend(d.send + peer * EXCHANGE_N, EXCHANGE_N, ncclInt32,
                          peer, d.comm, d.stream));
            NCCL(ncclRecv(d.recv + peer * EXCHANGE_N, EXCHANGE_N, ncclInt32,
                          peer, d.comm, d.stream));
        }
    }
    NCCL(ncclGroupEnd());
    check_output("GROUPED_ALL_TO_ALL", P * EXCHANGE_N, [](int rank, size_t i) {
        return personalized((int)(i / EXCHANGE_N), rank, i % EXCHANGE_N);
    });
}

void audit_halos() {
    // Linear domain: ranks 0/7 retain physical-boundary poison on outer ghosts.
    // Every generation has new interior values and distinct per-rank ghost poison.
    for (int generation = 1; generation <= 3; ++generation) {
        prepare(HALO_N + 2, [generation](int rank, size_t i) {
            return i == 0 || i == HALO_N + 1 ? halo_poison(generation, rank)
                                           : halo_value(generation, rank, i - 1);
        });
        NCCL(ncclGroupStart());
        for (int rank = 0; rank < P; ++rank) {
            CUDA(cudaSetDevice(rank));
            auto& d = devices[rank];
            if (rank > 0) {
                NCCL(ncclSend(d.send + 1, 1, ncclInt32, rank - 1, d.comm, d.stream));
                NCCL(ncclRecv(d.send, 1, ncclInt32, rank - 1, d.comm, d.stream));
            }
            if (rank + 1 < P) {
                NCCL(ncclSend(d.send + HALO_N, 1, ncclInt32, rank + 1, d.comm, d.stream));
                NCCL(ncclRecv(d.send + HALO_N + 1, 1, ncclInt32, rank + 1, d.comm, d.stream));
            }
        }
        NCCL(ncclGroupEnd());
        std::string label = "HALO_GENERATION_" + std::to_string(generation);
        check_output(label.c_str(), HALO_N + 2, [generation](int rank, size_t i) {
            if (i == 0)
                return rank == 0 ? halo_poison(generation, rank)
                                 : halo_value(generation, rank - 1, HALO_N - 1);
            if (i == HALO_N + 1)
                return rank == P - 1 ? halo_poison(generation, rank)
                                     : halo_value(generation, rank + 1, 0);
            return halo_value(generation, rank, i - 1);
        }, true);
    }
}

bool cleanup(bool abort_communicators) {
    bool ok = true;
    auto cuda_cleanup = [&ok](cudaError_t e, const char* call) {
        if (e != cudaSuccess) { fprintf(stderr, "Cleanup %s: %s\n", call, cudaGetErrorString(e)); ok = false; }
    };
    for (int rank = 0; rank < P; ++rank) {
        auto& d = devices[rank];
        if (!d.send && !d.recv && !d.stream && !d.comm) continue;
        cudaError_t selected = cudaSetDevice(rank);
        cuda_cleanup(selected, "cudaSetDevice");
        if (selected != cudaSuccess) continue;
        if (d.comm) {
            ncclResult_t e = abort_communicators ? ncclCommAbort(d.comm) : ncclCommDestroy(d.comm);
            if (e != ncclSuccess) { fprintf(stderr, "Communicator cleanup rank %d: %s\n", rank, ncclGetErrorString(e)); ok = false; }
            d.comm = nullptr;
        }
        if (d.stream) cuda_cleanup(cudaStreamDestroy(d.stream), "cudaStreamDestroy");
        if (d.send) cuda_cleanup(cudaFree(d.send), "cudaFree send");
        if (d.recv) cuda_cleanup(cudaFree(d.recv), "cudaFree recv");
        for (int peer = 0; peer < P; ++peer)
            if (enabled[rank][peer]) cuda_cleanup(cudaDeviceDisablePeerAccess(peer), "cudaDeviceDisablePeerAccess");
    }
    return ok;
}

int main() {
    try {
        int count = 0, nccl_version = 0;
        CUDA(cudaGetDeviceCount(&count));
        NCCL(ncclGetVersion(&nccl_version));
        printf("EIGHT_GPU_CORRECTNESS_AUDIT: visible=%d required=8 NCCL=%d\n", count, nccl_version);
        printf("No oversubscription. Using distinct CUDA-visible ordinals 0..7; record physical topology separately.\n");
        printf("Correctness only: no bandwidth/latency/saturation claim. Capability does not prove NVLink.\n");
        if (count < P) throw std::runtime_error("Requires eight actual CUDA-visible devices");
        for (int rank = 0; rank < P; ++rank) {
            cudaDeviceProp prop;
            CUDA(cudaGetDeviceProperties(&prop, rank));
            printf("GPU %d: %s CC=%d.%d memory_bytes=%zu\n", rank, prop.name, prop.major, prop.minor, prop.totalGlobalMem);
            CUDA(cudaSetDevice(rank));
            CUDA(cudaMalloc(&devices[rank].send, CAPACITY * sizeof(int)));
            CUDA(cudaMalloc(&devices[rank].recv, CAPACITY * sizeof(int)));
            CUDA(cudaStreamCreateWithFlags(&devices[rank].stream, cudaStreamNonBlocking));
        }
        for (int accessor = 0; accessor < P; ++accessor) {
            CUDA(cudaSetDevice(accessor));
            for (int owner = 0; owner < P; ++owner) {
                if (owner == accessor) continue;
                int can = 0;
                CUDA(cudaDeviceCanAccessPeer(&can, accessor, owner));
                capability[accessor][owner] = can != 0;
                if (can) {
                    cudaError_t e = cudaDeviceEnablePeerAccess(owner, 0);
                    if (e == cudaErrorPeerAccessAlreadyEnabled) {
                        cudaError_t cleared = cudaGetLastError();
                        if (cleared != cudaSuccess && cleared != cudaErrorPeerAccessAlreadyEnabled)
                            CUDA(cleared);
                    } else { CUDA(e); enabled[accessor][owner] = true; }
                }
            }
        }
        audit_peers();
        std::array<ncclComm_t, P> comms{};
        std::array<int, P> ordinals{{0,1,2,3,4,5,6,7}};
        ncclResult_t initialized = ncclCommInitAll(comms.data(), P, ordinals.data());
        for (int rank = 0; rank < P; ++rank) devices[rank].comm = comms[rank];
        NCCL(initialized);
        audit_collectives();
        audit_halos();
        bool cleaned = cleanup(false);
        bool ok = numerical_errors == 0 && cleaned;
        printf("EIGHT_GPU_AUDIT: %s; numerical_errors=%zu; cleanup=%s\n",
               ok ? "PASS" : "FAIL", numerical_errors, cleaned ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        fprintf(stderr, "EIGHT_GPU_AUDIT: FAIL: %s\n", error.what());
        cleanup(true);
        return 1;
    }
}
