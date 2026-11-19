// Chapter 7: two-process CUDA IPC with a checked nonuniform payload.
// 17_ipc_mem_handle.cu
// Linux/POSIX: fork happens before any CUDA initialization. Producer device 0
// stays alive until consumer device 1 closes its imported resources.
#include <cstdio>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
#include <cuda_runtime.h>

constexpr int COUNT = 4096;
static float expected(int i) { return float((i * 17) % 4093 - 2046) / 8.0f; }
static int error(cudaError_t status, const char* call) {
    if (status == cudaSuccess) return 0;
    fprintf(stderr, "%s: %s (code %d)\n", call, cudaGetErrorString(status), int(status));
    return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ? 2 : 1;
}
#define CUDA(call) do { int code = error((call), #call); if (code) return code; } while (0)
static bool transfer(int fd, void* data, size_t size, bool writing) {
    char* bytes = static_cast<char*>(data);
    while (size) {
        ssize_t done = writing ? write(fd, bytes, size) : read(fd, bytes, size);
        if (done < 0 && errno == EINTR) continue;
        if (done <= 0) return false;
        bytes += done; size -= size_t(done);
    }
    return true;
}
struct Handles { cudaIpcMemHandle_t memory;  };

static int consumer(int fd) {
    Handles handles{};
    if (!transfer(fd, &handles, sizeof(handles), false)) {
        fprintf(stderr, "IPC handle transport failed; producer did not publish resources.\n");
        return 1;
    }
    close(fd);
    CUDA(cudaSetDevice(1));
    void* imported = nullptr;
    CUDA(cudaIpcOpenMemHandle(&imported, handles.memory, cudaIpcMemLazyEnablePeerAccess));

    std::vector<float> output(COUNT);
    CUDA(cudaMemcpy(output.data(), imported, COUNT * sizeof(float), cudaMemcpyDeviceToHost));
    int mismatches = 0;
    for (int i = 0; i < COUNT; ++i) if (output[i] != expected(i)) {
        if (mismatches < 4) fprintf(stderr, "Mismatch[%d]: %.9g expected %.9g\n", i, output[i], expected(i));
        ++mismatches;
    }

    CUDA(cudaIpcCloseMemHandle(imported));
    printf("Consumer: %d / %d exact payload values correct.\n", COUNT - mismatches, COUNT);
    return mismatches ? 1 : 0;
}

static int producer(int fd) {
    int count = 0;
    CUDA(cudaGetDeviceCount(&count));
    if (count < 2) { fprintf(stderr, "BLOCKED: requires 2 visible GPUs; found %d.\n", count); return 2; }
    CUDA(cudaSetDevice(0));
    int peer = 0;
    CUDA(cudaDeviceCanAccessPeer(&peer, 1, 0));
    if (!peer) { fprintf(stderr, "BLOCKED: device 1 cannot access device 0 memory.\n"); return 2; }
    std::vector<float> input(COUNT);
    for (int i = 0; i < COUNT; ++i) input[i] = expected(i);
    float* memory = nullptr;
    CUDA(cudaMalloc(&memory, COUNT * sizeof(float)));

    CUDA(cudaMemcpy(memory, input.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice));
    // A pageable H2D cudaMemcpy may return before device DMA completes.
    // Complete the producer write before publishing a handle to another context.
    CUDA(cudaDeviceSynchronize());
    Handles handles{};
    CUDA(cudaIpcGetMemHandle(&handles.memory, memory));

    if (!transfer(fd, &handles, sizeof(handles), true)) {
        fprintf(stderr, "Could not send IPC handles.\n"); return 1;
    }
    close(fd);
    // main waits for child process exit before destroying producer resources.
    // Allocation remains live through cudaIpcCloseMemHandle in the consumer.
    return 0;
}

int main() {
    signal(SIGPIPE, SIG_IGN);
    int descriptors[2];
    if (pipe(descriptors) != 0) { perror("pipe"); return 1; }
    pid_t child = fork(); // No CUDA API has been called in this process yet.
    if (child < 0) { perror("fork"); close(descriptors[0]); close(descriptors[1]); return 1; }
    if (child == 0) {
        close(descriptors[1]);
        int code = consumer(descriptors[0]);
        fflush(nullptr);
        _exit(code);
    }
    close(descriptors[0]);
    int result = producer(descriptors[1]);
    // Closing also releases a consumer blocked on read if production failed.
    if (result != 0) close(descriptors[1]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        perror("waitpid"); return 1;
    }
    int child_result = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    if (result == 0) result = child_result;
    // Destroy the producer context only after the importing process is gone.
    if (result != 2) { int cleanup = error(cudaDeviceReset(), "cudaDeviceReset"); if (cleanup) result = 1; }
    if (result == 0) printf("PASS: two-process IPC memory, 4096 nonuniform values checked.\n");
    return result;
}
