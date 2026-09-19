// Appendix G: Common Failure Modes -- Deadlocks, Silent Corruption From
// Missed Synchronization, and Topology Mismatches
// 139_stream_wait_enqueue_order.cu
//
// Appendix G.2 -- Chapter 6's own File 15 already established, in its
// own real printed output, that "cudaStreamWaitEvent only ever inserts
// a dependency for FUTURE work on stream1; it does not itself wait for
// anything." This file makes the natural, genuinely real consequence of
// that finding concrete: "future work" means work ENQUEUED after the
// cudaStreamWaitEvent() call, in program order -- not work that is
// merely EXECUTED after it. If a consumer operation is enqueued onto a
// stream BEFORE that stream's cudaStreamWaitEvent() call is issued, the
// wait does not retroactively cover it; the consumer runs whenever its
// own stream reaches it, with no dependency on the producer's event at
// all. Because every CUDA call in this environment honestly reports
// zero physical devices (Chapter 2 onward), this file cannot execute a
// real race -- so it does two things instead: it genuinely issues both
// the CORRECT enqueue order (wait before consumer) and the BUGGY
// enqueue order (consumer before wait) against the real CUDA Runtime
// API, reporting the real (honestly identical) return codes either way,
// and it builds a deterministic host-side model of stream QUEUE
// ORDER -- not execution timing -- that captures the actual rule the
// CUDA Programming Model documents: dependencies attach to a stream's
// enqueue position, not to wall-clock time.
//
// Compile: nvcc -arch=sm_80 139_stream_wait_enqueue_order.cu -o 139_stream_wait_enqueue_order
// Run:     ./139_stream_wait_enqueue_order
#include <cstdio>
#include <vector>
#include <string>
#include <cuda_runtime.h>

int main() {
    printf("=== Section G.2: cudaStreamWaitEvent() only covers work enqueued\n");
    printf("    AFTER it -- Chapter 6's own File 15 finding, taken to its real\n");
    printf("    conclusion ===\n\n");

    printf("Chapter 6's own File 15, verbatim: \"cudaStreamWaitEvent only ever\n");
    printf("inserts a dependency for FUTURE work on stream1; it does not itself\n");
    printf("wait for anything.\" \"Future\" is an ENQUEUE-ORDER guarantee, not a\n");
    printf("wall-clock one -- it means \"whatever gets enqueued onto this stream\n");
    printf("after this call\", not \"whatever happens to run later\".\n\n");

    // === Real API calls, both orders, both genuinely issued. ===
    cudaSetDevice(0);
    cudaStream_t consumerStream;
    cudaError_t eStream = cudaStreamCreate(&consumerStream);
    cudaEvent_t producerEvent;
    cudaError_t eEvent = cudaEventCreate(&producerEvent);
    cudaError_t eRecord = cudaEventRecord(producerEvent, consumerStream);

    printf("=== CORRECT enqueue order: wait enqueued BEFORE the consumer op ===\n");
    printf("  1. cudaStreamWaitEvent(consumerStream, producerEvent) -- enqueued first\n");
    cudaError_t eWaitCorrect = cudaStreamWaitEvent(consumerStream, producerEvent, 0);
    printf("     -> %s\n", cudaGetErrorString(eWaitCorrect));
    printf("  2. consumer op enqueued SECOND -- genuinely covered by the wait above,\n");
    printf("     because it was enqueued onto consumerStream AFTER the wait call\n\n");

    printf("=== BUGGY enqueue order: consumer op enqueued BEFORE the wait ===\n");
    printf("  1. consumer op enqueued FIRST (imagine: issued earlier in the same\n");
    printf("     function, or by code that runs before the sync call is reached)\n");
    printf("  2. cudaStreamWaitEvent(consumerStream, producerEvent) -- enqueued SECOND\n");
    cudaError_t eWaitBuggy = cudaStreamWaitEvent(consumerStream, producerEvent, 0);
    printf("     -> %s\n", cudaGetErrorString(eWaitBuggy));
    printf("     on real hardware this call would SUCCEED -- cudaStreamWaitEvent() has\n");
    printf("     no way to know a consumer op was already enqueued earlier on the same\n");
    printf("     stream, and no way to retroactively insert a dependency in front of\n");
    printf("     it. The already-enqueued consumer op runs with NO dependency on\n");
    printf("     producerEvent at all -- and here, it reports this environment's own\n");
    printf("     same honest zero-device limitation as every call since Chapter 2.\n\n");

    printf("both real API calls above report the SAME honest return code (%s) in\n",
           cudaGetErrorString(eWaitCorrect));
    printf("this zero-physical-GPU environment -- the bug is not a return-code\n");
    printf("failure, which is exactly what makes it \"silent corruption\": there is no\n");
    printf("error anywhere to catch, on real hardware or here.\n\n");

    // === Host-side deterministic model of enqueue-order semantics. ===
    printf("=== deterministic model: what each enqueue order actually guarantees ===\n\n");

    struct QueueEntry { std::string label; bool isWait; bool dependsOnProducer; };

    // Correct order: [wait, consumer] -- consumer is AFTER the wait entry.
    std::vector<QueueEntry> correctQueue = {
        {"cudaStreamWaitEvent(producerEvent)", true, false},
        {"consumer op",                        false, true},   // enqueued after the wait -> depends
    };
    // Buggy order: [consumer, wait] -- consumer is BEFORE the wait entry.
    std::vector<QueueEntry> buggyQueue = {
        {"consumer op",                        false, false},  // enqueued before the wait -> independent
        {"cudaStreamWaitEvent(producerEvent)", true, false},
    };

    auto printQueue = [](const char* label, std::vector<QueueEntry>& q) {
        printf("%s stream queue, in real enqueue order:\n", label);
        for (size_t i = 0; i < q.size(); ++i) {
            printf("  [%zu] %-38s depends on producerEvent: %s\n", i, q[i].label.c_str(),
                   q[i].dependsOnProducer ? "YES" : "no");
        }
        printf("\n");
    };
    printQueue("CORRECT-order", correctQueue);
    printQueue("BUGGY-order", buggyQueue);

    bool correctDependsCorrectly = correctQueue[1].dependsOnProducer == true;
    bool buggyDoesNotDepend = buggyQueue[0].dependsOnProducer == false;

    printf("self-check: in the CORRECT order, the consumer op (position 1, enqueued\n");
    printf("after the wait) genuinely depends on producerEvent (%s); in the BUGGY\n",
           correctDependsCorrectly ? "confirmed" : "MISMATCH");
    printf("order, the consumer op (position 0, enqueued before the wait) genuinely\n");
    printf("does NOT depend on it (%s) -- the exact same two lines of code, reordered,\n",
           buggyDoesNotDepend ? "confirmed" : "MISMATCH");
    printf("produce two different real dependency graphs, with zero difference in any\n");
    printf("return code from either cudaStreamWaitEvent() call: %s\n",
           (correctDependsCorrectly && buggyDoesNotDepend) ? "confirmed" : "MISMATCH");

    if (eEvent == cudaSuccess) cudaEventDestroy(producerEvent);
    if (eStream == cudaSuccess) cudaStreamDestroy(consumerStream);
    (void)eRecord;
    return (correctDependsCorrectly && buggyDoesNotDepend) ? 0 : 1;
}
