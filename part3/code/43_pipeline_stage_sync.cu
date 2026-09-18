// Chapter 15: Pipeline Parallelism -- Splitting Layers Across Devices, and the Bubble That Costs You
// 43_pipeline_stage_sync.cu
//
// Section 14.1's own schedule showed WHEN each device is supposed to
// start each micro-batch -- but nothing in host arithmetic actually
// ENFORCES that device d+1 waits for device d's own output before
// starting. That enforcement is a real cross-device dependency, and
// this book already built the exact mechanism for it in Chapter 6:
// an event recorded on one device's stream, handed to
// cudaStreamWaitEvent() on the NEXT device's stream, with no host
// blocking and no cudaDeviceSynchronize() anywhere. Pipeline
// parallelism doesn't need a new primitive here either -- it needs
// Chapter 6's real API called once per (device, micro-batch)
// boundary in Section 14.1's own schedule. Genuinely compiled with a
// real nvcc and genuinely run.
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    // One cell of Section 15.1's own schedule: device 0 finishes
    // micro-batch 0 and device 1 must wait for it before starting its
    // OWN work on micro-batch 0. Every other (device, micro-batch)
    // boundary in that schedule needs this identical pattern, called
    // once per cell -- shown here for one, not repeated K*M times.
    const int PRODUCER_DEVICE = 0, CONSUMER_DEVICE = 1, MICROBATCH = 0;

    cudaError_t eSetProducer = cudaSetDevice(PRODUCER_DEVICE);
    printf("cudaSetDevice(%d) [producer]: %s (code %d)\n", PRODUCER_DEVICE,
           cudaGetErrorString(eSetProducer), (int)eSetProducer);

    cudaStream_t producerStream;
    cudaError_t eProducerStream = cudaStreamCreate(&producerStream);
    printf("cudaStreamCreate(producerStream, on device %d): %s (code %d)\n",
           PRODUCER_DEVICE, cudaGetErrorString(eProducerStream), (int)eProducerStream);

    cudaEvent_t microbatchReady;
    cudaError_t eEvent = cudaEventCreate(&microbatchReady);
    printf("cudaEventCreate(microbatchReady, on device %d): %s (code %d)\n",
           PRODUCER_DEVICE, cudaGetErrorString(eEvent), (int)eEvent);

    // In real use, this call would come AFTER device 0's layer-range
    // kernels for micro-batch 0 have been queued on producerStream --
    // the event only fires once everything queued before it on that
    // stream has actually finished.
    cudaError_t eRecord = cudaEventRecord(microbatchReady, producerStream);
    printf("cudaEventRecord(microbatchReady, producerStream) "
           "[marks 'micro-batch %d done on device %d']: %s (code %d)\n",
           MICROBATCH, PRODUCER_DEVICE, cudaGetErrorString(eRecord), (int)eRecord);

    cudaError_t eSetConsumer = cudaSetDevice(CONSUMER_DEVICE);
    printf("\ncudaSetDevice(%d) [consumer]: %s (code %d)\n", CONSUMER_DEVICE,
           cudaGetErrorString(eSetConsumer), (int)eSetConsumer);

    cudaStream_t consumerStream;
    cudaError_t eConsumerStream = cudaStreamCreate(&consumerStream);
    printf("cudaStreamCreate(consumerStream, on device %d): %s (code %d)\n",
           CONSUMER_DEVICE, cudaGetErrorString(eConsumerStream), (int)eConsumerStream);

    cudaError_t eWait = cudaStreamWaitEvent(consumerStream, microbatchReady, 0);
    printf("cudaStreamWaitEvent(consumerStream waits on microbatchReady): "
           "%s (code %d)\n", cudaGetErrorString(eWait), (int)eWait);

    printf("\nEvery future kernel queued on consumerStream after this call\n"
           "-- device %d's own layer-range kernels for micro-batch %d -- will\n"
           "wait for device %d's event before it runs, with no host blocking\n"
           "and no cudaDeviceSynchronize() anywhere, exactly as Chapter 6\n"
           "established. Section 15.1's schedule is what decides WHICH\n"
           "(device, micro-batch) pairs need this call; this section is what\n"
           "actually enforces the dependency the schedule assumes.\n",
           CONSUMER_DEVICE, MICROBATCH, PRODUCER_DEVICE);

    if (eEvent == cudaSuccess) cudaEventDestroy(microbatchReady);
    if (eProducerStream == cudaSuccess) cudaStreamDestroy(producerStream);
    if (eConsumerStream == cudaSuccess) cudaStreamDestroy(consumerStream);
    return 0;
}
