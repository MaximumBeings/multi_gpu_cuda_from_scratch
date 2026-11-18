// 67_nccl_captured_in_graph.cu
// Capture one initialized NCCL all-reduce and validate each replay on every rank.
#define OMPI_SKIP_MPICXX
#include <mpi.h>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
 fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); } } while(0)
#define NCCL_OK(call) do { ncclResult_t e=(call); if(e!=ncclSuccess) { \
 fprintf(stderr,"%s: %s\n",#call,ncclGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2); } } while(0)
int main(int argc, char **argv) {
 MPI_Init(&argc,&argv);
 int rank,size,localRank,count;
 MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
 MPI_Comm node;
 MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&node);
 MPI_Comm_rank(node,&localRank); MPI_Comm_free(&node);
 CUDA_OK(cudaGetDeviceCount(&count));
 if(localRank>=count) MPI_Abort(MPI_COMM_WORLD,3);
 CUDA_OK(cudaSetDevice(localRank));
 ncclUniqueId id;
 if(rank==0) NCCL_OK(ncclGetUniqueId(&id));
 MPI_Bcast(&id,sizeof(id),MPI_BYTE,0,MPI_COMM_WORLD);
 ncclComm_t comm; NCCL_OK(ncclCommInitRank(&comm,size,id,rank));
 cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream));
 constexpr int elements=1024,replays=3;
 float *input,*output;
 CUDA_OK(cudaMalloc(&input,elements*sizeof(float)));
 CUDA_OK(cudaMalloc(&output,elements*sizeof(float)));
 std::vector<float> host(elements,float(rank+1));
 CUDA_OK(cudaMemcpy(input,host.data(),elements*sizeof(float),cudaMemcpyHostToDevice));
 CUDA_OK(cudaMemset(output,0,elements*sizeof(float)));
 cudaGraph_t graph; cudaGraphExec_t exec;
 CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
 NCCL_OK(ncclAllReduce(input,output,elements,ncclFloat,ncclSum,comm,stream));
 CUDA_OK(cudaStreamEndCapture(stream,&graph));
 CUDA_OK(cudaGraphInstantiate(&exec,graph,0));
 int errors=0;
 float expected=float(size*(size+1)/2);
 for(int replay=0;replay<replays;++replay) {
  CUDA_OK(cudaGraphLaunch(exec,stream));
  CUDA_OK(cudaMemcpyAsync(host.data(),output,elements*sizeof(float),cudaMemcpyDeviceToHost,stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  for(float value:host) if(value!=expected) ++errors;
 }
 int totalErrors=0; MPI_Allreduce(&errors,&totalErrors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
 if(rank==0) printf("VALIDATION %s: %d ranks, %d graph replays, errors=%d\n",
  totalErrors?"FAIL":"PASS",size,replays,totalErrors);
 CUDA_OK(cudaGraphExecDestroy(exec)); CUDA_OK(cudaGraphDestroy(graph));
 CUDA_OK(cudaFree(input)); CUDA_OK(cudaFree(output)); CUDA_OK(cudaStreamDestroy(stream)); NCCL_OK(ncclCommDestroy(comm));
 MPI_Finalize(); return totalErrors?1:0;
}
