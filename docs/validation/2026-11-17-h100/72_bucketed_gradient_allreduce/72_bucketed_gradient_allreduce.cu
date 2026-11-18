// 72_bucketed_gradient_allreduce.cu
// Three steps, four initialized gradient buckets per step, checked on every rank.
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
 constexpr int steps=3,buckets=4,elements=1024;
 float *device; CUDA_OK(cudaMalloc(&device,elements*sizeof(float)));
 std::vector<float> host(elements);
 int errors=0;
 for(int step=0;step<steps;++step) for(int bucket=0;bucket<buckets;++bucket) {
  float input=float(rank+1+10*step+bucket);
  for(float &v:host) v=input;
  CUDA_OK(cudaMemcpyAsync(device,host.data(),elements*sizeof(float),cudaMemcpyHostToDevice,stream));
  NCCL_OK(ncclAllReduce(device,device,elements,ncclFloat,ncclSum,comm,stream));
  CUDA_OK(cudaMemcpyAsync(host.data(),device,elements*sizeof(float),cudaMemcpyDeviceToHost,stream));
  CUDA_OK(cudaStreamSynchronize(stream));
  float expected=float(size*(size+1)/2+size*(10*step+bucket));
  for(float v:host) if(v!=expected) ++errors;
  if(rank==0) printf("step=%d bucket=%d expected=%g checked=%d\n",step,bucket,expected,elements);
 }
 int totalErrors=0; MPI_Allreduce(&errors,&totalErrors,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
 if(rank==0) printf("VALIDATION %s: %d ranks, %d buckets, errors=%d\n",
  totalErrors?"FAIL":"PASS",size,steps*buckets,totalErrors);
 CUDA_OK(cudaFree(device)); CUDA_OK(cudaStreamDestroy(stream)); NCCL_OK(ncclCommDestroy(comm));
 MPI_Finalize(); return totalErrors?1:0;
}
