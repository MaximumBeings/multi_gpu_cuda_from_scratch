// 68_host_node_and_multi_process_graphs.cu
// A local graph host callback; no MPI or CUDA call is made inside the callback.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUDA_OK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
 fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
void CUDART_CB hostNode(void *argument) { ++*static_cast<int *>(argument); }
int main() {
 CUDA_OK(cudaSetDevice(0));
 int calls=0; cudaGraph_t graph; cudaGraphNode_t node;
 CUDA_OK(cudaGraphCreate(&graph,0));
 cudaHostNodeParams parameters{}; parameters.fn=hostNode; parameters.userData=&calls;
 CUDA_OK(cudaGraphAddHostNode(&node,graph,nullptr,0,&parameters));
 cudaGraphExec_t exec; CUDA_OK(cudaGraphInstantiate(&exec,graph,0));
 cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream));
 for(int i=0;i<3;++i) CUDA_OK(cudaGraphLaunch(exec,stream));
 CUDA_OK(cudaStreamSynchronize(stream)); // callback has completed before host reads calls
 int errors=(calls!=3);
 CUDA_OK(cudaGraphExecDestroy(exec)); CUDA_OK(cudaGraphDestroy(graph)); CUDA_OK(cudaStreamDestroy(stream));
 printf("API PROBE %s: local graph host callbacks=%d expected=3, errors=%d\n",errors?"FAIL":"PASS",calls,errors);
 return errors?1:0;
}
