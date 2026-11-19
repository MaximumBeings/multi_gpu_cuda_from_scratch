// 30_nccl_comm_init.cu
// API probe: valid communicator lifetimes and rejected malformed arguments.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>
int main() {
 int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count<1) return 2;
 std::vector<ncclComm_t> comms(count,nullptr);
 ncclResult_t result=ncclCommInitAll(comms.data(),count,nullptr);
 if(result!=ncclSuccess) { fprintf(stderr,"init: %s\n",ncclGetErrorString(result)); return 1; }
 int errors=0;
 for(auto comm:comms) if(ncclCommDestroy(comm)!=ncclSuccess) ++errors;
 ncclComm_t single=nullptr;
 result=ncclCommInitAll(&single,1,nullptr);
 if(result!=ncclSuccess) ++errors;
 else if(ncclCommDestroy(single)!=ncclSuccess) ++errors;
 ncclComm_t unused=nullptr;
 if(ncclCommInitAll(nullptr,1,nullptr)!=ncclInvalidArgument) ++errors;
 if(ncclCommInitAll(&unused,-1,nullptr)!=ncclInvalidArgument) ++errors;
 printf("API PROBE %s: %d devices, valid initialization/destruction and invalid argument checks, errors=%d\n",errors?"FAIL":"PASS",count,errors);
 return errors?1:0;
}
