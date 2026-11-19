// 31_nccl_collectives_mapped.cu
// Expected-error API probe; no collective is claimed to execute successfully.
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>
int main() {
 ncclComm_t comm=nullptr; cudaStream_t stream=nullptr; float *buf=nullptr;
 int errors=0;
 auto check=[&](const char *name,ncclResult_t value) {
  printf("%s: %s (expected invalid argument)\n",name,ncclGetErrorString(value));
  if(value!=ncclInvalidArgument) ++errors;
 };
 check("Broadcast",ncclBroadcast(buf,buf,4,ncclFloat,0,comm,stream));
 check("Reduce",ncclReduce(buf,buf,4,ncclFloat,ncclSum,0,comm,stream));
 check("AllReduce",ncclAllReduce(buf,buf,4,ncclFloat,ncclSum,comm,stream));
 check("AllGather",ncclAllGather(buf,buf,4,ncclFloat,comm,stream));
 check("ReduceScatter",ncclReduceScatter(buf,buf,4,ncclFloat,ncclSum,comm,stream));
 check("Send",ncclSend(buf,1,ncclFloat,0,comm,stream));
 check("Recv",ncclRecv(buf,1,ncclFloat,0,comm,stream));
 printf("API PROBE %s: NULL-communicator rejection only, errors=%d\n",errors?"FAIL":"PASS",errors);
 return errors?1:0;
}
