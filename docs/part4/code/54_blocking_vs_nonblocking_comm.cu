// 54_blocking_vs_nonblocking_comm.cu
// Blocking initialization/destruction plus nonblocking initialization/poll/abort API probe. No peer failure is injected.
#include <cstdio>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <nccl.h>
int main() {
 if(cudaSetDevice(0)!=cudaSuccess) return 2;
 ncclUniqueId id; if(ncclGetUniqueId(&id)!=ncclSuccess) return 1;
 ncclConfig_t blocking=NCCL_CONFIG_INITIALIZER; blocking.blocking=1;
 ncclComm_t blockingComm=nullptr;
 ncclResult_t first=ncclCommInitRankConfig(&blockingComm,1,id,0,&blocking);
 if(first!=ncclSuccess) { if(blockingComm) ncclCommAbort(blockingComm); return 1; }
 if(ncclCommDestroy(blockingComm)!=ncclSuccess) return 1;
 if(ncclGetUniqueId(&id)!=ncclSuccess) return 1;
 ncclConfig_t config=NCCL_CONFIG_INITIALIZER; config.blocking=0;
 ncclComm_t comm=nullptr;
 ncclResult_t init=ncclCommInitRankConfig(&comm,1,id,0,&config);
 if(init!=ncclSuccess && init!=ncclInProgress) {
  if(comm) ncclCommAbort(comm); fprintf(stderr,"init: %s\n",ncclGetErrorString(init)); return 1;
 }
 auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
 ncclResult_t state=ncclInProgress; int errors=0;
 do {
  ncclResult_t query=ncclCommGetAsyncError(comm,&state);
  if(query!=ncclSuccess) { ++errors; break; }
  if(state!=ncclInProgress) break;
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
 } while(std::chrono::steady_clock::now()<deadline);
 if(state!=ncclSuccess) ++errors;
 if(ncclCommAbort(comm)!=ncclSuccess) ++errors;
 printf("API PROBE %s: one-rank nonblocking init/poll/abort only; no fault recovery tested, errors=%d\n",errors?"FAIL":"PASS",errors);
 return errors?1:0;
}
