// 55_nccl_async_detect_and_recover.cu
// Nonblocking initialization/poll/abort API probe. No peer failure is injected.
#include <cstdio>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>
#include <nccl.h>
int main() {
 if(cudaSetDevice(0)!=cudaSuccess) return 2;
 ncclUniqueId id; if(ncclGetUniqueId(&id)!=ncclSuccess) return 1;
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
