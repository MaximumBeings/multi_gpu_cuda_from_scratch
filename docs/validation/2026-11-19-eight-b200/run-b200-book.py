from pathlib import Path
import json,os,subprocess,sys,time,datetime,hashlib
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'tools'))
import validate_examples as v
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
env=dict(os.environ)
nv=subprocess.check_output([str(root/'deps/bin/python'),'-c','import nvidia.nvshmem; print(list(nvidia.nvshmem.__path__)[0])'],text=True).strip()
lib=Path(nv)/'lib'
if not (lib/'libnvshmem_host.so').exists():(lib/'libnvshmem_host.so').symlink_to('libnvshmem_host.so.3')
env.update(NVSHMEM_HOME=nv,NVSHMEM_LIB=str(lib),NVSHMEM_BOOTSTRAP='MPI',CUDA_HOME='/usr',CUDA_INCDIR='/usr/include',CUDA_LIBDIR='/usr/lib/x86_64-linux-gnu',LD_LIBRARY_PATH=str(lib)+':/usr/lib/x86_64-linux-gnu',CUDA_DEVICE_ORDER='PCI_BUS_ID')
(root/'test-environment.json').write_text(json.dumps({k:env[k] for k in ['NVSHMEM_HOME','NVSHMEM_LIB','NVSHMEM_BOOTSTRAP','CUDA_HOME','CUDA_INCDIR','CUDA_LIBDIR','LD_LIBRARY_PATH','CUDA_DEVICE_ORDER']},indent=2)+'\n')
rows=v.inventory();assert len(rows)==145
children=[]
for worker in range(4):
 cmd=['python3','tools/validate_examples.py','--execute','--arch','sm_100','--mpi-ranks','8','--run-timeout','30','--compile-timeout','90','--output',f'results/part-{worker}']
 for row in rows[worker::4]:cmd+=['--match',row['source']]
 log=(root/f'worker-{worker}.log').open('w');children.append((worker,subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT),log))
summary=[]
for worker,p,log in children:
 code=p.wait();log.close();summary.append(dict(worker=worker,exit_code=code))
allrows=[]
for worker in range(4):allrows+=json.loads((root/f'results/part-{worker}/results.json').read_text())
assert len(allrows)==145 and len({r['source'] for r in allrows})==145
(root/'results/all-results.json').write_text(json.dumps(sorted(allrows,key=lambda r:r['source']),indent=2)+'\n')
# Full eight-device numerical harness runs alone after chapter workloads finish.
extra=[]
for name,args in [('eight-compile',['nvcc','-std=c++17','-O2','-arch=sm_100','tools/validate_eight_gpus.cu','-lnccl','-pthread','-o','eight-gpu-check']),('eight-run',['./eight-gpu-check']),('ipc-memcheck',['compute-sanitizer','--tool','memcheck','--target-processes','all','--error-exitcode','3',str(root/'results'/f'part-{next(i for i in range(4) if any(r["number"]==17 for r in rows[i::4]))}'/'17_ipc_mem_handle'/'example')])]:
 result=v.run(args,root/(name+'.log'),90,env);extra.append(dict(name=name,command=args,**result))
(root/'extra-results.json').write_text(json.dumps(extra,indent=2)+'\n')
from collections import Counter
out=dict(started_utc=start,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),workers=summary,status_counts=dict(Counter(r['status'] for r in allrows)),extra=extra)
(root/'run-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
