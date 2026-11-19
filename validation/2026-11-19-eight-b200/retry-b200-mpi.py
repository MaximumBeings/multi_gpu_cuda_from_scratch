from pathlib import Path
import os,subprocess,sys,json,time
root=Path(__file__).resolve().parent
while not (root/'run-summary.json').exists():time.sleep(1)
old=json.loads((root/'results/all-results.json').read_text())
retry=[r for r in old if r['status'] not in {'validation_pass','api_probe_pass','executed_unverified'}]
env=dict(os.environ);prefix=str(root/'mpi/usr');nv=subprocess.check_output([str(root/'deps/bin/python'),'-c','import nvidia.nvshmem; print(list(nvidia.nvshmem.__path__)[0])'],text=True).strip()
env.update(OPAL_PREFIX=prefix,PATH=prefix+'/bin:'+env['PATH'],NVSHMEM_HOME=nv,NVSHMEM_LIB=nv+'/lib',NVSHMEM_BOOTSTRAP='MPI',CUDA_HOME='/usr',CUDA_INCDIR='/usr/include',CUDA_LIBDIR='/usr/lib/x86_64-linux-gnu',LD_LIBRARY_PATH=nv+'/lib:'+prefix+'/lib/x86_64-linux-gnu:'+prefix+'/lib/x86_64-linux-gnu/openmpi/lib:/usr/lib/x86_64-linux-gnu',CUDA_DEVICE_ORDER='PCI_BUS_ID')
(root/'retry-environment.json').write_text(json.dumps({k:env[k] for k in ['OPAL_PREFIX','PATH','NVSHMEM_HOME','NVSHMEM_LIB','NVSHMEM_BOOTSTRAP','CUDA_HOME','CUDA_INCDIR','CUDA_LIBDIR','LD_LIBRARY_PATH','CUDA_DEVICE_ORDER']},indent=2)+'\n')
children=[]
for i in range(3):
 chunk=retry[i::3]
 if not chunk:continue
 cmd=['python3','tools/validate_examples.py','--execute','--arch','sm_100','--mpi-ranks','8','--run-timeout','25','--compile-timeout','60','--output',f'results/retry-{i}']
 for r in chunk:cmd+=['--match',r['source']]
 log=(root/f'retry-{i}.log').open('w');children.append((i,subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT),log))
for i,p,log in children:p.wait();log.close()
by={r['source']:r for r in old}
for i,p,log in children:
 for r in json.loads((root/f'results/retry-{i}/results.json').read_text()):by[r['source']]=r
rows=sorted(by.values(),key=lambda r:r['source']);(root/'results/latest-results.json').write_text(json.dumps(rows,indent=2)+'\n')
from collections import Counter
print(json.dumps(dict(Counter(r['status'] for r in rows)),indent=2),flush=True)
for r in rows:
 if r['status'] not in {'validation_pass','api_probe_pass','executed_unverified'}: print(r['status'],r['source'],r.get('reason',''),flush=True)
(root/'retry-complete.json').write_text(json.dumps(dict(status_counts=dict(Counter(r['status'] for r in rows))),indent=2)+'\n')
