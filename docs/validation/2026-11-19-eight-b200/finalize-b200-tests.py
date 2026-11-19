from pathlib import Path
import os,json,sys,time,subprocess,hashlib,datetime
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'tools'));import validate_examples as v
while not (root/'retry-complete.json').exists():time.sleep(1)
rows=json.loads((root/'results/latest-results.json').read_text());env=dict(os.environ);env.update(json.loads((root/'retry-environment.json').read_text()));env.update(OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_pml='ob1')
(root/'final-environment.json').write_text(json.dumps({k:env[k] for k in json.loads((root/'retry-environment.json').read_text())}|{'OMPI_MCA_btl':env['OMPI_MCA_btl'],'OMPI_MCA_pml':env['OMPI_MCA_pml']},indent=2)+'\n')
from concurrent.futures import ThreadPoolExecutor
bad=[r for r in rows if r['status'] not in {'validation_pass','api_probe_pass','executed_unverified'}]
def retry(r):
 stem=Path(r['source']).stem;folder=root/'results/final'/stem;folder.mkdir(parents=True)
 import shutil
 shutil.copy2(root/r['source'],folder/Path(r['source']).name)
 if r.get('compile',{}).get('exit_code')!=0:
  cmd=v.compile_command(r,folder/'example','sm_100',Path('/usr'),Path(env['NVSHMEM_HOME']),Path(env['NVSHMEM_LIB']))
  r['compile_command']=cmd;r['compile']=v.run(cmd,folder/'compile.log',120,env)
  if r['compile'].get('exit_code')!=0:r['status']='compile_failed';return r
  binary=folder/'example'
 else:
  old=Path(r['compile_command'][r['compile_command'].index('-o')+1]);binary=old
  shutil.copy2(old.parent/'compile.log',folder/'compile.log');r['compile_reused_from']=str(old.parent)
 cmd=([str(root/'mpi/usr/bin/mpirun'),'--oversubscribe','-np',str(r['processes'])] if 'mpi' in r['dependencies'] else [])+[str(binary)]
 if r['number']==57:cmd+=['1000000','sendrecv']
 if r['number']==138:cmd+=['fixed']
 r['run_command']=cmd;r['run']=v.run(cmd,folder/'run.log',60,env);text=(folder/'run.log').read_text();r['run_started_utc']=r['run']['started_utc']
 if r['run'].get('exit_code')!=0:r['status']='runtime_timeout' if r['run'].get('timed_out') else 'runtime_failed'
 elif r['number'] in v.VALIDATED:r['status']='validation_pass' if 'VALIDATION PASS' in text and 'errors=0' in text else 'validation_failed'
 elif r['number'] in v.API_PROBES:r['status']='api_probe_pass' if 'API PROBE PASS' in text and 'errors=0' in text else 'api_probe_failed'
 else:r['status']='executed_unverified'
 r['evidence_folder']=str(folder.relative_to(root));print(r['status'],r['source'],flush=True);return r
with ThreadPoolExecutor(max_workers=3) as pool:replaced=list(pool.map(retry,bad))
by={r['source']:r for r in rows};by.update({r['source']:r for r in replaced});rows=sorted(by.values(),key=lambda r:r['source'])
(root/'results/final-results.json').write_text(json.dumps(rows,indent=2)+'\n')
from collections import Counter
counts=dict(Counter(r['status'] for r in rows));(root/'final-complete.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2),flush=True)
