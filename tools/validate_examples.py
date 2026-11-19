#!/usr/bin/env python3
"""Capture reproducible compile/runtime evidence; execution is opt-in, never assumed validation.

Examples:
  python3 tools/validate_examples.py --output artifacts/h100-YYYYMMDD
  python3 tools/validate_examples.py --execute --arch sm_90 --output artifacts/h100-YYYYMMDD
  python3 tools/validate_examples.py --execute --arch sm_100 --output artifacts/b200-YYYYMMDD

No remote execution, package installation or automatic source mutation occurs here.
"""
import argparse, datetime, hashlib, json, os, pathlib, re, shlex, shutil, signal, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[1]
# These examples intentionally exercise invalid handles or asynchronous failure.
# Compile them, but isolate/review their runtime oracles before enabling them.
REVIEW={}
API_PROBES={14,15,16,30,31,43,54,55,61,68,130,134,135}
LAYOUT_VALIDATED={11,12,20,21,23,24,26,27,28,33,34,37,40,46,49}
VALIDATED={17,18,58,63,64,65,66,67,71,72,95,126,127,128,129,131,133,136,139}

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(args, logfile, timeout, env):
    start=time.monotonic(); started=utc()
    with logfile.open('w') as f:
        f.write('$ '+shlex.join([str(a) for a in args])+'\n'); f.flush()
        try:
            p=subprocess.Popen(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        except OSError as e:
            f.write(str(e)+'\n'); return {'exit_code':None,'launch_error':str(e),'seconds':time.monotonic()-start,'started_utc':started,'finished_utc':utc()}
        timed=False
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            code=p.returncode
        return {'exit_code':code,'timed_out':timed,'seconds':time.monotonic()-start,'started_utc':started,'finished_utc':utc()}
def detect_cuda():
    p=shutil.which('nvcc')
    return pathlib.Path(os.environ.get('CUDA_HOME') or (pathlib.Path(p).resolve().parent.parent if p else '/usr/local/cuda'))
def cuda_paths(cuda):
    inc=pathlib.Path(os.environ.get('CUDA_INCDIR') or (str(cuda/'include') if (cuda/'include/cuda_runtime.h').exists() else '/usr/include'))
    lib=pathlib.Path(os.environ.get('CUDA_LIBDIR') or (str(cuda/'lib64') if (cuda/'lib64').is_dir() else '/usr/lib/x86_64-linux-gnu'))
    return inc,lib
def inventory():
    # Exclude nested raw evidence trees from the standalone book inventory.
    rows=[]
    for p in sorted(ROOT.glob('docs/*/code/*')):
        if p.suffix not in ('.cu','.cpp','.c'):continue
        text=p.read_text();headers=re.findall(r'^\s*#\s*include\s*[<"]([^>\"]+)',text,re.M)
        n=re.match(r'(\d+)_',p.name);number=int(n[1]) if n else None
        deps=[k for k,h in [('mpi','mpi.h'),('nccl','nccl.h'),('nvshmem','nvshmem.h'),('cublas','cublasXt.h')] if h in headers]
        cuda=p.suffix=='.cu' or any(h.startswith(('cuda','cublas','nvshmem')) for h in headers)
        rows.append({'source':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'number':number,'dependencies':deps,'pthread':'pthread.h' in headers,'cuda':cuda,'has_main':bool(re.search(r'\bint\s+main\s*\(',text)),
          'minimum_gpus':2 if cuda else 0,'processes':1 if number==63 else 2 if 'mpi' in deps else 1,
          'runtime_review':REVIEW.get(number),'validation_oracle':'Checked event API lifecycle (no numerical kernel)' if number in {14,15,43} else 'PASS: full reference values' if number in LAYOUT_VALIDATED else 'API PROBE PASS, errors=0 (API scope only)' if number in API_PROBES else 'PASS: two-process IPC, 4096 nonuniform values checked' if number in {17,18} else 'VALIDATION PASS, errors=0' if number in VALIDATED else 'exit status only; numerical correctness unverified'})
    return rows
def compile_command(row,binary,arch,cuda,nvhome,nvlib):
    source=ROOT/row['source'];deps=row['dependencies'];suffix=source.suffix
    cuda_inc,cuda_lib=cuda_paths(cuda)
    compiler='nvcc' if suffix=='.cu' else ('mpicc' if suffix=='.c' else 'mpicxx') if 'mpi' in deps else ('gcc' if suffix=='.c' else 'g++')
    args=[compiler,'-std=c11' if suffix=='.c' else '-std=c++17','-O2']
    if suffix=='.cu':
        args+=['-arch='+arch]
        if 'mpi' in deps:args+=['-ccbin','mpicxx']
    elif row['cuda']:args+=['-I'+str(cuda_inc),'-L'+str(cuda_lib),'-Wl,-rpath,'+str(cuda_lib)]
    if 'nvshmem' in deps:
        args+=['-rdc=true','-I'+str(nvhome/'include'),'-L'+str(nvlib),'-Xlinker=-rpath='+str(nvlib)]
    if row.get('pthread'):args+=['-Xcompiler=-pthread'] if suffix=='.cu' else ['-pthread']
    if row['number']==130:args+=['-rdc=true']
    args+=[str(source),'-o',str(binary)]
    if 'nvshmem' in deps:args+=['-lnvshmem_host','-lnvshmem_device','-lcuda']
    if 'nccl' in deps:args+=['-lnccl']
    if 'cublas' in deps:args+=['-lcublas']
    if suffix!='.cu' and row['cuda']:args+=['-lcudart']
    if row['number']==130:args+=['-lcudadevrt']
    return args

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute',action='store_true',help='Run eligible programs, including GPU workloads; never enabled by default')
    ap.add_argument('--output',type=pathlib.Path,required=True)
    ap.add_argument('--arch',default='sm_90')
    ap.add_argument('--mpi-ranks',type=int,help='MPI ranks for multi-rank examples; single-PE example 63 stays at one')
    ap.add_argument('--match',action='append',default=[],help='Only paths matching any supplied substring; repeatable')
    ap.add_argument('--exclude',action='append',default=[],help='Exclude paths containing a supplied substring; repeatable')
    ap.add_argument('--compile-timeout',type=int,default=180)
    ap.add_argument('--run-timeout',type=int,default=45)
    ap.add_argument('--allow-reviewed',action='store_true',help='Run separately reviewed adversarial examples too')
    a=ap.parse_args();out=a.output.resolve()
    if a.mpi_ranks is not None and a.mpi_ranks < 1:ap.error('--mpi-ranks must be positive')
    if out.exists() and any(out.iterdir()):ap.error('Output directory must be new or empty; evidence is never overwritten')
    out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['OMPI_ALLOW_RUN_AS_ROOT']='1';env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM']='1'
    cuda=detect_cuda();cuda_inc,cuda_lib=cuda_paths(cuda);nvhome=pathlib.Path(env.get('NVSHMEM_HOME','/opt/nvshmem'))
    nvlib=pathlib.Path(env.get('NVSHMEM_LIB',str(nvhome/'lib')))
    env['LD_LIBRARY_PATH']=':'.join([str(nvlib),str(cuda_lib),env.get('LD_LIBRARY_PATH','')])
    system={ 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arch':a.arch,'execute':a.execute,'mpi_ranks':a.mpi_ranks,'CUDA_HOME':str(cuda),'CUDA_INCDIR':str(cuda_inc),'CUDA_LIBDIR':str(cuda_lib),
             'NVSHMEM_HOME':str(nvhome),'NVSHMEM_LIB':str(nvlib),'host':os.uname().nodename }
    probes={'uname':['uname','-a'],'git':['git','rev-parse','HEAD'],'git-status':['git','status','--short'],
      'nvcc':['nvcc','--version'],'mpi':['mpirun','--version'],'gpus':['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],
      'topology':['nvidia-smi','topo','-m'],'libraries':['ldconfig','-p']}
    for name,cmd in probes.items():system[name]=run(cmd,out/(name+'.log'),15,env)
    gpu_log=(out/'gpus.log').read_text();gpu_count=sum(bool(re.match(r'^(NVIDIA|Tesla|GeForce)',line)) for line in gpu_log.splitlines())
    system['visible_gpu_count']=gpu_count
    (out/'system.json').write_text(json.dumps(system,indent=2)+'\n')
    rows=inventory();(out/'inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
    results=[]
    for row in rows:
        if a.match and not any(x in row['source'] for x in a.match):continue
        if any(x in row['source'] for x in a.exclude):continue
        if a.mpi_ranks is not None and 'mpi' in row['dependencies'] and row['number'] != 63:
            row=dict(row,processes=a.mpi_ranks)
            if row['cuda']:row['minimum_gpus']=a.mpi_ranks
        result=dict(row);result['timestamp_utc']=utc();folder=out/pathlib.Path(row['source']).stem;folder.mkdir()
        shutil.copy2(ROOT/row['source'],folder/pathlib.Path(row['source']).name)
        binary=folder/'example'
        if not row['has_main']:
            result.update(status='fragment',reason='No standalone int main');results.append(result);continue
        cmd=compile_command(row,binary,a.arch,cuda,nvhome,nvlib);result['compile_command']=cmd
        missing=[]
        if not shutil.which(cmd[0]):missing.append(cmd[0])
        if 'mpi' in row['dependencies'] and not shutil.which('mpicxx' if cmd[0]=='nvcc' else cmd[0]):missing.append('MPI compiler')
        if 'nvshmem' in row['dependencies'] and not (nvhome/'include/nvshmem.h').exists():missing.append('NVSHMEM headers')
        if row['cuda'] and (ROOT/row['source']).suffix!='.cu' and not (cuda_inc/'cuda_runtime.h').exists():missing.append('CUDA headers for host compiler')
        if missing:result.update(status='dependency_blocked',reason=', '.join(missing))
        else:
            cr=run(cmd,folder/'compile.log',a.compile_timeout,env);result['compile']=cr;result['compile_started_utc']=cr['started_utc']
            if cr.get('exit_code')!=0:
                log=(folder/'compile.log').read_text()
                missing_dependency=re.search(r'(fatal error:.*(?:No such file|file not found)|cannot find -l|error while loading shared libraries)',log,re.I)
                result['status']='dependency_blocked' if missing_dependency else 'compile_timeout' if cr.get('timed_out') else 'compile_failed'
                if missing_dependency:result['reason']=missing_dependency[0]
            elif not a.execute:result['status']='compiled_not_executed'
            elif row['runtime_review'] and not a.allow_reviewed:result.update(status='runtime_review_required',reason=row['runtime_review'])
            elif row['minimum_gpus']>gpu_count:result.update(status='hardware_blocked',reason=f"Needs {row['minimum_gpus']} visible GPUs; found {gpu_count}")
            elif 'mpi' in row['dependencies'] and not shutil.which('mpirun'):result.update(status='dependency_blocked',reason='mpirun missing')
            else:
                cmd=([shutil.which('mpirun'),'--oversubscribe','-np',str(row['processes'])] if 'mpi' in row['dependencies'] else [])+[str(binary)]
                if row['number']==57:cmd+=['1000000','sendrecv']
                if row['number']==138:cmd+=['fixed']
                result['run_command']=cmd;rr=run(cmd,folder/'run.log',a.run_timeout,env);result['run']=rr;result['run_started_utc']=rr['started_utc']
                text=(folder/'run.log').read_text()
                if rr.get('timed_out'):result['status']='runtime_timeout'
                elif rr.get('exit_code')!=0:result['status']='runtime_failed'
                elif row['number'] in {17,18}:
                    result['status']='validation_pass' if 'PASS: two-process IPC' in text and '4096 nonuniform values checked.' in text else 'validation_failed'
                elif row['number'] in LAYOUT_VALIDATED:
                    result['status']='validation_pass' if re.search(r'^PASS:',text,re.M) else 'validation_failed'
                elif row['number']==14:
                    result['status']='api_probe_pass' if 'cudaEventSynchronize(event)' in text and 'no error' in text else 'api_probe_failed'
                elif row['number'] in {15,43}:
                    result['status']='api_probe_pass' if re.search(r'^PASS:',text,re.M) else 'api_probe_failed'
                elif row['number'] in API_PROBES:
                    result['status']='api_probe_pass' if 'API PROBE PASS' in text and re.search(r'errors=0\b',text) and 'API PROBE FAIL' not in text else 'api_probe_failed'
                elif row['number'] in VALIDATED:
                    result['status']='validation_pass' if 'VALIDATION PASS' in text and re.search(r'errors=0\b',text) and 'VALIDATION FAIL' not in text else 'validation_failed'
                else:result['status']='executed_unverified' # exit zero is not numerical proof
        result['finished_utc']=utc()
        results.append(result)
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(result['status'],row['source'],flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})}
    (out/'summary.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))
    return int(any(r['status'] in {'compile_failed','compile_timeout','runtime_failed','runtime_timeout','validation_failed','api_probe_failed'} for r in results))
if __name__=='__main__':sys.exit(main())
