from pathlib import Path
import subprocess,json,datetime,hashlib,os,shlex,signal
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
rows=[]
def run(name,args,env=None,timeout=120):
    selected=dict(os.environ);selected.update(env or {})
    started=now();log=root/(name+'.log')
    with log.open('w') as f:
        f.write('$ '+shlex.join(args)+'\n');f.flush()
        p=subprocess.Popen(args,cwd=root,env=selected,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        try:code=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(p.pid,signal.SIGKILL);p.wait();code=p.returncode
    row=dict(name=name,command=args,environment=env or {},started_utc=started,finished_utc=now(),exit_code=code,timed_out=timed_out,log=log.name,log_sha256=sha(log))
    rows.append(row);(root/'commands.json').write_text(json.dumps(rows,indent=2)+'\n')
    if code!=0 or timed_out:raise RuntimeError(name+' failed: '+log.read_text())
    return log.read_text().split('\n',1)[1]
for name,args in [('gpu',['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv']),('topology',['nvidia-smi','topo','-m']),('nvcc',['nvcc','--version']),('gcc',['gcc','--version']),('os',['cat','/etc/os-release']),('kernel',['uname','-r']),('sanitizer-version',['compute-sanitizer','--version'])]:run(name,args)
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();assert len(gpus)==2
run('compile',['nvcc','-std=c++17','-O2','-arch=sm_90','17_ipc_mem_handle.cu','-o','example'])
for direction,devices in [('forward',gpus),('reverse',gpus[::-1])]:
    env={'CUDA_VISIBLE_DEVICES':','.join(devices),'CUDA_DEVICE_ORDER':'PCI_BUS_ID'}
    for i in range(1,11):
        output=run(f'{direction}-{i:02}', ['./example'],env)
        assert 'Consumer: 4096 / 4096 exact payload values correct.' in output
        assert 'PASS: two-process IPC memory, 4096 nonuniform values checked.' in output
    output=run(direction+'-memcheck',['compute-sanitizer','--tool','memcheck','--target-processes','all','--error-exitcode','3','./example'],env)
    assert 'PASS: two-process IPC memory, 4096 nonuniform values checked.' in output
    assert 'Consumer: 4096 / 4096 exact payload values correct.' in output
    assert 'ERROR SUMMARY: 0 errors' in output
    assert not __import__('re').search(r'ERROR SUMMARY: [1-9]',output)
source=root/'17_ipc_mem_handle.cu'
(root/'results.json').write_text(json.dumps(dict(status='PASS',source='docs/part1/code/'+source.name,source_sha256=sha(source),arch='sm_90',gpu_uuids=gpus,normal_runs=20,memcheck_runs=2,payload_values_per_run=4096,scope='Two local H100 GPUs; producer/consumer directions reversed using explicit UUIDs. No multi-node or performance claim.',finished_utc=now()),indent=2)+'\n')
files={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name not in {'example','integrity.json'}}
(root/'integrity.json').write_text(json.dumps(dict(files=files),indent=2)+'\n')
print('PASS: 20 normal two-process IPC checks and two all-process memchecks; both GPU directions; exact source '+sha(source),flush=True)
