"""Verify the mixed-outcome eight-B200 record without concealing timeouts."""
from pathlib import Path
from datetime import datetime, timezone
import collections,hashlib,json,re,shlex
from publish_execution import sources
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/validation/2026-11-19-eight-b200'
TIMEOUTS={64,65,128}
PREFIX='/home/ubuntu/book-b200-20261119/'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def completed(record):return record.get('exit_code')==0 and not record.get('timed_out',False)
def folder(row):
    if row.get('evidence_folder'):name=row['evidence_folder']
    else:name=Path(row['compile_command'][row['compile_command'].index('-o')+1]).parent.as_posix().removeprefix(PREFIX)
    rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe B200 record path');return BASE/rel

def verify():
    inv=json.loads((BASE/'evidence-integrity.json').read_text())['files']
    actual={p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file()}-{'evidence-integrity.json'}
    require(set(inv)==actual,'B200 evidence inventory differs')
    for name,digest in inv.items():
        rel=Path(name);require(not rel.is_absolute() and '..' not in rel.parts and sha(BASE/rel)==digest,'B200 evidence changed: '+name)
    rows=json.loads((BASE/'results/final-results.json').read_text());current=sources(ROOT)
    require(len(rows)==145 and {r['source'] for r in rows}==set(current),'B200 source inventory differs')
    require(collections.Counter(r['status'] for r in rows)=={'validation_pass':31,'api_probe_pass':13,'executed_unverified':98,'runtime_timeout':3},'B200 outcomes differ')
    report=(ROOT/'docs/validation-b200.md').read_text()
    for row in rows:
        path=folder(row);source=current[row['source']]
        require(sha(source)==row['sha256'] and sha(path/source.name)==row['sha256'],'B200 source hash differs')
        require(completed(row['compile']),'B200 compilation did not complete')
        if source.suffix=='.cu':require('-arch=sm_100' in row['compile_command'],'Wrong B200 architecture')
        for stage in ('compile','run'):
            raw=(path/(stage+'.log')).read_text()
            require(raw.startswith('$ '+shlex.join(row[stage+'_command'])+'\n'),'B200 command/log mismatch')
            entry=row[stage];start,finish=(datetime.fromisoformat(entry[k]) for k in ('started_utc','finished_utc'))
            require(start.utcoffset()==timezone.utc.utcoffset(start) and finish.utcoffset()==timezone.utc.utcoffset(finish) and start<=finish and entry['started_utc'].startswith('2026-11-19T'),'B200 date mismatch')
        if row['number'] in TIMEOUTS:
            require(row['status']=='runtime_timeout' and row['run']['timed_out'] and row['run']['exit_code']!=0,'B200 timeout was silently promoted')
        else:require(completed(row['run']),'B200 completed status contradicts exit')
        output=(path/'run.log').read_text()
        if row['status']=='validation_pass':require(re.search(r'(?:VALIDATION PASS|^PASS:)',output,re.M),'B200 numerical result lacks passing check')
        require(row['run']['started_utc'] in report and source.name in report,'B200 report omits execution identity/date')
        if 'mpi' in row['dependencies']:require(row['processes']==(1 if row['number']==63 else 8),'Wrong B200 MPI participant count')
    harness=(BASE/'eight-run-fixed.log').read_text()
    pairs=set(re.findall(r'^PEER_COPY (\d)->(\d): PASS;.*errors=0$',harness,re.M))
    require(pairs=={(str(a),str(b)) for a in range(8) for b in range(8) if a!=b},'Incomplete B200 peer-pair checks')
    for label,count in [('ALL_REDUCE',32792),('ALL_GATHER',33088),('REDUCE_SCATTER',4136),('GROUPED_ALL_TO_ALL',16448),('HALO_GENERATION_1',8264),('HALO_GENERATION_2',8264),('HALO_GENERATION_3',8264)]:
        line=f'{label}: PASS; {count} values checked across 8 ranks; errors=0'
        require(line in harness and line in report,'Incomplete B200 collective/halo check')
    require('EIGHT_GPU_AUDIT: PASS; numerical_errors=0; cleanup=PASS' in harness,'B200 harness failed')
    extra=json.loads((BASE/'eight-fixed-results.json').read_text())
    require(len(extra)==2 and all(completed(r) for r in extra) and '-Xcompiler=-pthread' in extra[0]['command'],'B200 harness build/run mismatch')
    require(sha(ROOT/'tools/validate_eight_gpus.cu')==sha(BASE/'tools/validate_eight_gpus.cu'),'B200 harness source drift')
    mem=(BASE/'ipc-memcheck.log').read_text()
    require('ERROR SUMMARY: 0 errors' in mem and 'PASS: two-process IPC memory, 4096 nonuniform values checked.' in mem,'B200 IPC memory check failed')
    require((BASE/'setup-record/make-exit.txt').read_text().strip()=='0' and sha(BASE/'setup-record/Makefile.txt')==sha(ROOT/'docs/appendix/code/Makefile'),'B200 Makefile check mismatch')
    gpu=(BASE/'results/part-0/gpus.log').read_text();require(gpu.count('NVIDIA B200')==8 and '580.105.08' in gpu,'B200 GPU/driver identity differs')
    print('PASS: 145 B200 source/compile records, 142 completed executions, three explicit NVSHMEM timeouts, 56 peer pairs, four collectives, three halos and IPC memcheck.')
if __name__=='__main__':verify()
