#!/usr/bin/env python3
"""Check the November 18 rerun's additional claims against raw evidence."""
from pathlib import Path
import json,hashlib,collections
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    public=ROOT/'docs/validation/2026-11-18-h100'
    rows=json.loads((public/'results.json').read_text())
    assert len(rows)==145
    assert collections.Counter(r['status'] for r in rows)=={'validation_pass':34,'api_probe_pass':13,'executed_unverified':98}
    staged=next(r for r in rows if r['number']==58)
    assert staged['status']=='validation_pass' and staged['evidence_system']['visible_gpu_count']==2
    root=public/'systems/run-1/metadata'
    mem=json.loads((root/'correction-memcheck/memcheck.json').read_text())
    log=root/'correction-memcheck/memcheck.log'
    assert mem['exit_code']==0 and mem['zero_error_summaries']==2 and sha(log)==mem['log_sha256']
    assert log.read_text().count('ERROR SUMMARY: 0 errors')==2
    assert sha(root/'correction-memcheck/58_mpi_cuda_aware_query_and_fallback.cpp.txt')==staged['sha256']
    make=root/'makefile-check';commands=json.loads((make/'manifest.json').read_text())
    assert len(commands)==6 and all(r['exit_code']==0 and r['started_utc'].startswith('2026-11-18T') and sha(make/r['log'])==r['log_sha256'] for r in commands)
    assert sha(make/'Makefile.txt')==sha(ROOT/'docs/appendix/code/Makefile')
    for p in make.glob('hello_*.txt'):
        assert sha(p)==sha(ROOT/'docs/appendix/code'/p.name.removesuffix('.txt'))
    error=next(r for r in rows if r['number']==59)
    assert error['run_id']=='run-2' and error['run']['exit_code']==0
    assert '4.1.6' in (public/'systems/run-2/mpi.log').read_text()
    assert 'MPI_Send RETURNED rc=6 (MPI_ERR_RANK: invalid rank)' in (public/'59_mpi_errhandler_return/run.log').read_text()
    assert 'Segmentation fault' in (root/'mpi-parameter-check/vendor-failure.log').read_text()
    assert 'mpi_param_check:value:false' in (root/'mpi-parameter-check/vendor-ompi-info.log').read_text()
    print('PASS: 145 rerun classifications, staged-MPI numerical/memcheck results, five Makefile recipes and preserved vendor-MPI failure/checked retry agree.')
if __name__=='__main__':verify()
