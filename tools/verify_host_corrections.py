#!/usr/bin/env python3
"""Check exact CPU-only correction records without altering historical GPU evidence."""
import hashlib
import json
import shlex
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = '2026-11-18-host-corrections'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT, current=None, recorded=None):
    public = root/'docs/validation'/DATASET
    integrity = json.loads((public/'integrity.json').read_text())
    actual = {p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()} - {'integrity.json'}
    if actual != set(integrity['files']):
        raise ValueError('Host correction evidence inventory differs')
    for name, digest in integrity['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(public/rel) != digest:
            raise ValueError('Host correction evidence changed: '+name)
    summary = root/'docs/host-model-corrections.md'
    if sha(summary) != integrity['summary_sha256']:
        raise ValueError('Host correction summary differs')
    registry = json.loads((root/'tools/host_model_corrections.json').read_text())
    rows = json.loads((public/'results.json').read_text())
    if len(rows) != 3 or {r['number'] for r in rows} != {98,110,111}:
        raise ValueError('Expected exactly three host model corrections')
    if [{k: r[k] for k in registry[0]} for r in rows] != registry:
        raise ValueError('Host correction registry differs')
    system = json.loads((public/'system.json').read_text())
    if not all(value in summary.read_text() for value in (system['os'], system['machine'], system['compiler'].splitlines()[0])):
        raise ValueError('Host system differs from displayed record')
    result = {}
    for row in rows:
        rel = row['source']; source = root/rel
        folder = public/Path(rel).stem
        if sha(source) != row['current_sha256'] or sha(folder/source.name) != row['current_sha256']:
            raise ValueError('Host model source drift: '+rel)
        if current is not None and sha(current[rel]) != row['current_sha256']:
            raise ValueError('Host model current hash differs: '+rel)
        if recorded is not None and recorded.get(rel) != row['archived_sha256']:
            raise ValueError('Host model archived hash differs: '+rel)
        previous = None
        for stage in ('compile','run'):
            entry=row[stage];log=folder/(stage+'.log')
            if entry['exit_code'] != 0 or sha(log) != entry['log_sha256'] or not entry['started_utc'].startswith('2026-11-18T'):
                raise ValueError('Host model execution failed or changed: '+rel)
            if not log.read_text().startswith('$ '+shlex.join(row[stage+'_command'])+'\n'):
                raise ValueError('Host model command differs from captured log: '+rel)
            start, finish = (datetime.fromisoformat(entry[k]) for k in ('started_utc','finished_utc'))
            if start.tzinfo is None or finish < start or (previous is not None and start < previous):
                raise ValueError('Host model timestamp ordering differs: '+rel)
            previous = finish
        output=(folder/'run.log').read_text().split('\n',1)[1].rstrip()
        if 'MODEL CHECK PASS:' not in output or output not in summary.read_text():
            raise ValueError('Host model output missing: '+rel)
        page=(root/row['page']).read_text()
        if 'Corrected host model — November 18, 2026' not in page or f'../host-model-corrections.md#{row["number"]}' not in page:
            raise ValueError('Host model correction not disclosed beside historical output: '+rel)
        result[rel]=row['current_sha256']
    from verify_content_corrections import verify as verify_content_corrections
    result.update(verify_content_corrections(root, current, recorded))
    return result

if __name__=='__main__':
    verify()
    print('PASS: six exact corrected host models, immutable local compile/run logs, dates and visible corrections.')
