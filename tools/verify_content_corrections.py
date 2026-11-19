"""Verify new CPU content corrections while preserving all earlier GPU records."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
DATASET = '2026-11-19-content-review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT, current=None, recorded=None):
    public = root / 'docs/validation' / DATASET
    inventory = json.loads((public / 'integrity.json').read_text())
    actual = {p.relative_to(public).as_posix() for p in public.rglob('*') if p.is_file()} - {'integrity.json'}
    if actual != set(inventory['files']) or inventory['dataset'] != DATASET:
        raise ValueError('Content correction evidence inventory differs')
    for name, expected in inventory['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(public / rel) != expected:
            raise ValueError('Content correction evidence changed: ' + name)
    summary = root / 'docs/content-review-2026-11-19.md'
    if sha(summary) != inventory['summary_sha256']:
        raise ValueError('Content correction summary changed')
    text = summary.read_text()
    rows = json.loads((public / 'results.json').read_text())
    registry = json.loads((root / 'tools/content_review_corrections.json').read_text())
    if len(rows) != 3 or {r['number'] for r in rows} != {96, 113, 137}:
        raise ValueError('Expected exactly the three corrected CPU models')
    if [{key: row[key] for key in registry[0]} for row in rows] != registry:
        raise ValueError('Content correction registry differs')
    system = json.loads((public / 'system.json').read_text())
    if not all(value in text for value in (system['os'], system['machine'], system['compiler'].splitlines()[0])):
        raise ValueError('Displayed CPU system differs')
    result = {}
    for row in rows:
        rel = row['source']; source = root / rel
        folder = public / source.stem
        if sha(source) != row['current_sha256'] or sha(folder / source.name) != row['current_sha256']:
            raise ValueError('Current CPU source drift: ' + rel)
        if current is not None and sha(current[rel]) != row['current_sha256']:
            raise ValueError('Incorrect current revision: ' + rel)
        if recorded is not None and recorded[rel] != row['archived_sha256']:
            raise ValueError('Incorrect historical revision: ' + rel)
        previous = None
        for stage in ('compile', 'run'):
            entry = row[stage]; log = folder / (stage + '.log')
            if entry['exit_code'] != 0 or sha(log) != entry['log_sha256']:
                raise ValueError('Failed or changed CPU execution: ' + rel)
            if not log.read_text().startswith('$ ' + shlex.join(row[stage + '_command']) + '\n'):
                raise ValueError('Captured command differs: ' + rel)
            start, finish = (datetime.fromisoformat(entry[k]) for k in ('started_utc', 'finished_utc'))
            if (start.utcoffset() != timezone.utc.utcoffset(start) or finish.utcoffset() != timezone.utc.utcoffset(finish)
                    or not entry['started_utc'].startswith('2026-11-19T') or finish < start
                    or previous is not None and start < previous):
                raise ValueError('Invalid CPU execution times: ' + rel)
            previous = finish
        output = (folder / 'run.log').read_text().split('\n', 1)[1].rstrip()
        page = (root / row['page']).read_text()
        if ('MODEL CHECK PASS:' not in output or output not in text or output not in page
                or row['current_sha256'] not in text or row['current_sha256'] not in page
                or row['run']['started_utc'] not in text
                or '../content-review-2026-11-19.md#' + str(row['number']) not in page):
            raise ValueError('Current corrected result missing from book: ' + rel)
        result[rel] = row['current_sha256']
    return result


if __name__ == '__main__':
    verify()
    print('PASS: three exact CPU content corrections, dated commands, system, logs and displayed results.')
