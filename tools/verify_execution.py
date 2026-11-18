#!/usr/bin/env python3
"""Verify immutable execution evidence, current sources and displayed records; never run CUDA."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from approved_display import approved_display_source, approved_display_output
from publish_execution import ROOT, FENCES, SUCCESS, ARCHITECTURES, category, normalize, sha, sources, strip_command, supplements


def verify(root, public, docs=None, summary_text=None):
    errors = []
    integrity = json.loads((public / 'integrity.json').read_text())
    dataset, label = integrity['dataset'], integrity['label']
    if public.name != dataset or label not in ARCHITECTURES:
        errors.append('Dataset identity/architecture label mismatch.')
    inventory = integrity['files']
    actual = {str(p.relative_to(public)) for p in public.rglob('*') if p.is_file()} - {'integrity.json'}
    if set(inventory) != actual:
        errors.append('Evidence files differ from immutable inventory.')
    for relative, expected in inventory.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts:
            errors.append(f'Unsafe evidence path: {relative}')
        elif not (public / path).is_file() or sha(public / path) != expected:
            errors.append(f'Changed/missing evidence: {relative}')
    current = sources(root)
    rows = json.loads((public / 'results.json').read_text())
    recorded = {row['source']: row['sha256'] for row in rows}
    from pending_revalidation import pending
    revisions = pending(root, current, recorded)
    expected_current = dict(recorded)
    for rel, revision in revisions.items():
        expected_current[rel] = revision['current_sha256']
    if len(rows) != 145 or len(recorded) != 145 or len(current) != 145:
        errors.append('Expected exactly 145 unique standalone source records.')
    if {rel: sha(path) for rel, path in current.items()} != expected_current:
        errors.append('Execution manifest differs from current standalone source hashes.')
    summary_text = summary_text if summary_text is not None else (root / 'docs' / f'validation-{label}.md').read_text()
    if hashlib.sha256(summary_text.encode()).hexdigest() != integrity['summary_sha256']:
        errors.append('Summary hash differs from published inventory.')
    for row in rows:
        rel = row['source']
        name, stem = Path(rel).name, Path(rel).stem
        folder = public / stem
        if sha(folder / name) != row['sha256']:
            errors.append(f'Downloaded source mismatch: {rel}')
        if row.get('status') not in SUCCESS or row.get('compile', {}).get('exit_code') != 0 or row.get('run', {}).get('exit_code') != 0:
            errors.append(f'Unsuccessful/unexecuted record: {rel}')
        if row.get('compile', {}).get('timed_out') or row.get('run', {}).get('timed_out'):
            errors.append(f'Timed-out record: {rel}')
        system = row['evidence_system']
        if system.get('arch') != ARCHITECTURES[label]:
            errors.append(f'Architecture mismatch: {rel}')
        if not re.match(r'^\d{4}-\d\d-\d\dT.*(?:\+00:00|Z)$', row.get('captured_utc', '')) or not row.get('timestamp_scope'):
            errors.append(f'Missing UTC/timestamp scope: {rel}')
        if row['timestamp_scope'].startswith('run-batch') and row['captured_utc'] != system.get('utc'):
            errors.append(f'Batch timestamp mismatch: {rel}')
        compile_log = (folder / 'compile.log').read_text()
        run_log = (folder / 'run.log').read_text()
        for command, log, stage in [(row['compile_command'], compile_log, 'compile'), (row['run_command'], run_log, 'run')]:
            if not log.startswith('$ ' + shlex.join(command) + '\n'):
                errors.append(f'Captured {stage} command/log mismatch: {rel}')
        raw_output = strip_command(run_log).rstrip('\n')
        output = approved_display_output(name, raw_output)
        if re.search(r'^(?:VALIDATION FAIL|API PROBE FAIL|FAIL:)', raw_output, re.M):
            errors.append(f'Captured output reports failure: {rel}')
        if row['status'] == 'validation_pass' and not re.search(r'(?:VALIDATION PASS|PASS:)', output):
            errors.append(f'Numerical status lacks a passing oracle marker: {rel}')
        if row['status'] == 'api_probe_pass' and not ('API PROBE PASS' in output or row.get('number') in {14, 15, 43} and ('PASS:' in output or 'cudaEventSynchronize(event) [blocking wait]: no error' in output)):
            errors.append(f'API probe status lacks a passing marker: {rel}')
        page = root / row['page']
        text = docs[page] if docs and page in docs else page.read_text()
        exact = [m for m in FENCES.finditer(text) if m[3].strip() == approved_display_source(name, current[rel].read_text()).strip()]
        if len(exact) != 1:
            errors.append(f'Chapter must contain exactly one exact source fence: {rel}')
        marker = re.compile(r'<!-- execution:start:' + re.escape(label + ':' + name) + r' -->(.*?)<!-- execution:end:' + re.escape(label + ':' + name) + r' -->', re.S)
        sections = marker.findall(text)
        if len(sections) != 1:
            errors.append(f'Expected one displayed execution block: {rel}')
            continue
        section = sections[0]
        if rel in revisions and ('GPU revalidation pending' not in section
                or revisions[rel]['current_sha256'] not in section):
            errors.append(f'Corrected source needs a visible pending notice and exact new hash: {rel}')
        if label == 'a100' and ('<details' in section or not any(title in section for title in ('**Sample output**', '**Actual sample output (excerpt)**'))):
            errors.append(f'Latest A100 sample output must be inline and visible: {rel}')
        run_started = row.get('run_started_utc') or row.get('run', {}).get('started_utc')
        if output != raw_output and '**Actual sample output (excerpt)**' not in section:
            errors.append(f'Filtered actual output must be labeled as an excerpt: {rel}')
        if run_started and f'Executed UTC: `{run_started}`' not in section:
            errors.append(f'Displayed actual run-start timestamp differs: {rel}')
        if row['sha256'] not in section or row['captured_utc'] not in section or dataset not in section or row['timestamp_scope'] not in section:
            errors.append(f'Displayed execution metadata mismatch: {rel}')
        commands = shlex.join(normalize(row['compile_command'], rel)) + '\n' + shlex.join(normalize(row['run_command'], rel))
        displayed_commands = [m[3].rstrip('\n') for m in FENCES.finditer(section) if m[2].strip() == 'bash']
        displayed_output = [m[3].rstrip('\n') for m in FENCES.finditer(section) if m[2].strip() == 'text']
        if displayed_commands != [commands]:
            errors.append(f'Displayed normalized commands differ: {rel}')
        if displayed_output != [output]:
            errors.append(f'Displayed captured output differs: {rel}')
    supplemental = '\n'.join(supplements(root, public, label))
    if supplemental not in summary_text:
        errors.append('Summary system/supplemental facts differ from captured evidence.')
    counts = collections.Counter(category(row) for row in rows)
    for label_name, count in counts.items():
        if f'| {label_name} | {count} |' not in summary_text:
            errors.append(f'Summary classification count mismatch: {label_name}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dataset', help='Verify one immutable dataset; default checks all datasets.')
    args = parser.parse_args()
    root = args.root.resolve()
    public = root / 'docs/validation'
    directories = [public / args.dataset] if args.dataset else sorted(p.parent for p in public.glob('*/integrity.json'))
    if not directories:
        print('FAIL: no published execution dataset exists.')
        return 1
    errors = []
    for directory in directories:
        try:
            errors.extend(f'{directory.name}: {message}' for message in verify(root, directory))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'{directory.name}: {error}')
    for error in errors:
        print('FAIL:', error)
    if not errors:
        print(f'PASS: {len(directories)} dataset(s); 145 archived source records, commands, UTC scopes, outputs and explicitly pinned pending corrections agree.')
        print('Scope: evidence consistency only; no CUDA execution occurs in this verifier.')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
