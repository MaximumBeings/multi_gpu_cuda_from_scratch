"""Check exact pending code revisions separately from immutable execution records."""
import hashlib
import json
import re
from pathlib import Path


def pending(root, current, recorded):
    path = root / 'tools/gpu_revalidation.json'
    if not path.exists():
        return {}
    result = {}
    for row in json.loads(path.read_text()):
        key = row['source']
        if key in result or row['status'] not in {'pending_gpu_revalidation','verified_gpu_revalidation'} or not row['reason'].strip():
            raise ValueError('Invalid/duplicate pending correction')
        before, after = row['archived_sha256'], row['current_sha256']
        if before == after or not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (before, after)):
            raise ValueError('Invalid pending source hashes: ' + key)
        if hashlib.sha256(current[key].read_bytes()).hexdigest() != after:
            raise ValueError('Current correction differs from pinned hash: ' + key)
        if recorded.get(key) == after:
            continue  # A fresh dataset records the corrected revision directly.
        if recorded.get(key) != before:
            raise ValueError('Pending correction differs from pinned revisions: ' + key)
        if row['status'] == 'verified_gpu_revalidation':
            if key == 'docs/part1/code/17_ipc_mem_handle.cu':
                from verify_ipc_correction import verify as verify_ipc_correction
                checked = verify_ipc_correction(root)
                if row['dataset'] != '2026-11-19-h100-ipc' or checked['source_sha256'] != after:
                    raise ValueError('IPC correction differs from its successful GPU record')
                result[key] = row
                continue
            public = root/'docs/validation'/row['dataset']
            records = json.loads((public/'results.json').read_text())
            matches = [r for r in records if r['source'] == key]
            if len(matches) != 1: raise ValueError('Missing corrected execution record')
            record = matches[0]
            folder = public/Path(key).stem
            if (record['sha256'] != after or record['status'] != 'validation_pass'
                    or record['compile']['exit_code'] != 0 or record['run']['exit_code'] != 0
                    or hashlib.sha256((folder/Path(key).name).read_bytes()).hexdigest() != after
                    or 'VALIDATION PASS: host-staged MPI, 2 ranks, 4096 values/rank, errors=0' not in (folder/'run.log').read_text()):
                raise ValueError('Correction lacks exact successful GPU execution')
            inventory = json.loads((public/'integrity.json').read_text())['files']
            for relative, digest in inventory.items():
                item = Path(relative)
                if item.is_absolute() or '..' in item.parts or hashlib.sha256((public/item).read_bytes()).hexdigest() != digest:
                    raise ValueError('Corrected execution evidence changed')
        result[key] = row
    return result
