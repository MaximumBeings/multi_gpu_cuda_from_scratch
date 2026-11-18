"""Check exact pending code revisions separately from immutable execution records."""
import hashlib
import json
import re


def pending(root, current, recorded):
    path = root / 'tools/pending_gpu_revalidation.json'
    if not path.exists():
        return {}
    result = {}
    for row in json.loads(path.read_text()):
        key = row['source']
        if key in result or row['status'] != 'pending_gpu_revalidation' or not row['reason'].strip():
            raise ValueError('Invalid/duplicate pending correction')
        before, after = row['archived_sha256'], row['current_sha256']
        if before == after or not all(re.fullmatch(r'[0-9a-f]{64}', value) for value in (before, after)):
            raise ValueError('Invalid pending source hashes: ' + key)
        if recorded.get(key) != before or hashlib.sha256(current[key].read_bytes()).hexdigest() != after:
            raise ValueError('Pending correction differs from pinned revisions: ' + key)
        result[key] = row
    return result
