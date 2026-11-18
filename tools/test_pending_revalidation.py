"""Reject silent source drift in explicitly pending GPU corrections."""
import json
import tempfile
import unittest
from pathlib import Path
from pending_revalidation import pending
from publish_execution import sources

ROOT = Path(__file__).resolve().parents[1]

class PendingTests(unittest.TestCase):
    def test_pinned_current_and_historical_revisions(self):
        current = sources(ROOT)
        rows = json.loads((ROOT/'docs/validation/2026-11-17-a100/results.json').read_text())
        recorded = {p['source']: p['sha256'] for p in rows}
        revisions = pending(ROOT, current, recorded)
        self.assertEqual(len(revisions), 1)
        key = next(iter(revisions))
        with tempfile.TemporaryDirectory() as folder:
            bad = Path(folder)/'source.cpp'
            bad.write_bytes(current[key].read_bytes()+b'\nint unexpected_change;\n')
            changed = dict(current); changed[key] = bad
            with self.assertRaises(ValueError): pending(ROOT, changed, recorded)
        changed = dict(recorded); changed[key] = revisions[key]['current_sha256']
        with self.assertRaises(ValueError): pending(ROOT, current, changed)

if __name__ == '__main__': unittest.main()
