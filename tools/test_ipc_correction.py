"""Reject source drift or misleading role/sanitizer claims in the IPC rerun."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from verify_ipc_correction import ROOT, DATASET, SOURCE, verify


class IPCCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.public = self.root / 'docs/validation' / DATASET
        shutil.copytree(ROOT / 'docs/validation' / DATASET, self.public)
        for name in [SOURCE, 'docs/validation-h100-ipc.md',
                     'docs/part1/07-cuda-inter-process-communication.md']:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def update_inventory(self):
        path = self.public / 'integrity.json'
        record = json.loads(path.read_text())
        record['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in self.public.iterdir() if p.is_file() and p.name != 'integrity.json'}
        path.write_text(json.dumps(record))

    def test_complete_record(self):
        self.assertEqual(verify(self.root)['normal_runs'], 20)

    def test_changed_current_source_rejected(self):
        path = self.root / SOURCE
        path.write_text(path.read_text().replace('    CUDA(cudaDeviceSynchronize());\n', ''))
        with self.assertRaisesRegex(ValueError, 'Current IPC source differs'):
            verify(self.root)

    def test_duplicate_role_mapping_rejected_even_with_consistent_hashes(self):
        path = self.public / 'commands.json'
        commands = json.loads(path.read_text())
        forward = next(row for row in commands if row['name'] == 'forward-01')
        reverse = next(row for row in commands if row['name'] == 'reverse-01')
        reverse['environment'] = forward['environment'].copy()
        path.write_text(json.dumps(commands))
        self.update_inventory()
        with self.assertRaisesRegex(ValueError, 'Incorrect device-role mapping'):
            verify(self.root)

    def test_sanitizer_error_rejected_even_with_consistent_hashes(self):
        log = self.public / 'reverse-memcheck.log'
        log.write_text(log.read_text().replace('ERROR SUMMARY: 0 errors', 'ERROR SUMMARY: 1 errors'))
        path = self.public / 'commands.json'
        commands = json.loads(path.read_text())
        next(row for row in commands if row['name'] == 'reverse-memcheck')['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest()
        path.write_text(json.dumps(commands))
        self.update_inventory()
        with self.assertRaisesRegex(ValueError, 'IPC memcheck errors'):
            verify(self.root)


if __name__ == '__main__':
    unittest.main()
