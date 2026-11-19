"""Check formatter discovery without depending on the caller's shell PATH."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'code_formatting', Path(__file__).resolve().parents[1] / 'hooks/code_formatting.py')
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class FormatterLookupTests(unittest.TestCase):
    def tearDown(self):
        hook.executable.cache_clear()

    def test_wheel_binary_without_path_entry_or_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / 'clang_format'
            binary = package / 'data/bin/clang-format'
            binary.parent.mkdir(parents=True)
            binary.touch()
            with patch.object(hook, 'find_spec', return_value=SimpleNamespace(origin=str(package / '__init__.py'))), \
                 patch.object(hook.shutil, 'which', return_value='/wrong/version/clang-format'), \
                 patch.object(hook.sys, 'executable', str(Path(directory) / 'python')), \
                 patch.object(hook.subprocess, 'run', return_value=SimpleNamespace(stdout='clang-format version 19.1.7')) as run:
                self.assertEqual(hook.executable(), str(binary))
                self.assertEqual(run.call_args.args[0], [str(binary), '--version'])

    def test_missing_package_and_binary_explains_interpreter(self):
        with patch.object(hook, 'find_spec', return_value=None), \
             patch.object(hook.shutil, 'which', return_value=None), \
             patch.object(hook.sys, 'executable', '/missing/environment/python'):
            with self.assertRaisesRegex(RuntimeError, '/missing/environment/python -m pip'):
                hook.executable()

    def test_external_formatter_still_requires_pinned_version(self):
        with patch.object(hook, 'find_spec', return_value=None), \
             patch.object(hook.shutil, 'which', return_value='/external/clang-format'), \
             patch.object(hook.subprocess, 'run', return_value=SimpleNamespace(stdout='clang-format version 20.0.0')):
            with self.assertRaisesRegex(RuntimeError, 'requires clang-format 19.1.7'):
                hook.executable()


if __name__ == '__main__':
    unittest.main()
