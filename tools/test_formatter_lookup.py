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

    def test_wheel_binary_precedes_conflicting_path_formatter(self):
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

    def test_real_wheel_runs_without_scripts_on_path(self):
        # Exercise the installed pinned wheel, not a mocked subprocess.
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict('os.environ', {'PATH': directory}), \
             patch.object(hook.sys, 'executable', str(Path(directory) / 'python')):
            self.assertTrue(Path(hook.executable()).is_file())
            shown = hook.format_source('int main(){return 0;}\n', 'cpp')
            self.assertEqual(hook.lexical_signature(shown),
                             hook.lexical_signature('int main(){return 0;}\n'))

    def test_successful_external_path_fallback(self):
        with patch.object(hook, 'find_spec', return_value=None), \
             patch.object(hook.shutil, 'which', return_value='/external/clang-format'), \
             patch.object(hook.subprocess, 'run', return_value=SimpleNamespace(stdout='clang-format version 19.1.7')):
            self.assertEqual(hook.executable(), '/external/clang-format')

    def test_successful_interpreter_sibling_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'clang-format'
            binary.touch()
            with patch.object(hook, 'find_spec', return_value=None), \
                 patch.object(hook.shutil, 'which', return_value=None), \
                 patch.object(hook.sys, 'executable', str(Path(directory) / 'python')), \
                 patch.object(hook.subprocess, 'run', return_value=SimpleNamespace(stdout='clang-format version 19.1.7')):
                self.assertEqual(hook.executable(), str(binary))

    def test_missing_package_and_binary_explains_interpreter(self):
        with patch.object(hook, 'find_spec', return_value=None), \
             patch.object(hook.shutil, 'which', return_value=None), \
             patch.object(hook.sys, 'executable', '/missing/environment/python'):
            with self.assertRaisesRegex(RuntimeError, '/missing/environment/python -m pip'):
                hook.executable()

    def test_external_formatter_still_requires_exact_pinned_version(self):
        for version in ['20.0.0', '19.1.70', '19.1.7git', '19.1.7rc1', '']:
            with self.subTest(version=version), \
                 patch.object(hook, 'find_spec', return_value=None), \
                 patch.object(hook.shutil, 'which', return_value='/external/clang-format'), \
                 patch.object(hook.subprocess, 'run', return_value=SimpleNamespace(stdout='clang-format version ' + version)):
                hook.executable.cache_clear()
                with self.assertRaisesRegex(RuntimeError, 'requires clang-format 19.1.7'):
                    hook.executable()


if __name__ == '__main__':
    unittest.main()
