#!/usr/bin/env python3
"""CPU-only regression checks for the narrowly approved documentation views."""
import re
import unittest
from pathlib import Path
from approved_display import DISPLAY_EDITS, approved_display_source, approved_display_output

ROOT = Path(__file__).resolve().parents[1]


def tokens(text):
    """C/C++ lexical tokens; comment contents and whitespace do not affect code."""
    pattern = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_][A-Za-z_0-9]*|[^\s]'
    return [value for value in re.findall(pattern, text, re.S) if not value.startswith(('//', '/*'))]


def without_printf(text):
    values = tokens(text)
    result, index = [], 0
    while index < len(values):
        if values[index:index + 2] == ['printf', '(']:
            depth, index = 1, index + 2
            while depth:
                depth += (values[index] == '(') - (values[index] == ')')
                index += 1
            if values[index] != ';':
                raise AssertionError('Expected complete printf statement')
            index += 1
        else:
            result.append(values[index])
            index += 1
    return result


class ApprovedDisplayTests(unittest.TestCase):
    def test_exact_registry_and_operational_identity(self):
        for name, edits in DISPLAY_EDITS.items():
            with self.subTest(source=name):
                paths = list(ROOT.glob('docs/**/code/' + name))
                self.assertEqual(len(paths), 1)
                raw = paths[0].read_text()
                self.assertTrue(all(raw.count(old) == 1 for old, _ in edits))
                view = approved_display_source(name, raw)
                self.assertEqual(raw.count('\n'), view.count('\n'))
                self.assertEqual(without_printf(raw), without_printf(view))
                for old, new in edits:
                    if old.lstrip().startswith('printf('):
                        # Only text changes: argument tokens and format placeholders stay fixed.
                        self.assertEqual([t for t in tokens(old) if not t.startswith('"')],
                                         [t for t in tokens(new) if not t.startswith('"')])
                        self.assertEqual(re.findall(r'%(?:[-+ #0-9.*lhztj]*[A-Za-z%])', old),
                                         re.findall(r'%(?:[-+ #0-9.*lhztj]*[A-Za-z%])', new))
                    else:
                        self.assertTrue(all(not line.strip() or line.lstrip().startswith('//')
                                            for line in old.splitlines() + new.splitlines()))
                with self.assertRaises(ValueError):
                    approved_display_source(name, raw.replace(edits[0][0], '', 1))

    def test_unregistered_sources_are_byte_identical(self):
        raw = '// untouched\nint main(){return 7;}\n'
        self.assertEqual(approved_display_source('not_registered.cpp', raw), raw)

    def test_output_views_select_actual_lines_only(self):
        for public in ROOT.glob('docs/validation/*'):
            for name in DISPLAY_EDITS:
                log = public / Path(name).stem / 'run.log'
                if not log.exists():
                    continue
                from publish_execution import strip_command
                raw = strip_command(log.read_text())
                view = approved_display_output(name, raw)
                iterator = iter(raw.splitlines())
                for line in view.splitlines():
                    self.assertTrue(any(candidate == line for candidate in iterator), name)

    def test_verifier_rejects_unapproved_algorithm_change(self):
        from verify_execution import verify
        public = ROOT / 'docs/validation/2026-11-17-h100'
        page = next(ROOT.glob('docs/part0/03-*.md'))
        text = page.read_text()
        changed = text.replace('cudaGetDeviceCount(&deviceCount);', 'cudaGetDeviceCount(nullptr);', 1)
        self.assertNotEqual(text, changed)
        errors = verify(ROOT, public, {page: changed})
        self.assertTrue(any('exact source fence' in error for error in errors), errors)


if __name__ == '__main__':
    unittest.main()
