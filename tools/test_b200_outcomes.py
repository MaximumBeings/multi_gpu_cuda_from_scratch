"""A passing payload line cannot override an incomplete process execution."""
import unittest
from pathlib import Path
from verify_b200 import completed
class B200OutcomeTests(unittest.TestCase):
    def test_nested_capture_sources_are_not_extra_book_programs(self):
        from verify_b200 import ROOT
        from publish_execution import sources
        from validate_examples import inventory
        self.assertEqual(len(sources(ROOT)),145)
        self.assertEqual({row["source"] for row in inventory()},set(sources(ROOT)))

    def test_historical_harness_allows_only_the_build_comment_fix(self):
        from verify_b200 import ROOT
        from publish_execution import same_harness_with_build_comment_fix
        import tempfile
        old=next(ROOT.glob('docs/validation/2026-11-17-a100/systems/*/metadata/eight-device/validate_eight_gpus.cu'))
        current=ROOT/'tools/validate_eight_gpus.cu'
        self.assertTrue(same_harness_with_build_comment_fix(current,old))
        with tempfile.TemporaryDirectory() as directory:
            changed=Path(directory)/'changed.cu'
            changed.write_bytes(current.read_bytes().replace(b'constexpr int P = 8;',b'constexpr int P = 4;'))
            self.assertFalse(same_harness_with_build_comment_fix(changed,old))
    def test_zero_exit_complete(self):self.assertTrue(completed({'exit_code':0,'timed_out':False}))
    def test_timeout_is_not_a_completed_execution(self):
        self.assertFalse(completed({'exit_code':0,'timed_out':True}))
        self.assertFalse(completed({'exit_code':-15,'timed_out':True}))
    def test_failed_exit_is_not_completion(self):self.assertFalse(completed({'exit_code':1,'timed_out':False}))
if __name__=='__main__':unittest.main()
