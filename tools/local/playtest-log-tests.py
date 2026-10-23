#!/usr/bin/env python3
"""Focused parser, interrupted-record and PID identity tests (no game required)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from playtest_log import EventIndex, PlaytestMonitor, event


class EvidenceTests(unittest.TestCase):
    def test_partial_and_bad_records(self):
        index = EventIndex()
        for row in ('PLAYTEST_START', 'PLAYTEST_FRAMES frames=1\n',
                    'PLAYTEST_LONG_FRAME frame=1 start_ns=5 end_ns=2 elapsed_ns=-3\n'):
            index.add(row)
        self.assertEqual(index.malformed, 3)
        self.assertIsNone(index.summary()['whole_session_fps_including_menus_and_loading'])

    def test_pairing_hex_and_context(self):
        index = EventIndex()
        index.add('PLAYTEST_START version=1 pid=1 monotonic_ns=100 unix_ns=1000000000\n')
        index.add('PLAYTEST_COMPILE_BEGIN op=1 kind=ShaderCompile start_ns=100 id=abc\n')
        index.add('PLAYTEST_COMPILE_BEGIN op=2 kind=GraphicsPipeline start_ns=100\n')
        index.add('PLAYTEST_OPERATION op=1 kind=ShaderCompile start_ns=100 end_ns=6000100 elapsed_ns=6000000 id=abc context=file%20path%0Awith%25%3Dfields\n')
        report = index.summary()
        self.assertEqual([r['op'] for r in report['unfinished_compilations']], [2])
        self.assertEqual(report['slow_operation_counts'], {'ShaderCompile': 1})
        self.assertEqual(report['slowest_operations'][0]['id'], 0xabc)
        self.assertEqual(report['slowest_operations'][0]['context'], 'file path\nwith%=fields')
        self.assertEqual(report['slowest_operations'][0]['since_launch_seconds'], 0)

    def test_crash_tail_and_record_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            monitor = PlaytestMonitor(1, 'wrong', root)
            path = root / 'playtest.log'
            path.write_bytes(b'PLAYTEST_COMPILE_BEGIN op=9 kind=ShaderCompile start_ns=100')
            monitor.consume()
            self.assertFalse(monitor.index.pending)
            with path.open('ab') as stream:
                stream.write(b'\nPLAYTEST_FAULT time_hex=64 pc=900ac91ff sp=123 address=dead code_hex=b\nPARTIAL')
            monitor.consume(final=True)
            self.assertEqual(monitor.index.malformed, 1)
            self.assertEqual(monitor.index.faults[0]['pc'], 0x900ac91ff)
            self.assertIn(9, monitor.index.pending)

    def test_wrong_pid_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = PlaytestMonitor(1, 'wrong', directory)
            with self.assertRaises(ProcessLookupError):
                monitor.proc_sample(100)

    def test_exit_classification(self):
        spec = importlib.util.spec_from_file_location('supervisor', Path(__file__).with_name('debug-run.py'))
        supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supervisor)
        self.assertEqual(supervisor.classify(-6, False, '', '', 99)['status'], 'process_failure')
        self.assertEqual(supervisor.classify(0, False, 'VK_ERROR_DEVICE_LOST', '', 99)['status'], 'gpu_failure')
        self.assertEqual(supervisor.classify(0, False, '', 'NVRM: Xid pid=98 error', 99)['status'], 'inconclusive')
        self.assertFalse(supervisor.classify(0, False, '', '', 99)['gameplay_verified'])


if __name__ == '__main__':
    unittest.main()
