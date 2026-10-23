#!/usr/bin/env python3
"""Offline checks for A/B accounting and restoration; never touches a process."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('abba_switch', Path(__file__).with_name('abba-switch.py'))
abba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abba)
control_spec = importlib.util.spec_from_file_location('switch_control', Path(__file__).with_name('switch-control.py'))
switch_control = importlib.util.module_from_spec(control_spec)
control_spec.loader.exec_module(switch_control)


class AbbaTests(unittest.TestCase):
    def test_only_ready_retained_sessions_can_start_ab(self):
        ready = {'status': 'measured', 'stationary': True, 'retained': True}
        abba.require_stationary_receipt(ready)
        for change in ({'status': 'running'}, {'retained': False}, {'stationary': False},
                       {'walk_forward_seconds': 8}, {'finished_ns': 123},
                       {'original_save_restored': True}):
            with self.assertRaises(RuntimeError):
                abba.require_stationary_receipt(ready | change)
        abba.require_stationary_receipt(ready | {'walk_forward_seconds': 8,
                                                'forward_setup': {'held_seconds': 8.004}})

    def test_old_binary_counter_layout_is_rejected(self):
        listing = SimpleNamespace(stdout='00001000 00000068 B kyty_local_guest_readback_stats\n')
        with patch.object(switch_control.subprocess, 'run', return_value=listing):
            with self.assertRaisesRegex(RuntimeError, 'counter layout'):
                switch_control.symbol_offsets('fixture', {'kyty_local_guest_readback_stats'})

    def test_unregistered_array_is_not_a_scalar_switch(self):
        listing = SimpleNamespace(stdout='00001000 00000048 B new_unregistered_stats\n')
        with patch.object(switch_control.subprocess, 'run', return_value=listing):
            with self.assertRaisesRegex(RuntimeError, 'counter layout'):
                switch_control.symbol_offsets('fixture', {'new_unregistered_stats'})

    def test_stream_batch_counter_layout(self):
        listing = SimpleNamespace(stdout='00001000 00000048 B kyty_local_stream_batch_stats\n'
                                         '00002000 00000004 B kyty_local_stream_batch_mode\n')
        names = {'kyty_local_stream_batch_stats', 'kyty_local_stream_batch_mode'}
        with patch.object(switch_control.subprocess, 'run', return_value=listing):
            self.assertEqual(switch_control.symbol_offsets('fixture', names),
                {'kyty_local_stream_batch_stats': 0x1000, 'kyty_local_stream_batch_mode': 0x2000})

    def test_unequal_windows_use_total_time(self):
        result = abba.combine([
            {'value': 0, 'frames': 20, 'elapsed_ns': 10**9, 'fps': 20},
            {'value': 0, 'frames': 120, 'elapsed_ns': 3 * 10**9, 'fps': 40},
        ])['0']
        self.assertEqual(result['fps'], 35)
        self.assertAlmostEqual(result['ms_per_frame'], 1000 / 35)

    def test_reset_is_not_reported_as_activity(self):
        with self.assertRaises(RuntimeError):
            abba.counter_delta({'after': {'x': [9]}}, {'after': {'x': [5]}}, ['x'])

    def test_zero_frames_remain_in_total_time(self):
        stalled = {'value': 0, 'frames': 0, 'elapsed_ns': 2 * 10**9, 'fps': 0}
        result = abba.combine([stalled])['0']
        self.assertEqual(result['fps'], 0)
        self.assertIsNone(result['ms_per_frame'])
        self.assertIsNone(result['spread_percent'])
        result = abba.combine([stalled,
            {'value': 0, 'frames': 60, 'elapsed_ns': 3 * 10**9, 'fps': 20}])['0']
        self.assertEqual(result['fps'], 12)
        self.assertEqual(result['min_fps'], 0)
        self.assertAlmostEqual(result['ms_per_frame'], 1000 / 12)

    def exercise(self, failure=None, zero_first=False):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(out=Path(directory), receipt=Path('receipt'), symbol='mode',
                                   settle=0, seconds=1, counter=['x'])
            report = {'pid': 2, 'sequence': [1, 0], 'windows': []}
            state, counter = 7, 0
            writes = []

            def control(receipt, out, symbol, value=None, counters=()):
                nonlocal state, counter
                old = state
                if value is not None:
                    state = value
                    writes.append(value)
                    if failure == 'switch' and len(writes) == 1:
                        raise RuntimeError('fixture: write succeeded but response was lost')
                counter += 1
                return {'before': {symbol: [old]}, 'after': {symbol: [state], 'x': [counter]},
                        'monotonic_ns': counter * 1000}

            def measure(*unused):
                if failure == 'measurement':
                    raise RuntimeError('fixture: measurement interrupted')
                frames = 0 if zero_first and state == 1 else 20
                return {'measured_fps': frames, 'frames': frames, 'start_ns': 1, 'end_ns': 10**9 + 1}

            with patch.object(abba, 'control', control), patch.object(abba, 'measure', measure):
                if failure:
                    with self.assertRaises(RuntimeError):
                        abba.run_windows(args, report)
                else:
                    abba.run_windows(args, report)
            self.assertEqual(state, 7)
            self.assertEqual(writes[-1], 7)
            if not failure:
                self.assertEqual([w['counter_delta']['x'] for w in report['windows']], [[1], [1]])
                self.assertEqual(report['combined']['0']['fps'], 20)
                if zero_first:
                    self.assertEqual(report['windows'][0]['frames'], 0)
                    self.assertIsNone(report['combined']['1']['ms_per_frame'])
            self.assertTrue((args.out / 'report.json').is_file())

    def test_counters_belong_to_current_window(self):
        self.exercise()

    def test_zero_frame_window_continues_and_restores(self):
        self.exercise(zero_first=True)

    def test_failed_measurement_restores_initial_value(self):
        self.exercise('measurement')

    def test_uncertain_switch_restores_initial_value(self):
        self.exercise('switch')


if __name__ == '__main__':
    unittest.main()
