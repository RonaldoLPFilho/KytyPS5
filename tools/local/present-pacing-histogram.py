#!/usr/bin/env python3
"""Measure the per-frame presentation interval distribution of a retained session.

Reads the emulator's opt-in CLOCK_MONOTONIC present timestamp ring
(`kyty_local_present_timestamps`, enabled by `kyty_local_present_timing_mode=1`)
around one stationary measurement window and histograms the frame-to-frame
intervals in multiples of the 60 Hz vblank period (16.667 ms).

The owning process must be a retained stationary run whose result.json is passed
as RECEIPT. Values are read and written with switch-control.py; no debugger is used.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RING_WORDS = 2048
VB = 1000.0 / 60.0  # 16.6667 ms


def switch(receipt, out, sets=(), gets=()):
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ['sudo', '-n', 'python3', str(ROOT / 'tools/local/switch-control.py'),
               str(receipt), '--out', str(out)]
    for name, value in sets:
        command.extend(['--set', f'{name}={value}'])
    for name in gets:
        command.extend(['--get', name])
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f'switch-control failed: {done.stdout}\n{done.stderr}')
    return json.loads(out.read_text())


def ring_values(words):
    # switch-control.py already folds each pair of 32-bit words into one uint64.
    if len(words) != RING_WORDS:
        raise ValueError(f'ring read returned {len(words)} values, expected {RING_WORDS}')
    return list(words)


def intervals_for(first_index, last_index, ring):
    count = last_index - first_index
    if count <= 0:
        return []
    if count > RING_WORDS:
        count = RING_WORDS
        first_index = last_index - count
    stamps = [ring[i % RING_WORDS] for i in range(first_index, last_index)]
    return [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]


def histogram(intervals):
    buckets = {}
    for value in intervals:
        slot = int(round(value / VB))
        buckets[slot] = buckets.get(slot, 0) + 1
    return {str(k): buckets[k] for k in sorted(buckets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--value', type=int, required=True,
                        help='kyty_local_present_mode_override value for this window')
    parser.add_argument('--extra-set', action='append', default=[],
                        help='Additional SYMBOL=VALUE switch assignments')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--settle', type=float, default=3)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f'{args.out} already exists')
    args.out.mkdir(parents=True)

    report = {'label': args.label, 'value': args.value,
              'extra_set': args.extra_set, 'seconds': args.seconds,
              'settle': args.settle}
    sets = [('kyty_local_present_mode_override', args.value),
            ('kyty_local_present_timing_mode', 1)]
    for item in args.extra_set:
        name, value = item.split('=', 1)
        sets.append((name, int(value)))
    switch(args.receipt, args.out / 'set.json', sets=sets)
    time.sleep(args.settle)

    before = switch(args.receipt, args.out / 'before.json', gets=(
        'kyty_local_present_timestamp_index', 'kyty_local_present_stats'))
    index_before = before['after']['kyty_local_present_timestamp_index'][0]

    measured = subprocess.run(
        ['taskset', '-c', '8-15', 'python3', str(ROOT / 'tools/local/quick-benchmark-game.py'),
         '--pid', str(json.loads(args.receipt.read_text())['pid']), '--seconds',
         str(args.seconds), '--camera-frames', '0', '--label', args.label],
        cwd=ROOT, capture_output=True, text=True, timeout=args.seconds + 60)
    if measured.returncode != 0:
        raise RuntimeError(f'measurement failed: {measured.stdout}\n{measured.stderr}')
    phase = json.loads((Path(json.loads(measured.stdout)['evidence']) / 'result.json').read_text())
    phase = phase['phases'][0]

    after = switch(args.receipt, args.out / 'after.json', gets=(
        'kyty_local_present_timestamp_index', 'kyty_local_present_timestamps',
        'kyty_local_present_stats'))
    index_after = after['after']['kyty_local_present_timestamp_index'][0]
    ring = ring_values(after['after']['kyty_local_present_timestamps'])
    intervals = intervals_for(index_before, index_after, ring)

    report.update({
        'measured_fps': phase['measured_fps'], 'frames': phase['frames'],
        'elapsed_seconds': phase['elapsed_seconds'],
        'ring_first_index': index_before, 'ring_last_index': index_after,
        'ring_frames': index_after - index_before,
        'interval_count': len(intervals),
        'histogram_vblank_multiples': histogram(intervals),
        'mean_interval_ms': (sum(intervals) / len(intervals)) if intervals else None,
        'min_interval_ms': min(intervals) if intervals else None,
        'max_interval_ms': max(intervals) if intervals else None,
        'present_stats_delta': [b - a for a, b in zip(
            before['after']['kyty_local_present_stats'],
            after['after']['kyty_local_present_stats'])],
        'present_stats': after['after']['kyty_local_present_stats'],
    })
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in (
        'label', 'value', 'measured_fps', 'frames', 'histogram_vblank_multiples',
        'mean_interval_ms', 'interval_count', 'present_stats_delta')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
