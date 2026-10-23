#!/usr/bin/env python3
"""Post-process saved present-timestamp ring snapshots into a fine interval table.

Each argument is a directory produced by present-pacing-histogram.py containing
before.json (timestamp index) and after.json (timestamp index + ring). The fine
table shows whether frame-to-frame present intervals cluster on exact multiples
of the 60 Hz vblank period (16.667 ms) or form a continuous distribution.

No measurement is performed; this only reads already-captured snapshots.
"""
import argparse
import collections
import json
from pathlib import Path

RING_WORDS = 2048
VB = 1000.0 / 60.0


def load_intervals(directory):
    before = json.loads((directory / 'before.json').read_text())['after']
    after = json.loads((directory / 'after.json').read_text())['after']
    first = before['kyty_local_present_timestamp_index'][0]
    last = after['kyty_local_present_timestamp_index'][0]
    ring = after['kyty_local_present_timestamps']
    count = min(last - first, RING_WORDS)
    stamps = [ring[i % RING_WORDS] for i in range(last - count, last)]
    return [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]


def summarise(intervals):
    ordered = sorted(intervals)
    n = len(ordered)
    near = {k: sum(1 for v in intervals if abs(v - VB * k) <= 1.0) for k in (1, 2, 3, 4)}
    return {
        'count': n,
        'mean_ms': sum(intervals) / n,
        'min_ms': ordered[0],
        'p50_ms': ordered[n // 2],
        'p90_ms': ordered[int(n * 0.9)],
        'max_ms': ordered[-1],
        'within_1ms_of_vblank_multiple': {str(k): v for k, v in near.items()},
        'on_grid_fraction': sum(near.values()) / n,
        'fine_2ms_bins': {str(b): c for b, c in sorted(
            collections.Counter(int(v // 2) * 2 for v in intervals).items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = {}
    for directory in args.directories:
        intervals = load_intervals(directory)
        report[directory.name] = summarise(intervals)
        summary = report[directory.name]
        print(f"{directory.name:20s} n={summary['count']:4d} mean={summary['mean_ms']:6.2f} "
              f"min={summary['min_ms']:6.2f} p50={summary['p50_ms']:6.2f} "
              f"p90={summary['p90_ms']:6.2f} max={summary['max_ms']:6.2f} "
              f"on-grid={100 * summary['on_grid_fraction']:.1f}% "
              f"grid={summary['within_1ms_of_vblank_multiple']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
