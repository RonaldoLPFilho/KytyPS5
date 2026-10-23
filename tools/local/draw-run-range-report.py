#!/usr/bin/env python3
"""Interpret v22 reference-policy counters without inventing rejection causes.

Input: abba-switch.py report for kyty_local_draw_run_ranges_mode, modes 0/3.
Counter and FPS windows have different boundaries. Ratios use counter deltas;
per-frame counts are explicitly estimates, using counter duration and nearby FPS.
"""
import argparse
import json
from pathlib import Path

SYMBOL = 'kyty_local_draw_run_ranges_mode'
COUNTER = 'kyty_local_draw_run_ranges_stats'
NAMES = ('attempts', 'range_checks', 'nonunique_or_invalid_range',
         'tracked_image_exclusion', 'target_overlap', 'sampled_overlap_allowed',
         'other_image_exclusion', 'issued_runs', 'issued_draws')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report['symbol'] != SYMBOL or report.get('restoration_error'):
        parser.error('Require a restored reference-policy range-counter experiment')
    if report.get('restoration', {}).get('after', {}).get(SYMBOL) != [report['initial_value']]:
        parser.error('The initial switch value was not restored')
    rows = []
    for window in report['windows']:
        values = window['counter_delta'][COUNTER]
        if len(values) != 9 or any(type(v) is not int or v < 0 for v in values):
            parser.error('Invalid v22 counter layout or negative delta')
        if window['value'] == 0:
            if any(values):
                parser.error('Counters changed while reference diagnostics were disabled')
            continue
        if window['value'] != 3 or values[5] or values[6]:
            parser.error('Only mode 3 retains the reference policy and counts it')
        counts = dict(zip(NAMES, values))
        attempts, issued, draws = values[0], values[7], values[8]
        failures = attempts - issued
        classified = values[2] + values[3] + values[4]
        if (attempts <= 0 or not 0 <= classified <= failures
                or not 2 * issued <= draws <= 64 * issued):
            parser.error('Inconsistent counters; inspect raw snapshots and their boundaries')
        seconds = window['counter_elapsed_ns'] / 1e9
        estimated_frames = seconds * window['fps']
        if seconds <= 0 or estimated_frames <= 0:
            parser.error('Counter duration and adjacent frame rate must be positive')
        counts['failed_runs'] = failures
        counts['unclassified_failures'] = failures - classified
        counts['draw_preparations_avoided_by_issued_runs'] = draws - issued
        rows.append({
            'label': window['label'], 'counter_seconds': seconds,
            'adjacent_fps': window['fps'], 'counts': counts,
            'success_percent': 100 * issued / attempts,
            'mean_draws_per_issued_run': draws / issued if issued else None,
            'tracked_image_percent_of_failures': 100 * values[3] / failures if failures else None,
            'unclassified_percent_of_attempts': 100 * (failures - classified) / attempts,
            'estimated_counts_per_frame': {k: v / estimated_frames for k, v in counts.items()},
        })
    if not rows:
        parser.error('No mode 3 observation window')
    result = {
        'schema': 1, 'input': str(args.report.resolve()), 'pid': report['pid'],
        'binary_sha256': report['binary_sha256'], 'windows': rows,
        'limits': [
            'Counters start at TryDrawIndexRun; earlier PM4 scan failures are not counted.',
            'Unclassified failures include early guards, read observers, shader guards and final issue failure.',
            'No RefreshShaders counter, rejected-run length histogram or per-reason elapsed time exists here.',
            'Issued draw count includes active draws only. Avoided preparation count is relative to individual issuance.',
            'Counter arrays are read consecutively, not as one atomic renderer snapshot.',
            'Per-frame estimates use counter duration and an adjacent FPS window with different endpoints.',
            'The 0/3/0 FPS difference measures observation noise/overhead, not an optimization gain.',
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
