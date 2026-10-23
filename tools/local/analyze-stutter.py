#!/usr/bin/env python3
"""Correlate opt-in long-frame timings with shader and APR events on the same clock.

Overlapping APR operations are concurrent work, not proof they blocked a frame.
Unattributed time includes uninstrumented guest work and synchronization.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from playtest_log import event


def analyze(log, start_ns=0, end_ns=2**64):
    frames, operations, windows = {}, [], []
    incomplete_records = 0
    for line in log.splitlines():
        if line.startswith('PLAYTEST_'):
            try:
                tag, fields = event(line + '\n')
            except ValueError:
                incomplete_records += 1
                continue
            if tag == 'PLAYTEST_LONG_FRAME':
                fields['stages'] = []
                frames[(fields['frame'], fields['end_ns'])] = fields
            elif tag == 'PLAYTEST_OPERATION' and fields['elapsed_ns'] >= 5_000_000:
                operations.append(fields)
            elif tag == 'PLAYTEST_FRAMES':
                windows.append(fields)
            continue
        match = re.search(r'\b(LOCAL_LONG_FRAME_ITEM|LOCAL_LONG_FRAME|LOCAL_SLOW_OPERATION|LOCAL_RENDER_COST) (.+)', line)
        if not match:
            continue
        kind, body = match.groups()
        fields = dict(re.findall(r'(\w+)=([^\s]+)', body))
        required = {
            'LOCAL_LONG_FRAME': {'frame', 'start_ns', 'end_ns', 'elapsed_ns'},
            'LOCAL_LONG_FRAME_ITEM': {'frame', 'kind', 'count', 'inclusive_ns', 'exclusive_ns'},
            'LOCAL_SLOW_OPERATION': {'kind', 'start_ns', 'end_ns', 'elapsed_ns'},
            'LOCAL_RENDER_COST': {'frames', 'elapsed_ns', 'max_frame_ns'},
        }[kind]
        if not required <= fields.keys():
            incomplete_records += 1
            continue
        try:
            for key in fields.keys() - {'kind', 'id'}:
                fields[key] = int(fields[key])
        except ValueError:
            incomplete_records += 1
            continue
        if kind == 'LOCAL_LONG_FRAME':
            fields['stages'] = []
            frames[fields['frame']] = fields
        elif kind == 'LOCAL_LONG_FRAME_ITEM':
            if fields['frame'] in frames:
                frames[fields['frame']]['stages'].append(fields)
        elif kind == 'LOCAL_SLOW_OPERATION':
            operations.append(fields)
        else:
            windows.append(fields)
    selected = [f for f in frames.values() if f['start_ns'] >= start_ns and f['end_ns'] <= end_ns]
    for frame in selected:
        frame['stages'].sort(key=lambda s: s['exclusive_ns'], reverse=True)
        accounted = sum(s['exclusive_ns'] for s in frame['stages'])
        frame['unattributed_ns'] = max(0, frame['elapsed_ns'] - accounted)
        frame['overlapping_operations'] = [op for op in operations
            if op['start_ns'] < frame['end_ns'] and op['end_ns'] > frame['start_ns']]
    selected.sort(key=lambda f: f['elapsed_ns'], reverse=True)
    operation_counts = Counter()
    operation_max = Counter()
    for op in operations:
        if op['start_ns'] >= start_ns and op['end_ns'] <= end_ns:
            operation_counts[op['kind']] += 1
            operation_max[op['kind']] = max(operation_max[op['kind']], op['elapsed_ns'])
    return {'from_ns': start_ns, 'to_ns': end_ns, 'long_frame_threshold_ms': 100,
            'incomplete_records_skipped': incomplete_records,
            'long_frames': len(selected), 'over_500ms': sum(f['elapsed_ns'] >= 500_000_000 for f in selected),
            'over_1s': sum(f['elapsed_ns'] >= 1_000_000_000 for f in selected),
            'maximum_frame_ms': selected[0]['elapsed_ns'] / 1e6 if selected else None,
            'slow_operation_counts': dict(operation_counts), 'slow_operation_max_ns': dict(operation_max),
            'frames': selected, 'all_render_windows': windows,
            'limits': 'Only intervals >=100 ms and operations >=5 ms are logged. APR overlap is correlation, not causation; all_render_windows also includes startup.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--from-ns', type=int, default=0)
    p.add_argument('--to-ns', type=int, default=2**64)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.log.read_text(errors='replace'), a.from_ns, a.to_ns)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('frames', 'all_render_windows')}, indent=2))


if __name__ == '__main__':
    main()
