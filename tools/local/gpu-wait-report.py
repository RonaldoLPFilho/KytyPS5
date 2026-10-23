#!/usr/bin/env python3
"""Summarize complete renderer GPU-wait windows inside a measured interval.

wait_tick is the requested completion dependency, including download copies.
It is not a separately tracked shader producer tick. These diagnostics measure
host wait time, not GPU execution time or a guaranteed removable stall.
"""
import argparse
import json
from pathlib import Path
import re
import shlex


def analyze(log, start_ns=0, end_ns=2**64):
    windows = []
    window = None
    for line in log.splitlines():
        match = re.search(r'\b(LOCAL_GPU_WAIT_SUMMARY|LOCAL_GPU_WAIT) (.*)', line)
        if not match:
            continue
        kind, body = match.groups()
        values = dict(token.split('=', 1) for token in shlex.split(body) if '=' in token)
        values = {k: v if k == 'site' else int(v) for k, v in values.items()}
        if kind.endswith('SUMMARY'):
            window = values | {'sites': []}
            windows.append(window)
        elif window:
            window['sites'].append(values)
    selected = [w for w in windows if w['start_ns'] >= start_ns and w['end_ns'] <= end_ns]
    frames = sum(w['frames'] for w in selected)
    sites = {}
    for w in selected:
        for item in w['sites']:
            site = sites.setdefault(item['site'], {'site': item['site'], 'calls': 0, 'blocked': 0,
                                                  'elapsed_ns': 0, 'maximum_ns': 0})
            for key in ('calls', 'blocked', 'elapsed_ns'):
                site[key] += item[key]
            site['maximum_ns'] = max(site['maximum_ns'], item['maximum_ns'])
    for item in sites.values():
        item['ms_per_frame'] = item['elapsed_ns'] / frames / 1e6 if frames else None
        item['calls_per_frame'] = item['calls'] / frames if frames else None
    return {'frames': frames, 'windows': len(selected),
            'overflow': sum(w['overflow'] for w in selected),
            'sites': sorted(sites.values(), key=lambda s: s['elapsed_ns'], reverse=True),
            'details': selected, 'limits': __doc__.strip()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--from-ns', type=int, default=0)
    p.add_argument('--to-ns', type=int, default=2**64)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.log.read_text(errors='replace'), a.from_ns, a.to_ns)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}, indent=2))


if __name__ == '__main__':
    main()
