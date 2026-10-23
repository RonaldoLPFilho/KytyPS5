#!/usr/bin/env python3
"""Summarise SHADER_PROFILE blocks (third control-file digit = 1) and diff two runs.

    shader-profile-report.py RUN [--against OTHER_RUN] [--top 40]

RUN is a benchmark output directory (its result.json names the evidence folder) or
a stdout.log.  Windows are averaged; shaders are keyed by call kind plus the hash
of their first 256 code bytes, which is stable across runs, unlike the address.
With --against, the table shows what RUN has that OTHER_RUN does not (e.g. the
cost the XPR world adds: RUN = normal, OTHER_RUN = r_disableXPRRendering).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

WINDOW = re.compile(r'SHADER_PROFILE_WINDOW frames=(\d+) wall_ms_per_frame=([\d.]+) '
                    r'profiled_ms_per_frame=([\d.]+) calls_per_frame=([\d.]+)')
LINE = re.compile(r'SHADER_PROFILE kind=(\w+) a=(0x[0-9a-f]+) b=(0x[0-9a-f]+) ha=([0-9a-f]+) hb=([0-9a-f]+) '
                  r'ms_per_frame=([\d.]+) calls_per_frame=([\d.]+) items_per_frame=([\d.]+)')


def log_path(path):
    path = Path(path)
    if path.is_file():
        return path
    result = json.loads((path / 'run' / 'result.json').read_text()) if (path / 'run').is_dir() \
        else json.loads((path / 'result.json').read_text())
    return Path(result['evidence']) / 'stdout.log'


def load(path):
    windows, table = [], defaultdict(lambda: [0.0, 0.0, 0.0, ''])
    for line in log_path(path).read_text(errors='replace').splitlines():
        if m := WINDOW.search(line):
            windows.append(tuple(float(x) for x in m.groups()))
        elif m := LINE.search(line):
            kind, a, b, ha, hb, ms, calls, items = m.groups()
            entry = table[(kind, ha, hb)]
            entry[0] += float(ms); entry[1] += float(calls); entry[2] += float(items); entry[3] = f'{a}/{b}'
    n = max(len(windows), 1)
    for entry in table.values():
        entry[0] /= n; entry[1] /= n; entry[2] /= n
    return windows, table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--against')
    parser.add_argument('--top', type=int, default=40)
    args = parser.parse_args()
    windows, table = load(args.run)
    if not windows:
        raise SystemExit('no SHADER_PROFILE_WINDOW lines')
    n = len(windows)
    wall, prof, calls = (sum(w[i] for w in windows) / n for i in (1, 2, 3))
    print(f'{n} windows: wall {wall:.2f} ms/frame, profiled {prof:.2f} ms/frame, {calls:.0f} calls/frame, '
          f'all shaders listed')
    other = load(args.against)[1] if args.against else {}
    rows = []
    for key, (ms, c, items, addr) in table.items():
        o = other.get(key, [0.0, 0.0, 0.0, ''])
        rows.append((ms - o[0], ms, c, c - o[1], items, key, addr, key in other))
    rows.sort(key=lambda r: -r[0])
    by_kind = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        by_kind[r[5][0]][0] += r[1]; by_kind[r[5][0]][1] += r[2]
    print('by kind: ' + ', '.join(f'{k} {v[0]:.2f} ms/{v[1]:.0f} calls' for k, v in sorted(by_kind.items(), key=lambda x: -x[1][0])))
    header = 'delta_ms  ms/frame  calls  dcalls  items/frame  kind       ha               hb               in_other  addr'
    print(header)
    for d, ms, c, dc, items, (kind, ha, hb), addr, present in rows[:args.top]:
        print(f'{d:8.3f}  {ms:8.3f}  {c:6.1f}  {dc:6.1f}  {items:11.1f}  {kind:9s}  {ha}  {hb}  {"yes" if present else "no ":8s}  {addr}')
    if args.against:
        only = sum(r[1] for r in rows if not r[7])
        total_delta = sum(r[0] for r in rows)
        print(f'\nshaders absent from the other run: {only:.2f} ms/frame; net delta of listed entries {total_delta:.2f} ms/frame')


if __name__ == '__main__':
    main()
