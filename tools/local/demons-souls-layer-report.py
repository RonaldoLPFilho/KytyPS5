#!/usr/bin/env python3
"""Summarise the per-layer FPS measurements produced by demons-souls-layer-bench
or by benchmark-entry.py runs driven through tools/local/demons-souls-layers.py.

    demons-souls-layer-report.py --dir _Build/layer-bench

Reads every bench-<layer>/result.json (and the loose <label>.json files) and
prints the stationary FPS, the delta against the baseline run and the frames
counted, plus the screenshot each run captured so the removal can be eyeballed.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dir', type=Path, default=ROOT / '_Build/layer-bench')
parser.add_argument('--baseline', default='baseline')
arguments = parser.parse_args()

rows = []


def render_cpu_ms(directory, frames):
    """Render-thread CPU milliseconds per frame from benchmark-entry --cpu-times.

    Prefer Kyty.Gpu; fall back to the busiest thread if the name changed, and
    report nothing rather than a wrong number.
    """
    before_path, after_path = directory / 'cpu-before.json', directory / 'cpu-after.json'
    if not (before_path.is_file() and after_path.is_file() and frames):
        return None
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    hz = after.get('clock_ticks_per_second') or before.get('clock_ticks_per_second')
    best = None
    for tid, entry in after.get('threads', {}).items():
        old = before.get('threads', {}).get(tid)
        if not old:
            continue
        ticks = (entry['user_ticks'] + entry['system_ticks']) - (old['user_ticks'] + old['system_ticks'])
        if ticks <= 0:
            continue
        seconds = ticks / hz
        if entry.get('name') == 'Kyty.Gpu':
            return seconds * 1000.0 / frames
        if best is None or seconds > best[0]:
            best = (seconds, seconds * 1000.0 / frames)
    return best[1] if best else None


def add(label, fps, frames, seconds, screenshot, status, cpu_ms=None):
    rows.append({'label': label, 'fps': fps, 'frames': frames, 'seconds': seconds,
                 'screenshot': screenshot, 'status': status, 'cpu_ms': cpu_ms})


for directory in sorted(arguments.dir.glob('bench-*')):
    result = directory / 'result.json'
    if not result.is_file():
        continue
    data = json.loads(result.read_text())
    measurement = data.get('measurement') or {}
    add(directory.name[len('bench-'):], measurement.get('fps'),
        measurement.get('frames'), measurement.get('elapsed_seconds'),
        str(directory), data.get('status'),
        render_cpu_ms(directory, measurement.get('frames')))

for loose in sorted(arguments.dir.glob('*.json')):
    data = json.loads(loose.read_text())
    if 'measured_fps' not in data:
        continue
    add(data['label'], data.get('measured_fps'), data.get('frames'),
        data.get('elapsed_seconds'), data.get('screenshot'),
        'measured' if data.get('ok') else 'failed')

baseline = next((r for r in rows if r['label'] == arguments.baseline and r['fps']), None)
print(f'{"configuration":16} {"FPS":>7} {"vs baseline":>15} {"render ms/frame":>16} {"saved":>8}')
for row in sorted(rows, key=lambda r: -(r['cpu_ms'] or 0)):
    label = row['label']
    fps = '-' if row['fps'] is None else f'{row["fps"]:.2f}'
    delta = '-'
    if baseline and row['fps'] and label != arguments.baseline:
        delta = f'{row["fps"] - baseline["fps"]:+.2f} ({(row["fps"] / baseline["fps"] - 1) * 100:+.1f}%)'
    cpu = '-' if row['cpu_ms'] is None else f'{row["cpu_ms"]:.2f}'
    saving = '-'
    if row['cpu_ms'] and baseline and baseline['cpu_ms'] and label != arguments.baseline:
        saving = f'{baseline["cpu_ms"] - row["cpu_ms"]:+.2f}'
    print(f'{label:16} {fps:>7} {delta:>15} {cpu:>16} {saving:>8}')

if baseline:
    print(f'\nbaseline reference: {baseline["fps"]:.2f} FPS over {baseline["seconds"]:.1f}s '
          f'({baseline["frames"]} frames), render thread '
          f'{baseline["cpu_ms"]:.2f} ms/frame' if baseline['cpu_ms'] else '')
print('\n"render ms/frame" is the Kyty.Gpu thread\'s CPU time per presented frame; it is')
print('not quantised by the FIFO swapchain, so it shows work removed below one vblank.')
