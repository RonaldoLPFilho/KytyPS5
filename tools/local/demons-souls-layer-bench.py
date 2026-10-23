#!/usr/bin/env python3
"""Measure Demon's Souls FPS and render-thread cost with a render layer switched off.

This is a thin wrapper around the project's own entry benchmark so the numbers are
comparable with everything in docs/BENCHMARKING.md: it restores the fixed save,
Continues, settles, measures a stationary window and then puts your save back.
It only adds the layer switch and the summary.

    # one configuration
    demons-souls-layer-bench.py --label no-particles --disable particles
    demons-souls-layer-bench.py --label baseline

    # the whole set, baseline first, then report
    demons-souls-layer-bench.py --sweep
    demons-souls-layer-report.py

Why two metrics: the swapchain is FIFO, so frames/elapsed is quantised to whole
vblanks and can hide several ms of render work. The `Kyty.Gpu` thread's CPU
seconds per frame is the sensitive control variable, so both are recorded.
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAYERS_TOOL = ROOT / 'tools/local/demons-souls-layers.py'


def load_layers_tool():
    spec = importlib.util.spec_from_file_location('demons_souls_layers', LAYERS_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layers = load_layers_tool()

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--label', help='output name; defaults to baseline or no-<layer>')
parser.add_argument('--disable', help='comma separated layer names to switch off')
parser.add_argument('--sweep', action='store_true', help='measure baseline then every layer')
parser.add_argument('--config', type=Path,
                    default=ROOT / '_Build/profiles/manual-current-launch.json')
parser.add_argument('--save-baseline', type=Path,
                    default=ROOT / '_Build/walk-fps-20260914/save-baseline')
parser.add_argument('--out-dir', type=Path, default=ROOT / '_Build/layer-bench')
parser.add_argument('--seconds', type=int, default=20)
parser.add_argument('--keep-layers', action='store_true',
                    help='leave the last layer switch installed afterwards')
arguments = parser.parse_args()

if not arguments.config.is_file():
    parser.error(f'launch config not found: {arguments.config}')
if not (arguments.save_baseline / 'PPSA01341').is_dir():
    parser.error(f'fixed save baseline not found: {arguments.save_baseline}')


def render_ms_per_frame(directory, frames):
    before, after = directory / 'cpu-before.json', directory / 'cpu-after.json'
    if not (before.is_file() and after.is_file() and frames):
        return None
    old = json.loads(before.read_text())
    new = json.loads(after.read_text())
    hz = new.get('clock_ticks_per_second') or old.get('clock_ticks_per_second')
    for tid, entry in new.get('threads', {}).items():
        previous = old.get('threads', {}).get(tid)
        if not previous or entry.get('name') != 'Kyty.Gpu':
            continue
        ticks = (entry['user_ticks'] + entry['system_ticks']) - \
                (previous['user_ticks'] + previous['system_ticks'])
        return ticks / hz * 1000.0 / frames
    return None


def measure(label, disabled):
    """Run one configuration through benchmark-entry.py and summarise it."""
    if disabled:
        subprocess.run([sys.executable, str(LAYERS_TOOL), 'apply', '--disable', disabled],
                       check=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.run([sys.executable, str(LAYERS_TOOL), 'clear'], check=True,
                       stdout=subprocess.DEVNULL)

    out = arguments.out_dir / f'bench-{label}'
    if out.exists():
        shutil.rmtree(out)
    print(f'--- {label}: disabled={disabled or "none"} ---', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/local/benchmark-entry.py'),
                    '--config', str(arguments.config),
                    '--save-baseline', str(arguments.save_baseline),
                    '--out', str(out),
                    '--seconds', str(arguments.seconds),
                    '--cpu-times'], check=True)

    result = json.loads((out / 'result.json').read_text())
    measurement = result.get('measurement') or {}
    fps = measurement.get('fps')
    cpu = render_ms_per_frame(out, measurement.get('frames'))
    print(f'{label:16} fps={fps if fps is None else round(fps, 2)}  '
          f'render_ms_per_frame={cpu if cpu is None else round(cpu, 2)}  '
          f'save_restored={result.get("original_save_restored")}', flush=True)
    return {'label': label, 'fps': fps, 'render_ms_per_frame': cpu,
            'frames': measurement.get('frames')}


summary = []
try:
    if arguments.sweep:
        summary.append(measure('baseline', None))
        for name in layers.LAYERS:
            summary.append(measure(f'no-{name}', name))
    else:
        label = arguments.label or (f'no-{arguments.disable}' if arguments.disable else 'baseline')
        summary.append(measure(label, arguments.disable))
finally:
    if not arguments.keep_layers:
        subprocess.run([sys.executable, str(LAYERS_TOOL), 'clear'], check=False,
                       stdout=subprocess.DEVNULL)

print('\n' + json.dumps(summary, indent=1))
print('\nnext: python3 tools/local/demons-souls-layer-report.py --dir ' + str(arguments.out_dir))
