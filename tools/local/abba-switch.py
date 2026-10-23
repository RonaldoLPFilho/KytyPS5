#!/usr/bin/env python3
"""Alternate one runtime switch inside a retained stationary process and measure.

Each window is a separate zero-input measurement of the same owned process, so
only the switch differs between them. Combining windows per state uses frames and
elapsed time, never an average of per-window rates.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def control(receipt, out, symbol, value=None, counters=()):
    command = ['sudo', '-n', 'python3', str(ROOT / 'tools/local/switch-control.py'), str(receipt),
               '--out', str(out / 'switch.json')]
    if value is not None:
        command.extend(['--set', f'{symbol}={value}'])
    else:
        command.extend(['--get', symbol])
    for counter in counters:
        command.extend(['--get', counter])
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f'Switch control failed: {done.stdout}\n{done.stderr}')
    return json.loads((out / 'switch.json').read_text())


def measure(pid, seconds, label, out):
    done = subprocess.run(['taskset', '-c', '8-15', 'python3',
                           str(ROOT / 'tools/local/quick-benchmark-game.py'), '--pid', str(pid),
                           '--seconds', str(seconds), '--camera-frames', '0', '--label', label],
                          cwd=ROOT, capture_output=True, text=True, timeout=seconds + 60)
    if done.returncode != 0:
        raise RuntimeError(f'Measurement failed: {done.stdout}\n{done.stderr}')
    payload = json.loads(done.stdout)
    evidence = Path(payload['evidence'])
    phase = json.loads((evidence / 'result.json').read_text())['phases'][0]
    (out / f'{label}.json').write_text(json.dumps({'evidence': str(evidence), **phase}, indent=2) + '\n')
    return phase


def counter_delta(before, after, counters):
    result = {}
    for name in counters:
        first, last = before['after'][name], after['after'][name]
        if len(first) != len(last) or any(b < a for a, b in zip(first, last)):
            raise RuntimeError(f'Counter {name} reset or changed size during measurement')
        result[name] = [b - a for a, b in zip(first, last)]
    return result


def combine(windows):
    combined = {}
    for window in windows:
        state = combined.setdefault(window['value'], {'frames': 0, 'elapsed_ns': 0,
                                                     'windows': 0, 'window_fps': []})
        state['frames'] += window['frames']
        state['elapsed_ns'] += window['elapsed_ns']
        state['windows'] += 1
        state['window_fps'].append(window['fps'])
    for state in combined.values():
        state['fps'] = state['frames'] * 1e9 / state['elapsed_ns']
        # A cold pipeline can stall an entire window. Keep its elapsed time and
        # zero frames; no completed frame means that ms/frame is undefined.
        state['ms_per_frame'] = (state['elapsed_ns'] / state['frames'] / 1e6
                                 if state['frames'] else None)
        state['min_fps'] = min(state['window_fps'])
        state['max_fps'] = max(state['window_fps'])
        state['spread_percent'] = (100 * (state['max_fps'] - state['min_fps']) / state['fps']
                                   if state['fps'] else None)
    return {str(k): v for k, v in sorted(combined.items())}


def require_stationary_receipt(receipt):
    if receipt.get('finished_ns') is not None or receipt.get('original_save_restored'):
        raise RuntimeError('The retained benchmark session has already ended')
    if (receipt.get('status') != 'measured' or receipt.get('retained') is not True
            or receipt.get('stationary') is not True):
        raise RuntimeError('Wait for benchmark-entry to finish setup and retain the stationary process')
    if receipt.get('walk_forward_seconds', 0) and not receipt.get('forward_setup'):
        raise RuntimeError('The requested forward setup has no completed input receipt')


def run_windows(args, report):
    initial = None
    try:
        initial_dir = args.out / 'initial'
        initial_dir.mkdir()
        initial = control(args.receipt, initial_dir, args.symbol)['after'][args.symbol][0]
        report['initial_value'] = initial
        for index, value in enumerate(report['sequence']):
            label = f'{index}-{args.symbol}-{value}'
            switch_dir = args.out / f'switch-{label}'
            switch_dir.mkdir(parents=True, exist_ok=True)
            switch = control(args.receipt, switch_dir, args.symbol, value)
            time.sleep(args.settle)
            snapshots = []
            for name in ('before', 'after'):
                directory = switch_dir / name
                directory.mkdir()
                snapshots.append(control(args.receipt, directory, args.symbol, counters=args.counter))
                if name == 'before':
                    phase = measure(report['pid'], args.seconds, label, args.out)
            if any(sample['after'][args.symbol] != [value] for sample in snapshots):
                raise RuntimeError('Switch changed during the measurement window')
            report['windows'].append({
                'index': index, 'value': value, 'label': label,
                'fps': phase['measured_fps'], 'frames': phase['frames'],
                'elapsed_ns': phase['end_ns'] - phase['start_ns'], 'switch': switch,
                'counters_before': snapshots[0], 'counters_after': snapshots[1],
                'counter_delta': counter_delta(*snapshots, args.counter),
                'counter_elapsed_ns': snapshots[1]['monotonic_ns'] - snapshots[0]['monotonic_ns'],
            })
            report['combined'] = combine(report['windows'])
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f"{label}: {phase['measured_fps']:.4f} FPS ({phase['frames']} frames)", flush=True)
    finally:
        if initial is not None:
            restore_dir = args.out / 'restore'
            restore_dir.mkdir(exist_ok=True)
            try:
                report['restoration'] = control(args.receipt, restore_dir, args.symbol, initial)
            except Exception as exc:
                report['restoration_error'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path, help='result.json of a retained stationary session')
    parser.add_argument('--symbol', required=True, help='Switch symbol, e.g. kyty_local_lod_stats_period')
    parser.add_argument('--sequence', required=True,
                        help='Comma-separated switch values in measurement order, e.g. 0,1,1,0')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--settle', type=float, default=3, help='Seconds after each switch before timing')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--counter', action='append', default=[], help='Counter array to sample each window')
    args = parser.parse_args()
    values = [int(token) for token in args.sequence.split(',')]
    if not values or any(value < 0 or value > 0xffffffff for value in values):
        parser.error('Switch values must fit an unsigned 32-bit integer')
    if args.seconds <= 0 or args.settle < 0:
        parser.error('Measurement seconds must be positive and settle must be nonnegative')
    receipt = json.loads(args.receipt.read_text())
    require_stationary_receipt(receipt)
    args.out.mkdir(parents=True, exist_ok=False)
    pid = receipt['pid']
    report = {'receipt': str(args.receipt.resolve()), 'pid': pid, 'symbol': args.symbol,
              'sequence': values, 'seconds': args.seconds, 'settle': args.settle,
              'binary_sha256': receipt['binary_sha256'], 'windows': []}
    run_windows(args, report)
    print(json.dumps(report['combined'], indent=2))


if __name__ == '__main__':
    main()
