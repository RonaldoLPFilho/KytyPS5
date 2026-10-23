#!/usr/bin/env python3
"""Measure one runtime switch across values inside a retained stationary process.

Values repeat in the given order so that a slow drift in the scene shows up as
disagreement between repeats of the same value rather than as a fake difference
between values. Per-value results combine frames and elapsed time. Counter arrays
are sampled around each window so a value's mechanism can be confirmed, not assumed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def switch(receipt, out, assignments, counters):
    command = ['sudo', '-n', 'python3', str(ROOT / 'tools/local/switch-control.py'), str(receipt),
               '--out', str(out)]
    for name, value in assignments:
        command.extend(['--set', f'{name}={value}'])
    for counter in counters:
        command.extend(['--get', counter])
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f'Switch failed: {done.stdout}\n{done.stderr}')
    return json.loads(out.read_text())


def measure(pid, seconds, label):
    done = subprocess.run(['taskset', '-c', '8-15', 'python3',
                           str(ROOT / 'tools/local/quick-benchmark-game.py'), '--pid', str(pid),
                           '--seconds', str(seconds), '--camera-frames', '0', '--label', label],
                          cwd=ROOT, capture_output=True, text=True, timeout=seconds + 60)
    if done.returncode != 0:
        raise RuntimeError(f'Measurement failed: {done.stdout}\n{done.stderr}')
    evidence = Path(json.loads(done.stdout)['evidence'])
    return json.loads((evidence / 'result.json').read_text())['phases'][0] | {'evidence': str(evidence)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--values', required=True, help='Comma-separated values, e.g. 32,256,2048')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=15)
    parser.add_argument('--settle', type=float, default=2)
    parser.add_argument('--counter', action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    values = [int(token) for token in args.values.split(',')]
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(args.receipt.read_text())
    pid = receipt['pid']
    report = {'receipt': str(args.receipt.resolve()), 'pid': pid, 'symbol': args.symbol,
              'values': values, 'repeats': args.repeats, 'seconds': args.seconds,
              'binary_sha256': receipt['binary_sha256'], 'windows': []}
    for repeat in range(args.repeats):
        for value in values:
            label = f'{args.symbol}-{value}-r{repeat}'
            before = switch(args.receipt, args.out / f'before-{label}.json',
                            [(args.symbol, value)], args.counter)
            time.sleep(args.settle)
            phase = measure(pid, args.seconds, label)
            after = switch(args.receipt, args.out / f'after-{label}.json', [], args.counter)
            deltas = {name: [b - a for a, b in zip(before['after'][name], after['after'][name])]
                      for name in args.counter}
            window = {'repeat': repeat, 'value': value, 'fps': phase['measured_fps'],
                      'frames': phase['frames'], 'elapsed_ns': phase['end_ns'] - phase['start_ns'],
                      'evidence': phase['evidence'],
                      'counters_per_frame': {name: [d / phase['frames'] for d in delta]
                                             for name, delta in deltas.items()}}
            report['windows'].append(window)
            (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            extra = '  '.join(f'{name.split("kyty_local_")[-1]}={[round(x, 1) for x in per]}'
                              for name, per in window['counters_per_frame'].items())
            print(f'{label}: {phase["measured_fps"]:.4f} FPS  {extra}', flush=True)
    combined = {}
    for window in report['windows']:
        state = combined.setdefault(window['value'], {'frames': 0, 'elapsed_ns': 0, 'windows': 0})
        state['frames'] += window['frames']
        state['elapsed_ns'] += window['elapsed_ns']
        state['windows'] += 1
    for state in combined.values():
        state['fps'] = state['frames'] * 1e9 / state['elapsed_ns']
    report['combined'] = {str(k): v for k, v in sorted(combined.items())}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print()
    for value, state in sorted(combined.items()):
        print(f'{args.symbol}={value:<8} {state["fps"]:.4f} FPS   '
              f'({state["frames"]} frames over {state["windows"]} windows)')


if __name__ == '__main__':
    main()
