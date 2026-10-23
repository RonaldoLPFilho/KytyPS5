#!/usr/bin/env python3
"""Stage-3 iteration loop (docs/PERF-STAGE3-GOAL.md): one command per step, timed.

    iter.py up [--env NAME=VALUE ...] [--settle 3]   build PGO, stop our session, stage, boot
    iter.py ab VARIANT [VARIANT ...] [--rounds 3] [--seconds 5]
                                                     same-process A/B of runtime switches;
                                                     VARIANT is name=value[+name=value...]
    iter.py measure [--n 4] [--seconds 5]            repeated measurements (noise floor)
    iter.py down                                     stop our session, check the saves

Only our own live session is ever stopped; any other emulator process aborts `up`.
"""
import argparse
import importlib.util
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / '_Build/hist/build-pgo'
USER_SAVES = ROOT / '_Build/layer-bench/save-backups/20260926-171436'
LIVE_BENCH = [sys.executable, str(ROOT / 'tools/local/live-bench.py')]


def game():
    spec = importlib.util.spec_from_file_location('fp', ROOT / 'tools/local/freeze-probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Game()


def emulator_running():
    return subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True).returncode == 0


def our_session():
    return (ROOT / '_Build/layer-bench/live-session.json').exists()


def down(_args=None):
    if our_session():
        subprocess.run(LIVE_BENCH + ['stop'], check=True)
    same = subprocess.run(['diff', '-rq', str(ROOT / '_SaveData'), str(USER_SAVES)],
                          capture_output=True).returncode == 0
    print('saves equal the user backup' if same else 'WARNING: saves differ from the user backup', flush=True)
    if not same:
        raise SystemExit(1)


def up(args):
    t0 = time.monotonic()
    if emulator_running() and not our_session():
        raise SystemExit('an emulator that is not our live session is running; leave it alone')
    if our_session():
        down()
    build = subprocess.run(['ninja', '-C', str(BUILD), 'kyty_emulator'], capture_output=True, text=True)
    if build.returncode != 0:
        print(build.stdout[-4000:], build.stderr[-2000:])
        raise SystemExit('build failed')
    t1 = time.monotonic()
    stage = [sys.executable, str(ROOT / 'tools/local/iter-stage.py')]
    for item in args.env:
        stage += ['--env', item]
    subprocess.run(stage, check=True)
    subprocess.run(LIVE_BENCH + ['start', '--config', str(ROOT / '_Build/iter/launch.json'),
                                 '--settle', str(args.settle)], check=True)
    t2 = time.monotonic()
    print(f'build {t1 - t0:.0f} s, boot {t2 - t1:.0f} s', flush=True)


def measure_once(g, label, seconds):
    lines = g.send([f'measure {seconds} {label}'], timeout=seconds + 30)
    m = next(line for line in lines if 'LIVE_MEASURE' in line)
    return float(re.search(r'fps=([\d.]+)', m)[1]), float(re.search(r'render_ms=([\d.]+)', m)[1])


def measure(args):
    g = game()
    fps, render = [], []
    for i in range(args.n):
        f, r = measure_once(g, f'n{i}', args.seconds)
        fps.append(f)
        render.append(r)
    g.send([])
    print(f'fps {statistics.mean(fps):.2f} (sd {statistics.pstdev(fps):.2f})  '
          f'render_ms {statistics.mean(render):.2f}  fps={fps}')


def ab(args):
    g = game()
    try:
        for variant in args.variants:
            pairs = [(item.split('=')[0], int(item.split('=')[1], 0)) for item in variant.split('+')]
            originals = [(name, g.read(name)) for name, _ in pairs]
            results = {'A': [], 'B': []}
            try:
                for r in range(args.rounds):
                    for which, values in (('A', originals), ('B', pairs)):
                        g.write(values)
                        time.sleep(args.settle)
                        results[which].append(measure_once(g, f'{which}{r}', args.seconds))
            finally:
                g.write(originals)
            fa, fb = (statistics.mean(x[0] for x in results[k]) for k in 'AB')
            ra, rb = (statistics.mean(x[1] for x in results[k]) for k in 'AB')
            # Paired per round (B right after A): robust to scene level shifts between rounds.
            dfps = [b[0] - a[0] for a, b in zip(results['A'], results['B'])]
            dren = [b[1] - a[1] for a, b in zip(results['A'], results['B'])]
            before = '+'.join(f'{n}={v}' for n, v in originals)
            print(f'{before} -> {variant}: fps {fa:.2f} -> {fb:.2f} ({fb - fa:+.2f}, {100 * (fb - fa) / fa:+.1f}%), '
                  f'render_ms {ra:.2f} -> {rb:.2f} ({rb - ra:+.2f}); paired median fps {statistics.median(dfps):+.2f} '
                  f'render {statistics.median(dren):+.2f}  '
                  f'A={[x[0] for x in results["A"]]} B={[x[0] for x in results["B"]]}', flush=True)
    finally:
        g.send([])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('up')
    p.add_argument('--env', action='append', default=[])
    p.add_argument('--settle', type=float, default=3)
    p.set_defaults(func=up)
    p = sub.add_parser('ab')
    p.add_argument('variants', nargs='+')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--settle', type=float, default=1.5)
    p.set_defaults(func=ab)
    p = sub.add_parser('measure')
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--seconds', type=float, default=5)
    p.set_defaults(func=measure)
    sub.add_parser('down').set_defaults(func=down)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
