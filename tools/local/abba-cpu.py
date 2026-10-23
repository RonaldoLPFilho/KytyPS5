#!/usr/bin/env python3
"""Same-process A/B/B/A judged by render-thread CPU per frame, not by FPS.

`abba-switch.py` judges a change by frames/elapsed. But the swapchain is FIFO, so
that interval is quantised to whole vblanks: `interval = ceil(work / 16.67) * 16.67`.
A change that removes several ms of render-thread work therefore moves *nothing*
until the frame drops a whole vblank class -- which is why a campaign can log
"0 FPS gain" over many rounds while real work is being removed.

This tool measures the control variable instead: the render thread's CPU seconds
divided by the frames it produced, in the same retained stationary process. That
is proportional to the work actually removed, so sub-vblank progress accumulates
visibly. FPS is still reported, but only as context.

Usage (run unprivileged; perf and switch-control elevate themselves):
  python3 tools/local/abba-cpu.py <receipt.json> \
      --symbol kyty_local_xxx --sequence 2,0,0,2,2,0,0,2 \
      --seconds 12 --settle 3 --out _Build/<name>
"""
import argparse, json, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def render_tid(pid):
    for task in Path(f'/proc/{pid}/task').iterdir():
        try:
            if (task / 'comm').read_text().strip() == 'Kyty.Gpu':
                return int(task.name)
        except OSError:
            continue
    raise RuntimeError('render thread (Kyty.Gpu) not found')


def read_switch(receipt, symbol):
    out = subprocess.run(['sudo', '-n', 'python3', str(ROOT / 'tools/local/switch-control.py'),
                          str(receipt), '--get', symbol],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r'"before":\s*\{\s*"[^"]+":\s*\[\s*(-?\d+)', out)
    if not m:
        raise RuntimeError(f'cannot read {symbol}:\n{out[:400]}')
    return int(m.group(1))


def write_switch(receipt, symbol, value):
    out = subprocess.run(['sudo', '-n', 'python3', str(ROOT / 'tools/local/switch-control.py'),
                          str(receipt), '--set', f'{symbol}={value}'],
                         capture_output=True, text=True, check=True).stdout
    m = re.search(r'"after":\s*\{\s*"[^"]+":\s*\[\s*(-?\d+)', out)
    if not m or int(m.group(1)) != value:
        raise RuntimeError(f'write of {symbol}={value} not confirmed:\n{out[:400]}')


def measure(receipt, pid, seconds):
    """One window: render-thread CPU ms and frames produced, over the same interval.

    Only perf runs as root. The benchmark stays an unprivileged child because it
    shells out to git, which refuses to run as root inside a user-owned checkout.
    """
    tid = render_tid(pid)
    perf = subprocess.Popen(['sudo', '-n', 'perf', 'stat', '-t', str(tid), '-e', 'task-clock',
                             '--', 'sleep', str(seconds)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    bench = subprocess.Popen([sys.executable, str(ROOT / 'tools/local/quick-benchmark-game.py'),
                              '--pid', str(pid), '--seconds', str(seconds),
                              '--camera-frames', '0', '--label', 'abba-cpu'],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    perf_out = perf.communicate(timeout=seconds + 120)[0]
    out = bench.communicate(timeout=seconds + 120)[0]
    m = re.search(r'([0-9,]+(?:\.\d+)?)\s+msec\s+task-clock', perf_out)
    f = re.search(r'"frames":\s*(\d+)', out)
    e = re.search(r'"elapsed_seconds":\s*([0-9.]+)', out)
    if not (m and f and e):
        raise RuntimeError(f'cannot parse measurement:\nperf:\n{perf_out[-400:]}\nbench:\n{out[-600:]}')
    return {'cpu_ms': float(m.group(1).replace(',', '')), 'frames': int(f.group(1)),
            'elapsed_s': float(e.group(1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('receipt', type=Path)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--sequence', required=True, help='comma-separated switch values, e.g. 2,0,0,2')
    ap.add_argument('--seconds', type=int, default=12)
    ap.add_argument('--settle', type=float, default=3.0)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()

    receipt = json.loads(a.receipt.read_text())
    pid = receipt['pid']
    sequence = [int(x) for x in a.sequence.split(',')]
    if len(sequence) < 4:
        ap.error('need at least 4 windows; the point is an interleaved sequence')

    original = read_switch(a.receipt, a.symbol)
    print(f'pid={pid} render_tid={render_tid(pid)} {a.symbol} original={original}', flush=True)
    print(f'sequence={sequence} seconds={a.seconds} settle={a.settle}', flush=True)

    windows = []
    restored = False
    try:
        for index, value in enumerate(sequence):
            write_switch(a.receipt, a.symbol, value)
            time.sleep(a.settle)
            w = measure(a.receipt, pid, a.seconds)
            w.update(index=index, value=value, cpu_per_frame=w['cpu_ms'] / w['frames'],
                     fps=w['frames'] / w['elapsed_s'])
            windows.append(w)
            print(f"  [{index}] {a.symbol}={value}  frames={w['frames']:4}  "
                  f"cpu={w['cpu_ms']:9.2f} ms  CPU/frame={w['cpu_per_frame']:7.3f} ms  "
                  f"fps={w['fps']:6.2f}", flush=True)
    finally:
        write_switch(a.receipt, a.symbol, original)
        restored = read_switch(a.receipt, a.symbol) == original
        print(f'restored {a.symbol}={original} ok={restored}', flush=True)

    combined = {}
    for w in windows:
        c = combined.setdefault(w['value'], {'cpu_ms': 0.0, 'frames': 0, 'elapsed_s': 0.0})
        c['cpu_ms'] += w['cpu_ms']
        c['frames'] += w['frames']
        c['elapsed_s'] += w['elapsed_s']
    for value, c in combined.items():
        c['cpu_per_frame'] = c['cpu_ms'] / c['frames']
        c['fps'] = c['frames'] / c['elapsed_s']
        c['windows'] = [round(w['cpu_per_frame'], 3) for w in windows if w['value'] == value]

    order = sorted(combined, key=lambda v: combined[v]['cpu_per_frame'])
    report = {'symbol': a.symbol, 'sequence': sequence, 'seconds': a.seconds,
              'original': original, 'restored': restored, 'windows': windows,
              'combined': {str(k): v for k, v in combined.items()}}

    print()
    print(f'{"value":>8} {"windows":>4} {"CPU/frame ms":>13} {"spread":>8} {"fps":>7}')
    for value in sorted(combined):
        c = combined[value]
        spread = (max(c['windows']) - min(c['windows'])) / c['cpu_per_frame'] * 100 if len(c['windows']) > 1 else 0.0
        print(f'{value:>8} {c["frames"]:>4}windows {c["cpu_per_frame"]:>13.3f} {spread:>7.1f}% {c["fps"]:>7.2f}')
    if len(order) == 2:
        lo, hi = order
        d = combined[hi]['cpu_per_frame'] - combined[lo]['cpu_per_frame']
        pct = d / combined[hi]['cpu_per_frame'] * 100
        dfps = combined[lo]['fps'] - combined[hi]['fps']
        print(f'\n{lo} vs {hi}: CPU/frame {d:+.3f} ms ({pct:+.2f}%), fps {dfps:+.2f}')
        print(f'  (lowest CPU/frame is the better setting)')
        report['delta_cpu_per_frame_ms'] = d
        report['delta_percent'] = pct
        report['delta_fps'] = dfps

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'report: {a.out / "report.json"}')


if __name__ == '__main__':
    sys.exit(main())
