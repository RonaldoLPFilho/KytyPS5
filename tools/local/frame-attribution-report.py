#!/usr/bin/env python3
"""Per-frame render/guest attribution for one stationary retained window.

Turns on `kyty_local_frame_attribution_mode=1` plus any extra switches (for
example `kyty_local_route_a_ceiling_mode=4`) in one owned retained session, runs
one stationary `quick-benchmark-game.py` window, and reads the per-frame ring and
monotonic accumulators back with switch-control.py. Reports the distributions of
the frame interval, render-thread busy time (inside GuestGpu::Process), the
timeline-wait time nested inside that busy time, the queue-wait idle time, and
the guest wait at the frame boundary.

The process must be a retained stationary run whose result.json is passed as
RECEIPT. No debugger is used.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RING_FRAMES = 1024
RING_FIELDS = 12
# Ring field offsets, matching KYTY_LOCAL_FRAME_ATTR_FIELDS in graphicsRun.cpp.
F_INTERVAL, F_RENDER_BUSY, F_RENDER_SUBS = 0, 1, 2
F_GPU_WAIT, F_GUEST_WAIT = 3, 4
F_QUEUE_WAIT, F_QUEUE_COUNT = 5, 6
F_BOUNDARY, F_CMD, F_CMD_COUNT = 7, 8, 9
F_PRESENT_TID, F_DONE_COUNT = 10, 11

COUNTERS = ('kyty_local_frame_attr_index', 'kyty_local_frame_attr_stats',
            'kyty_local_frame_attr_render_busy_ns',
            'kyty_local_frame_attr_render_submissions',
            'kyty_local_frame_attr_render_gpu_wait_ns',
            'kyty_local_frame_attr_render_queue_wait_ns',
            'kyty_local_frame_attr_render_queue_wait_count',
            'kyty_local_frame_attr_render_command_ns',
            'kyty_local_frame_attr_render_command_count',
            'kyty_local_frame_attr_guest_wait_total_ns',
            'kyty_local_frame_attr_guest_wait_count',
            'kyty_local_frame_attr_done_count')


def switch(receipt, out, sets=(), gets=()):
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ['sudo', '-n', 'python3', str(ROOT / 'tools/local/switch-control.py'),
               str(receipt), '--out', str(out)]
    for name, value in sets:
        command.extend(['--set', f'{name}={value}'])
    for name in gets:
        command.extend(['--get', name])
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    if done.returncode != 0:
        raise RuntimeError(f'switch-control failed: {done.stdout}\n{done.stderr}')
    return json.loads(out.read_text())


def scalar(report, name):
    return report['after'][name][0]


def stats(swap, name):
    return report_values(swap['after'][name])


def report_values(words):
    return list(words)


def distribution(values, scale=1e6, unit='ms'):
    if not values:
        return None
    ordered = sorted(values)
    def pick(q):
        index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[index] / scale
    return {f'mean_{unit}': sum(ordered) / len(ordered) / scale,
            f'median_{unit}': pick(.5), f'p10_{unit}': pick(.1),
            f'p90_{unit}': pick(.9), f'min_{unit}': ordered[0] / scale,
            f'max_{unit}': ordered[-1] / scale, 'count': len(ordered)}


def extract(index_before, index_after, ring):
    count = index_after - index_before
    if count <= 0:
        return []
    if count > RING_FRAMES:
        count = RING_FRAMES
        index_before = index_after - count
    frames = []
    for slot in range(index_before, index_after):
        base = (slot % RING_FRAMES) * RING_FIELDS
        frames.append([ring[base + field] for field in range(RING_FIELDS)])
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--settle', type=float, default=3)
    parser.add_argument('--set', action='append', default=[], metavar='SYMBOL=VALUE',
                        help='Extra switch assignments for this window')
    parser.add_argument('--drop-frames', type=int, default=3,
                        help='Leading ring entries to discard after a switch change')
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f'{args.out} already exists')
    args.out.mkdir(parents=True)
    receipt = json.loads(args.receipt.read_text())
    if receipt.get('status') not in ('measured', 'diagnostics-only'):
        raise SystemExit(f'retained run is not measured: {receipt.get("status")}')

    sets = [('kyty_local_frame_attribution_mode', 1)]
    for item in args.set:
        name, value = item.split('=', 1)
        sets.append((name, int(value)))
    switch(args.receipt, args.out / 'set.json', sets=sets)
    time.sleep(args.settle)

    before = switch(args.receipt, args.out / 'before.json', gets=COUNTERS)
    index_before = scalar(before, 'kyty_local_frame_attr_index')
    stats_before = stats(before, 'kyty_local_frame_attr_stats')

    measured = subprocess.run(
        ['taskset', '-c', '8-15', 'python3', str(ROOT / 'tools/local/quick-benchmark-game.py'),
         '--pid', str(receipt['pid']), '--seconds', str(args.seconds),
         '--camera-frames', '0', '--label', args.label],
        cwd=ROOT, capture_output=True, text=True, timeout=args.seconds + 90)
    if measured.returncode != 0:
        raise RuntimeError(f'measurement failed: {measured.stdout}\n{measured.stderr}')
    phase = json.loads((Path(json.loads(measured.stdout)['evidence']) / 'result.json').read_text())
    phase = phase['phases'][0]

    after = switch(args.receipt, args.out / 'after.json',
                   gets=COUNTERS + ('kyty_local_frame_attr_ring',))
    index_after = scalar(after, 'kyty_local_frame_attr_index')
    stats_after = stats(after, 'kyty_local_frame_attr_stats')
    ring = after['after']['kyty_local_frame_attr_ring']
    frames = extract(index_before, index_after, ring)[args.drop_frames:]

    interval = [f[F_INTERVAL] for f in frames]
    render_busy = [f[F_RENDER_BUSY] for f in frames]
    render_subs = [f[F_RENDER_SUBS] for f in frames]
    gpu_wait = [f[F_GPU_WAIT] for f in frames]
    guest_wait = [f[F_GUEST_WAIT] for f in frames]
    queue_wait = [f[F_QUEUE_WAIT] for f in frames]
    command = [f[F_CMD] for f in frames]
    done_count = [f[F_DONE_COUNT] for f in frames]

    totals = {name: scalar(after, name) - scalar(before, name) for name in COUNTERS[2:]
              if name != 'kyty_local_frame_attr_stats'}
    interval_sum = sum(interval)
    busy_sum = sum(render_busy)
    ratio = (busy_sum / interval_sum) if interval_sum else None
    per_frame_ratio = [b / i for b, i in zip(render_busy, interval) if i > 0]

    report = {
        'label': args.label, 'set': sets, 'seconds': args.seconds, 'settle': args.settle,
        'measured_fps': phase['measured_fps'], 'frames': phase['frames'],
        'elapsed_seconds': phase['elapsed_seconds'],
        'fps_from_interval': (1e9 / (interval_sum / len(interval))) if interval else None,
        'ring_first_index': index_before, 'ring_last_index': index_after,
        'ring_recorded': index_after - index_before, 'ring_used': len(frames),
        'interval': distribution(interval),
        'render_busy': distribution(render_busy),
        'render_gpu_wait_inside_busy': distribution(gpu_wait),
        'render_queue_wait_idle': distribution(queue_wait),
        'render_command_path': distribution(command),
        'render_submissions': distribution(render_subs, scale=1.0, unit='count'),
        'done_boundaries_per_frame': distribution(done_count, scale=1.0, unit='count'),
        'presents_by_tid': sorted({f[F_PRESENT_TID] for f in frames}),
        'guest_wait_at_boundary': distribution(guest_wait),
        'ratio_render_busy_over_interval': ratio,
        'ratio_render_busy_over_interval_per_frame_mean':
            (sum(per_frame_ratio) / len(per_frame_ratio)) if per_frame_ratio else None,
        'render_busy_minus_gpu_wait_over_interval':
            ((busy_sum - sum(gpu_wait)) / interval_sum) if interval_sum else None,
        'queue_wait_over_interval': (sum(queue_wait) / interval_sum) if interval_sum else None,
        'command_over_interval': (sum(command) / interval_sum) if interval_sum else None,
        'accumulator_delta': totals,
        'stats_before': stats_before, 'stats_after': stats_after,
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in (
        'label', 'set', 'measured_fps', 'frames', 'ring_used', 'interval',
        'render_busy', 'render_gpu_wait_inside_busy', 'render_queue_wait_idle',
        'guest_wait_at_boundary', 'render_submissions',
        'done_boundaries_per_frame', 'presents_by_tid',
        'ratio_render_busy_over_interval',
        'render_busy_minus_gpu_wait_over_interval', 'queue_wait_over_interval',
        'command_over_interval')}, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
