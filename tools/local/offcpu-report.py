#!/usr/bin/env python3
"""Turn `perf record -e sched:sched_switch -g` into blocked time per call stack.

For one thread, every switch-out is paired with its next switch-in; the gap is
time the thread was not running, charged to the stack it switched out on.  That
separates "waiting for work" from "waiting for the GPU" from "preempted", which
an on-CPU profile cannot show at all.

    offcpu-report.py sched.data --tid 1234 [--frames 6] [--seconds 8]
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict

SWITCH = re.compile(r'^\s*(\S.*?)\s+(\d+)\s+\[(\d+)\]\s+([\d.]+):\s+sched:(sched_switch|sched_wakeup):\s+(.*)$')
FIELD = re.compile(r'(\w+)=(\S+)')
# perf 6.x prints the compact form:  prev_comm:prev_pid [prio] STATE ==> next_comm:next_pid [prio]
COMPACT = re.compile(r'^(.*):(\d+)\s+\[\d+\]\s+(\S+)\s+==>\s+(.*):(\d+)\s+\[\d+\]')


def parse_trace(trace):
    fields = dict(FIELD.findall(trace.replace(' ==> ', ' ')))
    if 'prev_pid' in fields:
        return fields
    compact = COMPACT.match(trace.strip())
    if compact:
        return {'prev_comm': compact.group(1), 'prev_pid': compact.group(2),
                'prev_state': compact.group(3), 'next_comm': compact.group(4),
                'next_pid': compact.group(5)}
    return fields


def events(path):
    text = subprocess.run(['perf', 'script', '-i', path],
                          capture_output=True, text=True, check=True).stdout
    current = None
    for line in text.splitlines():
        match = SWITCH.match(line)
        if match:
            if current:
                yield current
            kind = match.group(5)
            fields = parse_trace(match.group(6)) if kind == 'sched_switch' else {}
            current = {'time': float(match.group(4)), 'kind': kind, 'fields': fields,
                       'comm': match.group(1), 'stack': []}
        elif current is not None and line.strip():
            # "<addr> <symbol>+<off> (<dso>)"
            parts = line.strip().split(None, 1)
            frame = parts[1] if len(parts) > 1 else parts[0]
            current['stack'].append(re.sub(r'\+0x[0-9a-f]+', '', frame))
    if current:
        yield current


def signature(stack, depth):
    keep = []
    for frame in stack:
        name = re.sub(r'\(.*', '', frame).strip()
        if not name or name.startswith('[unknown]') or name.startswith('0x'):
            continue
        if any(skip in name for skip in ('__schedule', 'schedule', 'futex_', 'do_syscall',
                                         'entry_SYSCALL', '__x64_sys', 'syscall_exit')):
            continue
        keep.append(name.replace('Libs::Graphics::', ''))
        if len(keep) == depth:
            break
    return ' <- '.join(keep) or '(no user frames)'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data')
    parser.add_argument('--tid', required=True)
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--top', type=int, default=14)
    arguments = parser.parse_args()

    blocked = defaultdict(float)
    counts = defaultdict(int)
    by_state = defaultdict(float)
    wakers = defaultdict(float)
    waker_counts = defaultdict(int)
    pending = None
    waker = None
    first = last = None
    for event in events(arguments.data):
        fields = event['fields']
        first = event['time'] if first is None else first
        last = event['time']
        if event['kind'] == 'sched_wakeup':
            # Who made the thread runnable again, and from where.
            waker = f"{event['comm']}: {signature(event['stack'], arguments.frames)}"
            continue
        if fields.get('prev_pid') == arguments.tid:
            pending = (event['time'], fields.get('prev_state', '?'),
                       signature(event['stack'], arguments.frames))
            waker = None
        elif fields.get('next_pid') == arguments.tid and pending:
            start, state, sig = pending
            gap = event['time'] - start
            key = f'[{state}] {sig}'
            blocked[key] += gap
            counts[key] += 1
            by_state[state] += gap
            if state.startswith('S') or state.startswith('D'):
                wkey = waker or '(no wakeup recorded)'
                wakers[wkey] += gap
                waker_counts[wkey] += 1
            pending = None
    if first is None:
        print('no sched_switch events for that tid', file=sys.stderr)
        return 1
    span = last - first
    total = sum(blocked.values())
    print(f'window {span:.2f}s; thread off-CPU {total:.2f}s = {100 * total / span:.1f}% of wall')
    for state, seconds in sorted(by_state.items(), key=lambda kv: -kv[1]):
        label = {'S': 'blocked (sleep)', 'R': 'preempted', 'R+': 'preempted',
                 'D': 'uninterruptible'}.get(state, state)
        print(f'  {label:18} {seconds:7.3f}s  {100 * seconds / span:5.1f}%')
    print()
    print(f"{'off-CPU s':>9} {'%wall':>6} {'count':>7}  stack at switch-out")
    for key, seconds in sorted(blocked.items(), key=lambda kv: -kv[1])[:arguments.top]:
        print(f'{seconds:9.3f} {100 * seconds / span:6.1f} {counts[key]:7d}  {key[:230]}')
    print()
    print(f"{'blocked s':>9} {'%wall':>6} {'count':>7}  woken by")
    for key, seconds in sorted(wakers.items(), key=lambda kv: -kv[1])[:arguments.top]:
        print(f'{seconds:9.3f} {100 * seconds / span:6.1f} {waker_counts[key]:7d}  {key[:230]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
