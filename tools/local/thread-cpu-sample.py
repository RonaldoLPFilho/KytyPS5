#!/usr/bin/env python3
"""Per-thread CPU over a window, for a running emulator.

The frame counter says how fast a configuration runs; this says how the work is
spread. Together they separate "the two layers overlap" from "they serialise":
if guest and renderer threads each sit near a full core, they are running at the
same time; if only one is busy at a time, they are not.

    thread-cpu-sample.py --seconds 10 [--pattern kyty_emulator]
"""
import argparse
import os
import sys
import time

HZ = os.sysconf('SC_CLK_TCK')


def snapshot(pid):
    threads = {}
    for tid in os.listdir(f'/proc/{pid}/task'):
        try:
            stat = open(f'/proc/{pid}/task/{tid}/stat').read()
            name = stat[stat.index('(') + 1:stat.rindex(')')]
            fields = stat[stat.rindex(')') + 2:].split()
            threads[tid] = (name, int(fields[11]) + int(fields[12]))
        except (OSError, ValueError):
            continue
    return threads


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--pattern', default='kyty_emulator')
    parser.add_argument('--top', type=int, default=8)
    arguments = parser.parse_args()

    # Match on the resolved executable, not the command line: a `pgrep -f` for
    # the binary path also matches this sampler's own argv and the shell that
    # launched it, which is how an earlier version reported 0 cores busy.
    pid = None
    for entry in sorted(os.listdir('/proc'), key=lambda name: name.isdigit() and int(name) or 0):
        if not entry.isdigit():
            continue
        try:
            if arguments.pattern in os.readlink(f'/proc/{entry}/exe'):
                pid = entry
        except OSError:
            continue
    if pid is None:
        print(f'no running process whose executable matches {arguments.pattern!r}',
              file=sys.stderr)
        return 1

    before = snapshot(pid)
    start = time.monotonic()
    time.sleep(arguments.seconds)
    after = snapshot(pid)
    wall = time.monotonic() - start

    rows = sorted(((after[tid][1] - before[tid][1]) / HZ, after[tid][0])
                  for tid in after if tid in before)[::-1]
    total = sum(seconds for seconds, _ in rows)
    print(f'{wall:.1f}s window, total process CPU {total:.1f}s = {total / wall:.2f} cores busy')
    for seconds, name in rows[:arguments.top]:
        if seconds < 0.05:
            continue
        print(f'  {name:20}{seconds:7.2f}s{100 * seconds / wall:6.0f}% of a core')
    return 0


if __name__ == '__main__':
    sys.exit(main())
