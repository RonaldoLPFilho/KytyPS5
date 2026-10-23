#!/usr/bin/env python3
"""Sample the emulator window title during boot: it carries "frame: N, fps: M".

The frame counter is what separates an intro animation that *blocks* the boot
from one that is merely drawn while the level streams in.  If the emulator sits
at the 60 Hz vblank cap while a screen is up, that screen costs nothing and
removing it would only replace it with black; if the frame counter crawls, the
presentation really is in the way.

Round 11 used this to settle the question for the Demon's Souls loading screen -
see docs/compatibility/demons-souls-boot-patch.md.

    demons-souls-boot-frames.py [--seconds 90] [--interval 0.5]

Run it alongside ./run-boot-skip.sh; it prints one line per sample and a summary
of frames-per-second between samples at the end.
"""
import argparse
import re
import subprocess
import sys
import time

TITLE = re.compile(r'(0x[0-9a-f]+) "(\[Source build[^"]*)"')
COUNTER = re.compile(r'frame: (\d+), fps: (\d+)')


def window_title():
    try:
        tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
    except (subprocess.SubprocessError, OSError):
        return None
    match = TITLE.search(tree)
    return match.group(2) if match else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--interval', type=float, default=0.5)
    arguments = parser.parse_args(argv)

    start = time.monotonic()
    samples = []
    while time.monotonic() - start < arguments.seconds:
        title = window_title()
        if title:
            counter = COUNTER.search(title)
            if counter:
                elapsed = time.monotonic() - start
                frame, fps = int(counter.group(1)), int(counter.group(2))
                samples.append((elapsed, frame))
                print(f'{elapsed:6.1f}s frame={frame:<6d} fps={fps}', flush=True)
        time.sleep(arguments.interval)

    if len(samples) < 2:
        print('no frame counter seen; was the game running?', file=sys.stderr)
        return 1
    span = samples[-1][0] - samples[0][0]
    drawn = samples[-1][1] - samples[0][1]
    print(f'\n{drawn} frames over {span:.1f}s = {drawn / span:.1f} fps average', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
