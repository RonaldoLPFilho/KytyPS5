#!/usr/bin/env python3
"""Short live-process FPS screening; optional camera or forward-walk setup.

No perf collection or binary archival. Frame counts measure presentation, not
simulation speed or correctness. Equal camera input does not guarantee identical
views; use --camera-frames 0 for stationary A/B and inspect screenshots. The
outbound turn is frame bounded; the return matches its measured wall time,
since the game's camera speed need not be tied to presented frame counts.
Forward setup holds W for the requested wall time, releases it, and settles for
10 seconds. Its moving phase is not a stationary FPS measurement.
"""
import argparse
import ctypes as C
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--seconds', type=float, default=5, help='Seconds per stationary view (2–60)')
    p.add_argument('--camera-frames', type=int, default=0, help='Outbound H frames, then F for equal elapsed time; default 0 sends no input (0–30)')
    p.add_argument('--forward-seconds', type=float, default=0,
                   help='Setup only: hold W for this wall time, then release and settle 10s (0–10; default 0 sends no input)')
    p.add_argument('--settle-seconds', type=float, default=10,
                   help='Seconds to settle after the forward setup (default 10; iteration runs use less)')
    p.add_argument('--label', default='quick')
    p.add_argument('--screenshots', action='store_true', help='Capture each sampled view outside its timing interval')
    a = p.parse_args()
    if not math.isfinite(a.seconds) or not 2 <= a.seconds <= 60 or not 0 <= a.camera_frames <= 30:
        p.error('Invalid duration or camera frame count')
    if not math.isfinite(a.forward_seconds) or not 0 <= a.forward_seconds <= 10:
        p.error('Forward setup must be 0–10 seconds')
    if a.forward_seconds and a.camera_frames:
        p.error('Forward setup and camera setup are mutually exclusive')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', a.label):
        p.error('Invalid label')
    proc = Path('/proc') / str(a.pid)
    if Path(os.readlink(proc / 'exe')).name.removesuffix(' (deleted)') != 'kyty_emulator':
        p.error('Expected a live kyty_emulator')
    identity = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    binary_stat = (proc / 'exe').stat()
    lock_dir = ROOT / '_Build/automation'
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = (lock_dir / 'input.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
    windows = re.findall(r'(0x[0-9a-f]+) "[^\n]*frame: \d+, fps: [0-9.]+"[^\n]*\("kyty_emulator"', tree)
    matches = []
    for candidate in windows:
        prop = subprocess.check_output(['xprop', '-id', candidate, '_NET_WM_PID'], text=True, timeout=5)
        if re.search(rf'= {a.pid}\s*$', prop):
            matches.append(int(candidate, 16))
    if len(matches) != 1:
        raise RuntimeError('Expected one frame-counter window belonging to the supplied PID')
    window = matches[0]
    x, t = C.CDLL('libX11.so.6'), C.CDLL('libXtst.so.6')
    x.XOpenDisplay.restype = C.c_void_p
    x.XFetchName.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_void_p)]
    x.XFree.argtypes = [C.c_void_p]
    x.XCloseDisplay.argtypes = [C.c_void_p]
    x.XSetInputFocus.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_ulong]
    x.XGetInputFocus.argtypes = [C.c_void_p, C.POINTER(C.c_ulong), C.POINTER(C.c_int)]
    x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
    x.XKeysymToKeycode.restype = C.c_uint
    x.XFlush.argtypes = [C.c_void_p]
    # The default Xlib handler exits the interpreter on a destroyed window,
    # bypassing finally/keyup. Convert protocol errors into Python failures.
    error_type = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_void_p)
    x_errors = []
    @error_type
    def x_error(display, event):
        x_errors.append(True)
        return 0
    x.XSetErrorHandler.argtypes = [error_type]
    x.XSetErrorHandler(x_error)
    t.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
    display = x.XOpenDisplay(None)
    if not display:
        raise RuntimeError('No X11 display')
    out = ROOT / '_Build/profiles' / (datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + a.label)
    out.mkdir(parents=True)
    result = vars(a) | {'status': 'running', 'gameplay_verified': False,
        'pid_start_ticks': identity, 'window': hex(window), 'phases': [],
        'binary': {'path': os.readlink(proc / 'exe'), 'inode': binary_stat.st_ino,
                   'size': binary_stat.st_size, 'mtime_ns': binary_stat.st_mtime_ns},
        'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'working_tree': subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True),
        'command': (proc / 'cmdline').read_bytes().decode().split('\0')[:-1],
        'environment': [s for s in (proc / 'environ').read_bytes().decode().split('\0') if s.startswith('KYTY_')],
        'note': 'Live debug toggles are not reflected in environment; retain toggle JSON separately.'}
    held = set()
    started = time.monotonic()
    last_frame = None

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def sample():
        nonlocal last_frame
        if (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] != identity:
            raise RuntimeError('Process identity changed')
        name = C.c_void_p()
        try:
            if not x.XFetchName(display, window, C.byref(name)) or not name.value:
                raise RuntimeError('Window title unavailable')
            match = re.search(r'frame: (\d+), fps: ([0-9.]+)', C.string_at(name).decode())
        finally:
            if name.value:
                x.XFree(name)
        if not match:
            raise RuntimeError('Frame counter unavailable')
        frame = int(match[1])
        if x_errors:
            raise RuntimeError('X11 window became unavailable')
        if last_frame is not None and frame < last_frame:
            raise RuntimeError('Frame counter reset')
        last_frame = frame
        now = time.monotonic_ns()
        return {'seconds': now / 1e9 - started, 'monotonic_ns': now,
                'frame': frame, 'title_fps': float(match[2])}

    def phase(label, key=None, hold_seconds=None):
        values = [sample()]
        entry = {'name': label, 'key': key, 'samples': values,
                 'role': 'setup' if key else 'measurement'}
        result['phases'].append(entry)
        code = x.XKeysymToKeycode(display, ord(key)) if key else None
        try:
            if code:
                held.add(code)
                t.XTestFakeKeyEvent(display, code, 1, 0)
                x.XFlush(display)
                entry['keydown_ns'] = time.monotonic_ns()
            while True:
                time.sleep(.01 if key else .1)
                if key:
                    focus, revert = C.c_ulong(), C.c_int()
                    x.XGetInputFocus(display, C.byref(focus), C.byref(revert))
                    if focus.value != window:
                        raise RuntimeError('Focus changed during setup input')
                values.append(sample())
                elapsed = values[-1]['seconds'] - values[0]['seconds']
                frames = values[-1]['frame'] - values[0]['frame']
                if ((key and (elapsed >= hold_seconds if hold_seconds is not None else frames >= a.camera_frames))
                        or (not key and elapsed >= a.seconds)):
                    break
                if key and elapsed >= 10:
                    raise RuntimeError('Setup input exceeded 10 seconds; stalled or too slow')
        finally:
            if code:
                t.XTestFakeKeyEvent(display, code, 0, 0)
                x.XFlush(display)
                entry['keyup_ns'] = time.monotonic_ns()
                if 'keydown_ns' in entry:
                    entry['held_seconds'] = (entry['keyup_ns'] - entry['keydown_ns']) / 1e9
                held.discard(code)
        entry.update(elapsed_seconds=elapsed, frames=frames, measured_fps=frames / elapsed,
                     first_frame=values[0]['frame'], last_frame=values[-1]['frame'],
                     start_ns=values[0]['monotonic_ns'], end_ns=values[-1]['monotonic_ns'])
        changed_at = values[0]['seconds']
        longest = 0.0
        for before, after in zip(values, values[1:]):
            longest = max(longest, after['seconds'] - changed_at)
            if after['frame'] != before['frame']:
                changed_at = after['seconds']
        entry['longest_observed_counter_gap_seconds'] = longest
        if not key and a.screenshots:
            subprocess.run(['import', '-window', hex(window), '-resize', '1280x720>',
                            str(out / (label + '.png'))], check=True, timeout=10,
                           env=os.environ | {'MAGICK_THREAD_LIMIT': '2'})

    try:
        if a.forward_seconds:
            if a.screenshots:
                subprocess.run(['import', '-window', hex(window), '-resize', '1280x720>',
                                str(out / 'before-forward.png')], check=True, timeout=10,
                               env=os.environ | {'MAGICK_THREAD_LIMIT': '2'})
            x.XSetInputFocus(display, window, 1, 0)
            x.XFlush(display)
            phase('walk-forward', 'w', a.forward_seconds)
            time.sleep(a.settle_seconds)
        if a.camera_frames:
            x.XSetInputFocus(display, window, 1, 0)
            x.XFlush(display)
            phase('turn-right', 'h')
            time.sleep(1)
            phase('right-view')
            phase('turn-back', 'f', result['phases'][0]['elapsed_seconds'])
            time.sleep(1)
        phase('starting-view')
        result['status'] = 'completed_unverified'
    except BaseException as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        for code in held:
            t.XTestFakeKeyEvent(display, code, 0, 0)
        x.XFlush(display)
        x.XCloseDisplay(display)
        result['total_seconds'] = time.monotonic() - started
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'evidence': str(out), 'status': result['status'],
            'total_seconds': result['total_seconds'], 'views': [
                {k: v for k, v in entry.items() if k != 'samples'} for entry in result['phases']]}, indent=2), flush=True)


if __name__ == '__main__':
    main()
