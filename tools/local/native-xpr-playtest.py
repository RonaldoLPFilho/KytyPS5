#!/usr/bin/env python3
"""Drive a running live-bench game session with keyboard input (XTest).

    native-xpr-playtest.py hold KEY SECONDS [KEY SECONDS ...]   hold keys in turn
    native-xpr-playtest.py play MINUTES                          scripted walking/turning

Keys follow hostInput.cpp: w/a/s/d move, f/h turn the camera, t/g tilt it,
l circle (roll/dash), q/e L1/R1. `play` avoids menus and items. It prints the
frame counter from the window title once per action and fails when the
counter stops or the process disappears.
"""
import ctypes as C
import random
import re
import subprocess
import sys
import time
from pathlib import Path


def game_pid():
    out = subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True, text=True).stdout.split()
    if len(out) != 1:
        raise SystemExit('expected exactly one kyty_emulator')
    return int(out[0])


def game_window(pid):
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
    for candidate in re.findall(r'(0x[0-9a-f]+) "[^\n]*frame: \d+, fps: [0-9.]+"[^\n]*\("kyty_emulator"', tree):
        prop = subprocess.check_output(['xprop', '-id', candidate, '_NET_WM_PID'], text=True, timeout=5)
        if re.search(rf'= {pid}\s*$', prop):
            return int(candidate, 16)
    raise SystemExit('game window not found')


class Input:
    def __init__(self):
        self.pid    = game_pid()
        self.window = game_window(self.pid)
        self.x      = C.CDLL('libX11.so.6')
        self.t      = C.CDLL('libXtst.so.6')
        self.x.XOpenDisplay.restype = C.c_void_p
        self.x.XSetInputFocus.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_ulong]
        self.x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
        self.x.XKeysymToKeycode.restype = C.c_uint
        self.x.XFlush.argtypes = [C.c_void_p]
        self.x.XFetchName.argtypes = [C.c_void_p, C.c_ulong, C.POINTER(C.c_void_p)]
        self.x.XFree.argtypes = [C.c_void_p]
        self.t.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
        self.display = self.x.XOpenDisplay(None)
        if not self.display:
            raise SystemExit('no X11 display')
        self.held = set()

    def frame(self):
        if not Path(f'/proc/{self.pid}').exists():
            raise SystemExit('emulator exited')
        name = C.c_void_p()
        try:
            if not self.x.XFetchName(self.display, self.window, C.byref(name)) or not name.value:
                return None
            match = re.search(r'frame: (\d+), fps: ([0-9.]+)', C.string_at(name).decode())
        finally:
            if name.value:
                self.x.XFree(name)
        return (int(match[1]), float(match[2])) if match else None

    def hold(self, keys, seconds):
        self.x.XSetInputFocus(self.display, self.window, 1, 0)
        codes = [self.x.XKeysymToKeycode(self.display, ord(key)) for key in keys]
        try:
            for code in codes:
                self.held.add(code)
                self.t.XTestFakeKeyEvent(self.display, code, 1, 0)
            self.x.XFlush(self.display)
            time.sleep(seconds)
        finally:
            self.release()

    def release(self):
        for code in list(self.held):
            self.t.XTestFakeKeyEvent(self.display, code, 0, 0)
        self.held.clear()
        self.x.XFlush(self.display)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    game = Input()
    if sys.argv[1] == 'hold':
        pairs = sys.argv[2:]
        for key, seconds in zip(pairs[0::2], pairs[1::2]):
            game.hold(key, float(seconds))
        print('frame', game.frame())
        return
    if sys.argv[1] != 'play':
        raise SystemExit(__doc__)
    minutes = float(sys.argv[2])
    rng     = random.Random(20260925)
    actions = [('w', 1.5, 4.0), ('s', 1.0, 2.5), ('a', 0.8, 2.0), ('d', 0.8, 2.0), ('h', 0.4, 1.5),
               ('f', 0.4, 1.5), ('t', 0.2, 0.6), ('g', 0.2, 0.6), ('wh', 1.0, 2.5), ('wf', 1.0, 2.5),
               ('wl', 1.0, 3.0), ('l', 0.1, 0.2)]
    end        = time.monotonic() + minutes * 60
    last       = game.frame()
    last_moved = time.monotonic()
    step       = 0
    try:
        while time.monotonic() < end:
            keys, low, high = rng.choice(actions)
            game.hold(keys, rng.uniform(low, high))
            time.sleep(rng.uniform(0.2, 1.0))
            current = game.frame()
            now     = time.monotonic()
            if current and last and current[0] > last[0]:
                last_moved = now
            elif now - last_moved > 20:
                raise SystemExit(f'frame counter stalled at {current}')
            last = current or last
            step += 1
            if step % 20 == 0:
                print(f'{time.strftime("%H:%M:%S")} step {step} frame/fps {current}', flush=True)
    finally:
        game.release()
    print('play done, steps', step, 'frame/fps', game.frame())


if __name__ == '__main__':
    main()
