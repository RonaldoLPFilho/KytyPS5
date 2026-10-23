#!/usr/bin/env python3
"""Finish the first-boot prompts so Demon's Souls lands in gameplay unattended.

``demons-souls-boot-skip.py`` makes the engine leave the splash screens, the
opening cinematic, the title screen and the main menu behind by booting in the
DemoLoopA game flow, which drops straight into a new game.  What is left are the
one-off prompts that a new game always asks for:

    language select -> character creation -> offline confirmation -> loading

This driver answers those (and skips the post-load cinematic) with injected
controller input, then stops as soon as the gameplay HUD appears.  It only ever
sends buttons; it never reads or writes game files.

Cross is ``j`` in Kyty's default keyboard map; Down is the Down arrow.
"""
import argparse
import ctypes as C
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]

DOWN = 0xff54
CROSS = ord('j')

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--timeout', type=float, default=420)
parser.add_argument('--pid', type=int, help='only drive the window of this emulator process')
arguments = parser.parse_args()

ocr_root = ROOT / '_Build/debug-tools/ocr'
ocr = shutil.which('tesseract') or str(ocr_root / 'usr/bin/tesseract')
if not Path(ocr).is_file():
    sys.exit('Tesseract missing; install tesseract-ocr / tesseract-ocr-eng')

env = os.environ.copy()
if ocr.startswith(str(ocr_root)):
    env['LD_LIBRARY_PATH'] = str(ocr_root / 'usr/lib/x86_64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
    env['TESSDATA_PREFIX'] = str(ocr_root / 'usr/share/tesseract-ocr/5/tessdata')

x = C.CDLL('libX11.so.6')
t = C.CDLL('libXtst.so.6')
x.XOpenDisplay.restype = C.c_void_p
x.XFlush.argtypes = [C.c_void_p]
x.XRaiseWindow.argtypes = [C.c_void_p, C.c_ulong]
x.XSetInputFocus.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_ulong]
x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
x.XKeysymToKeycode.restype = C.c_uint
t.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
display = x.XOpenDisplay(None)
if not display:
    sys.exit('no X11 display')

work = ROOT / '_Build' / 'boot-skip' / 'quickstart'
work.mkdir(parents=True, exist_ok=True)


def window_id():
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
    pattern = r'(0x[0-9a-f]+) "\[Source build[^\n]*Demon[^\n]*\("kyty_emulator"'
    for match in re.finditer(pattern, tree):
        if arguments.pid is None:
            return int(match.group(1), 16)
        prop = subprocess.check_output(['xprop', '-id', match.group(1), '_NET_WM_PID'],
                                       text=True, timeout=5)
        if re.search(r'=\s*' + str(arguments.pid) + r'\s*$', prop):
            return int(match.group(1), 16)
    return None


def press(window, key, duration=0.15):
    window_raise(window)
    code = x.XKeysymToKeycode(display, key)
    t.XTestFakeKeyEvent(display, code, 1, 0)
    x.XFlush(display)
    try:
        time.sleep(duration)
    finally:
        t.XTestFakeKeyEvent(display, code, 0, 0)
        x.XFlush(display)


def window_raise(window):
    x.XRaiseWindow(display, window)
    x.XSetInputFocus(display, window, 1, 0)
    x.XFlush(display)


def screenshot(window, path):
    subprocess.run(['import', '-window', hex(window), '-resize', '1280x720>', str(path)],
                   check=True, timeout=15)
    return Image.open(path).convert('RGB')


def read_text(path):
    return subprocess.check_output([ocr, str(path), 'stdout', '--psm', '11'],
                                   env=env, stderr=subprocess.DEVNULL, text=True,
                                   timeout=10).lower()


def screen_text(shot, frame):
    """Full-screen OCR plus a contrast-stretched crop of the left menu column.

    The character-creation buttons sit on a dark panel that the plain full-screen
    pass reads poorly, which is exactly where the Finalise label lives.
    """
    menu = frame.crop((0, 0, int(frame.width * .28), frame.height))
    menu = ImageOps.autocontrast(ImageOps.grayscale(menu)).resize((720, 1440))
    menu_path = shot.with_name(shot.stem + '-menu.png')
    menu.save(menu_path)
    return ' '.join((read_text(shot) + ' ' + read_text(menu_path)).split())


def gameplay_hud(frame):
    red = green = 0
    for y in range(5, min(100, frame.height // 6)):
        row = [frame.getpixel((xx, y)) for xx in range(40, min(300, frame.width // 3))]
        red += sum(r > 60 and r > g * 1.5 and r > b * 1.5 for r, g, b in row) > 80
        green += sum(g > 35 and g > r * 1.25 and g > b * 1.25 for r, g, b in row) > 80
    return red >= 2 and green >= 2


start = time.monotonic()
window = None
finalised = False
body_done = False
settings_done = False
saw_character_creation = False
hud_frames = 0
last_blind = 0.0
language_done = False
iteration = 0

print('waiting for the emulator window...', flush=True)
while time.monotonic() - start < arguments.timeout:
    if window is None:
        window = window_id()
        if window is None:
            time.sleep(2)
            continue
        print(f'{time.monotonic() - start:6.1f}s window={hex(window)}', flush=True)

    shot = work / f'{iteration:04d}.png'
    iteration += 1
    frame = screenshot(window, shot)
    text = screen_text(shot, frame)
    if 'character creation' in text or 'body type' in text or 'finalise' in text or 'finalize' in text:
        saw_character_creation = True

    # The HUD is the finish line.  Require three consecutive frames so an intro
    # frame with red/green pixels cannot end the run early; do NOT require that
    # character creation was seen, because the boot-skip patch removes it.
    if gameplay_hud(frame):
        hud_frames += 1
        if hud_frames >= 3:
            print(f'{time.monotonic() - start:6.1f}s gameplay HUD detected', flush=True)
            print('QUICKSTART_OK', flush=True)
            sys.exit(0)
        time.sleep(1)
        continue
    hud_frames = 0

    if 'language' in text and not language_done:
        print(f'{time.monotonic() - start:6.1f}s confirming language', flush=True)
        press(window, CROSS)
        language_done = True
        time.sleep(2)
        continue

    if not settings_done and ('adjust settings' in text or 'brightness' in text
                              or ('contrast' in text and 'saturation' in text)):
        # First-boot brightness/contrast calibration; default sliders are fine.
        print(f'{time.monotonic() - start:6.1f}s confirming brightness settings', flush=True)
        press(window, CROSS)
        settings_done = True
        time.sleep(2)
        continue

    if not body_done and ('body type' in text or 'bodytype' in text):
        # The default body is what a normal confirm would pick.
        print(f'{time.monotonic() - start:6.1f}s accepting the default body type', flush=True)
        press(window, CROSS)
        body_done = True
        time.sleep(2)
        continue

    if not finalised and ('finalise' in text or 'finalize' in text or 'saved creations' in text):
        # The default class/name/gift are already sensible; walk the cursor down
        # to the Finalise button and commit.
        print(f'{time.monotonic() - start:6.1f}s finalising the default character', flush=True)
        press(window, DOWN, 3.0)
        time.sleep(0.5)
        press(window, CROSS)
        finalised = True
        time.sleep(5)
        continue

    if finalised and 'offline' in text:
        print(f'{time.monotonic() - start:6.1f}s confirming offline mode', flush=True)
        press(window, CROSS)
        time.sleep(2)
        continue

    # Nothing recognisable: nudge the game forward (startup logos, cinematics,
    # loading screens) but do not spam once the character exists.
    if time.monotonic() - last_blind > (3 if not finalised else 15):
        print(f'{time.monotonic() - start:6.1f}s skipping: {text[:70]}', flush=True)
        press(window, CROSS, 1.0)
        last_blind = time.monotonic()
    time.sleep(0.5)

print('QUICKSTART_TIMEOUT', flush=True)
sys.exit(2)
