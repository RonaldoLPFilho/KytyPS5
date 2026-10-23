#!/usr/bin/env python3
"""Drive the English Demon's Souls new-game flow in Kyty's X11 window.
Requires ImageMagick import, X11/XTest and Tesseract English OCR. Extracted
Tesseract packages in _Build/debug-tools/ocr are also supported. Uses Kyty's
DEFAULT keyboard mapping. No game files or saves are modified by this script.
The game itself creates the selected new character. Ctrl-C releases all keys.
"""
import argparse
import ctypes as C
import datetime
import os
from pathlib import Path
import re
import shutil
import signal
import fcntl
import json
import csv
import io
import subprocess
import time
from PIL import Image, ImageOps, ImageStat

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--stop-at', choices=['character', 'smoke', 'gameplay'], default='smoke')
p.add_argument('--fast-skip', action='store_true', help='Poll faster while retaining long intro-skip presses; omit OCR while waiting for the gameplay HUD')
p.add_argument('--stall-seconds', type=int, default=0, help='Stop input if the presentation counter stays unchanged this long; 0 disables (does not classify a crash)')
p.add_argument('--resume', action='store_true', help='Choose Continue from an existing save instead of creating a character')
p.add_argument('--observe-gameplay', action='store_true', help='Only watch for the gameplay HUD in an already loading/running game; send no input')
p.add_argument('--timeout', type=float, default=600)
p.add_argument('--pid', type=int, help='Only control the window belonging to this emulator process')
p.add_argument('--name', default='debug')
p.add_argument('--marker-before-finalise', type=Path, help='Write an explicit diagnostic trigger immediately before finalising the character')
a = p.parse_args()
if a.stall_seconds < 0:
    p.error('--stall-seconds must be nonnegative')
if a.observe_gameplay and a.stop_at != 'gameplay':
    p.error('--observe-gameplay requires --stop-at gameplay')
if not re.fullmatch('[a-z]{1,12}', a.name):
    p.error('--name must be 1–12 lowercase ASCII letters')
ocr_root = ROOT / '_Build/debug-tools/ocr'
ocr = shutil.which('tesseract') or str(ocr_root / 'usr/bin/tesseract')
if not Path(ocr).is_file():
    p.error('Tesseract missing; install tesseract-ocr and tesseract-ocr-eng, or extract their packages under _Build/debug-tools/ocr')
env = os.environ.copy()
if ocr.startswith(str(ocr_root)):
    env['LD_LIBRARY_PATH'] = str(ocr_root / 'usr/lib/x86_64-linux-gnu') + ':' + env.get('LD_LIBRARY_PATH', '')
    env['TESSDATA_PREFIX'] = str(ocr_root / 'usr/share/tesseract-ocr/5/tessdata')
out = ROOT / '_Build/automation' / datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
out.mkdir(parents=True)
x, t = C.CDLL('libX11.so.6'), C.CDLL('libXtst.so.6')
x.XOpenDisplay.restype = C.c_void_p
x.XCloseDisplay.argtypes = [C.c_void_p]
x.XSetInputFocus.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_ulong]
x.XRaiseWindow.argtypes = [C.c_void_p, C.c_ulong]
x.XFlush.argtypes = [C.c_void_p]
x.XKeysymToKeycode.argtypes = [C.c_void_p, C.c_ulong]
x.XKeysymToKeycode.restype = C.c_uint
t.XTestFakeKeyEvent.argtypes = [C.c_void_p, C.c_uint, C.c_int, C.c_ulong]
d = x.XOpenDisplay(None)
if not d:
    raise SystemExit('No X11 display')
held = set()

def press(key=ord('j'), duration=.15, *, resume_prompt=False):
    # Continue has no cinematic to skip. A blind held Cross can reach gameplay
    # and open gestures; closing it can also change the saved entry view.
    if a.resume and finalised and not resume_prompt:
        raise RuntimeError('Input after Continue requires an observed loading prompt')
    if resume_prompt and (key != ord('j') or duration > .15):
        raise RuntimeError('Only a short confirmation is allowed for the loading prompt')
    event = {'monotonic_ns': time.monotonic_ns(), 'key': key, 'duration': duration,
             'after_continue': a.resume and finalised, 'resume_prompt': resume_prompt}
    x.XRaiseWindow(d, window)
    x.XSetInputFocus(d, window, 1, 0)
    k = x.XKeysymToKeycode(d, key)
    t.XTestFakeKeyEvent(d, k, 1, 0); held.add(k); x.XFlush(d)
    try:
        time.sleep(duration)
    finally:
        t.XTestFakeKeyEvent(d, k, 0, 0); held.discard(k); x.XFlush(d)
        event['released_ns'] = time.monotonic_ns()
        with (out / 'input.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
    time.sleep(.25)

def log(message):
    line = f'{time.monotonic()-start:6.1f}s {message}'
    print(line, flush=True)
    with (out / 'events.txt').open('a') as f:
        f.write(line + '\n')
    with (out / 'events.jsonl').open('a') as f:
        f.write(json.dumps({'monotonic_ns': time.monotonic_ns(), 'unix_ns': time.time_ns(),
                            'pid': a.pid, 'event': message}) + '\n')

def select_main_menu(shot, frame):
    # Menu order is Continue, Load Game, New Game, Settings when saves exist.
    # Held arrows wrap; observe the highlighted row before confirming instead.
    tsv = subprocess.check_output([ocr, str(shot), 'stdout', '--psm', '11', 'tsv'],
        env=env, stderr=subprocess.DEVNULL, text=True, timeout=10)
    rows = {}
    for word in csv.DictReader(io.StringIO(tsv), delimiter='\t'):
        label = word['text'].lower().strip()
        label = {'loadgame': 'load', 'newgame': 'new'}.get(label, label)
        if label not in ('continue', 'load', 'new', 'settings'):
            continue
        xx, yy, ww, hh = (int(word[k]) for k in ('left', 'top', 'width', 'height'))
        if xx > frame.width * .5 or not frame.height * .25 < yy < frame.height * .7:
            continue
        center = yy + hh // 2
        glow = frame.crop((int(frame.width*.21), max(0, center-10),
                           int(frame.width*.357), min(frame.height, center+10)))
        pixels = [glow.getpixel((gx, gy)) for gy in range(glow.height) for gx in range(glow.width)]
        rows[label] = sum(max(0, min(g, b)-r) for r,g,b in pixels) / len(pixels)
    target = 'continue' if a.resume else 'new'
    if target not in rows:
        log(f'Waiting for main-menu target {target} to become readable: ' + str(rows))
        return False
    ranking = sorted(rows, key=rows.get, reverse=True)
    if rows[ranking[0]] < 5 or (len(ranking) > 1 and rows[ranking[0]]-rows[ranking[1]] < 3):
        log('Waiting for an unambiguous main-menu highlight: ' + str(rows))
        return False
    selected = ranking[0]
    if selected == target:
        log(f'Main menu verified: {target}; confirming.')
        press()
        return True
    log(f'Main menu highlight is {selected}; moving one row toward {target}.')
    press(0xff54)
    return False

lock = (ROOT / '_Build/automation/input.lock').open('w')
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('Another automation run owns the game input')
def stop(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
start = time.monotonic()
pid_identity = ((Path('/proc') / str(a.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
                if a.pid else None)
counter = None
counter_changed = start
window = None
named = False
finalised = a.observe_gameplay
gray_frames = 0
hud_frames = 0
waiting_for_gameplay = a.observe_gameplay
last_unknown_key = 0
iteration = 0
try:
    while time.monotonic() - start < a.timeout:
        tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
        candidates = list(re.finditer(r'(0x[0-9a-f]+) "\[Source build[^\n]*Demon[^\n]*\("kyty_emulator"', tree))
        if a.pid:
            current_identity = (Path('/proc') / str(a.pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            if current_identity != pid_identity:
                raise SystemExit('Owned emulator identity changed')
            candidates = [item for item in candidates if re.search(r'=\s*' + str(a.pid) + r'\s*$',
                subprocess.check_output(['xprop', '-id', item[1], '_NET_WM_PID'], text=True, timeout=5))]
        if len(candidates) > 1:
            raise SystemExit('Multiple game windows found; use --pid to select the owned process')
        match = candidates[0] if candidates else None
        if not match:
            if window:
                raise SystemExit('Emulator window closed before completion; see ' + str(out))
            time.sleep(2); continue
        window = int(match[1], 16)
        counter_match = re.search(r'frame: (\d+), fps:', match[0])
        if counter_match:
            current_counter = int(counter_match[1])
            if counter != current_counter:
                counter, counter_changed = current_counter, time.monotonic()
            elif a.stall_seconds and time.monotonic() - counter_changed >= a.stall_seconds:
                log(f'Presentation counter {counter} unchanged for {a.stall_seconds}s; stopping input, not classifying a crash.')
                raise SystemExit('Presentation stalled; last screenshots and OCR: ' + str(out))
        shot = out / f'{iteration:04d}.png'
        # OCR/HUD coordinates were designed for 720p. Avoid encoding full 4K PNGs
        # on every poll, which can time out and contend with the running game.
        subprocess.run(['import', '-window', hex(window), '-resize', '1280x720>', str(shot)],
                       check=True, timeout=10)
        iteration += 1
        frame = Image.open(shot).convert('RGB')
        stats = ImageStat.Stat(frame)
        gray = max(stats.mean) - min(stats.mean) < 2 and 25 < stats.mean[0] < 210 and 3 < stats.stddev[0] < 45
        gray_frames = gray_frames + 1 if finalised and gray else 0
        # HUD detection is an endpoint for resume automation, not visual correctness.
        if finalised:
            red_rows = 0
            green_rows = 0
            for y in range(5, min(100, frame.height // 6)):
                row = [frame.getpixel((xx, y)) for xx in range(40, min(300, frame.width // 3))]
                red_rows += sum(r > 60 and r > g * 1.5 and r > b * 1.5 for r, g, b in row) > 80
                green_rows += sum(g > 35 and g > r * 1.25 and g > b * 1.25 for r, g, b in row) > 80
            hud_frames = hud_frames + 1 if red_rows >= 2 and green_rows >= 2 else 0
            if hud_frames >= 2:
                if a.resume and not a.observe_gameplay:
                    # Reject an unexpected overlay instead of changing gameplay
                    # state to conceal input that invalidated the saved entry.
                    overlay = frame.crop((int(frame.width * .76), int(frame.height * .13),
                                          frame.width, int(frame.height * .25)))
                    overlay = ImageOps.autocontrast(ImageOps.grayscale(overlay)).resize((720, 240))
                    overlay_path = out / 'hud-overlay.png'
                    overlay.save(overlay_path)
                    overlay_text = subprocess.check_output([ocr, str(overlay_path), 'stdout', '--psm', '6'],
                        env=env, stderr=subprocess.DEVNULL, text=True, timeout=10).lower()
                    if 'gestures' in overlay_text or 'do nothing' in overlay_text:
                        raise RuntimeError('Unexpected gesture selector after Continue; entry measurement invalid')
                log('Gameplay HUD detected; movement and image correctness still need verification.')
                break
        if gray_frames >= 3:
            if a.stop_at != 'gameplay':
                log('Gray smoke transition detected; this does not verify gameplay.')
                break
            if not waiting_for_gameplay:
                log('Gray smoke transition detected; waiting for gameplay HUD.')
                waiting_for_gameplay = True
        if waiting_for_gameplay:
            if (not a.observe_gameplay and not a.resume and not gray and hud_frames == 0 and
                    time.monotonic() - last_unknown_key > 8):
                log('Skipping the post-load cinematic with held Cross.')
                press(duration=2.5)
                last_unknown_key = time.monotonic()
            time.sleep(.5 if a.fast_skip else 2)
            continue
        text = subprocess.check_output([ocr, str(shot), 'stdout', '--psm', '11'], env=env, stderr=subprocess.DEVNULL, text=True, timeout=10).lower()
        menu = Image.open(shot)
        menu = menu.crop((0, 0, int(menu.width * .28), menu.height)).resize((720, 1440))
        menu = ImageOps.autocontrast(ImageOps.grayscale(menu))
        menu.save(out / 'menu.png')
        text += '\n' + subprocess.check_output([ocr, str(out / 'menu.png'), 'stdout', '--psm', '6'], env=env, stderr=subprocess.DEVNULL, text=True, timeout=10).lower()
        shot.with_suffix('.txt').write_text(text)
        if 'outpost' in text or 'passage' in text:
            if not a.resume and not named and not a.observe_gameplay:
                raise SystemExit('Reached a saved game without creating the requested new character')
            log('Playable-area title detected.')
            if a.stop_at != 'gameplay':
                break
            finalised = True
            waiting_for_gameplay = True
            continue
        if 'journey' in text or ('nexus' in text and 'would you' in text):
            log('Tutorial Yes/No prompt detected.')
            press()
        elif "enter your player's name" in ' '.join(text.split()):
            log('Entering test character name.')
            press(0xff08, 2)
            for c in a.name: press(ord(c), .06)
            press(0xff0d)
            named = True
        elif 'finalise' in text or 'finalize' in text or 'saved creations' in text:
            if not named and ('name' in text):
                log('Opening character name input.')
                press(0xff52, 2.5); press()
            else:
                log('Character settings detected.')
                if a.stop_at == 'character': break
                if a.marker_before_finalise:
                    a.marker_before_finalise.write_text('character-finalise\n')
                    log('Diagnostic trigger written: ' + str(a.marker_before_finalise))
                press(0xff54, 3); press()
                finalised = True
                time.sleep(4)
        elif 'body type' in text or 'bodytype' in text or ('type a' in text and 'type b' in text):
            log('Choosing default body type.'); press()
        elif 'offline' in text:
            log('Continuing offline.'); press(resume_prompt=a.resume and finalised)
            if a.resume and finalised:
                log('Continue confirmed; observing loading and gameplay without further input.')
                waiting_for_gameplay = True
        elif 'new game' in text or (a.resume and 'load game' in text and 'settings' in text):
            if a.resume and ('continue' in text or ('load game' in text and 'settings' in text)):
                if select_main_menu(shot, frame):
                    finalised = True
            elif a.resume:
                raise SystemExit('No Continue option detected; existing save required for --resume')
            else:
                select_main_menu(shot, frame)
        elif 'press any' in text:
            log('Title screen.'); press()
        elif 'language' in text and 'english' in text:
            log('Confirming English.'); press()
        elif a.resume and finalised:
            # Keep OCR active until a possible offline prompt has been handled.
            # Unknown loading screens must never trigger a gameplay button.
            pass
        elif time.monotonic() - last_unknown_key > (1 if a.fast_skip else 7):
            log('Waiting/skipping startup or cutscene: ' + ' '.join(text.split())[:100])
            press(duration=2 if finalised else 2.5); last_unknown_key = time.monotonic()
        time.sleep(.5 if a.fast_skip else 2)
    else:
        raise SystemExit('Timed out; screenshots and OCR: ' + str(out))
finally:
    for k in held: t.XTestFakeKeyEvent(d, k, 0, 0)
    x.XFlush(d); x.XCloseDisplay(d)
print('Evidence: ' + str(out), flush=True)
