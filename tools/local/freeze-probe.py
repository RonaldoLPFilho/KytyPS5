#!/usr/bin/env python3
"""Find which performance path causes a visual glitch that is on screen right now.

    freeze-probe.py [--settle 1.5] [--only LABEL,LABEL]

Run it while the glitch is visible (for example the frozen character model, or the
spot where the invisible character stands) and leave the controller alone until it
finishes (about one minute). It toggles one suspect switch at a time in the running
game, takes a screenshot after each change, restores the switch, and writes the
screenshots plus a contact sheet to _Build/freeze-probe/<time>/. The screenshot in
which the glitch disappears names the culprit.

The game must have been started with KYTY_LIVE_FILE, e.g.
    python3 tools/local/play-demons-souls.py --checkpoint-config _Build/release-33fps-fix-20260926/launch-probe.json
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / '_Build/layer-bench'

# (label, [(symbol, test value), ...]); every probe is restored before the next one.
PROBES = [
    ('native-xpr-draws-skipped', [('kyty_local_native_xpr_debug', 2)]),
    ('native-xpr-off', [('kyty_local_native_xpr_mode', 0)]),
    ('33fps-switches-off', [('kyty_local_async_write_readback_mode', 0), ('kyty_local_readback_detach_mode', 0),
                            ('kyty_local_write_window_handoff_mode', 0)]),
    ('async-write-readback-off', [('kyty_local_async_write_readback_mode', 0)]),
    ('readback-detach-off', [('kyty_local_readback_detach_mode', 0)]),
    ('copy-feedback-off', [('kyty_local_copy_feedback_mode', 0)]),
    ('frame-pipeline-off', [('kyty_local_frame_pipeline_mode', 0)]),
    ('backing-read-off', [('kyty_local_backing_read_mode', 0)]),
    ('buffer-residency-off', [('kyty_local_buffer_residency_mode', 0)]),
    ('draw-run-ranges-off', [('kyty_local_draw_run_ranges_mode', 0)]),
    ('stream-upload-off', [('kyty_local_stream_upload_mode', 0)]),
    ('srt-native-off', [('kyty_local_srt_native_mode', 0)]),
    ('srt-predicates-off', [('kyty_local_srt_predicate_mode', 0)]),
    ('specialization-guard-off', [('kyty_local_specialization_guard_mode', 0)]),
    ('preparation-lookup-off', [('kyty_local_preparation_lookup_mode', 0)]),
    ('preparation-trim-off', [('kyty_local_preparation_trim_mode', 0)]),
    ('binding-scratch-off', [('kyty_local_binding_scratch_mode', 0)]),
    ('pipeline-index-off', [('kyty_local_pipeline_index_mode', 0)]),
    ('async-lod-stats-off', [('kyty_local_async_lod_stats_mode', 0)]),
    ('image-barrier-dedupe-off', [('kyty_local_image_barrier_dedupe', 0)]),
]

# --diag (diagnostic build only): re-upload what the tracker missed, then everything.
DIAG_PROBES = [
    ('invalidate-all-buffers', [('kyty_local_diag_invalidate_all', 1)]),
]


class Game:
    def __init__(self):
        pids = subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True, text=True).stdout.split()
        if len(pids) != 1:
            raise SystemExit(f'expected one running kyty_emulator, found {len(pids)}')
        self.pid = int(pids[0])
        environ = Path(f'/proc/{self.pid}/environ').read_bytes().decode(errors='replace').split('\0')
        live = [item.split('=', 1)[1] for item in environ if item.startswith('KYTY_LIVE_FILE=')]
        if not live:
            raise SystemExit('the game was started without KYTY_LIVE_FILE; use launch-probe.json')
        self.live = Path(live[0])
        self.log = Path(os.readlink(f'/proc/{self.pid}/fd/1'))
        exe = os.readlink(f'/proc/{self.pid}/exe')
        nm = subprocess.run(['nm', '--defined-only', exe], capture_output=True, text=True).stdout
        self.symbols = {m[2]: int(m[1], 16) for m in re.finditer(r'^([0-9a-f]+) [bBdD] (kyty_local_\w+)$', nm, re.M)}
        self.base = None
        for line in Path(f'/proc/{self.pid}/maps').read_text().splitlines():
            fields = line.split()
            if len(fields) >= 6 and fields[5] == exe and int(fields[2], 16) == 0:
                self.base = int(fields[0].split('-')[0], 16)
                break
        if self.base is None:
            raise SystemExit('load base not found')

    def address(self, name):
        if name not in self.symbols:
            raise SystemExit(f'symbol {name} not in this binary')
        return self.base + self.symbols[name]

    def send(self, lines, timeout=5.0):
        text = self.live.read_text() if self.live.exists() else 'id 0'
        match = re.match(r'id (\d+)', text)
        command_id = (int(match[1]) if match else 0) + 1
        start = self.log.stat().st_size
        self.live.write_text(f'id {command_id}\n' + '\n'.join(lines) + '\n')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.log.open('rb') as log:
                log.seek(start)
                tail = log.read().decode(errors='replace')
            if f'LIVE_DONE id={command_id}' in tail:
                return [line for line in tail.splitlines() if f'id={command_id}' in line]
            time.sleep(0.05)
        raise SystemExit(f'the game did not answer command {command_id}')

    def read(self, name):
        address = self.address(name)
        for line in self.send([f'peek {address:#x} 4']):
            match = re.search(r'bytes=([0-9a-f]+)', line)
            if match:
                return int.from_bytes(bytes.fromhex(match[1][:8]), 'little')
        raise SystemExit(f'could not read {name}')

    def write(self, values):
        self.send([f'poke32 {self.address(name):#x} {value:x}' for name, value in values])


def screenshot(target):
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
    match = re.search(r'(0x[0-9a-f]+) "\[Source build', tree)
    if not match:
        raise SystemExit('game window not found')
    subprocess.run(['timeout', '-s', 'KILL', '10', 'import', '-window', match[1], '-resize', '1280x720', str(target)],
                   check=True)


def contact_sheet(files, target):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    width, height, columns = 640, 360, 4
    rows = (len(files) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * width, rows * height), 'black')
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(files):
        image = Image.open(path).convert('RGB').resize((width, height))
        x, y = (i % columns) * width, (i // columns) * height
        sheet.paste(image, (x, y))
        draw.rectangle([x, y, x + 300, y + 16], fill='black')
        draw.text((x + 3, y + 2), path.stem, fill='yellow')
    sheet.save(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--settle', type=float, default=1.5, help='seconds between a change and its screenshot')
    parser.add_argument('--only', help='comma-separated probe labels')
    parser.add_argument('--diag', action='store_true', help='diagnostic build: heal and invalidate probes only')
    args = parser.parse_args()
    probes = DIAG_PROBES if args.diag else PROBES
    if args.diag:
        args.settle = max(args.settle, 3.0)
    if args.only:
        wanted = set(args.only.split(','))
        probes = [probe for probe in PROBES if probe[0] in wanted]
    game = Game()
    out = ROOT / '_Build/freeze-probe' / time.strftime('%Y%m%d-%H%M%S')
    out.mkdir(parents=True)
    files = []
    try:
        shot = out / '00-before.png'
        screenshot(shot)
        files.append(shot)
        print(f'screenshots: {out}', flush=True)
        for index, (label, changes) in enumerate(probes, 1):
            original = [(name, game.read(name)) for name, _ in changes]
            try:
                game.write(changes)
                time.sleep(args.settle)
                shot = out / f'{index:02d}-{label}.png'
                screenshot(shot)
                files.append(shot)
                print(f'{index:02d} {label}', flush=True)
            finally:
                game.write(original)
            time.sleep(1.0)
        shot = out / f'{len(probes) + 1:02d}-after.png'
        screenshot(shot)
        files.append(shot)
    finally:
        # The live thread runs a file's commands once per new id, also right after a
        # later launch: never leave a poke (an address of this process) behind.
        game.send([])
    contact_sheet(files, out / 'sheet.png')
    print(f'done: {out}/sheet.png')


if __name__ == '__main__':
    main()
