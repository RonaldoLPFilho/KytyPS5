#!/usr/bin/env python3
"""Derive and install the Demon's Souls guest-code patch that strips the first-boot
screens and the character creator out of the boot flow.

Installed patch
---------------
It is derived from the game's own `decrypted/eboot.bin`: the expected bytes are
read out of the file and compared before anything is generated, so a different
build refuses instead of being corrupted.  The cheat file also carries the title
id, version and process name, which the emulator validates before applying.

1. `remove the character-creation transition` - `0x7c0a4f`
   `FrontEndFlow` builds the transition into the character-creation step and calls
   `0xde27d0`:

       0x7c0a4f  mov  rdi,[r15]
       0x7c0a52  mov  eax,0xfa0a1f01
       0x7c0a57  lea  rsi,[0x2e9eb90]     ; FrontEndCharacterCreator step object
       0x7c0a66  call 0xde27d0            ; transition
       0x7c0a6b  inc  dword ptr [rbx+0xc]

   Replacing the 5-byte call with NOPs removes the creator entirely.

2. `skip the new-profile screens` - `0x7c9254`
   The front-end flow is a state machine: the flow object holds a handler table
   at `flow + state * 0x10` and every handler advances `[flow+0xc]` itself.  The
   first-boot branch is state 44 (`0x7c9190`), which copies the console's own
   language/subtitle settings into the profile and then hands over to the first
   of four screens:

       0x7c9254  mov dword ptr [rbx+0xc],0x2d   ; 45 = FE_NewProfileLanguage
                 ; 46 wait -> 49 FE_NewProfileBrightnessHdr -> 50 wait
                 ;         -> 47 FE_NewProfileSettings -> 48 wait -> 27

   Writing 27 (`0x1b`) instead of 45 lands on exactly the state that state 48
   hands control to when the player finishes those four screens, so all of them
   disappear at once and the flow continues where it normally would.  The
   console settings are still copied, because that happens before this store.

3. `remove the new-profile language screen` - `0x7c930c`
   The same transition shape as the character creator, in state 45.  Site 2
   means state 45 is never entered, so this is now only a backstop: if a build
   ever reaches the language screen by another edge (state 50 goes back to 45
   when the brightness screen is cancelled), it still does not appear.

Rejected candidate, kept for the record
---------------------------------------
At `0x3d137` boot reads a named setting `nointro` and hands `!value` to three
intro subsystems, with the not-found path defaulting the value to 1:

       0x3d137  lea  rsi,[0x23fcd4f]      ; "nointro"
       0x3d13e  call 0x818930             ; lookup by name
       0x3d143  test rax,rax
       0x3d146  je   0x3d15d
       0x3d154  call 0x8160b0             ; value
       0x3d159  xor  al,1                 ; intro enabled = !nointro
       0x3d15d  mov  al,1                 ; not-found path: enabled

`nointro` is not a registered convar (the name occurs once in the whole image and
its hash occurs nowhere), so the lookup always misses.  Patching the not-found
path to `xor al,al` was built and tested and **did not remove the opening
cinematic**, so it is not shipped.  The remaining intro is played through the UI
(`DsUIMovie` / `DsUILogo` are UI classes resolved at `0x78eddc`-`0x78ee15`) and the
movie resource names are not referenced from code at all, so it is not reachable
by patching this C++ transition the way the character creator was.

Five candidates for the remaining "On the first day" card were built, applied and
measured; every one of them left the screen and the timeline unchanged:

    0x7c11be  always take the boot flow's own intro-skip branch
    0x7c7eea  publish `HasPlayed_TutorialCinematic` as 1
    0x526c3   make the subtitle event listener ignore every "show" event
    0x50a90   make the subtitle-text UI listener return immediately
    0x55e498  stop the caption dialog picking the HistoryCinematicSubtitle scene

The card is drawn by the UI script layer, not by the C++ movie-subtitle pipeline:
gdb shows `0x55e498` is never reached during boot, and suppressing the subtitle
slot entirely does not remove the text.  `demons-souls-gdb-probe.py` reproduces
the trace; section 8 of the compatibility note has the details.

Usage
-----
    demons-souls-patch.py install [--game DIR]     # write the cheat JSON
    demons-souls-patch.py status  [--game DIR]
    demons-souls-patch.py remove  [--game DIR]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / 'tools' / 'local' / 'patches' / 'skip-boot-flow.json'

TITLE_ID = 'PPSA01341'
APP_VER = '01.007.000'
PROCESS = 'eboot.bin'
FILE_DELTA = 0x4000         # va = file offset - 0x4000 for this PT_LOAD layout


def nop(blob, offset, length):
    patched = bytearray(blob)
    patched[offset:offset + length] = b'\x90' * length
    return patched


def xor_al_al(blob, offset, _length):
    """mov al,1 -> xor al,al: force the boolean to zero."""
    patched = bytearray(blob)
    patched[offset:offset + 2] = b'\x30\xc0'
    return patched


def set_state(offset, state):
    """Rewrite the immediate of `mov dword ptr [rbx+0xc],imm32`."""
    def transform(blob):
        patched = bytearray(blob)
        if bytes(patched[offset:offset + 3]) != b'\xc7\x43\x0c':
            raise SystemExit(f'expected a flow-state store at +{offset:#x}')
        patched[offset + 3:offset + 7] = state.to_bytes(4, 'little')
        return patched
    return transform


# name, va, length, expected byte prefix, transform
SITES = (
    ('remove the character-creation transition', 0x7c0a4f, 32,
     bytes.fromhex('498b3fb8011f0afa488d3532e16d02488d55d0'),
     lambda blob: nop(blob, 0x17, 5)),
    ('skip the new-profile screens', 0x7c9254, 16,
     bytes.fromhex('c7430c2d000000498b07483b45d8'),
     set_state(0x0, 27)),
    ('remove the new-profile language screen', 0x7c930c, 32,
     bytes.fromhex('488d35e520b402b8011f0afa488d55cc488945cce8ab946100'),
     lambda blob: nop(blob, 0x14, 5)),
)


def default_game_dir():
    from_env = os.environ.get('KYTY_GAME', '')
    if from_env and Path(from_env).is_dir():
        return Path(from_env)
    for config in sorted(ROOT.glob('_Build/*/launch-*.json')):
        try:
            command = json.loads(config.read_text())['command']
        except (OSError, ValueError, KeyError):
            continue
        for index, argument in enumerate(command[:-1]):
            if argument == '--game':
                candidate = Path(command[index + 1])
                if candidate.is_dir():
                    return candidate
    return None


def probe(game):
    """Return the list of (name, va, original, patched) for every site."""
    eboot = game / 'decrypted' / 'eboot.bin'
    if not eboot.is_file():
        raise SystemExit(f'missing {eboot}; it is needed to derive the patch')
    data = eboot.read_bytes()
    entries = []
    for name, va, length, prefix, transform in SITES:
        blob = data[va + FILE_DELTA:va + FILE_DELTA + length]
        if len(blob) != length or not blob.startswith(prefix):
            raise SystemExit(
                f'{name}: eboot at {va:#x} does not match the expected bytes; '
                f'refusing to patch a different build')
        patched = bytes(transform(blob))
        if patched == blob:
            raise SystemExit(f'{name}: transform changed nothing; refusing')
        entries.append((name, va, blob, patched))
    return entries


def build_plan(game):
    entries = probe(game)
    return {
        'id': TITLE_ID,
        'version': APP_VER,
        'process': PROCESS,
        'mods': [{
            'name': f'kyty-boot-skip: {name}',
            'enabled': True,
            'memory': [{'offset': f'{va:x}', 'off': original.hex(), 'on': patched.hex()}],
        } for name, va, original, patched in entries],
    }


def install(arguments):
    plan = build_plan(arguments.game)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(plan, indent=1) + '\n')
    print(f'patch written : {OUT_PATH}')
    for mod in plan['mods']:
        entry = mod['memory'][0]
        print(f'  {entry["offset"]}: {mod["name"]}')
    print(f'launch with   : ./run.sh --game-patch {OUT_PATH.relative_to(ROOT)}')


def status(arguments):
    print(f'game     : {arguments.game}')
    print(f'patch    : {"present" if OUT_PATH.is_file() else "not generated"} ({OUT_PATH})')
    if not OUT_PATH.is_file():
        return
    plan = json.loads(OUT_PATH.read_text())
    print(f'mods     : {len(plan["mods"])}')
    try:
        fresh = build_plan(arguments.game)
        print(f'up to date: {fresh == plan}')
        if fresh != plan:
            for mod in fresh['mods']:
                print(f'  would be {mod["memory"][0]["offset"]}: {mod["name"]}')
    except SystemExit as error:
        print(f'up to date: no ({error})')


def remove(_arguments):
    if OUT_PATH.is_file():
        OUT_PATH.unlink()
        print(f'removed {OUT_PATH}')
    else:
        print('no patch file to remove')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['install', 'status', 'remove'])
    parser.add_argument('--game', type=Path, help='game directory (app0)')
    arguments = parser.parse_args(argv)

    if arguments.game is None:
        arguments.game = default_game_dir()
    if arguments.game is None or not arguments.game.is_dir():
        parser.error('could not determine the game directory; pass --game')
    arguments.game = arguments.game.resolve()

    return {'install': install, 'status': status, 'remove': remove}[arguments.action](arguments)


if __name__ == '__main__':
    sys.exit(main())
