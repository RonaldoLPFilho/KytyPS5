#!/usr/bin/env python3
"""Make Demon's Souls (PPSA01341) open inside the game's starting corridor.

What "skipping the intro" means here
------------------------------------
The retail package plays roughly two minutes of splash logos and the opening
cinematic, then shows the title screen, the main menu and character creation
before the player ever reaches the Tutorial (world 8-1), which is the corridor
the game opens in.

This profile makes the engine leave all of that behind:

* ``cp11_debug_skipSplashScreens``  drops the publisher/developer logo screens.
* ``cp11_gameflowtype=DemoLoopA``   boots the engine's demo/kiosk game flow
  instead of ``FullGame``.  That flow never enters the front end, so the opening
  cinematic, the title screen and the main menu do not appear at all; it goes
  straight into a new game, i.e. the Tutorial corridor.
* ``cp11_demoModeFile``/``cp11_demoCycle`` point that flow at the shipped
  ``misc/demo_tutorial.txt`` so the Tutorial is also what attract mode would run.

What this profile alone leaves behind are the one-off prompts a new game always
asks for (language select, brightness, settings, body type, class/token
confirmation).  ``tools/local/demons-souls-patch.py`` removes those from the
guest code, so with the patch there is nothing left to answer;
``tools/local/demons-souls-quickstart.py`` still watches the screen and stops at
the gameplay HUD (and can answer the prompts when the patch is off).
``run-boot-skip.sh`` runs both.

Why this edits a game file instead of passing an emulator option
----------------------------------------------------------------
The Bluepoint "cp11" engine builds its own command line; the emulator never
forwards its argv to the guest (``src/loader/runtimeLinker.cpp`` hands the guest
a fixed ``argv == {"KytyEmu"}``).  The engine instead reads, in order,

    $/misc/GameCommandLineArgs.txt
    $/PackageCmdLineArgs.txt          <- documented "user defined cmd line args"
    #/CommandLineArgs.txt

and dispatches every ``+Name=Value`` line through its console.  ``PackageCmdLineArgs.txt``
is loaded last, so entries added there win over the shipped defaults.  That is
the only supported injection surface, and this tool keeps it reversible.

Usage
-----
    demons-souls-boot-skip.py install [--game DIR]
    demons-souls-boot-skip.py remove  [--game DIR]
    demons-souls-boot-skip.py status  [--game DIR]

``--game`` falls back to ``$KYTY_GAME`` and then to the ``--game`` argument of
the validated launch config under ``_Build/*/launch-*.json``.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BEGIN = '// >>> kyty-boot-skip >>> (managed by tools/local/demons-souls-boot-skip.py)'
END = '// <<< kyty-boot-skip <<<'
ARGS_NAME = 'PackageCmdLineArgs.txt'

# Verified against PPSA01341 01.007.000: this set boots the DemoLoopA game flow,
# which skips the opening cinematic, the title screen and the main menu and
# starts a new game (the Tutorial corridor).
BOOT_SKIP_CONVARS = (
    '+cp11_debug_skipSplashScreens=true',
    '+cp11_gameflowtype=DemoLoopA',
    '+cp11_demoModeFile=$/misc/demo_tutorial.txt',
    '+cp11_demoCycle=$/misc/demo_tutorial.txt',
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


def strip_block(text):
    """Remove a previously installed managed block, keeping everything else."""
    kept, inside = [], False
    for line in text.splitlines():
        if line.strip() == BEGIN:
            inside = True
            continue
        if line.strip() == END:
            inside = False
            continue
        if not inside:
            kept.append(line)
    return '\n'.join(kept).rstrip('\n') + '\n'


def managed_block():
    return '\n'.join([BEGIN, *BOOT_SKIP_CONVARS, END])


def install(arguments):
    args_path = arguments.game / ARGS_NAME
    if not args_path.is_file():
        raise SystemExit(f'missing engine command line file: {args_path}')
    original = args_path.read_text()
    args_path.write_text(strip_block(original).rstrip('\n') + '\n' + managed_block() + '\n')
    print(f'boot-skip installed in {args_path}')
    for line in BOOT_SKIP_CONVARS:
        print(f'  {line}')
    print('now run tools/local/demons-souls-quickstart.py (or run-boot-skip.sh) to')
    print('finish the first-boot prompts and land in the Tutorial corridor.')


def remove(arguments):
    args_path = arguments.game / ARGS_NAME
    if not args_path.is_file():
        raise SystemExit(f'missing engine command line file: {args_path}')
    original = args_path.read_text()
    stripped = strip_block(original)
    if stripped == original:
        print('boot-skip was not installed')
        return
    args_path.write_text(stripped)
    print(f'boot-skip removed from {args_path}')


def status(arguments):
    args_path = arguments.game / ARGS_NAME
    text = args_path.read_text() if args_path.is_file() else ''
    print(f'game      : {arguments.game}')
    print(f'boot-skip : {"installed" if BEGIN in text else "not installed"}')
    if BEGIN in text:
        for line in text.splitlines():
            if line.startswith('+cp11_'):
                print(f'  {line}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['install', 'remove', 'status'])
    parser.add_argument('--game', type=Path, help='game directory (app0)')
    arguments = parser.parse_args(argv)

    if arguments.game is None:
        arguments.game = default_game_dir()
    if arguments.game is None or not arguments.game.is_dir():
        parser.error('could not determine the game directory; pass --game')
    arguments.game = arguments.game.resolve()

    return {'install': install, 'remove': remove, 'status': status}[arguments.action](arguments)


if __name__ == '__main__':
    sys.exit(main())
