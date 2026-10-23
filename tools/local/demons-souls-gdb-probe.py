#!/usr/bin/env python3
"""Run the emulator under gdb with breakpoints on *guest* code addresses.

Kyty executes the PS5 module natively in its own process, so gdb can break on
guest code exactly like host code - this is by far the fastest way to answer
"which code path actually does this?", and it is what round 11 used to trace the
"On the first day" card (see docs/compatibility/demons-souls-boot-patch.md).

Three things make it work:

* ``ptrace_scope`` is 1 on this machine, so gdb has to *launch* the emulator
  rather than attach to it.  This script therefore rebuilds the validated launch
  config's environment and command line as gdb commands.
* The main module is the first one loaded, so its base is
  ``SYSTEM_RESERVED + CODE_BASE_OFFSET`` = ``0x900000000`` (see
  ``src/loader/runtimeLinker.cpp``), and the eboot's first PT_LOAD has
  ``p_vaddr == 0``.  A guest address from the recon notes is therefore just
  ``0x900000000 + va``.
* Kyty uses SIGSEGV for its GPU memory tracking and SIGILL in the guest, so both
  must be passed through or the session stops on the first frame.

Guest addresses only become valid once the module is mapped, so every breakpoint
is installed from a breakpoint on ``Loader::GamePatch::Apply``, which runs after
the module is loaded and before the guest starts.

Usage
-----
    demons-souls-gdb-probe.py --break 526a0 --break 485220
    demons-souls-gdb-probe.py --break 485220 --condition '*(unsigned int *)$rdi == 0x53eea816'
    demons-souls-gdb-probe.py --break 55e498 --frames 12      # walk the rbp chain
    demons-souls-gdb-probe.py --print-script                  # just show the gdb script

``--condition`` applies to the breakpoint it follows on the command line.  Every
hit prints the guest return address, so the caller is identified even though the
guest has no symbols; ``--frames N`` additionally walks N guest rbp frames,
which is how the UIManager/MessageQueue chain behind the card was recovered.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / '_Build' / 'profiles' / 'manual-current-launch.json'
MIMALLOC = (ROOT / '_Build/profiles/libraries'
            / 'b0f267c17975ee39391331d0f02dd3682233a67e8464095db1407cd4e37f3853'
            / 'libmimalloc.so')
GUEST_BASE = 0x900000000
GUEST_END = 0x902a00000          # comfortably past the module, for "is this guest code?"


def launch_commands(patch):
    config = json.loads(CONFIG.read_text())
    environment = dict(config['environment'])
    if config.get('startup_control') == 'native':
        environment.setdefault('KYTY_NATIVE_CHECKPOINT', '1')
        environment.setdefault('KYTY_NATIVE_RENDER_COST', '0')
        environment.setdefault('KYTY_NATIVE_VERIFY_RESOURCES', '0')
    command = config['command']
    arguments = command[command.index('--') + 1:]
    emulator = arguments[0]
    if MIMALLOC.is_file():
        environment['LD_PRELOAD'] = str(MIMALLOC)
    digest = subprocess.run(['sha256sum', emulator], capture_output=True, text=True,
                            check=True).stdout.split()[0]
    environment['KYTY_DRIVER_CACHE_KEY'] = digest

    lines = ['set pagination off', 'set confirm off', 'set print thread-events off']
    lines += [f'set environment {key} {value}' for key, value in environment.items()]
    lines.append(f'file {emulator}')
    rest = list(arguments[1:])
    if patch is not None:
        rest += ['--game-patch', str(patch)]
    lines.append('set args ' + ' '.join(shlex.quote(argument) for argument in rest))
    # Kyty's own memory tracking and the guest both raise these; stopping on them
    # ends the session before the game ever draws a frame.
    lines += [f'handle {signal} nostop noprint pass'
              for signal in ('SIGSEGV', 'SIGILL', 'SIGBUS', 'SIGFPE', 'SIGUSR1', 'SIGUSR2')]
    return lines


def breakpoint_commands(sites, frames, stop_after):
    body = []
    for address, condition in sites:
        guard = f' if {condition}' if condition else ''
        body.append(f'  break *($base + {address:#x}){guard}')
        body.append('  commands')
        body.append('    silent')
        body.append('    set $hits = $hits + 1')
        body.append(f'    printf "HIT %d at guest {address:#x}, called from guest 0x%llx\\n", '
                    '$hits, (*(unsigned long long *)$rsp) - $base')
        if frames:
            # The guest keeps rbp frames, so walking them recovers the call chain
            # that gdb's own unwinder cannot produce without symbols.
            body.append('    set $fr = $rbp')
            body.append('    set $n = 0')
            body.append(f'    while $n < {frames} && $fr > 0x1000 && $fr < 0xffffffffffff')
            body.append('      set $ra = *(unsigned long long *)($fr + 8)')
            body.append(f'      if $ra > {GUEST_BASE:#x} && $ra < {GUEST_END:#x}')
            body.append('        printf "    frame %2d: guest 0x%llx\\n", $n, $ra - $base')
            body.append('      else')
            body.append('        printf "    frame %2d: host  0x%llx\\n", $n, $ra')
            body.append('      end')
            body.append('      set $fr = *(unsigned long long *)$fr')
            body.append('      set $n = $n + 1')
            body.append('    end')
        body.append(f'    if $hits >= {stop_after}')
        body.append('      kill')
        body.append('      quit')
        body.append('    end')
        body.append('    continue')
        body.append('  end')
    return body


def build_script(arguments):
    lines = launch_commands(arguments.game_patch)
    sites = []
    conditions = list(arguments.condition)
    for address in arguments.brk:
        sites.append((int(address, 16), conditions.pop(0) if conditions else None))
    lines += [
        '',
        '# Guest addresses are only mapped once the module is loaded; this runs',
        '# after that and before the guest starts.',
        'break Loader::GamePatch::Apply',
        'commands',
        '  silent',
        f'  set $base = {GUEST_BASE:#x}',
        '  set $hits = 0',
        '  delete',
    ]
    lines += breakpoint_commands(sites, arguments.frames, arguments.stop_after)
    lines += ['  continue', 'end', 'run']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--break', dest='brk', action='append', default=[], metavar='VA',
                        help='guest virtual address in hex, as the recon notes give it')
    parser.add_argument('--condition', action='append', default=[], metavar='EXPR',
                        help='gdb condition for the preceding --break')
    parser.add_argument('--frames', type=int, default=0,
                        help='walk this many guest rbp frames on each hit')
    parser.add_argument('--stop-after', type=int, default=8,
                        help='quit after this many hits (default 8)')
    parser.add_argument('--game-patch', type=Path,
                        default=ROOT / 'tools/local/patches/skip-boot-flow.json',
                        help='cheat file to pass through, or "none"')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--print-script', action='store_true')
    arguments = parser.parse_args(argv)

    if str(arguments.game_patch) == 'none':
        arguments.game_patch = None
    elif not arguments.game_patch.is_file():
        parser.error(f'missing cheat file {arguments.game_patch}; '
                     'generate it with demons-souls-patch.py install, or pass --game-patch none')
    if not arguments.brk and not arguments.print_script:
        parser.error('nothing to probe; pass at least one --break')

    script = build_script(arguments)
    if arguments.print_script:
        sys.stdout.write(script)
        return 0

    with tempfile.NamedTemporaryFile('w', suffix='.gdb', delete=False) as handle:
        handle.write(script)
        script_path = handle.name
    try:
        return subprocess.run(['gdb', '-batch', '-x', script_path],
                              timeout=arguments.timeout).returncode
    except subprocess.TimeoutExpired:
        print('probe timed out; no breakpoint was reached in time', file=sys.stderr)
        subprocess.run(['pkill', '-f', 'kyty_emulator'], check=False)
        return 2
    finally:
        os.unlink(script_path)


if __name__ == '__main__':
    sys.exit(main())
