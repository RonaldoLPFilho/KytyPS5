#!/usr/bin/env python3
"""One boot, many experiments: drive a retained emulator session through KYTY_LIVE_FILE.

The emulator's live-control thread (graphicsRun.cpp) polls a command file and prints
LIVE_* lines; this tool writes numbered command batches and waits for their results.

    live-bench.py start                      boot to the official viewpoint and keep it
    live-bench.py run "set 000310" "measure 6 rec"
    live-bench.py ab --a "set 000000" --b "set 000310" --rounds 3 --seconds 6
    live-bench.py convar r_fadeOutInstanceOverDistance        # locate a convar's value
    live-bench.py shot NAME                  screenshot of the game window
    live-bench.py stop                       end the session (the save is restored)

A variant is a ';'-separated list of commands (set / poke32 / poke8 / pokef / peek /
sleep).  `ab` settles each variant, measures it, alternates A/B for N rounds and reports
the mean, so drift inside the session cancels out.
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / '_Build/layer-bench'
LIVE_FILE = BENCH / 'live.cmd'
SESSION = BENCH / 'live-session.json'
CONFIG = ROOT / '_Build/release-33fps-fix-20260926/launch-live.json'
SAVE = ROOT / '_Build/walk-fps-20260914/save-baseline'


def session():
    if not SESSION.exists():
        raise SystemExit('no live session; run `live-bench.py start` first')
    return json.loads(SESSION.read_text())


def stdout_log(info):
    result = json.loads((Path(info['out']) / 'result.json').read_text())
    return Path(result['evidence']) / 'stdout.log'


def next_id():
    text = LIVE_FILE.read_text() if LIVE_FILE.exists() else 'id 0'
    match = re.match(r'id (\d+)', text)
    return (int(match[1]) if match else 0) + 1


def send(lines, timeout=600):
    """Write one command batch; return its LIVE_* lines once LIVE_DONE arrives."""
    info = session()
    log = stdout_log(info)
    command_id = next_id()
    start = log.stat().st_size if log.exists() else 0
    LIVE_FILE.write_text(f'id {command_id}\n' + '\n'.join(lines) + '\n')
    deadline = time.monotonic() + timeout
    collected = []
    while time.monotonic() < deadline:
        with log.open('rb') as handle:
            handle.seek(start)
            chunk = handle.read().decode(errors='replace')
        collected = [l for l in chunk.splitlines() if l.startswith('LIVE_') and f'id={command_id}' in l
                     or l.startswith('Unhandled host exception')]
        if any(l.startswith('Unhandled') for l in collected):
            raise SystemExit('emulator crashed:\n' + '\n'.join(collected))
        if any(l.startswith(f'LIVE_DONE id={command_id}') for l in collected):
            return collected
        time.sleep(0.2)
    raise SystemExit(f'no LIVE_DONE for id {command_id} within {timeout}s (is the session alive?)')


def parse_measure(line):
    return {k: v for k, v in re.findall(r'(\w+)=([^\s]+)', line)}


def cmd_start(args):
    if running_session() or subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True).returncode == 0:
        raise SystemExit('a live session or emulator is already running; run `live-bench.py stop` first')
    # Startup occasionally hangs in poll() right after audio init (host audio/X), before
    # any of our code runs; the harness then times out.  Just try again.
    for attempt in range(1, args.attempts + 1):
        try:
            return start_once(args)
        except SystemExit as error:
            print(f'start attempt {attempt} failed: {error}', flush=True)
    raise SystemExit('could not start a live session')


def running_session():
    """A session whose supervisor is still alive, or None."""
    if not SESSION.exists():
        return None
    info = json.loads(SESSION.read_text())
    try:
        os.kill(info['bench_pid'], 0)
    except (ProcessLookupError, KeyError):
        return None
    return info


def backup_saves():
    """Keep every pre-session save state outside the session directory: the
    session's own saves-before snapshot dies with `rm -rf` of that directory."""
    stamp = time.strftime('%Y%m%d-%H%M%S')
    target = BENCH / 'save-backups' / stamp
    subprocess.run(['cp', '-a', str(ROOT / '_SaveData'), str(target)], check=True)
    return target


def start_once(args):
    out = BENCH / 'live-session'
    if running_session() or subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True).returncode == 0:
        raise SystemExit('a live session or emulator is already running; run `live-bench.py stop` first')
    (BENCH / 'save-backups').mkdir(parents=True, exist_ok=True)
    print(f'saves backed up to {backup_saves()}', flush=True)
    subprocess.run(['rm', '-rf', str(out)], check=True)
    LIVE_FILE.write_text('id 0\n')
    command = [sys.executable, str(ROOT / 'tools/local/benchmark-entry.py'), '--config', str(args.config),
               '--save-baseline', str(getattr(args, 'save', None) or SAVE), '--out', str(out / 'run'), '--seconds', '5', '--cpu-times',
               '--keep-seconds', str(args.keep_seconds), '--walk-forward-seconds', str(args.walk),
               '--settle-seconds', str(args.settle)]
    (out).mkdir(parents=True, exist_ok=True)
    log = (out / 'bench.log').open('w')
    bench = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    SESSION.write_text(json.dumps({'out': str(out / 'run'), 'bench_pid': bench.pid}) + '\n')
    result = out / 'run' / 'result.json'
    for _ in range(300):
        if result.exists() and '"retained": true' in result.read_text():
            print(f'live session ready (bench pid {bench.pid}); stdout: {stdout_log(session())}')
            return
        if bench.poll() is not None:
            raise SystemExit(f'benchmark exited early; see {out / "bench.log"}')
        time.sleep(2)
    raise SystemExit('session did not become ready in 10 minutes')


def cmd_stop(_args):
    info = session()
    (Path(info['out']) / 'stop').touch()
    for _ in range(120):
        try:
            os.kill(info['bench_pid'], 0)
        except ProcessLookupError:
            break
        time.sleep(1)
    result = json.loads((Path(info['out']) / 'result.json').read_text())
    print('session ended; save restored:', result.get('original_save_restored'))
    SESSION.unlink()


def cmd_run(args):
    for line in send(args.commands):
        print(line)


def variant_lines(text):
    lines = []
    for part in (part.strip() for part in text.split(';')):
        if not part:
            continue
        if part.startswith('sym '):
            _, name, value = part.split()
            lines.append(f'poke32 {symbol_address(name):#x} {int(value, 0):x}')
        else:
            lines.append(part)
    return lines


def cmd_ab(args):
    variants = [('A', variant_lines(args.a)), ('B', variant_lines(args.b))]
    if args.c:
        variants.append(('C', variant_lines(args.c)))
    results = {name: [] for name, _ in variants}
    for round_index in range(args.rounds):
        for name, lines in variants:
            reply = send(lines + [f'sleep {args.settle}', f'measure {args.seconds} {name}{round_index}'])
            measure = parse_measure(next(l for l in reply if l.startswith('LIVE_MEASURE')))
            results[name].append(measure)
            print(f'round {round_index} {name}: fps {measure["fps"]:>6}  render {measure["render_ms"]:>6} ms  '
                  f'record {measure["record_ms"]:>6} ms  busy {measure["render_busy"]}  front {measure.get("front_ms", "-")} ms', flush=True)
            for line in reply:
                if line.startswith(('LIVE_ATTR', 'LIVE_PROTECT', 'LIVE_IMAGES', 'LIVE_HOT', 'LIVE_DELTA', 'LIVE_REUSE', 'LIVE_REPEAT', 'LIVE_ROUTEB', 'LIVE_NATIVE')):
                    print('    ' + ' '.join(f'{k}={v}' for k, v in parse_measure(line).items()
                                            if k not in ('id', 'label')), flush=True)
    print()
    for name, lines in variants:
        fps = [float(m['fps']) for m in results[name]]
        render = [float(m['render_ms']) for m in results[name]]
        print(f'{name}: fps mean {statistics.mean(fps):.2f} (min {min(fps):.2f} max {max(fps):.2f})  '
              f'render {statistics.mean(render):.2f} ms/frame   [{"; ".join(lines)}]')
    if args.restore:
        send(variant_lines(args.restore))


def cmd_convar(args):
    """Print a convar object's address and the first bytes, to find its value field."""
    sys.path.insert(0, str(ROOT / 'tools/local'))
    import importlib.util
    spec = importlib.util.spec_from_file_location('layers', ROOT / 'tools/local/demons-souls-layers.py')
    layers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layers)
    doc = layers.__doc__ or ''
    match = re.search(rf'\b{re.escape(args.name)}\s+(0x[0-9a-f]+)', doc)
    address = args.address or (match[1] if match else None)
    if not address:
        raise SystemExit('address unknown; pass --address (static convar object VA)')
    for line in send([f'peek {address} {args.bytes}']):
        print(line)


def symbol_address(name):
    """Absolute address of an emulator global in the live process (PIE base + nm vaddr)."""
    info = session()
    result = json.loads((Path(info['out']) / 'result.json').read_text())
    pid = result['pid']
    exe = os.readlink(f'/proc/{pid}/exe')
    out = subprocess.run(['nm', '--defined-only', exe], capture_output=True, text=True).stdout
    match = re.search(rf'^([0-9a-f]+) [bBdDrR] {re.escape(name)}$', out, re.M)
    if not match:
        raise SystemExit(f'symbol {name} not found')
    base = None
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[5] == exe and int(fields[2], 16) == 0:
            base = int(fields[0].split('-')[0], 16)
            break
    if base is None:
        raise SystemExit('load base not found')
    return base + int(match[1], 16)


def cmd_sym(args):
    address = symbol_address(args.name)
    lines = [f'peek {address:#x} {args.bytes}']
    if args.value is not None:
        lines.insert(0, f'poke32 {address:#x} {int(args.value, 0):x}')
    for line in send(lines):
        print(line)


def cmd_shot(args):
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
    match = re.search(r'(0x[0-9a-f]+) "\[Source build', tree)
    if not match:
        raise SystemExit('game window not found')
    subprocess.run([sys.executable, str(BENCH / 'raise-window.py'), match[1]], check=False)
    time.sleep(0.5)
    target = BENCH / 'live-session' / f'{args.name}.png'
    # `import` grabs the X server while it captures; a stuck one freezes every X client
    # (including the emulator's window creation), so it must never outlive 10 seconds.
    subprocess.run(['timeout', '-s', 'KILL', '10', 'import', '-window', match[1], '-resize', '1280x720',
                    str(target)], check=True)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('start'); p.add_argument('--walk', type=float, default=10)
    p.add_argument('--settle', type=float, default=10, help='settle after the HUD and after the walk (default 10)')
    p.add_argument('--save', type=Path, help='save baseline directory (default: the official scene)')
    p.add_argument('--keep-seconds', type=int, default=7200); p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--config', type=Path, default=CONFIG)
    p.set_defaults(func=cmd_start)
    sub.add_parser('stop').set_defaults(func=cmd_stop)
    p = sub.add_parser('run'); p.add_argument('commands', nargs='+'); p.set_defaults(func=cmd_run)
    p = sub.add_parser('ab'); p.add_argument('--a', required=True); p.add_argument('--b', required=True)
    p.add_argument('--c'); p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seconds', type=float, default=6); p.add_argument('--settle', type=float, default=2)
    p.add_argument('--restore', help='commands to run afterwards'); p.set_defaults(func=cmd_ab)
    p = sub.add_parser('convar'); p.add_argument('name'); p.add_argument('--address')
    p.add_argument('--bytes', type=int, default=64); p.set_defaults(func=cmd_convar)
    p = sub.add_parser('shot'); p.add_argument('name'); p.set_defaults(func=cmd_shot)
    p = sub.add_parser('sym', help='read (and optionally set, 32-bit) an emulator global by symbol name')
    p.add_argument('name'); p.add_argument('value', nargs='?'); p.add_argument('--bytes', type=int, default=8)
    p.set_defaults(func=cmd_sym)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
