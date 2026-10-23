#!/usr/bin/env python3
"""Bounded, input-free native profile with frame, thread, GPU and screenshot evidence."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid', type=int, required=True)
p.add_argument('--seconds', type=int, default=30)
p.add_argument('--label', default='baseline')
p.add_argument('--registers', action='store_true', help='Include user registers for instruction-level diagnosis; adds sampling overhead')
p.add_argument('--callgraph', action='store_true', help='Capture 32 KiB DWARF stacks to attribute library/JIT costs to callers; diagnostic overhead only')
p.add_argument('--event', choices=['cycles:u', 'cycles', 'cycles:k', 'mem-loads'], default='cycles:u',
               help='Use cycles for kernel costs; mem-loads samples user load use-latency (diagnostics only)')
target = p.add_mutually_exclusive_group()
target.add_argument('--tid', type=int, help='Sample one thread; otherwise sample the entire process')
target.add_argument('--renderer', action='store_true', help='Find and sample the unique Kyty.Gpu thread')
a = p.parse_args()
if not 5 <= a.seconds <= 120 or not re.fullmatch(r'[a-zA-Z0-9_-]+', a.label):
    p.error('seconds must be 5–120; label must contain letters, digits, _ or -')
lock_dir = ROOT / '_Build/automation'
lock_dir.mkdir(parents=True, exist_ok=True)
input_lock = (lock_dir / 'input.lock').open('a')
fcntl.flock(input_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
proc = Path('/proc') / str(a.pid)
exe = proc / 'exe'
if Path(os.readlink(exe)).name.removesuffix(' (deleted)') != 'kyty_emulator':
    p.error('Expected a live, unchanged kyty_emulator binary')
identity = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
if a.renderer:
    renderers = []
    for task in (proc / 'task').iterdir():
        try:
            if (task / 'comm').read_text().strip() == 'Kyty.Gpu':
                renderers.append(int(task.name))
        except FileNotFoundError:
            pass
    if len(renderers) != 1:
        p.error('Expected exactly one live Kyty.Gpu thread')
    a.tid = renderers[0]
if a.tid and not (proc / 'task' / str(a.tid)).exists():
    p.error('TID must belong to the supplied PID')
if a.event == 'mem-loads' and not a.tid:
    p.error('Memory-load diagnostics require a specific renderer TID')
out = ROOT / '_Build/profiles' / (datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + a.label)
out.mkdir(parents=True)

def run(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10)

def run_report(command, output):
    # Reports do not need source-line/inline expansion. Bound symbolization and
    # terminate its private root-owned process group on timeout, not the game.
    report_proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    try:
        result = report_proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        subprocess.run(['sudo', '-n', 'kill', '-TERM', '--', f'-{report_proc.pid}'],
                       timeout=5, check=False)
        try:
            report_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            subprocess.run(['sudo', '-n', 'kill', '-KILL', '--', f'-{report_proc.pid}'],
                           timeout=5, check=False)
            report_proc.wait(timeout=5)
        raise
    if result:
        raise subprocess.CalledProcessError(result, command)


def window():
    matches = re.findall(r'(0x[0-9a-f]+) "[^\n]*frame: (\d+), fps: ([0-9.]+)"[^\n]*\("kyty_emulator"', run(['xwininfo', '-root', '-tree']))
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one game frame counter')
    return matches[0]

def threads():
    result = {}
    for task in (proc / 'task').iterdir():
        try:
            s = (task / 'stat').read_text().rsplit(')', 1)[1].split()
            result[task.name] = {'name': (task / 'comm').read_text().strip(), 'ticks': int(s[11]) + int(s[12]), 'cpu': int(s[36])}
        except FileNotFoundError:
            pass
    return result

binary_bytes = exe.read_bytes()
binary_hash = hashlib.sha256(binary_bytes).hexdigest()
archive = ROOT / '_Build/profiles/binaries' / binary_hash / 'kyty_emulator'
archive.parent.mkdir(parents=True, exist_ok=True)
if archive.exists():
    if hashlib.sha256(archive.read_bytes()).hexdigest() != binary_hash:
        raise RuntimeError(f'Archived binary hash mismatch: {archive}')
else:
    with archive.open('xb') as saved_binary:
        saved_binary.write(binary_bytes)
del binary_bytes
metadata = vars(a) | {'binary': os.readlink(exe), 'sha256': binary_hash, 'archived_binary': str(archive),
    'command': (proc / 'cmdline').read_bytes().decode().split('\0')[:-1],
    'environment': [v for v in (proc / 'environ').read_bytes().decode().split('\0') if v.startswith('KYTY_')],
    'commit': run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).strip(), 'gameplay_verified': False}
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
(out / 'working-tree.patch').write_text(run(['git', '-C', str(ROOT), 'diff']))
# git diff omits new source files. Keep a bounded source-only snapshot alongside it.
untracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '--others', '--exclude-standard', '-z']).decode().split('\0')
for name in filter(None, untracked):
    relative = Path(name)
    source = ROOT / relative
    if ((relative.parts[0] in {'src', 'scripts', 'tests'} or relative.parent == Path('3rdparty')) and
        relative.suffix in {'.h', '.hpp', '.cpp', '.c', '.py', '.sh'} and
        not source.is_symlink() and source.stat().st_size <= 1024 * 1024):
        destination = out / 'untracked-source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
w, _, _ = window()
subprocess.run(['import', '-window', w, str(out / 'before.png')], check=True)
first = threads()
samples = []
started = time.monotonic()
log = (out / 'perf.log').open('w')
record = (['sudo', '-n', 'perf', 'mem', '-t', 'load', 'record', '--all-user',
           '--no-buildid', '--ldlat', '30', '-c', '100000' if a.callgraph else '2000'] if a.event == 'mem-loads' else
          ['sudo', '-n', 'perf', 'record', '--no-buildid', '-F', '199', '-e', a.event])
record += [
    *(['--call-graph', 'dwarf,32768'] if a.callgraph else []),
    *(['--user-regs=AX,BX,CX,DX,SI,DI,BP,SP,IP,R8,R9,R10,R11,R12,R13,R14,R15'] if a.registers else []),
    ('--tid' if a.event == 'mem-loads' else '-t') if a.tid else '-p',
    str(a.tid or a.pid), '-o', str(out / 'perf.data'), '--', 'sleep', str(a.seconds)]
metadata['perf_command'] = record
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
perf = subprocess.Popen(record, stdout=log, stderr=log, start_new_session=True)
try:
    while time.monotonic() - started < a.seconds:
        if perf.poll() not in (None, 0):
            raise RuntimeError(f'perf record failed; see {out / "perf.log"}')
        if (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] != identity:
            raise RuntimeError('Process identity changed')
        current, frame, fps = window()
        if current != w:
            raise RuntimeError('Window changed')
        gpu = run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw', '--format=csv,noheader,nounits']).strip()
        samples.append({'seconds': time.monotonic()-started, 'frame': int(frame), 'title_fps': float(fps), 'gpu': gpu})
        time.sleep(1)
    end = threads()
    elapsed = time.monotonic() - started
    # DWARF captures can contain tens of MiB of stacks. Allow bounded final
    # build-ID/header processing without extending the measured CPU interval.
    perf.wait(timeout=60 if a.callgraph or a.event == 'mem-loads' else 10)
finally:
    try:
        if perf.poll() is None:
            # perf is root-owned under sudo. Its private process group contains
            # only this capture and its duration helper, never the game.
            subprocess.run(['sudo','-n','kill','-INT','--',f'-{perf.pid}'],timeout=5,check=False)
            try:
                perf.wait(timeout=10)
            except subprocess.TimeoutExpired:
                subprocess.run(['sudo','-n','kill','-KILL','--',f'-{perf.pid}'],timeout=5,check=False)
                perf.wait(timeout=5)
    finally:
        log.close()
        (out / 'samples.json').write_text(json.dumps(samples, indent=2))
if perf.returncode != 0:
    raise RuntimeError(f'perf record failed with {perf.returncode}; see {out / "perf.log"}')
cpu = sorted([{'tid': tid, 'name': v['name'], 'cpu_percent': 100 * (v['ticks'] - first[tid]['ticks']) / os.sysconf('SC_CLK_TCK') / elapsed}
    for tid, v in end.items() if tid in first], key=lambda x: -x['cpu_percent'])
if any(y['frame'] < x['frame'] for x, y in zip(samples, samples[1:])):
    raise RuntimeError('Frame counter reset')
result = {'measured_fps': (samples[-1]['frame']-samples[0]['frame'])/(samples[-1]['seconds']-samples[0]['seconds']),
    'threads': cpu, 'perf_returncode': perf.returncode, 'evidence': str(out), 'gameplay_verified': False}
lost = re.search(r'Processed \d+ events and lost (\d+) chunks', (out / 'perf.log').read_text())
result['record_lost_chunks'] = int(lost[1]) if lost else 0
result['profile_usable_for_quantitative_attribution'] = not bool(lost and int(lost[1]))
out_of_order = re.search(r'(\d+) out of order events recorded', (out / 'perf.log').read_text())
result['record_out_of_order_events'] = int(out_of_order[1]) if out_of_order else 0
(out / 'result.json').write_text(json.dumps(result, indent=2))
subprocess.run(['import', '-window', w, str(out / 'after.png')], check=True)
mapping = Path('/tmp') / f'perf-{a.pid}.map'
if not mapping.is_symlink() and mapping.is_file() and mapping.stat().st_size <= 64 * 1024 * 1024:
    (out / mapping.name).write_bytes(mapping.read_bytes())
analysis_data = out / 'perf.data'
with (out / 'code-map-filter.json').open('w') as filter_log:
    filtered = subprocess.run(['sudo', '-n', 'python3', str(ROOT / 'tools/local/perf-code-maps.py'),
                               str(analysis_data), str(out / 'perf-code.data')],
                              stdout=filter_log, stderr=subprocess.STDOUT, timeout=30)
if filtered.returncode == 0:
    analysis_data = out / 'perf-code.data'
result['analysis_data'] = str(analysis_data)
result['code_map_filter_returncode'] = filtered.returncode
(out / 'result.json').write_text(json.dumps(result, indent=2))
with (out / 'perf.txt').open('w') as report:
    run_report(['sudo', '-n', 'perf', 'report', '--stdio', '--no-children', '--no-inline', '-g', 'none', '-i', str(analysis_data), '--sort', 'comm,dso,symbol', '--percent-limit', '0.5'], report)
with (out / 'perf-dso.txt').open('w') as report:
    run_report(['sudo', '-n', 'perf', 'report', '--stdio', '--no-children', '--no-inline', '-g', 'none', '-i', str(analysis_data), '--sort', 'dso', '--percent-limit', '0.1'], report)
if a.registers:
    with (out / 'registers.txt').open('w') as report:
        run_report(['sudo', '-n', 'perf', 'script', '--no-inline', '-i', str(analysis_data), '-F', 'ip,sym,symoff,uregs'], report)
if a.callgraph:
    with (out / 'callchains.txt').open('w') as report:
        run_report(['sudo', '-n', 'perf', 'script', '--no-inline', '-i', str(analysis_data),
                        '-F', 'comm,pid,tid,time,event,ip,sym,symoff,dso'],
                       report)
if a.event == 'mem-loads':
    # Code-only copies omit data mappings. Memory reports must use the retained
    # raw capture so sampled data addresses keep their original attribution.
    with (out / 'memory-report.txt').open('w') as report:
        run_report(['sudo', '-n', 'perf', 'mem', '-i', str(out / 'perf.data'),
                    'report', '--stdio', '--no-children', '--no-inline', '-g', 'none',
                    '--percent-limit', '0.5'], report)
    with (out / 'memory-samples.txt').open('w') as report:
        run_report(['sudo', '-n', 'perf', 'script', '--no-inline', '-i', str(out / 'perf.data'),
                    '-F', 'comm,pid,tid,time,event,ip,sym,symoff,addr,weight,data_src,dso'], report)
    result['memory_note'] = ('Intel load use-latency includes pipeline queueing; '
                             'sample weights are not exclusive wall time or FPS savings.')
    (out / 'result.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if k != 'threads'} | {'busiest_threads': cpu[:4]}, indent=2))
