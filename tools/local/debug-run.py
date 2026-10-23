#!/usr/bin/env python3
"""Bounded local debug runs with PID-scoped kernel evidence and a JSON result.

An exit code of zero is never treated as proof of correct gameplay.
Commands after -- are executed directly, without a shell.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

from debug_cpu_policy import (atomic_json, pin_threads, renderer_for_request, start_ticks,
                              thread_overrides_for_request, validate_request)
from playtest_log import PlaytestMonitor

ROOT = Path(__file__).resolve().parents[2]
GPU_ERROR = re.compile(r'VK_ERROR_DEVICE_LOST|ErrorDeviceLost|CTX SWITCH TIMEOUT|GPU[- ]?AV.*ERROR', re.I)
FATAL = re.compile(r'\bFATAL\b|Assertion .*failed|SPIR-V validation failed|Unhandled host exception', re.I)


def classify(returncode, timeout, log, kernel, pid, kernel_available=True, shader_tests=False):
    expected_rejections = 0
    if shader_tests and returncode == 0 and log.rstrip().endswith('ShaderRecompilerComputeTests: all cases passed'):
        # Ignore only the exact diagnostic followed by its successful negative
        # test marker. Other validation failures/device errors remain failures.
        log, expected_rejections = re.subn(
            r'^SPIR-V validation failed: hash=0x0000000000000000 stage=4 '
            r'reason=writable FLAT/GLOBAL addresses require GPU ownership tracking\n'
            r' in [^\n]+\n\[host\]\s+WritableFlatStoreRejection\s+ok$',
            '[expected writable-flat rejection]', log, flags=re.M)
    xid = [line for line in kernel.splitlines() if 'NVRM: Xid' in line]
    own = [line for line in xid if re.search(rf'\bpid={pid}(?:\D|$)', line)]
    if own or GPU_ERROR.search(log): status = 'gpu_failure'
    elif timeout: status = 'timeout'
    elif returncode != 0 or FATAL.search(log): status = 'process_failure'
    elif xid or not kernel_available: status = 'inconclusive'
    else: status = 'completed_unverified'
    return {'status': status, 'returncode': returncode, 'timed_out': timeout,
            'kernel_available': kernel_available, 'pid_xid': own,
            'other_xid': [line for line in xid if line not in own], 'gameplay_verified': False,
            'expected_test_rejections': expected_rejections}


def command_output(command, timeout=15):
    try:
        p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as exc: return -1, str(exc)


def kernel_log():
    for command in (['dmesg', '--color=never'], ['sudo', '-n', 'dmesg', '--color=never']):
        rc, text = command_output(command)
        if rc == 0: return True, text
    return False, text


def stop_group(process):
    # The child has its own session. Never match/kill other emulator processes.
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=3)
    except subprocess.TimeoutExpired: pass
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def cpu_affinity(pid):
    """Read a launch-time affinity snapshot; a missing value is not an empty mask."""
    if not hasattr(os, 'sched_getaffinity'):
        return {'cpus': None, 'error': 'sched_getaffinity is unavailable'}
    try:
        return {'cpus': sorted(os.sched_getaffinity(pid))}
    except OSError as exc:
        return {'cpus': None, 'error': str(exc)}


def acquire_run_lock(path):
    """Separate from input.lock: measurements may attach to the one running game."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('Another debug run is active; concurrent GPU runs invalidate measurements') from None
    return lock


def pin_render_threads(pid, cpu, allowed):
    """Only adjust threads of the child owned by this run; parent masks stay unchanged."""
    return pin_threads(pid, cpu, allowed - {cpu})[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', default='experiment')
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--render-cpu', type=int, help='Pin the child Kyty.Gpu thread and exclude this CPU from its other threads for this bounded run')
    p.add_argument('--sample-gpu', action='store_true', help='Record GPU-wide utilization/memory at 200 ms intervals')
    p.add_argument('--playtest-log', action='store_true', help='Low-overhead manual gameplay timeline, process samples and stall snapshots')
    p.add_argument('--shader-tests', action='store_true', help='Recognize the exact successful writable-flat negative test diagnostic')
    p.add_argument('--driver-cache', action='store_true', help='Enable a binary-hash-scoped local Vulkan cache for a direct emulator command')
    p.add_argument('--artifact', type=Path, action='append', default=[], help='Hash a shader/capture/binary used by this experiment')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: p.error('Provide a command after --')
    if args.shader_tests and (Path(command[0]).name != 'shader_recompiler_compute_tests' or
                             command[1:] not in ([], ['--linear-srt'])):
        p.error('--shader-tests requires the direct complete shader_recompiler_compute_tests suite')
    if not math.isfinite(args.timeout) or args.timeout <= 0: p.error('--timeout must be finite and positive')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label): p.error('Use letters, digits, underscore or hyphen in --label')
    allowed_cpus = None
    if args.render_cpu is not None:
        binary = (ROOT / command[0]).resolve()
        if binary.name != 'kyty_emulator' or not binary.is_file() or not hasattr(os, 'sched_getaffinity'):
            p.error('--render-cpu requires a direct Linux kyty_emulator command')
        allowed_cpus = set(os.sched_getaffinity(0))
        if args.render_cpu not in allowed_cpus or len(allowed_cpus) < 2:
            p.error('--render-cpu must be allowed and leave a CPU for the other child threads')
    child_env = os.environ.copy()
    if args.driver_cache:
        binary = (ROOT / command[0]).resolve()
        if binary.name != 'kyty_emulator' or not binary.is_file():
            p.error('--driver-cache requires a direct kyty_emulator command')
        child_env['KYTY_DRIVER_CACHE_KEY'] = file_hash(binary)
    # Validate/hash inputs before creating evidence or launching any child process.
    # Relative artifacts, like the command, are rooted at this repository.
    artifacts = {}
    try:
        for path in args.artifact:
            path = (ROOT / path).resolve()
            artifacts[str(path)] = file_hash(path)
    except OSError as exc:
        p.error(f'Cannot read artifact: {exc}')
    try:
        run_lock = acquire_run_lock(ROOT / '_Build/automation/debug-run.lock')
    except RuntimeError as exc:
        p.error(str(exc))
    stamp = dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out = ROOT / '_Build/experiments' / (stamp + '-' + args.label)
    out.mkdir(parents=True)
    if args.playtest_log:
        child_env['KYTY_PLAYTEST_LOG'] = str(out / 'playtest.log')
    _, revision = command_output(['git', 'rev-parse', 'HEAD'])
    _, patch = command_output(['git', 'diff', '--binary'])
    (out / 'working-tree.patch').write_text(patch)
    _, untracked = command_output(['git', 'ls-files', '--others', '--exclude-standard'])
    (out / 'untracked-files.txt').write_text(untracked)
    _, gpu = command_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'])
    (out / 'gpu.txt').write_text(gpu)
    metadata = {'command': command, 'cwd': str(ROOT), 'revision': revision.strip(), 'timeout_seconds': args.timeout,
                'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'artifacts': artifacts,
                'render_cpu': args.render_cpu, 'shader_tests': args.shader_tests,
                'environment': {k: v for k, v in child_env.items() if (k.startswith(('KYTY_', 'VK_', 'GFXRECON_')) or k in ('ANISO', 'REPEAT', 'READ_OUTPUT', 'DUMP_PIPELINE_CACHE', '__GL_SHADER_DISK_CACHE'))}}
    metadata['allowed_cpus'] = sorted(allowed_cpus) if allowed_cpus is not None else None
    metadata['supervisor'] = {'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
                              'initial_cpu_affinity': cpu_affinity(0)}
    executable = ROOT / command[0] if '/' in command[0] else Path(shutil.which(command[0]) or command[0])
    if executable.is_file(): metadata['executable_sha256'] = file_hash(executable)
    available_before, before = kernel_log()
    (out / 'kernel-before.txt').write_text(before)
    start = time.monotonic(); timed_out = False; interrupted = False
    process = None
    sampler = None
    sample_log = None
    monitor = None
    received_signal = None
    affinity_updates = 0
    def interrupted_signal(signum, frame):
        nonlocal received_signal
        received_signal = signal.Signals(signum).name
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, interrupted_signal)
    previous_int = signal.signal(signal.SIGINT, interrupted_signal)
    try:
        if args.sample_gpu or args.playtest_log:
            sample_log = (out / 'gpu-samples.csv').open('w')
            try:
                fields = 'timestamp,index,uuid,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw'
                sampler = subprocess.Popen(['nvidia-smi', '--query-gpu=' + fields,
                    '--format=csv', '--loop-ms=' + ('200' if args.sample_gpu else '1000')], stdout=sample_log, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as exc:
                sample_log.write(str(exc))
        with (out / 'stdout.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=child_env)
            metadata['pid'] = process.pid
            metadata['initial_cpu_affinity'] = cpu_affinity(process.pid)
            # Keep the host ELF load bias for symbolizing fatal addresses after
            # exit; a PID alone cannot reconstruct ASLR once /proc disappears.
            try:
                proc = Path('/proc') / str(process.pid)
                metadata['pid_start_ticks'] = (proc/'stat').read_text().rsplit(')', 1)[1].split()[19]
                (out/'startup-maps.txt').write_text((proc/'maps').read_text())
            except (OSError, IndexError) as exc:
                metadata['startup_maps_error'] = str(exc)
            (out / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'PID {process.pid}; evidence: {out}', flush=True)
            if args.playtest_log and metadata.get('pid_start_ticks'):
                monitor = PlaytestMonitor(process.pid, metadata['pid_start_ticks'], out)
                monitor.start()
            try:
                if args.render_cpu is None:
                    process.wait(timeout=args.timeout)
                else:
                    deadline = time.monotonic() + args.timeout
                    policy_identity = {'pid': process.pid, 'start_ticks': metadata['pid_start_ticks'],
                                       'binary_sha256': metadata['executable_sha256']}
                    last_ack = None
                    while process.poll() is None:
                        # The supervisor also applies phase policies to threads
                        # created later. An external one-shot taskset would be
                        # silently undone by this loop every two seconds.
                        request = {'identity': policy_identity, 'token': 'default',
                                   'worker_cpus': sorted(allowed_cpus - {args.render_cpu})}
                        error = None
                        renderer = None
                        renderer_cpu = args.render_cpu
                        overrides = {}
                        try:
                            policy_path = out / 'cpu-policy.json'
                            if policy_path.exists():
                                request = json.loads(policy_path.read_text())
                            selected_cpu = renderer_for_request(request, policy_identity, allowed_cpus, args.render_cpu)
                            workers = validate_request(request, policy_identity, allowed_cpus, selected_cpu)
                            overrides = thread_overrides_for_request(request, allowed_cpus, selected_cpu)
                            renderer_cpu = selected_cpu
                            renderer = request.get('renderer')
                        except (OSError, ValueError) as exc:
                            error = str(exc)
                            workers = allowed_cpus - {args.render_cpu}
                            renderer_cpu = args.render_cpu
                            renderer = None
                            overrides = {}
                        try:
                            updates, masks, names = pin_threads(process.pid, renderer_cpu, workers, renderer, overrides)
                            affinity_updates += updates
                        except (OSError, ValueError) as exc:
                            error = str(exc)
                            masks, names = {}, {}
                        ack = {'identity': policy_identity,
                               'token': request.get('token') if isinstance(request, dict) else None,
                               'worker_cpus': sorted(workers), 'affinities': masks,
                               'renderer_cpu': renderer_cpu,
                               'thread_overrides': [
                                   {'tid': tid, 'start_ticks': item['start_ticks'], 'cpus': sorted(item['cpus'])}
                                   for tid, item in overrides.items()],
                               'thread_names': names, 'error': error}
                        if ack != last_ack:
                            atomic_json(out / 'cpu-policy-applied.json', ack)
                            last_ack = ack
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, args.timeout)
                        try:
                            process.wait(timeout=min(2, remaining))
                        except subprocess.TimeoutExpired:
                            continue
            except subprocess.TimeoutExpired:
                timed_out = True; stop_group(process)
            except KeyboardInterrupt:
                interrupted = True; stop_group(process)
    except OSError as exc:
        (out / 'stdout.log').write_text(str(exc))
    finally:
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGINT, previous_int)
        if sampler is not None: stop_group(sampler)
        if sample_log is not None: sample_log.close()
        if process is not None and process.poll() is None: stop_group(process)
    # Allow already queued kernel messages to arrive, without a long blind wait.
    time.sleep(.5)
    available_after, after = kernel_log()
    old_lines = set(before.splitlines()) if available_before else set()
    delta = '\n'.join(line for line in after.splitlines() if line not in old_lines)
    (out / 'kernel-new.txt').write_text(delta)
    log = (out / 'stdout.log').read_text(errors='replace')
    result = classify(process.returncode if process else 127, timed_out, log, delta,
                      process.pid if process else -1, available_before and available_after, args.shader_tests)
    if interrupted: result['status'] = 'interrupted'
    result.update({'elapsed_seconds': round(time.monotonic() - start, 3), 'evidence': str(out),
                   'ended_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'interrupt_signal': received_signal,
                   'render_cpu': args.render_cpu, 'affinity_updates': affinity_updates})
    code = result['returncode']
    result['exit_signal'] = signal.Signals(-code).name if code < 0 and -code in signal.valid_signals() else None
    if args.playtest_log:
        result['gpu_sampler_returncode'] = sampler.returncode if sampler else None
        if monitor:
            try:
                monitor.finish(result)
                result['playtest_summary'] = str(out / 'playtest-summary.json')
            except (OSError, RuntimeError) as exc:
                result['playtest_summary_error'] = str(exc)
        else:
            result['playtest_summary_error'] = 'Child exited before its PID identity could be captured'
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'completed_unverified' else 1)


if __name__ == '__main__': main()
