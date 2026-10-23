#!/usr/bin/env python3
"""Measure the reviewed build or the frozen checkpoint in the same stationary Continue scene."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from debug_cpu_policy import atomic_json, start_ticks

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def capture_startup_failure(result, out):
    """Preserve a live startup wait graph before the owned launcher is stopped."""
    pid = result.get('pid')
    identity = result.get('pid_start_ticks')
    expected_sha = result.get('binary_sha256')
    if not pid or not identity or result.get('gameplay_hud_reached'):
        return
    proc = Path('/proc') / str(pid)
    diagnostic = {'timing': False, 'pid': pid, 'pid_start_ticks': identity,
                  'binary_sha256': expected_sha}
    result['startup_failure_diagnostic'] = diagnostic
    try:
        if start_ticks(pid) != identity or sha(proc / 'exe') != expected_sha:
            diagnostic['status'] = 'identity_changed'
            return
        maps = out / 'startup-failure-maps.txt'
        maps.write_text((proc / 'maps').read_text())
        script = out / 'startup-failure.gdb'
        script.write_text(f'''set pagination off
set print elements 16
python
import gdb,pathlib,hashlib
try:
    p=pathlib.Path('/proc/{pid}')
    assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=={identity!r}
    assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()=={expected_sha!r}
    gdb.execute('thread apply all bt 25')
    print('STARTUP_WAIT_GRAPH_CAPTURED')
finally:
    gdb.execute('detach')
end
''')
        log = out / 'startup-failure-stacks.log'
        # Keep even partial output if GDB fails or reaches its own deadline.
        with log.open('w') as stream:
            check = subprocess.run(['sudo', '-n', 'gdb', '-q', '-nx', '-batch',
                '-iex', 'set print thread-events off', '-p', str(pid), '-x', str(script)],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=30)
        diagnostic.update(returncode=check.returncode, log=str(log), maps=str(maps),
            status='captured' if check.returncode == 0 and
                'STARTUP_WAIT_GRAPH_CAPTURED' in log.read_text() else 'capture_failed')
    except (OSError, subprocess.SubprocessError) as exc:
        diagnostic.update(status='capture_failed', error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('reviewed', 'checkpoint'), required=True)
    parser.add_argument('--binary', type=Path, help='Override reviewed executable only')
    parser.add_argument('--local-environment', type=Path,
                        help='Explicit JSON environment for a local reviewed-build experiment')
    parser.add_argument('--label', required=True)
    parser.add_argument('--seconds', type=int, default=20, choices=range(10, 61))
    parser.add_argument('--diagnostics-only', action='store_true',
                        help='Skip FPS windows and collect requested diagnostics; never a performance result')
    parser.add_argument('--warmup', type=int, default=20, choices=range(0, 61))
    parser.add_argument('--keep-seconds', type=int, default=0, choices=range(0, 901),
                        help='Leave the owned process alive briefly for separate diagnostic sampling')
    parser.add_argument('--profile-after', action='store_true',
                        help='Capture a separate 8-second CPU profile after all uninstrumented FPS samples')
    parser.add_argument('--render-cost-after', type=int, default=0, choices=range(0, 61),
                        help='Collect separate renderer stage timings; requires a local timer build')
    parser.add_argument('--render-cost-mode', type=int, default=1, choices=(1, 2, 3),
                        help='Local timer mode: all stages (1), readback/wait (2), or per-shader SRT attribution (3)')
    parser.add_argument('--verify-idle', action='store_true',
                        help='Verify the reviewed portable idle patch before warming and timing')
    parser.add_argument('--verify-copy', action='store_true',
                        help='Verify the reviewed title-specific coherent-copy hook too')
    parser.add_argument('--camera-after', action='store_true',
                        help='Perform a short camera round trip after stationary FPS samples')
    parser.add_argument('--unrestricted-cpus', action='store_true',
                        help='Remove checkpoint CPU isolation for a comparison with the portable build')
    parser.add_argument('--render-cpu', type=int,
                        help='Local reviewed-build comparison: isolate Kyty.Gpu on this CPU')
    args = parser.parse_args()
    if args.render_cpu is not None and (args.variant != 'reviewed' or args.unrestricted_cpus):
        parser.error('--render-cpu is for the reviewed variant and excludes --unrestricted-cpus')
    if args.diagnostics_only and not (args.render_cost_after or args.profile_after):
        parser.error('--diagnostics-only requires a diagnostic capture')
    if args.verify_copy: args.verify_idle = True
    if args.render_cpu is not None: args.verify_idle = True
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,50}', args.label):
        parser.error('Invalid label')
    if subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True).returncode == 0:
        parser.error('Close the existing emulator first; it will not be adopted')
    config = json.loads((ROOT/'_Build/profiles/manual-current-launch.json').read_text())
    original = config['command']
    game_dir = original[original.index('--game')+1]
    reviewed_settings = {}
    expected_render_cpu = None
    if args.variant == 'checkpoint':
        if args.binary: parser.error('--binary is only valid for the reviewed variant')
        binary = Path(original[original.index('--')+1])
        command = [str(ROOT/'run-demons-souls.sh')]
        if args.unrestricted_cpus: command.append('--unrestricted-cpus')
        elif '--render-cpu' in original[:original.index('--')]:
            expected_render_cpu = int(original[original.index('--render-cpu')+1])
    else:
        receipt = ROOT/'_Build/profiles/reviewed-current.json'
        if args.binary is None and receipt.is_file():
            reviewed_settings = json.loads(receipt.read_text())
            binary = Path(reviewed_settings['binary']).resolve()
            if sha(binary) != reviewed_settings['binary_sha256']:
                parser.error('Reviewed binary receipt fingerprint mismatch')
            for library in reviewed_settings.get('libraries', []):
                if sha(Path(library['path'])) != library['sha256']:
                    parser.error('Reviewed allocator receipt fingerprint mismatch')
        else:
            binary = (args.binary or ROOT/'_Build/upstream-clean/_Build/review/kyty_emulator').resolve()
        expected_render_cpu = args.render_cpu
        affinity_args = ['--render-cpu', str(args.render_cpu)] if args.render_cpu is not None else []
        # The owned diagnostic retention begins after startup and measurements.
        # Its deadline must not outlive the outer process supervisor.
        supervisor_seconds = max(900, 420 + args.warmup + 3 * (args.seconds + 40) +
                                 args.keep_seconds + (480 if args.profile_after else 0) +
                                 args.render_cost_after + (90 if args.camera_after else 0))
        command = [sys.executable, str(TOOLS/'debug-run.py'), '--label', args.label,
                   '--timeout', str(supervisor_seconds), '--driver-cache', *affinity_args, '--', str(binary), '--game', game_dir,
                   '--screen-width', '1280', '--screen-height', '720', '--console-language', '1',
                   '--shader-validation', 'false', '--shader-log-direction', 'Silent',
                   '--printf-direction', 'Silent']
    expected_sha = sha(binary)
    if args.render_cost_after:
        symbols = subprocess.run(['nm', '--defined-only', str(binary)],
                                 capture_output=True, text=True, check=True).stdout
        if not any(line.split()[-1:] == ['kyty_local_render_cost_mode'] for line in symbols.splitlines()):
            parser.error('Requested renderer diagnostic symbol is absent from this binary')
    out = ROOT/'_Build/profiles'/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'-'+args.label)
    out.mkdir(parents=True)
    result = vars(args) | {'binary': str(binary), 'binary_sha256': expected_sha,
                         'command': command, 'status': 'running', 'fps_phases': [],
                         'effective_render_cpu': expected_render_cpu}
    if args.local_environment:
        result['local_environment'] = str(args.local_environment.resolve())
    if reviewed_settings:
        result['source_commit'] = reviewed_settings['source_commit']
    result['save_files_before'] = {str(p.relative_to(ROOT)):sha(p)
                                  for p in (ROOT/'_SaveData').rglob('*') if p.is_file()}
    def save():
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    print('Evidence:',out,flush=True)
    launcher = None
    environment = {k:v for k,v in os.environ.items() if not k.startswith('KYTY_')}
    environment.update(reviewed_settings.get('environment', {}))
    if args.local_environment:
        if args.variant != 'reviewed':
            parser.error('--local-environment requires --variant reviewed')
        selected = json.loads(args.local_environment.read_text())
        if not isinstance(selected, dict) or any(not isinstance(k, str) or not isinstance(v, str) or
                not k.startswith('KYTY_') for k, v in selected.items()):
            parser.error('Local environment accepts only explicit KYTY_* string values')
        environment.update(selected)
        result['local_environment'] = {'path': str(args.local_environment.resolve()),
                                       'sha256': sha(args.local_environment), 'values': selected}
        save()
    try:
        with (out/'launch.log').open('w') as log:
            launcher = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,
                                        stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+90
            pid=None
            while time.monotonic()<deadline:
                if launcher.poll() is not None: raise RuntimeError('Launcher exited before game startup')
                candidates=subprocess.run(['pgrep','-x','kyty_emulator'],capture_output=True,text=True).stdout.split()
                for candidate in candidates:
                    ancestor=int(candidate)
                    for _ in range(12):
                        if ancestor == launcher.pid:
                            pid=int(candidate)
                            break
                        try: ancestor=int((Path('/proc')/str(ancestor)/'stat').read_text().rsplit(')',1)[1].split()[1])
                        except (FileNotFoundError,IndexError,ValueError): break
                        if ancestor<=1: break
                    if pid: break
                if pid: break
                time.sleep(.2)
            if not pid: raise RuntimeError('No owned emulator appeared')
            proc=Path('/proc')/str(pid)
            identity=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
            if sha(proc/'exe') != expected_sha: raise RuntimeError('Executable fingerprint mismatch')
            result.update(pid=pid,pid_start_ticks=identity)
            def affinity_snapshot():
                masks = {}
                for task in (proc/'task').iterdir():
                    try:
                        masks[task.name] = {'name': (task/'comm').read_text().strip(),
                                            'cpus': sorted(os.sched_getaffinity(int(task.name)))}
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                if args.unrestricted_cpus and any(
                        item['cpus'] != sorted(os.sched_getaffinity(0)) for item in masks.values()):
                    raise RuntimeError('A game thread is unexpectedly pinned during the unrestricted comparison')
                if expected_render_cpu is not None:
                    renderer_tid = result.get('renderer_tid')
                    renderers = [item for tid,item in masks.items()
                                 if (int(tid) == renderer_tid if renderer_tid else item['name'] == 'Kyty.Gpu')]
                    if len(renderers) != 1 or renderers[0]['cpus'] != [expected_render_cpu]:
                        raise RuntimeError('Renderer affinity differs from the requested comparison policy')
                    if any(expected_render_cpu in item['cpus'] for tid,item in masks.items()
                           if (int(tid) != renderer_tid if renderer_tid else item['name'] != 'Kyty.Gpu')):
                        raise RuntimeError('Another game thread shares the isolated renderer CPU')
                return masks
            # Preserve the actual child's relevant launch settings; the outer
            # launcher may add an allocator or driver cache after our Popen.
            child_environment = dict(item.split(b'=', 1) for item in
                (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
            result['runtime_environment'] = {
                key.decode(): value.decode(errors='replace')
                for key,value in child_environment.items()
                if key.startswith(b'KYTY_') or key == b'LD_PRELOAD'}
            result['allocator_libraries'] = {
                path: sha(Path(path)) for path in
                result['runtime_environment'].get('LD_PRELOAD','').split(':')
                if path and 'mimalloc' in Path(path).name and Path(path).is_file()}
            save()
            print('Game PID:',pid,flush=True)
            if args.variant == 'checkpoint':
                deadline=time.monotonic()+160
                while time.monotonic()<deadline:
                    output=(out/'launch.log').read_text()
                    if '8 项优化已在启动阶段启用' in output: break
                    if '未完整启用' in output or launcher.poll() is not None:
                        raise RuntimeError('Frozen checkpoint settings were not restored')
                    time.sleep(.2)
                else: raise RuntimeError('Frozen checkpoint restore timed out')
            with (out/'resume.log').open('w') as log_resume:
                resume = subprocess.Popen([sys.executable,str(TOOLS/'demons-souls-smoke.py'),
                    '--resume','--fast-skip','--stop-at','gameplay','--timeout','240',
                    '--stall-seconds','90'],cwd=ROOT,stdout=log_resume,stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 260
                    while resume.poll() is None:
                        if launcher.poll() is not None or not proc.exists():
                            raise RuntimeError('Owned emulator exited while loading; see launch.log')
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Continue automation timed out')
                        time.sleep(.2)
                    if resume.returncode:
                        raise RuntimeError(f'Continue automation failed ({resume.returncode})')
                finally:
                    if resume.poll() is None:
                        resume.terminate()
                        resume.wait(timeout=10)
            result['resume_output']=(out/'resume.log').read_text()
            result['gameplay_hud_reached'] = True
            if args.verify_idle:
                copy_verification = ''
                if args.verify_copy:
                    copy_verification = '''
copy_site=int(gdb.parse_and_eval("'Loader::DemonsSoulsCopy::(anonymous namespace)::site'"))
assert copy_site
patch_address=int(gdb.parse_and_eval("&'Loader::DemonsSoulsCopy::(anonymous namespace)::patch'"))
saved=bytes(gdb.selected_inferior().read_memory(patch_address,13))
assert bytes(gdb.selected_inferior().read_memory(copy_site,13))==saved
print('PORTABLE_COPY_VERIFIED')
'''
                script = f'''set pagination off
python
import gdb,pathlib,hashlib
p=pathlib.Path('/proc/{pid}')
assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=={identity!r}
assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()=={expected_sha!r}
cave=int(gdb.parse_and_eval("'Loader::DemonsSoulsIdle::(anonymous namespace)::cave'"))
site=int(gdb.parse_and_eval("'Loader::DemonsSoulsIdle::(anonymous namespace)::site'"))
assert cave and site
expected=bytes([0xe8])+(cave-site-5).to_bytes(4,'little',signed=True)
assert bytes(gdb.selected_inferior().read_memory(site,5))==expected
gdb.execute('x/16i '+str(cave))
print('PORTABLE_IDLE_VERIFIED')
{copy_verification}
original=gdb.selected_thread()
for thread in gdb.selected_inferior().threads():
    thread.switch()
    try:
        if bool(gdb.parse_and_eval("'Libs::Graphics::g_gpu_thread'")):
            print('RENDERER_TID='+str(thread.ptid[1]))
    except gdb.error:
        pass
original.switch()
end
detach
'''
                (out/'verify-idle.gdb').write_text(script)
                check=subprocess.run(['sudo','-n','gdb','-q','-nx','-batch','-iex',
                    'set print thread-events off','-p',str(pid),'-x',str(out/'verify-idle.gdb')],
                    capture_output=True,text=True,timeout=30)
                (out/'verify-idle.log').write_text(check.stdout+check.stderr)
                if check.returncode or 'PORTABLE_IDLE_VERIFIED' not in check.stdout:
                    raise RuntimeError('Portable idle patch was not verified; FPS timing cancelled')
                result['portable_idle_verified']=True
                if args.verify_copy:
                    if 'PORTABLE_COPY_VERIFIED' not in check.stdout:
                        raise RuntimeError('Coherent copy hook was not verified')
                    result['portable_copy_verified']=True
                tids=re.findall(r'^RENDERER_TID=(\d+)$', check.stdout, re.M)
                if len(tids)==1: result['renderer_tid']=int(tids[0])
            if args.render_cpu is not None:
                if not result.get('renderer_tid'):
                    raise RuntimeError('A unique renderer TLS identity is required for the clean build')
                match = re.search(rf'^PID {pid}; evidence: (.+)$',
                                  (out/'launch.log').read_text(), re.M)
                if not match: raise RuntimeError('Cannot locate the owned CPU policy supervisor')
                evidence = Path(match.group(1)).resolve()
                if not evidence.is_relative_to(ROOT/'_Build/experiments'):
                    raise RuntimeError('Unexpected CPU policy evidence directory')
                token = 'benchmark-renderer-' + identity
                policy = {'identity': {'pid':pid, 'start_ticks':identity, 'binary_sha256':expected_sha},
                          'token':token, 'worker_cpus': sorted(os.sched_getaffinity(0)-{args.render_cpu}),
                          'renderer': {'tid':result['renderer_tid'],
                                       'start_ticks':start_ticks(result['renderer_tid'])}}
                atomic_json(evidence/'cpu-policy.json', policy)
                deadline = time.monotonic()+10
                while time.monotonic()<deadline:
                    ack_path = evidence/'cpu-policy-applied.json'
                    ack = json.loads(ack_path.read_text()) if ack_path.exists() else {}
                    if ack.get('token') == token:
                        if ack.get('error'): raise RuntimeError(ack['error'])
                        affinity_snapshot()
                        result['applied_cpu_policy'] = policy
                        break
                    time.sleep(.2)
                else: raise RuntimeError('Renderer CPU policy was not acknowledged')
            save()
            print('Gameplay HUD reached; warming for',args.warmup,'seconds',flush=True)
            (out/'process-maps.txt').write_text((proc/'maps').read_text())
            if result.get('renderer_tid') and hasattr(os, 'sched_getaffinity'):
                result['renderer_affinity_before_timing'] = sorted(
                    os.sched_getaffinity(result['renderer_tid']))
            time.sleep(args.warmup)
            for index in range(0 if args.diagnostics_only else 3):
                if (proc/'stat').read_text().rsplit(')',1)[1].split()[19] != identity:
                    raise RuntimeError('Emulator identity changed')
                before=set((ROOT/'_Build/profiles').iterdir())
                affinity_before = affinity_snapshot()
                capture=subprocess.run([sys.executable,str(TOOLS/'quick-benchmark-game.py'),'--pid',str(pid),
                    '--seconds',str(args.seconds),'--camera-frames','0','--screenshots',
                    '--label',args.label+'-sample'+str(index+1)],cwd=ROOT,text=True,capture_output=True,
                    timeout=args.seconds+35)
                (out/f'sample{index+1}.log').write_text(capture.stdout+capture.stderr)
                capture.check_returncode()
                created=[p for p in (ROOT/'_Build/profiles').iterdir() if p not in before and
                         p.name.endswith(args.label+'-sample'+str(index+1))]
                if len(created)!=1: raise RuntimeError('Ambiguous measurement output')
                measurement=json.loads((created[0]/'result.json').read_text())
                result['fps_phases'].append({'path':str(created[0]),'result':measurement,
                    'affinity_before': affinity_before, 'affinity_after': affinity_snapshot()})
                save()
                print('Measurement',index+1,[{'fps':p['measured_fps'],'frames':p['frames'],'seconds':p['elapsed_seconds']} for p in measurement['phases']],flush=True)
            result['status']='diagnostics-only' if args.diagnostics_only else 'measured'
            save()
            if args.camera_after:
                print('Stationary samples complete; checking a camera round trip', flush=True)
                camera=subprocess.run([sys.executable,str(TOOLS/'quick-benchmark-game.py'),
                    '--pid',str(pid),'--seconds','5','--camera-frames','8','--screenshots',
                    '--label',args.label+'-camera'],cwd=ROOT,capture_output=True,text=True,timeout=45)
                (out/'camera.log').write_text(camera.stdout+camera.stderr)
                result['camera_returncode']=camera.returncode
                if camera.returncode == 0: result['camera']=json.loads(camera.stdout)
                save()
                camera.check_returncode()
            if args.profile_after:
                print('FPS measurement complete; collecting separate CPU diagnostics', flush=True)
                with (out/'profile.log').open('w') as profile_log:
                    profile = subprocess.run([sys.executable, str(TOOLS/'profile-game.py'),
                        '--pid', str(pid), '--seconds', '8', '--callgraph', '--event', 'cycles',
                        *(['--tid',str(result['renderer_tid'])] if result.get('renderer_tid') else []),
                        '--label', args.label+'-cost'], cwd=ROOT, stdout=profile_log,
                        stderr=subprocess.STDOUT, timeout=420)
                result['profile_returncode'] = profile.returncode
                save()
            if args.render_cost_after:
                def set_render_cost(value):
                    script = f'''set pagination off
python
import gdb,pathlib,hashlib
p=pathlib.Path('/proc/{pid}')
assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=={identity!r}
assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()=={expected_sha!r}
address=int(gdb.parse_and_eval('&kyty_local_render_cost_mode'))
gdb.selected_inferior().write_memory(address,({value}).to_bytes(4,'little'))
assert bytes(gdb.selected_inferior().read_memory(address,4))==({value}).to_bytes(4,'little')
print('RENDER_COST_MODE={value}')
end
detach
'''
                    script_path=out/f'render-cost-{value}.gdb'
                    script_path.write_text(script)
                    check=subprocess.run(['sudo','-n','gdb','-q','-nx','-batch','-iex',
                        'set print thread-events off','-p',str(pid),'-x',str(script_path)],
                        capture_output=True,text=True,timeout=30)
                    (out/f'render-cost-{value}.log').write_text(check.stdout+check.stderr)
                    if check.returncode or f'RENDER_COST_MODE={value}' not in check.stdout:
                        raise RuntimeError(f'Could not set local renderer timer to {value}')
                print('Collecting separate renderer stage timings',flush=True)
                set_render_cost(args.render_cost_mode)
                try:
                    time.sleep(args.render_cost_after)
                finally:
                    if proc.exists(): set_render_cost(0)
                result['render_cost_collected']=True
                save()
                if args.render_cost_mode == 3:
                    # Consume the report while its plan addresses still belong to this
                    # owned process. Do not rely on a later interactive keep window.
                    match = re.search(rf'^PID {pid}; evidence: (.+)$',
                                      (out/'launch.log').read_text(), re.M)
                    if not match:
                        raise RuntimeError('Cannot locate owned SRT diagnostic log')
                    evidence = Path(match.group(1)).resolve()
                    if not evidence.is_relative_to(ROOT/'_Build/experiments'):
                        raise RuntimeError('Unexpected SRT evidence directory')
                    report = subprocess.run([sys.executable, str(TOOLS/'srt-cost-report.py'),
                        str(evidence/'stdout.log'), '--output', str(out/'srt-cost.json')],
                        capture_output=True, text=True, timeout=15)
                    (out/'srt-cost-report.log').write_text(report.stdout+report.stderr)
                    report.check_returncode()
                    capture = subprocess.run([sys.executable, str(TOOLS/'capture-srt-plans.py'),
                        str(out/'summary.json'), str(out/'srt-cost.json'), '--output', str(out/'srt-plans')],
                        capture_output=True, text=True, timeout=35)
                    (out/'srt-plans.log').write_text(capture.stdout+capture.stderr)
                    result['srt_plan_capture_returncode'] = capture.returncode
                    save()
            if args.keep_seconds:
                result['diagnostics_deadline_monotonic'] = time.monotonic() + args.keep_seconds
                result['diagnostics_ready_at'] = datetime.datetime.now().isoformat()
                save()
                print('Available for separate diagnostics:',pid,'for',args.keep_seconds,'seconds',flush=True)
                while launcher.poll() is None:
                    remaining = result['diagnostics_deadline_monotonic'] - time.monotonic()
                    if remaining <= 0:
                        break
                    time.sleep(min(1.0, remaining))
                result['diagnostics_end_reason'] = (
                    'owned_launcher_exited' if launcher.poll() is not None else 'retention_deadline')
                result['diagnostics_finished_at'] = datetime.datetime.now().isoformat()
                save()
    except BaseException as exc:
        result['status']='failed'
        result['error']=str(exc)
        if isinstance(exc, Exception) and launcher and launcher.poll() is None:
            capture_startup_failure(result, out)
        raise
    finally:
        if launcher and launcher.poll() is None:
            launcher.terminate()
            try: launcher.wait(timeout=30)
            except subprocess.TimeoutExpired:
                result['stop_error']='Launcher did not stop in 30 seconds; no unrelated process was signaled'
        if launcher: result['launcher_returncode']=launcher.poll()
        save()
        print('Finished:',out,flush=True)


if __name__=='__main__': main()
