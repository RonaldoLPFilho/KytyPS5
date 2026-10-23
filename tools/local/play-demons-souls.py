#!/usr/bin/env python3
"""Play the selected, fingerprinted checkpoint with its native optimization settings."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time

from game_recording import GameRecorder, find_game_window, fit_4k_window
from debug_cpu_policy import atomic_json, start_ticks

ROOT = Path(__file__).resolve().parents[2]
AOT_SHA = '297dd1cec0cb03572acc2ce640fe1275d8a17644508979b8201b08749ce33334'
MIMALLOC_SHA = 'b0f267c17975ee39391331d0f02dd3682233a67e8464095db1407cd4e37f3853'
NATIVE_MODES = {
    'kyty_local_srt_native_mode': 1,
    'kyty_local_srt_predicate_mode': 1,
    'kyty_local_preparation_scratch_mode': 1,
    'kyty_local_preparation_lookup_mode': 1,
    'kyty_local_preparation_trim_mode': 1,
    'kyty_local_buffer_residency_mode': 1,
    'kyty_local_pipeline_index_mode': 1,
    'kyty_local_copy_feedback_mode': 1,
    'kyty_local_srt_native_diagnostics': 0,
    'kyty_local_render_cost_mode': 0,
}


def mapped_files(proc):
    try:
        lines = (proc / 'maps').read_text().splitlines()
    except FileNotFoundError:
        # Let the supervisor finish recording an early child exit.
        return set()
    return {fields[5] for line in lines if len(fields := line.split(maxsplit=5)) == 6}


def startup_script(proc, identity, library, modes, binary_sha, find_renderer, aot_sha=AOT_SHA):
    # The frozen build exposes these atomic switches to GDB. Resolve every
    # symbol before writing, and detach before allowing manual gameplay.
    return f'''set pagination off
python
import gdb,pathlib,hashlib,json
try:
    p=pathlib.Path({str(proc)!r})
    assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=={identity!r}
    assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()=={binary_sha!r}
    loaded={{fields[5] for line in p.joinpath('maps').read_text().splitlines()
            if len(fields:=line.split(maxsplit=5))==6}}
    assert {str(library)!r} in loaded
    assert hashlib.sha256(pathlib.Path({str(library)!r}).read_bytes()).hexdigest()=={aot_sha!r}
    modes={modes!r}
    addresses={{name:int(gdb.parse_and_eval('&'+name)) for name in modes}}
    inferior=gdb.selected_inferior()
    for name,value in modes.items():
        inferior.write_memory(addresses[name],value.to_bytes(4,'little'))
    actual={{name:int.from_bytes(bytes(inferior.read_memory(address,4)),'little')
            for name,address in addresses.items()}}
    assert actual==modes
    print('NATIVE_CHECKPOINT_READY='+json.dumps(actual))
    if {find_renderer!r}:
        original=gdb.selected_thread()
        renderers=[]
        for thread in inferior.threads():
            thread.switch()
            try:
                if bool(gdb.parse_and_eval("'Libs::Graphics::g_gpu_thread'")):
                    tid=thread.ptid[1]
                    ticks=p.joinpath('task',str(tid),'stat').read_text().rsplit(')',1)[1].split()[19]
                    renderers.append({{'tid':tid,'start_ticks':ticks}})
            except gdb.error:
                pass
        original.switch()
        assert len(renderers)==1, 'Expected one renderer TLS identity: '+str(renderers)
        print('NATIVE_RENDERER='+json.dumps(renderers[0]))
finally:
    gdb.execute('detach')
end
'''


def switches_renderer(proc, modes, out, status, binary_sha, library, aot_sha, identity, evidence):
    """Cleaned builds: the emulator reports its environment switches in one line."""
    output = (evidence / 'stdout.log').read_text(errors='replace')
    lines = [line for line in output.splitlines() if line.startswith('Performance switches: ')]
    status.write('\n'.join(lines) + '\n')
    status.flush()
    actual = dict(item.split('=', 1) for item in lines[-1][len('Performance switches: '):].split()) if lines else {}
    if len(lines) != 1 or actual != modes:
        raise RuntimeError(f'优化开关未完整启用（{actual} != {modes}），请检查 {out}/baseline.log')
    (out / 'startup-modes.json').write_text(json.dumps({
        'pid': int(proc.name), 'pid_start_ticks': identity, 'binary_sha256': binary_sha,
        'library': str(library), 'aot_sha256': aot_sha, 'modes': actual, 'startup_control': 'switches',
    }, indent=2) + '\n')
    tids = [int(task.name) for task in (proc / 'task').iterdir()
            if task.joinpath('comm').read_text().strip() == 'Kyty.Gpu']
    if len(tids) != 1:
        raise RuntimeError('启动时未定位到唯一渲染线程，无法验证绑核')
    return {'tid': tids[0], 'start_ticks': start_ticks(tids[0])}


def enable_native_checkpoint(proc, identity, library, modes, out, status, binary_sha, render_cpu, evidence,
                             native_startup=False, aot_sha=AOT_SHA):
    if native_startup == 'switches':
        renderer = switches_renderer(proc, modes, out, status, binary_sha, library, aot_sha, identity, evidence)
        if render_cpu is None:
            return
        apply_render_policy(proc, identity, binary_sha, render_cpu, evidence, out, renderer)
        return
    if native_startup:
        output = (evidence / 'stdout.log').read_text(errors='replace')
        status.write('\n'.join(line for line in output.splitlines() if line.startswith('NATIVE_')) + '\n')
        returncode = 0
    else:
        script = out / 'startup-modes.gdb'
        script.write_text(startup_script(proc, identity, library, modes, binary_sha, render_cpu is not None, aot_sha))
        result = subprocess.run(['sudo', '-n', 'gdb', '-q', '-nx', '-batch',
            '-iex', 'set print thread-events off', '-p', proc.name, '-x', str(script)],
            capture_output=True, text=True, timeout=30)
        output, returncode = result.stdout, result.returncode
        status.write(output + result.stderr)
    status.flush()
    records = re.findall(r'^NATIVE_CHECKPOINT_READY=(.+)$', output, re.M)
    if returncode or len(records) != 1 or json.loads(records[0]) != modes:
        raise RuntimeError(f'优化开关未完整启用，请检查 {out}/baseline.log')
    (out / 'startup-modes.json').write_text(json.dumps({
        'pid': int(proc.name), 'pid_start_ticks': identity, 'binary_sha256': binary_sha,
        'library': str(library), 'aot_sha256': aot_sha, 'modes': json.loads(records[0]),
        'startup_control': 'native' if native_startup else 'debugger',
    }, indent=2) + '\n')
    if render_cpu is None:
        return
    renderers = re.findall(r'^NATIVE_RENDERER=(.+)$', output, re.M)
    if len(renderers) != 1:
        raise RuntimeError('启动时未定位到唯一渲染线程，无法验证绑核')
    renderer = json.loads(renderers[0])
    if native_startup:
        task = proc / 'task' / str(renderer['tid'])
        if task.joinpath('comm').read_text().strip() != 'Kyty.Gpu':
            raise RuntimeError('模拟器回报的渲染线程身份不符')
        renderer['start_ticks'] = start_ticks(renderer['tid'])
    if start_ticks(renderer['tid']) != renderer['start_ticks']:
        raise RuntimeError('渲染线程身份已变化')
    apply_render_policy(proc, identity, binary_sha, render_cpu, evidence, out, renderer)


def apply_render_policy(proc, identity, binary_sha, render_cpu, evidence, out, renderer):
    token = 'startup-renderer-' + identity
    workers = sorted(os.sched_getaffinity(0) - {render_cpu})
    policy = {'identity': {'pid': int(proc.name), 'start_ticks': identity, 'binary_sha256': binary_sha},
              'token': token, 'worker_cpus': workers, 'renderer': renderer}
    atomic_json(evidence / 'cpu-policy.json', policy)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        path = evidence / 'cpu-policy-applied.json'
        ack = json.loads(path.read_text()) if path.exists() else {}
        if ack.get('token') == token:
            if ack.get('error'):
                raise RuntimeError(ack['error'])
            masks = ack.get('affinities', {})
            expected = {str(renderer['tid']): [render_cpu]}
            if (masks.get(str(renderer['tid'])) != [render_cpu] or
                    any(mask != expected.get(tid, workers) for tid, mask in masks.items()) or
                    sorted(os.sched_getaffinity(renderer['tid'])) != [render_cpu]):
                raise RuntimeError('CPU 策略已应答，但实际线程亲和性不符')
            atomic_json(out / 'startup-cpu-policy.json', ack)
            print(f'已验证绑核：渲染线程 {renderer["tid"]} 使用 CPU {render_cpu}，其余游戏线程使用 {workers}。', flush=True)
            return
        time.sleep(.2)
    raise RuntimeError('CPU 绑核策略未在期限内应答')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='只显示启动配置，不启动游戏')
    parser.add_argument('--checkpoint-config', type=Path,
                        default=ROOT / '_Build/profiles/manual-current-launch.json',
                        help='使用指定的带 SHA256 指纹检查点配置，供候选版本验证')
    parser.add_argument('--unrestricted-cpus', action='store_true',
                        help='本地对照测试：取消固定检查点的 CPU 绑核，保留其余配置')
    resolution = parser.add_mutually_exclusive_group()
    resolution.add_argument('--2k', dest='two_k', action='store_true', help='2560×1440 输出（普通缩放，非 DLSS）')
    resolution.add_argument('--4k', dest='four_k', action='store_true', help='3840×2160 输出（普通缩放，非 DLSS）')
    parser.add_argument('--output-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--record', nargs='?', const='auto', metavar='FILE.mp4',
                        help='启动后录制游戏窗口；默认保存至 recordings/，可指定新 MP4 路径')
    parser.add_argument('--record-fps', type=int, default=30, choices=(15, 30, 60), help='录像帧率，默认 30')
    parser.add_argument('--no-audio', action='store_true', help='录像不含扬声器声音')
    hud = parser.add_mutually_exclusive_group()
    hud.add_argument('--record-hud', dest='record_hud', action='store_true', default=True,
                     help='录像叠加游戏 FPS / 帧数 / 停滞时间（默认开启）')
    hud.add_argument('--no-record-hud', dest='record_hud', action='store_false', help='录制干净画面，不叠加帧率信息')
    parser.add_argument('--precompile-shaders', action='store_true', help='预编译全部已收集的 shader 和管线后退出，不进入游戏')
    parser.add_argument('--shader-warmup-seconds', type=int, default=0, help='启动预编译秒数预算，0 表示全部完成（默认）')
    parser.add_argument('--diagnose-stutter', action='store_true', help='额外启用完整帧阶段计时（会影响 FPS，普通游玩无需添加）')
    parser.add_argument('--no-playtest-log', action='store_true', help='对照测量时关闭默认的轻量游玩日志')
    parser.add_argument('--verify-native-resources', action='store_true', help='候选版本校验：逐次比对本机资源输出与原求值器（不用于 FPS 测量）')
    args, extra = parser.parse_known_args()
    args.allocator = "mimalloc"
    if not 0 <= args.shader_warmup_seconds <= 3600:
        parser.error("--shader-warmup-seconds must be between 0 and 3600")
    if args.precompile_shaders and args.shader_warmup_seconds:
        parser.error("--precompile-shaders requires an unlimited warmup budget")
    if args.two_k or args.four_k:
        size = [2560, 1440] if args.two_k else [3840, 2160]
        option = '--2k' if args.two_k else '--4k'
        if args.output_size and args.output_size != size:
            parser.error(f'{option} cannot be combined with a different --output-size')
        args.output_size = size
    recording = None
    if args.record is not None:
        recording = (ROOT / 'recordings' / (datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.mp4')
                     if args.record == 'auto' else Path(args.record).expanduser().resolve())
        if recording.suffix.lower() != '.mp4' or recording.exists():
            parser.error('--record needs a new .mp4 file; existing videos are never overwritten')
        for tool in ('ffmpeg', 'ffprobe', 'xwininfo', 'xprop'):
            if not shutil.which(tool):
                parser.error(f'Recording requires {tool}')
    config = json.loads(args.checkpoint_config.read_text())
    binary_sha = config['binary_sha256']
    aot_sha = config.get('aot_sha256', AOT_SHA)
    args.checkpoint = config['checkpoint']
    if (not re.fullmatch('[0-9a-f]{64}', binary_sha) or not re.fullmatch('[0-9a-f]{64}', aot_sha)
            or not re.fullmatch('[a-z0-9-]+', args.checkpoint)):
        parser.error('Invalid checkpoint identity')
    config['startup_modes'] = NATIVE_MODES | {'kyty_local_render_cost_mode': int(args.diagnose_stutter)}
    if 'KYTY_FRAME_PIPELINE' in config['environment']:
        frame_pipeline = config['environment']['KYTY_FRAME_PIPELINE']
        if frame_pipeline not in ('0', '1', '2'):
            parser.error('KYTY_FRAME_PIPELINE must be 0, 1, or 2')
        config['startup_modes']['kyty_local_frame_pipeline_mode'] = int(frame_pipeline)
    if 'KYTY_GUEST_READBACK_GUARDS' in config['environment']:
        readback_guards = config['environment']['KYTY_GUEST_READBACK_GUARDS']
        if readback_guards not in ('0', '1'):
            parser.error('KYTY_GUEST_READBACK_GUARDS must be 0 or 1')
        config['startup_modes']['kyty_local_guest_readback_guards_mode'] = int(readback_guards)
    native_startup = config.get('startup_control') == 'native'
    switches_startup = config.get('startup_control') == 'switches'
    if switches_startup:
        switch_names = {'KYTY_SRT_NATIVE', 'KYTY_SRT_PREDICATES', 'KYTY_PREPARATION_SCRATCH',
                        'KYTY_PREPARATION_LOOKUP', 'KYTY_PREPARATION_TRIM', 'KYTY_SPECIALIZATION_GUARD',
                        'KYTY_PIPELINE_INDEX', 'KYTY_BINDING_SCRATCH', 'KYTY_DRAW_RUN_RANGES',
                        'KYTY_BUFFER_RESIDENCY', 'KYTY_COPY_FEEDBACK', 'KYTY_ASYNC_LOD_STATS',
                        'KYTY_BACKING_READ', 'KYTY_STREAM_UPLOAD', 'KYTY_FRAME_PIPELINE',
                        'KYTY_IMAGE_BARRIER_DEDUPE', 'KYTY_IMAGE_POOL', 'KYTY_PENDING_DRAIN',
                        'KYTY_DISPATCH_BATCH', 'KYTY_VULKAN_RECORDING', 'KYTY_DEFERRED_SUBMIT',
                        'KYTY_NATIVE_XPR', 'KYTY_WRITE_WINDOW_HANDOFF', 'KYTY_READBACK_DETACH',
                        'KYTY_ASYNC_WRITE_READBACK',
                        'KYTY_READBACK_SLOTS', 'KYTY_ASYNC_UPLOAD', 'KYTY_BDA_DIRTY_REGIONS', 'KYTY_GLOBAL_BARRIER_DEDUPE',
                        'KYTY_RANGE_SET_FAST', 'KYTY_IMAGE_GRANULES', 'KYTY_NATIVE_XPR_PREDICT', 'KYTY_NATIVE_IMAGE_PROOF', 'KYTY_TEXTURE_RESOLVE_PAGES', 'KYTY_ASYNC_REPROTECT', 'KYTY_READBACK_NARROW', 'KYTY_READBACK_QUEUE'}
        config['startup_modes'] = {name: value for name, value in config['environment'].items()
                                   if name in switch_names}
    for env_name, symbol, choices in () if switches_startup else (
        ('KYTY_BACKING_READ', 'kyty_local_backing_read_mode', ('0', '1', '2')),
        ('KYTY_STREAM_UPLOAD', 'kyty_local_stream_upload_mode', ('0', '1', '2', '3', '4', '5')),
        ('KYTY_DRAW_RUN_RANGES', 'kyty_local_draw_run_ranges_mode', ('0', '1', '2', '3')),
        ('KYTY_DIRECT_MULTI_DRAW', 'kyty_local_direct_multi_draw_mode', ('0', '1', '2', '3', '4')),
        ('KYTY_BDA_RANGES', 'kyty_local_bda_ranges_mode', ('0', '1', '2', '3')),
        ('KYTY_BINDING_SCRATCH', 'kyty_local_binding_scratch_mode', ('0', '1')),
        ('KYTY_BINDING_REPEAT_DIAGNOSTICS', 'kyty_local_binding_repeat_diagnostics', ('0', '1', '2')),
        ('KYTY_DESCRIPTOR_PLAN', 'kyty_local_descriptor_plan_mode', ('0', '1', '2')),
        ('KYTY_SPECIALIZATION_MEMO', 'kyty_local_specialization_memo_mode', ('0', '1', '2')),
        ('KYTY_SPECIALIZATION_GUARD', 'kyty_local_specialization_guard_mode', tuple(str(i) for i in range(7))),
        ('KYTY_GRAPHICS_STATE', 'kyty_local_graphics_state_mode', ('0', '1', '2', '3')),
        ('KYTY_IMAGE_BARRIER_DEDUPE', 'kyty_local_image_barrier_dedupe', ('0', '1')),
        ('KYTY_PENDING_DRAIN', 'kyty_local_pending_drain_mode', ('0', '1')),
        ('KYTY_DRAW_RUN_PRECHECK', 'kyty_local_draw_run_precheck_mode', ('0', '1')),
        ('KYTY_WRITE_MASK_STATS', 'kyty_local_write_mask_stats_mode', ('0', '1')),
        # Experiment switches (default off); the emulator acknowledges them only when set.
        ('KYTY_HEADLESS', 'kyty_local_headless_mode', ('0', '1')),
        ('KYTY_FEEDBACK_PREFETCH', 'kyty_local_feedback_prefetch_mode', ('0', '1', '2')),
        ('KYTY_COPY_FEEDBACK', 'kyty_local_copy_feedback_mode', ('0', '1', '2')),
        ('KYTY_IMAGE_POOL', 'kyty_local_image_pool_mode', ('0', '1')),
        ('KYTY_VULKAN_RECORDING', 'kyty_local_vulkan_recording_mode', ('0', '1', '2', '3', '4')),
        ('KYTY_DEFERRED_SUBMIT', 'kyty_local_deferred_submit_mode', ('0', '1')),
    ):
        if env_name in config['environment']:
            value = config['environment'][env_name]
            if value not in choices:
                parser.error(f'{env_name} must be one of {", ".join(choices)}')
            config['startup_modes'][symbol] = int(value)
    if args.precompile_shaders and config.get('shader_warmup_version', 1) < 2:
        parser.error('This checkpoint does not support pipeline precompilation; select a version 2 cache checkpoint')
    config['environment']['KYTY_SHADER_WARMUP_SECONDS'] = str(args.shader_warmup_seconds)
    if args.precompile_shaders:
        config['environment'].update(KYTY_SHADER_WARMUP='1', KYTY_SHADER_WARMUP_ONLY='1')
    if args.verify_native_resources:
        if not native_startup:
            parser.error('Resource verification requires a native startup checkpoint')
        config['startup_modes'].update(kyty_local_srt_native_mode=2, kyty_local_srt_predicate_mode=2,
                                       kyty_local_srt_native_diagnostics=1)
    if native_startup:
        config['environment'].update(KYTY_NATIVE_CHECKPOINT='1',
                                     KYTY_NATIVE_RENDER_COST=str(int(args.diagnose_stutter)),
                                     KYTY_NATIVE_VERIFY_RESOURCES=str(int(args.verify_native_resources)))
    allocator_library = None
    if args.allocator == 'mimalloc':
        allocator_library = ROOT / '_Build/profiles/libraries' / MIMALLOC_SHA / 'libmimalloc.so'
        if (not allocator_library.is_file() or
                hashlib.sha256(allocator_library.read_bytes()).hexdigest() != MIMALLOC_SHA):
            parser.error(f'Allocator fingerprint mismatch or missing library: {allocator_library}')
        preload = config['environment'].get('LD_PRELOAD', os.environ.get('LD_PRELOAD', ''))
        config['environment']['LD_PRELOAD'] = str(allocator_library) + (':' + preload if preload else '')
        config['external_libraries'] = [{'path': str(allocator_library), 'sha256': MIMALLOC_SHA,
                                        'source': 'https://github.com/microsoft/mimalloc', 'tag': 'v3.5.1'}]
    command = config['command']
    command[1] = str(Path(__file__).with_name('debug-run.py'))
    if not args.no_playtest_log and '--playtest-log' not in command[:command.index('--')]:
        command.insert(command.index('--'), '--playtest-log')
    if args.no_playtest_log and '--playtest-log' in command[:command.index('--')]:
        command.remove('--playtest-log')
    config['playtest_logging'] = {'enabled': not args.no_playtest_log,
                                 'native_supported': config.get('playtest_log_version', 0) >= 1,
                                 'full_stage_timing': args.diagnose_stutter}
    if args.unrestricted_cpus and '--render-cpu' in command[:command.index('--')]:
        index = command.index('--render-cpu')
        del command[index:index + 2]
    render_cpu = (int(command[command.index('--render-cpu') + 1])
                  if '--render-cpu' in command[:command.index('--')] else None)
    cpu_affinity = config.get('cpu_affinity') if not args.unrestricted_cpus else None
    if cpu_affinity is not None:
        if not isinstance(cpu_affinity, list) or not cpu_affinity or any(type(cpu) is not int or cpu < 0 for cpu in cpu_affinity):
            parser.error('Invalid checkpoint CPU affinity')
        cpu_affinity = sorted(set(cpu_affinity) & os.sched_getaffinity(0))
        if len(cpu_affinity) < 2 or (render_cpu is not None and render_cpu not in cpu_affinity):
            parser.error('Checkpoint CPU affinity is unavailable in this session')
        config['cpu_affinity'] = cpu_affinity
    binary = Path(command[command.index('--') + 1]).resolve()
    library = Path(config['environment']['KYTY_SRT_AOT_LIBRARY']).resolve()
    for path, expected in [(binary, binary_sha), (library, aot_sha)]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            parser.error(f'Checkpoint fingerprint mismatch: {path}')
    if os.environ.get('KYTY_GAME_DIR'):
        command[command.index('--game') + 1] = os.environ['KYTY_GAME_DIR']
    if not Path(command[command.index('--game') + 1]).is_dir():
        parser.error('Game directory is unavailable; check the game disk')
    if args.output_size:
        if not all(320 <= value <= 8192 for value in args.output_size):
            parser.error('Output dimensions must be between 320 and 8192')
        for option, value in zip(('--screen-width', '--screen-height'), args.output_size):
            command[command.index(option) + 1] = str(value)
    command.extend(extra[1:] if extra[:1] == ['--'] else extra)
    environment = {k: v for k, v in os.environ.items() if not k.startswith('KYTY_')}
    environment.update(config['environment'])
    if 'KYTY_LIVE_FILE' in environment and not args.dry_run:
        # The live thread runs the file's last commands on start: a poke left by an
        # earlier process names an address of that process.
        Path(environment['KYTY_LIVE_FILE']).write_text('id 0\n')
    if args.dry_run:
        print(json.dumps(config | {'binary_sha256': binary_sha, 'checkpoint': args.checkpoint, 'aot_sha256': aot_sha,
                                   'allocator': args.allocator,
                                   'input': 'manual; eight native optimizations applied once the AOT library loads',
                                   'diagnose_stutter': args.diagnose_stutter,
                                   'recording': {'path': str(recording), 'fps': args.record_fps,
                                                 'hud': args.record_hud,
                                                 'speaker_audio': not args.no_audio} if recording else None}, indent=2))
        return
    if cpu_affinity is not None:
        os.sched_setaffinity(0, cpu_affinity)
    if subprocess.run(['pgrep', '-x', 'kyty_emulator'], capture_output=True).returncode == 0:
        parser.error('An emulator is already running; close it before launching this checkpoint')
    out = ROOT / '_Build/run-logs' / (datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-manual-' + args.checkpoint)
    out.mkdir(parents=True)
    (out / 'launch.json').write_text(json.dumps(config, indent=2) + '\n')
    session = {'state': 'launching', 'directory': str(out), 'binary_sha256': binary_sha,
               'checkpoint': args.checkpoint, 'checkpoint_config': str(args.checkpoint_config.resolve()),
               'unix_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
    atomic_json(out / 'session.json', session)
    if not args.precompile_shaders:
        latest = out.parent / f'.latest-{os.getpid()}'
        latest.symlink_to(out.name, target_is_directory=True)
        latest.replace(out.parent / 'latest')
    receipt = config.get('build_receipt')
    if receipt and Path(receipt).is_file():
        shutil.copy2(receipt, out / 'build-receipt.json')
    driver = None
    evidence = None
    recorder = GameRecorder(recording, args.record_fps, not args.no_audio, args.record_hud) if recording else None

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with (out / 'driver.log').open('w') as log, (out / 'baseline.log').open('w') as status:
            driver = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            print(f'已启动 {args.checkpoint}，宿主分配器：{args.allocator}。日志：{out}', flush=True)
            print('正在预编译已收集的 shader 和管线。' if args.precompile_shaders else
                  '正在初始化优化配置，完成后可自行选择 New Game 或 Continue。', flush=True)
            proc = None
            for _ in range(225):
                match = re.search(r'^PID (\d+); evidence: (.+)$', (out / 'driver.log').read_text(), re.M)
                if match:
                    proc = Path('/proc') / match[1]
                    evidence = Path(match[2]).resolve()
                    if not evidence.is_relative_to(ROOT / '_Build/experiments'):
                        raise RuntimeError('Unexpected supervisor evidence path')
                    metadata = json.loads((evidence / 'run.json').read_text())
                    identity = str(metadata['pid_start_ticks'])
                    (out / 'evidence').symlink_to(evidence, target_is_directory=True)
                    (out / 'summary.md').symlink_to(evidence / 'playtest-summary.md')
                    (out / 'summary.json').symlink_to(evidence / 'playtest-summary.json')
                    session.update(state='running', evidence=str(evidence), pid=int(proc.name), pid_start_ticks=identity)
                    atomic_json(out / 'session.json', session)
                    break
                if driver.poll() is not None:
                    raise RuntimeError(f'Launch failed; inspect {out}/driver.log')
                time.sleep(.2)
            if proc is None:
                raise RuntimeError('Supervisor did not identify the game')

            def check_identity():
                if (driver.poll() is not None or
                        (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19] != identity or
                        hashlib.sha256((proc / 'exe').read_bytes()).hexdigest() != binary_sha):
                    raise RuntimeError('Launched game identity changed')

            check_identity()
            if allocator_library:
                maps = (proc / 'maps').read_text()
                (out / 'allocator-maps.txt').write_text(maps)
                loaded = {fields[5] for line in maps.splitlines()
                          if len(fields := line.split(maxsplit=5)) == 6}
                if str(allocator_library.resolve()) not in loaded:
                    raise RuntimeError('The verified allocator was not loaded by the emulator')
            if args.precompile_shaders:
                seen = 0
                game_log = evidence / 'stdout.log'
                while True:
                    if game_log.exists():
                        lines = game_log.read_text(errors='replace').splitlines()
                        for line in lines[seen:]:
                            if 'warmup:' in line or 'Shader precompile:' in line:
                                print(line, flush=True)
                        seen = len(lines)
                    if driver.poll() is not None:
                        break
                    try:
                        driver.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
                if driver.returncode != 0 or 'Shader precompile: complete' not in game_log.read_text(errors='replace'):
                    raise RuntimeError(f'Shader precompilation failed; inspect {game_log}')
                return
            if recorder or args.output_size == [3840, 2160]:
                window = None
                deadline = time.monotonic() + 30
                while driver.poll() is None and time.monotonic() < deadline:
                    window = find_game_window(proc.name)
                    if window:
                        break
                    time.sleep(.2)
                if window:
                    if args.output_size == [3840, 2160]:
                        try:
                            fit_4k_window(window)
                        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                            print(f'4K 窗口定位未完成：{exc}', flush=True)
                    if recorder:
                        try:
                            recorder.start(window)
                        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                            print(f'无法启动录像，游戏继续运行：{exc}', flush=True)
                else:
                    print('未找到游戏窗口，未启动录像或调整窗口位置。', flush=True)
            # This build integrates the idle/copy fixes and has no legacy cheat
            # marker or checkpoint toggle symbols. Wait for its AOT library.
            ready = False
            game_log = evidence / 'stdout.log'
            deadline = time.monotonic() + 600
            while driver.poll() is None and time.monotonic() < deadline:
                if str(library) in mapped_files(proc):
                    acknowledgement = (game_log.read_text(errors='replace')
                                       if native_startup or switches_startup else '')
                    if switches_startup:
                        if 'Performance switches: ' in acknowledgement:
                            ready = True
                            break
                    elif not native_startup or ('NATIVE_CHECKPOINT_READY=' in acknowledgement and
                                                'NATIVE_RENDERER=' in acknowledgement):
                        ready = True
                        break
                time.sleep(.2)
            if ready:
                check_identity()
                enable_native_checkpoint(proc, identity, library, config['startup_modes'], out, status,
                                         binary_sha, render_cpu, evidence,
                                         'switches' if switches_startup else native_startup, aot_sha)
                print('8 项优化已在启动阶段启用，现可正常操作。', flush=True)
                if not args.no_playtest_log:
                    native_log = evidence / 'playtest.log'
                    try:
                        with native_log.open() as stream:
                            header = stream.readline()
                        confirmed = header.startswith('PLAYTEST_START ') and f'pid={proc.name} ' in header
                    except OSError:
                        confirmed = False
                    session['native_playtest_log_confirmed'] = confirmed
                    atomic_json(out / 'session.json', session)
                    print(('轻量游玩日志已开启' if confirmed else '未确认原生帧日志，进程/GPU 采样继续；请检查 stdout.log 和版本') +
                          f'；退出后自动汇总：{out}/summary.md', flush=True)
                if args.diagnose_stutter:
                    (out / 'clock.json').write_text(json.dumps({'monotonic_ns': time.monotonic_ns(),
                        'unix_ns': time.time_ns(), 'pid': int(proc.name), 'pid_start_ticks': identity}, indent=2))
                    print(f'帧阶段计时已启用：{game_log}（LOCAL_RENDER_COST）。', flush=True)
            elif driver.poll() is None:
                raise RuntimeError(f'AOT 库未在启动期限内加载，优化未启用。详情：{game_log}')
            else:
                raise RuntimeError(f'模拟器在初始化完成前退出。详情：{game_log}')
            while driver.poll() is None:
                if recorder:
                    recorder.poll()
                try:
                    driver.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        if recorder:
            try:
                recorder.stop()
            except (OSError, subprocess.SubprocessError) as exc:
                print(f'录像收尾出错：{exc}；继续关闭游戏。', flush=True)
        for child in (driver,):
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Keep the supervisor alive so it can finish game cleanup.
                    print(f'退出仍在处理中，PID {child.pid}；日志：{out}', flush=True)
        session['state'] = 'finished' if driver is None or driver.poll() is not None else 'finishing'
        if evidence and (evidence / 'result.json').exists():
            result = json.loads((evidence / 'result.json').read_text())
            session['result'] = result
            print(f"本次退出：{result['status']}，退出码 {result['returncode']}。日志：{out}", flush=True)
            if not args.no_playtest_log:
                print(f'游玩汇总：{out}/summary.md', flush=True)
        atomic_json(out / 'session.json', session)


if __name__ == '__main__':
    main()
