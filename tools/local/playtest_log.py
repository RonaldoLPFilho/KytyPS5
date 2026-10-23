"""Low-rate, read-only gameplay evidence. Never attaches a debugger or sends input."""
from collections import Counter
import datetime as dt
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import unquote

from debug_cpu_policy import atomic_json


def event(line):
    if not line.endswith('\n') or not line.startswith('PLAYTEST_'):
        raise ValueError('Incomplete or unknown record')
    kind, body = line.rstrip('\n').split(' ', 1)
    fields = {}
    for key, value in re.findall(r'(\w+)=([^\s]+)', body):
        if key in ('kind', 'name', 'context'):
            fields[key] = unquote(value)
        else:
            fields[key] = int(value, 16 if key in ('id', 'aux', 'guest_pc', 'base', 'pc', 'sp', 'address') or key.endswith('_hex') else 10)
    required = {
        'PLAYTEST_START': ('monotonic_ns', 'unix_ns', 'pid', 'version'),
        'PLAYTEST_FRAMES': ('frames', 'frame', 'start_ns', 'end_ns', 'elapsed_ns', 'max_frame_ns'),
        'PLAYTEST_LONG_FRAME': ('frame', 'start_ns', 'end_ns', 'elapsed_ns'),
        'PLAYTEST_COMPILE_BEGIN': ('op', 'kind', 'start_ns'),
        'PLAYTEST_OPERATION': ('op', 'kind', 'start_ns', 'end_ns', 'elapsed_ns'),
        'PLAYTEST_FAULT': ('time_hex', 'pc', 'sp', 'address', 'code_hex'),
        'PLAYTEST_MODULE': ('name', 'base', 'size'),
        'PLAYTEST_EXIT': ('end_ns', 'dropped'),
    }
    if kind not in required or any(k not in fields for k in required[kind]):
        raise ValueError('Missing event fields')
    if 'elapsed_ns' in fields and (fields['elapsed_ns'] < 0 or fields['end_ns'] - fields['start_ns'] != fields['elapsed_ns']):
        raise ValueError('Invalid event interval')
    return kind, fields


class EventIndex:
    """Keep bounded summaries while the full evidence stays in the append log."""
    def __init__(self):
        self.start = None
        self.last_frame_ns = None
        self.frame_windows = self.frames = self.elapsed_ns = 0
        self.long_frames = self.over_500ms = self.over_1s = self.maximum_ns = 0
        self.pending = {}
        self.slow_counts, self.slow_max, self.compile_counts = Counter(), Counter(), Counter()
        self.longest, self.slowest, self.faults, self.modules = [], [], [], []
        self.malformed = self.dropped = 0
        self.exit_record = None

    def add(self, line):
        try:
            kind, row = event(line)
        except (ValueError, OverflowError):
            self.malformed += 1
            return
        self.dropped = max(self.dropped, row.get('dropped', 0))
        if kind == 'PLAYTEST_START':
            self.start = row
        elif kind == 'PLAYTEST_FRAMES':
            self.frame_windows += 1
            self.frames += row['frames']
            self.elapsed_ns += row['elapsed_ns']
            self.last_frame_ns = row['end_ns']
        elif kind == 'PLAYTEST_LONG_FRAME':
            self.last_frame_ns = row['end_ns']
            self.long_frames += 1
            self.over_500ms += row['elapsed_ns'] >= 500_000_000
            self.over_1s += row['elapsed_ns'] >= 1_000_000_000
            self.maximum_ns = max(self.maximum_ns, row['elapsed_ns'])
            self.longest = sorted(self.longest + [row], key=lambda r: r['elapsed_ns'], reverse=True)[:100]
        elif kind == 'PLAYTEST_COMPILE_BEGIN':
            if len(self.pending) < 16384:
                self.pending[row['op']] = row
            else:
                self.malformed += 1
        elif kind == 'PLAYTEST_OPERATION':
            if row['op']:
                self.pending.pop(row['op'], None)
                self.compile_counts[row['kind']] += 1
            if row['elapsed_ns'] >= 5_000_000:
                self.slow_counts[row['kind']] += 1
                self.slow_max[row['kind']] = max(self.slow_max[row['kind']], row['elapsed_ns'])
                self.slowest = sorted(self.slowest + [row], key=lambda r: r['elapsed_ns'], reverse=True)[:100]
        elif kind == 'PLAYTEST_FAULT':
            self.faults = (self.faults + [row])[-32:]
        elif kind == 'PLAYTEST_MODULE':
            self.modules = (self.modules + [row])[-256:]
        else:
            self.exit_record = row

    def summary(self):
        def stamped(row):
            row = dict(row)
            stamp = row.get('start_ns', row.get('time_hex'))
            if self.start and stamp is not None:
                row['since_launch_seconds'] = (stamp - self.start['monotonic_ns']) / 1e9
                unix_ns = self.start['unix_ns'] + stamp - self.start['monotonic_ns']
                row['wall_time'] = dt.datetime.fromtimestamp(unix_ns / 1e9, dt.timezone.utc).astimezone().isoformat()
            return row
        return {
            'telemetry_available': self.start is not None, 'clock': self.start,
            'presentation_windows': self.frame_windows,
            'whole_session_fps_including_menus_and_loading': self.frames * 1e9 / self.elapsed_ns if self.elapsed_ns else None,
            'long_frames': self.long_frames, 'over_500ms': self.over_500ms, 'over_1s': self.over_1s,
            'maximum_long_frame_ms': self.maximum_ns / 1e6 if self.long_frames else None,
            'slow_operation_counts': dict(self.slow_counts), 'slow_operation_max_ns': dict(self.slow_max),
            'completed_compilation_counts': dict(self.compile_counts),
            'unfinished_compilations': [stamped(r) for r in self.pending.values()],
            'longest_frames': [stamped(r) for r in self.longest],
            'slowest_operations': [stamped(r) for r in self.slowest],
            'faults': [stamped(r) for r in self.faults], 'guest_modules': self.modules,
            'dropped_records': self.dropped, 'malformed_records': self.malformed,
            'exit_record': self.exit_record,
            'limits': ['Frame intervals include menus, loading and window pauses; they are not automatically gameplay stutters.',
                       'Concurrent IO/compile overlap is correlation, not proof of a blocked frame. Nested IO durations must not be added.',
                       'Only compilation has begin/end pairs; other operations are logged on completion at >=5 ms.',
                       'The final partial FPS window and a stall still in progress have no completed frame interval.',
                       'GPU samples are device-wide. Thread snapshots show kernel wait locations, not user-space stack traces.']}


class PlaytestMonitor:
    def __init__(self, pid, identity, out):
        self.proc, self.identity, self.out = Path('/proc') / str(pid), str(identity), Path(out)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name='playtest-log', daemon=True)
        self.index = EventIndex()
        self.started_ns = time.monotonic_ns()
        self.offset = 0
        self.pending_bytes = b''
        self.last_maps = None
        self.maps_at = self.snapshot_at = 0
        self.snapshots = 0
        self.errors = []

    def consume(self, final=False):
        try:
            with (self.out / 'playtest.log').open('rb') as stream:
                stream.seek(self.offset)
                # Bound each live read; drain everything only after the child exits.
                while True:
                    data = stream.read(1024 * 1024)
                    if not data:
                        break
                    self.offset += len(data)
                    lines = (self.pending_bytes + data).split(b'\n')
                    self.pending_bytes = lines.pop()
                    for line in lines:
                        self.index.add(line.decode('utf-8', errors='replace') + '\n')
                    if len(self.pending_bytes) > 16384:
                        self.pending_bytes = b''
                        self.index.malformed += 1
                    if not final:
                        break
        except FileNotFoundError:
            pass
        if final and self.pending_bytes:
            self.index.malformed += 1
            self.pending_bytes = b''

    def proc_sample(self, now):
        stat = (self.proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[19] != self.identity:
            raise ProcessLookupError('PID identity changed')
        row = {'monotonic_ns': now, 'unix_ns': time.time_ns(), 'state': stat[0],
               'user_ticks': int(stat[11]), 'system_ticks': int(stat[12]),
               'minor_faults': int(stat[7]), 'major_faults': int(stat[9]),
               'threads': int(stat[17]), 'rss_bytes': int(stat[21]) * os.sysconf('SC_PAGE_SIZE'),
               'clock_ticks_per_second': os.sysconf('SC_CLK_TCK')}
        for name in ('io', 'status'):
            try:
                values = dict(re.findall(r'^(\w+):\s*(.+)$', (self.proc / name).read_text(), re.M))
                wanted = ('rchar', 'wchar', 'read_bytes', 'write_bytes', 'syscr', 'syscw') if name == 'io' else (
                    'VmRSS', 'VmHWM', 'VmSwap', 'voluntary_ctxt_switches', 'nonvoluntary_ctxt_switches')
                row[name] = {k: values[k] for k in wanted if k in values}
            except OSError as exc:
                row[name + '_error'] = str(exc)
        meminfo = Path('/proc/meminfo').read_text()
        row['system_memory'] = dict(re.findall(r'^(MemAvailable|SwapFree|SwapTotal):\s*(.+)$', meminfo, re.M))
        if now - self.maps_at >= 15_000_000_000:
            self.maps_at = now
            maps = (self.proc / 'maps').read_text()
            if maps != self.last_maps:
                (self.out / 'latest-maps.tmp').write_text(maps)
                (self.out / 'latest-maps.tmp').replace(self.out / 'latest-maps.txt')
                atomic_json(self.out / 'latest-maps-clock.json', {'monotonic_ns': now, 'unix_ns': row['unix_ns']})
                self.last_maps = maps
        return row

    def snapshot(self, now, reason):
        # A periodic heartbeat can lag the last actual present by up to one
        # second. Describe this as an observation, never diagnose a deadlock.
        record = {'monotonic_ns': now, 'unix_ns': time.time_ns(), 'reason': reason,
                  'last_present_heartbeat_ns': self.index.last_frame_ns,
                  'unfinished_compilations': list(self.index.pending.values()), 'threads': {}}
        for task in (self.proc / 'task').iterdir():
            fields = {}
            for name in ('comm', 'stat', 'wchan', 'syscall'):
                try:
                    fields[name] = (task / name).read_text().strip()
                except OSError as exc:
                    fields[name + '_error'] = str(exc)
            record['threads'][task.name] = fields
        with (self.out / 'stalls.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        self.snapshot_at = now
        self.snapshots += 1

    def run(self):
        try:
            with (self.out / 'process-samples.jsonl').open('w', buffering=1) as stream:
                while not self.stop_event.is_set():
                    now = time.monotonic_ns()
                    self.consume()
                    try:
                        row = self.proc_sample(now)
                        age = now - (self.index.last_frame_ns or self.started_ns)
                        row['last_present_heartbeat_ns'] = self.index.last_frame_ns
                        threshold = 3_000_000_000 if self.index.last_frame_ns else 30_000_000_000
                        if age >= threshold and now - self.snapshot_at >= 10_000_000_000:
                            self.snapshot(now, 'present_heartbeat_gap' if self.index.last_frame_ns else 'startup_without_present_heartbeat')
                        stream.write(json.dumps(row) + '\n')
                    except (FileNotFoundError, ProcessLookupError):
                        break
                    except OSError as exc:
                        stream.write(json.dumps({'monotonic_ns': now, 'error': str(exc)}) + '\n')
                    self.stop_event.wait(1)
        except Exception as exc:
            self.errors.append(repr(exc))

    def start(self):
        self.thread.start()

    def finish(self, result):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Evidence worker did not stop; live logs remain on disk')
        self.consume(final=True)
        report = self.index.summary() | {'process_result': result, 'thread_snapshots': self.snapshots,
                                        'collector_errors': self.errors}
        atomic_json(self.out / 'playtest-summary.json', report)
        maximum = report['maximum_long_frame_ms']
        text = (f"# 游玩日志汇总\n\n退出分类：`{result['status']}`；退出码：{result['returncode']}"
                f"；信号：{result.get('exit_signal') or '无'}。\n\n"
                f"轻量记录：{'已启用' if report['telemetry_available'] else '未收到，请检查二进制版本及日志错误'}。"
                f"超过 100 ms 的呈现间隔 {report['long_frames']} 次；超过 1 秒 {report['over_1s']} 次；"
                f"最长 {f'{maximum:.1f} ms' if maximum is not None else '未记录'}。\n\n"
                f"未结束的编译 {len(report['unfinished_compilations'])} 项；线程快照 {self.snapshots} 份；"
                f"丢弃记录 {report['dropped_records']} 条；不完整记录 {report['malformed_records']} 条。\n\n"
                "菜单、加载、切出窗口也可能产生长间隔，需结合游玩时刻判断。详见 `playtest-summary.json`；"
                "原始时间线 `playtest.log`，错误及寄存器 `stdout.log`，线程快照 `stalls.jsonl`，"
                "进程内存/读写 `process-samples.jsonl`，显存/利用率 `gpu-samples.csv`，"
                "地址映射 `latest-maps.txt`，驱动错误 `kernel-new.txt`。\n")
        (self.out / 'playtest-summary.md').write_text(text)
        return report
