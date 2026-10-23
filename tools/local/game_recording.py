"""Optional X11 window capture for the manual launcher; no emulator hooks."""
import ctypes as C
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from recording_hud import RecordingHud


def find_game_window(pid):
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True, timeout=5)
    candidates = re.findall(r'(0x[0-9a-f]+) [^\n]*\("kyty_emulator"', tree)
    matches = []
    for window in candidates:
        props = subprocess.run(['xprop', '-id', window, '_NET_WM_PID'],
                               capture_output=True, text=True, timeout=5)
        if re.search(r'=\s*' + re.escape(str(pid)) + r'\s*$', props.stdout):
            matches.append(int(window, 16))
    if len(matches) > 1:
        raise RuntimeError('Multiple windows belong to the launched emulator')
    return matches[0] if matches else None


def window_size(window):
    info = subprocess.check_output(['xwininfo', '-id', hex(window)], text=True, timeout=5)
    return tuple(int(re.search(r'\b' + name + r':\s*(\d+)', info)[1])
                 for name in ('Width', 'Height'))


def fit_4k_window(window):
    """Remove decorations and center a 4K client so its bottom is not off-screen."""
    x = C.CDLL('libX11.so.6')
    x.XOpenDisplay.restype = C.c_void_p
    for name, types, result in [
        ('XCloseDisplay', [C.c_void_p], C.c_int),
        ('XDefaultScreen', [C.c_void_p], C.c_int),
        ('XDisplayWidth', [C.c_void_p, C.c_int], C.c_int),
        ('XDisplayHeight', [C.c_void_p, C.c_int], C.c_int),
        ('XInternAtom', [C.c_void_p, C.c_char_p, C.c_int], C.c_ulong),
        ('XChangeProperty', [C.c_void_p, C.c_ulong, C.c_ulong, C.c_ulong, C.c_int,
                             C.c_int, C.c_void_p, C.c_int], C.c_int),
        ('XMoveWindow', [C.c_void_p, C.c_ulong, C.c_int, C.c_int], C.c_int),
        ('XChangeWindowAttributes', [C.c_void_p, C.c_ulong, C.c_ulong, C.c_void_p], C.c_int),
        ('XFlush', [C.c_void_p], C.c_int),
    ]:
        function = getattr(x, name)
        function.argtypes, function.restype = types, result
    display = x.XOpenDisplay(None)
    if not display:
        raise RuntimeError('No X11 display')
    try:
        screen = x.XDefaultScreen(display)
        width, height = window_size(window)
        sw, sh = x.XDisplayWidth(display, screen), x.XDisplayHeight(display, screen)
        if width > sw or height > sh:
            raise RuntimeError(f'{width}×{height} exceeds this desktop ({sw}×{sh})')
        atom = x.XInternAtom(display, b'_MOTIF_WM_HINTS', 0)
        hints = (C.c_ulong * 5)(2, 0, 0, 0, 0)  # decorations flag, no decorations
        x.XChangeProperty(display, window, atom, atom, 32, 0, hints, 5)
        x.XFlush(display)
        time.sleep(.2)  # Allow the window manager to remove its frame first.
        # Move this client once past the panel work-area clamp. Restore normal
        # window management immediately; Alt-Tab and normal close keep working.
        class Attributes(C.Structure):
            _fields_ = [(name, kind) for name, kind in (
                ('background_pixmap', C.c_ulong), ('background_pixel', C.c_ulong),
                ('border_pixmap', C.c_ulong), ('border_pixel', C.c_ulong),
                ('bit_gravity', C.c_int), ('win_gravity', C.c_int), ('backing_store', C.c_int),
                ('backing_planes', C.c_ulong), ('backing_pixel', C.c_ulong),
                ('save_under', C.c_int), ('event_mask', C.c_long),
                ('do_not_propagate_mask', C.c_long), ('override_redirect', C.c_int),
                ('colormap', C.c_ulong), ('cursor', C.c_ulong))]
        attributes = Attributes()
        attributes.override_redirect = 1
        x.XChangeWindowAttributes(display, window, 1 << 9, C.byref(attributes))
        try:
            x.XMoveWindow(display, window, (sw - width) // 2, (sh - height) // 2)
        finally:
            attributes.override_redirect = 0
            x.XChangeWindowAttributes(display, window, 1 << 9, C.byref(attributes))
        x.XFlush(display)
    finally:
        x.XCloseDisplay(display)


class GameRecorder:
    def __init__(self, output, fps=30, audio=True, hud=True):
        self.output = Path(output)
        self.fps, self.audio = fps, audio
        self.process = self.log = None
        self.hud_requested, self.hud = hud, None
        self.reported = False
        self.metadata = {'output': str(self.output), 'capture_fps': fps,
                         'encoder': 'h264_nvenc', 'status': 'not_started'}

    def start(self, window):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise RuntimeError(f'Refusing to overwrite video: {self.output}')
        # Speaker monitor only. Never fall back to the default (microphone) source.
        audio = self.audio
        if audio:
            try:
                probe = subprocess.run(
                    ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'pulse',
                     '-i', '@DEFAULT_MONITOR@', '-t', '0.2', '-f', 'null', '-'],
                    capture_output=True, timeout=6)
                audio = probe.returncode == 0
                if not audio:
                    self.metadata['audio_probe_error'] = probe.stderr.decode(errors='replace')
            except subprocess.TimeoutExpired:
                audio = False
                self.metadata['audio_probe_error'] = 'Speaker monitor probe timed out'
            if not audio:
                print('扬声器录音源不可用，本次仅录画面。', flush=True)
        size = window_size(window)
        filters = ['pad=ceil(iw/2)*2:ceil(ih/2)*2']
        self.metadata['hud'] = {'requested': self.hud_requested, 'enabled': False}
        if self.hud_requested:
            try:
                probe = subprocess.run(['ffmpeg', '-hide_banner', '-h', 'filter=drawtext'],
                                       capture_output=True, text=True, timeout=5, check=True)
                if 'drawtext AVOptions:' not in probe.stdout:
                    raise RuntimeError('FFmpeg drawtext filter unavailable')
                self.hud = RecordingHud(window, self.output.with_suffix('.frames.jsonl'))
                self.hud.start()
                filters.append(self.hud.filter(*size))
                self.metadata['hud'].update(enabled=True, sample_hz=5,
                    source='game window presentation counter; independent 1-second rate',
                    telemetry=str(self.hud.telemetry), started_at=self.hud.started_at,
                    started_monotonic_ns=self.hud.started_ns)
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                if self.hud:
                    self.hud.stop()
                    self.hud = None
                self.metadata['hud']['error'] = str(exc)
                print(f'录像 HUD 无法启用，继续录制画面：{exc}', flush=True)
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
                   '-thread_queue_size', '8', '-f', 'x11grab', '-window_id', str(window),
                   '-framerate', str(self.fps), '-draw_mouse', '0',
                   '-i', os.environ.get('DISPLAY', ':0')]
        if audio:
            command += ['-thread_queue_size', '512', '-f', 'pulse', '-i', '@DEFAULT_MONITOR@']
        command += ['-map', '0:v:0']
        if audio:
            command += ['-map', '1:a:0', '-c:a', 'aac', '-b:a', '192k',
                        '-af', 'aresample=async=1:first_pts=0']
        command += ['-filter_threads', '2', '-vf', ','.join(filters),
                    '-c:v', 'h264_nvenc', '-preset', 'p1', '-tune', 'll',
                    '-rc', 'vbr', '-cq', '21', '-b:v', '0', '-bf', '0',
                    '-g', str(self.fps * 2), '-pix_fmt', 'yuv420p', '-threads', '2',
                    '-fps_mode', 'cfr', '-r', str(self.fps),
                    # Fragmented MP4 remains recoverable if the game or terminal crashes.
                    '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
                    '-frag_duration', '2000000', str(self.output)]
        self.metadata.update(window=hex(window), input_size=size, command=command,
                             audio_source='@DEFAULT_MONITOR@' if audio else None,
                             status='recording', started_at=time.time())
        self.log = self.output.with_suffix('.capture.log').open('w')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self.log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        except OSError:
            self.log.close()
            if self.hud:
                self.hud.stop()
            raise
        self.write_metadata()
        print(f'正在录像：{self.output}（NVENC，{self.fps} fps，' +
              ('含扬声器声音）' if audio else '无声音）'), flush=True)
        if self.hud:
            print('录像顶部显示游戏 FPS / Frame / Idle，采样记录保存为 .frames.jsonl。', flush=True)

    def write_metadata(self):
        self.output.with_suffix('.capture.json').write_text(json.dumps(self.metadata, indent=2) + '\n')

    def poll(self):
        if self.process is not None and self.process.poll() is not None and not self.reported:
            self.reported = True
            if self.hud:
                self.hud.stop()
            print(f'录像已停止，游戏继续运行；详情：{self.output.with_suffix(".capture.log")}', flush=True)

    def stop(self):
        if self.process is None:
            if self.hud:
                self.hud.stop()
            return
        forced = False
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'q\n')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    forced = True
                    self.process.kill()
                    self.process.wait()
        if self.hud:
            self.hud.stop()
            self.metadata['hud']['error'] = self.hud.error
        self.process.stdin.close()
        self.log.close()
        self.metadata.update(returncode=self.process.returncode, forced_stop=forced,
                             bytes=self.output.stat().st_size if self.output.exists() else 0)
        try:
            probe = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                                    '-of', 'json', str(self.output)], capture_output=True,
                                   text=True, timeout=10, check=True)
            media = json.loads(probe.stdout)
            self.metadata['media'] = media
            valid = any(s['codec_type'] == 'video' and int(s.get('width', 0)) > 0
                        for s in media['streams']) and float(media['format'].get('duration', 0)) > 0
        except (OSError, subprocess.SubprocessError, ValueError, KeyError):
            valid = False
        self.metadata['status'] = ('saved' if self.process.returncode == 0 else 'saved_partial') if valid else 'failed'
        self.write_metadata()
        print(('录像已保存：' if valid else '录像失败，请检查日志：') + str(self.output), flush=True)
