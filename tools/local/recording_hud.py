"""Read Kyty's presentation counter without attaching to the game process."""
from collections import deque
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time


class FrameRate:
    """One-second presentation rate and time since the counter last changed."""
    def __init__(self):
        self.samples = deque()
        self.changed_at = None

    def sample(self, now, frame):
        if self.samples and frame < self.samples[-1][1]:
            self.samples.clear()
        if not self.samples or frame != self.samples[-1][1]:
            self.changed_at = now
        self.samples.append((now, frame))
        while len(self.samples) > 2 and self.samples[1][0] <= now - 1:
            self.samples.popleft()
        elapsed = now - self.samples[0][0]
        fps = (frame - self.samples[0][1]) / elapsed if elapsed >= .5 else None
        return fps, now - self.changed_at


class RecordingHud:
    """Update an atomic drawtext file at 5 Hz; the game never waits for this worker."""
    def __init__(self, window, telemetry):
        self.window = window
        self.telemetry = Path(telemetry)
        # Generated /tmp names avoid passing a user filename through filter syntax.
        self.directory = tempfile.TemporaryDirectory(prefix='kyty-record-hud-', dir='/tmp')
        self.text_path = Path(self.directory.name) / 'hud.txt'
        self.stop_event = threading.Event()
        self.thread = None
        self.error = None
        self.started_ns = time.monotonic_ns()
        self.started_at = time.time()
        self.write_text('GAME FPS: --  |  Frame: --  |  Idle: --')

    def write_text(self, text):
        temporary = self.text_path.with_suffix('.next')
        temporary.write_text(text, encoding='utf-8')
        temporary.replace(self.text_path)

    def start(self):
        self.log = self.telemetry.open('x', encoding='utf-8')
        self.thread = threading.Thread(target=self.run, name='recording-hud', daemon=True)
        self.thread.start()

    def run(self):
        rate = FrameRate()
        try:
            while not self.stop_event.is_set():
                began = time.monotonic()
                try:
                    result = subprocess.run(
                        ['xprop', '-id', hex(self.window), '_NET_WM_NAME', 'WM_NAME'],
                        capture_output=True, text=True, timeout=.5)
                    match = re.search(r'frame: (\d+), fps: ([0-9.]+)', result.stdout)
                    if result.returncode or not match:
                        raise ValueError('Window frame counter unavailable')
                    now_ns = time.monotonic_ns()
                    frame, title_fps = int(match[1]), float(match[2])
                    fps, idle = rate.sample(now_ns / 1e9, frame)
                    value = f'{fps:.1f}' if fps is not None else '--'
                    self.write_text(f'GAME FPS: {value}  |  Frame: {frame}  |  Idle: {idle:.1f}s')
                    row = {'monotonic_ns': now_ns, 'elapsed_seconds': (now_ns-self.started_ns)/1e9,
                           'frame': frame, 'fps_1s': fps, 'title_fps': title_fps, 'idle_seconds': idle}
                except (subprocess.SubprocessError, ValueError) as exc:
                    # Missing/stalled X11 queries are not evidence of a game freeze.
                    rate = FrameRate()
                    self.write_text('GAME FPS: --  |  Frame: --  |  Counter unavailable')
                    row = {'monotonic_ns': time.monotonic_ns(), 'error': str(exc)}
                self.log.write(json.dumps(row) + '\n')
                self.log.flush()
                self.stop_event.wait(max(0, .2 - (time.monotonic() - began)))
        except Exception as exc:
            self.error = str(exc)
            try:
                self.write_text('GAME FPS: --  |  Frame: --  |  HUD unavailable')
            except OSError:
                pass
        finally:
            self.log.close()

    def filter(self, width, height):
        # Center avoids the game's top-left health bars and top-right soul count.
        # Scale for 720p/1440p/2160p; cap by width for narrow custom windows.
        size = max(10, min(48, height // 45, width // 36))
        return (f'drawtext=font=monospace:textfile={self.text_path}:reload=1:expansion=none:'
                f'fontsize={size}:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=8:'
                'x=(w-tw)/2:y=12:fix_bounds=1')

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = 'HUD worker did not stop within two seconds'
                return  # Do not remove the text file while a writer still owns it.
        self.directory.cleanup()
