# Local tools

For day-to-day launching and the required assets see the [current build](../../docs/CURRENT-PLAYABLE.md),
for every experiment's conclusion the [experiment log](../../docs/EXPERIMENTS.md), and for the test
procedure [benchmarking](../../docs/BENCHMARKING.md).

## Running and measuring

- `./run.sh` in the repository root, or `play-demons-souls.py --2k`: the same release configuration.
  The latter supports run logs, recording and shader warmup options.
- `benchmark-entry.py`: backs up the user's save, loads the fixed baseline, continues normally,
  positions, measures standing still, and restores the original save after a normal exit.
- `quick-benchmark-game.py`: a short measurement/screenshot of a running process; no camera input by
  default.
- `switch-control.py`, `abba-switch.py`, `abba-cpu.py`: same-process switch and CPU/frame comparisons
  bound to the process identity/fingerprint.
- `debug-run.py`, `debug_cpu_policy.py`, `demons-souls-smoke.py`, `close-game-window.py`: processes,
  CPU pinning, menus and a clean shutdown.
- `game_recording.py`, `recording_hud.py`, `playtest_log.py`: recording and run logs.
- Windows: `windows/bench-run.ps1` (launch to the HUD from the fixed save), `windows/ab-spot.ps1`
  (same-process ABBA of a runtime switch), `windows/ab-exe.ps1` (ABBA of two builds),
  `windows/walk-run.ps1` (the walk route), `windows/prof-spot.ps1` (render-thread profile),
  `windows/pgo-train.ps1` (PGO profile refresh).
- Hitches on screen: with `KYTY_HITCH_LOG_MS` the run log's SLOW Frame lines carry `unix_ms`;
  `windows/record.ps1` records the emulator window with wall-clock timestamps (ffmpeg 8+:
  `winget install Gyan.FFmpeg`), `windows/capture.ps1` takes stills without ffmpeg, and `hitch-sheet.py`
  puts the screen before and after each slow frame side by side; `hitch-prof.py` splits a render-thread
  profile into the slow frames' samples and the rest.

## Building and diagnostics

- `compile-srt-aot.py` (Linux) / `compile-srt-aot-windows.py` (Windows DLL): rebuild the SRT AOT
  library from exported plans; the ABI header is `src/local/SrtAotAbi.h`.
- `extract-game-shaders.py`, `capture-srt-plans.py`, `shader_precompile_audit.py`: resource
  extraction, plan export and warmup audits.
- [static-precompile](static-precompile/README.md): collects every shader/pipeline statically from the
  game files; a standalone program (`precompile-windows.ps1`, several processes) precompiles them into a
  static pipeline cache kept across builds; includes coverage comparisons with recorded caches.
- `profile-game.py`, `perf-code-maps.py`, `prof-inline.py` and the `*-report.py` scripts: on-demand
  diagnostics; instrumented FPS is never a net gain.
- `stage-candidate.py`: makes a configuration for an independent candidate without overwriting the
  release one.
- `run-reviewed.py`, `benchmark-build.py`: advanced entry points for independent builds; they need
  their own binary/receipt given explicitly and are not the day-to-day entry.
- [PGO](PGO.md): the current profile inputs and notes on retraining.

The local sources CMake and the tests refer to (vulkan-recording, native-resource, ...) are all in
`src/local/`; optional experiments are off by default. The old experiment sources of this folder
(shader-preparation, prepared-draw, render-cost/playtest), old copies of `src/local` files, `*.cmake`
fragments with their test sources, and `build-pbr-full-overlay.py` were no longer referenced and were
removed on 2026-09-28. Tools for the finished PT/SRT capture replays, the old checkpoint byte-patch
tools and large run outputs were removed as well.

A new experiment only appends one line to the experiment log. Keep reusable tools and delete old
output folders promptly; never delete process locks in use, release libraries, saves or build inputs.
