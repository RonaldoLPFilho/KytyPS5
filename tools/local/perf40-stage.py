#!/usr/bin/env python3
"""Stage a measurement binary: perf40-stage.py TAG [BUILD_DIR]
Copies kyty_emulator to _Build/perf40/TAG/, seeds its driver pipeline cache from the
validated clean build and writes launch-live.json (33 FPS switch set + KYTY_LIVE_FILE)."""
import hashlib, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
tag = sys.argv[1]
build = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / '_Build/hist/build'
out = ROOT / '_Build/perf40' / tag
out.mkdir(parents=True, exist_ok=True)
binary = out / 'kyty_emulator'
shutil.copy2(build / 'kyty_emulator', binary)
sha = hashlib.sha256(binary.read_bytes()).hexdigest()
cache = ROOT / '_PipelineCache/local'
seed = cache / 'b8964ecbdbe5833ddb8cfe805f98af8511ef775a91d7a48d756a9d84c0476939'
if seed.is_dir() and not (cache / sha).exists():
    shutil.copytree(seed, cache / sha)
config = json.loads((ROOT / '_Build/release-33fps-20260926/launch.json').read_text())  # 33 FPS set
config['binary_sha256'] = sha
config['checkpoint'] = f'perf40-{tag}'
config['environment']['KYTY_LIVE_FILE'] = str(ROOT / '_Build/layer-bench/live.cmd')
cmd = config['command']
cmd[cmd.index('--label') + 1] = f'perf40-{tag}'
cmd[cmd.index('--') + 1] = str(binary)
(out / 'launch-live.json').write_text(json.dumps(config, indent=1) + '\n')
print(out / 'launch-live.json', sha[:12])
