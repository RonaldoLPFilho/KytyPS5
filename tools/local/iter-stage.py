#!/usr/bin/env python3
"""Stage the PGO build for one fast iteration (stage-3 loop, docs/PERF-STAGE3-GOAL.md).

    iter-stage.py [--build DIR] [--env NAME=VALUE ...]

Copies kyty_emulator to _Build/iter/, hands the previous iteration's driver pipeline cache
to the new binary hash by renaming it (copying 200 MB per build otherwise; only a cache this
script created is ever renamed), and writes _Build/iter/launch.json: the stage-3 control set
(_Build/perf40g/launch.json: the 33 FPS switches plus the four 40 FPS switches) with
KYTY_LIVE_FILE, shader warmup on 16 threads and any --env overrides.  Stop the game first.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '_Build/iter'
CACHE = ROOT / '_PipelineCache/local'
CONTROL = ROOT / '_Build/perf40g/launch.json'
SEED = CACHE / '8061f01f27323cfaf0f6edb25912fae6b1cda1f0fc2d94f63722f746a50cbb58'  # perf40g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--build', type=Path, default=ROOT / '_Build/hist/build-pgo')
    p.add_argument('--env', action='append', default=[], metavar='NAME=VALUE')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    binary = OUT / 'kyty_emulator'
    shutil.copy2(a.build / 'kyty_emulator', binary)
    sha = hashlib.sha256(binary.read_bytes()).hexdigest()
    owned = OUT / 'cache-sha'
    previous = owned.read_text().strip() if owned.exists() else ''
    target = CACHE / sha
    if not target.exists():
        if previous and (CACHE / previous).is_dir():
            # A release package keeps its binary's cache: `pinned` makes this a copy.
            if (CACHE / previous / 'pinned').exists():
                shutil.copytree(CACHE / previous, target, ignore=shutil.ignore_patterns('pinned'))
            else:
                (CACHE / previous).rename(target)
        else:
            shutil.copytree(SEED, target)
    owned.write_text(sha + '\n')
    # The emulator scopes a driver cache to its executable (signature line: revision and
    # binary SHA); for iteration builds re-sign it so the driver still gets its pipelines
    # (the driver validates its own payload; a changed shader only misses).
    version = next(a.build.rglob('kytyGitVersion.h')).read_text()
    revision = re.search(r'KYTY_GIT_REVISION "([0-9a-f]{40})"', version)[1]
    with (target / 'PPSA01341.bin').open('r+b') as cache:
        head = cache.read(512)
        end = head.index(b'\n')
        fields = head[:end].decode().split(':')
        if fields[0] == 'KytyPC1' and fields[2] == 'local':
            fields[1], fields[3] = revision, sha
            line = ':'.join(fields).encode()
            if len(line) == end:
                cache.seek(0)
                cache.write(line)
    config = json.loads(CONTROL.read_text())
    config['binary_sha256'] = sha
    config['checkpoint'] = 'iter'
    environment = config['environment']
    environment['KYTY_LIVE_FILE'] = str(ROOT / '_Build/layer-bench/live.cmd')
    environment['KYTY_SHADER_WARMUP_THREADS'] = '16'
    for item in a.env:
        name, value = item.split('=', 1)
        environment[name] = value
    command = config['command']
    command[command.index('--label') + 1] = 'iter'
    command[command.index('--') + 1] = str(binary)
    (OUT / 'launch.json').write_text(json.dumps(config, indent=1) + '\n')
    print(OUT / 'launch.json', sha[:12])


if __name__ == '__main__':
    main()
