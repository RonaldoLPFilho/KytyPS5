#!/usr/bin/env python3
"""Fingerprint the current build and derive a launch config from an existing one.

The derived config keeps the parent's environment, CPU policy and AOT identity so
that a candidate differs from its parent only by the emulator binary. Record the
source fingerprints next to it, because the launcher only verifies the binary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / '_Build/linux-pr500/kyty_emulator')
    parser.add_argument('--parent-config', type=Path,
                        default=ROOT / '_Build/profiles/manual-current-launch.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, action='append', default=[],
                        help='Source file whose fingerprint identifies this candidate')
    args = parser.parse_args()
    if not args.checkpoint.replace('-', '').isalnum() or not args.checkpoint.islower():
        parser.error('Checkpoint identity must be lowercase alphanumeric with dashes')
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.parent_config.read_text())
    binary_sha = digest(args.binary)
    staged = ROOT / '_Build/profiles/binaries' / binary_sha / 'kyty_emulator'
    staged.parent.mkdir(parents=True, exist_ok=True)
    if staged.exists() and digest(staged) != binary_sha:
        parser.error(f'Staged path already holds different contents: {staged}')
    if not staged.exists():
        staged.write_bytes(args.binary.read_bytes())
        staged.chmod(0o755)
    command = list(config['command'])
    command[command.index('--') + 1] = str(staged)
    command[command.index('--label') + 1] = args.checkpoint
    config.update(checkpoint=args.checkpoint, binary_sha256=binary_sha, command=command)
    config['parent_config'] = str(args.parent_config.resolve())
    config['parent_binary_sha256'] = json.loads(args.parent_config.read_text())['binary_sha256']
    config['build_receipt'] = str((args.out / 'build-receipt.json').resolve())
    config_path = args.out / 'candidate-launch.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    receipt = {
        'checkpoint': args.checkpoint,
        'binary': str(staged), 'binary_sha256': binary_sha,
        'built_from': str(args.binary.resolve()),
        'parent_config': config['parent_config'],
        'parent_binary_sha256': config['parent_binary_sha256'],
        'base_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip(),
        'working_tree_dirty': bool(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                                                  capture_output=True, text=True,
                                                  check=True).stdout.strip()),
        'source_sha256': {str(Path(path)): digest(path) for path in args.source},
        'config': str(config_path.resolve()),
    }
    (args.out / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'config': str(config_path), 'binary_sha256': binary_sha,
                      'staged': str(staged)}, indent=2))


if __name__ == '__main__':
    main()
