#!/usr/bin/env python3
"""Launch the reviewed source build, without the frozen checkpoint's local patches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--binary', type=Path, help='Use an explicit build without local receipt settings')
parser.add_argument('--dry-run', action='store_true')
args, forwarded = parser.parse_known_args()
receipt_path = ROOT / '_Build/profiles/reviewed-current.json'
receipt = {}
environment = {key: value for key, value in os.environ.items() if not key.startswith('KYTY_')}
if args.binary is None and receipt_path.is_file():
    receipt = json.loads(receipt_path.read_text())
    args.binary = Path(receipt['binary'])
    def verify(path, expected):
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                parser.error(f'Local build receipt fingerprint mismatch: {path}')
    verify(args.binary, receipt['binary_sha256'])
    for library in receipt.get('libraries', []):
        verify(Path(library['path']), library['sha256'])
    environment.update(receipt.get('environment', {}))
if args.binary is None:
    args.binary = ROOT / '_Build/upstream-clean/_Build/review' / (
        'kyty_emulator.exe' if os.name == 'nt' else 'kyty_emulator')
if not args.binary.is_file():
    parser.error('Build the reviewed source first, or specify --binary PATH')
command = [str(args.binary.resolve()), *(forwarded[1:] if forwarded[:1] == ['--'] else forwarded)]
if '--game' not in command:
    game = os.environ.get('KYTY_GAME_DIR') or receipt.get('game_directory')
    if game:
        command.extend(['--game', game])
if args.dry_run:
    print(json.dumps({'command': command, 'source_commit': receipt.get('source_commit'),
                      'local_receipt': str(receipt_path) if receipt else None,
                      'environment': receipt.get('environment', {})}, indent=2))
else:
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))
