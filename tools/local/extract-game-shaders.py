#!/usr/bin/env python3
"""Extract Demon's Souls CSDR shader bundles without modifying the game.

This is a checked parser for the observed bundle layout, not a byte-pattern scan.
Raw RDNA2 and AGC metadata are sufficient to inventory shader programs, but do
not provide live resource specializations, attachments or pipeline state. The
runtime warmup cache records those separately. Extraction is not precompilation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct


def parse_bundle(data):
    if len(data) < 20:
        raise ValueError('truncated bundle header')
    if data[-12:-8] != b'RDSC' or struct.unpack_from('<I', data, len(data) - 4)[0] != len(data) - 12:
        raise ValueError('invalid CSDR footer or payload length')
    _, count = struct.unpack_from('<2I', data)
    if not 0 < count <= 16384 or 8 + count * 20 > len(data):
        raise ValueError('unsupported or truncated entry table')
    table_end = 8 + count * 20
    cursor = (table_end + 7) & ~7
    if any(b != 0xff for b in data[table_end:cursor]):
        raise ValueError("invalid table alignment padding")
    entries = []
    for i in range(count):
        entry = 8 + i * 20
        stage, flags, code_bytes, header_bytes, relative = struct.unpack_from('<5I', data, entry)
        begin = entry + 16 + relative
        header = begin + code_bytes
        end = header + header_bytes
        if begin != cursor or code_bytes == 0 or code_bytes % 4 or header_bytes < 96 or end > len(data) - 12:
            raise ValueError(f'entry {i}: non-contiguous or out-of-bounds payload')
        if data[header:header + 4] != b'1234':
            raise ValueError(f'entry {i}: missing AGC header')
        saved_header, saved_code = struct.unpack_from('<2I', data, header + 64)
        binary_type = data[header + 90]
        if saved_code != code_bytes or saved_header != header_bytes or binary_type > 8:
            raise ValueError(f'entry {i}: inconsistent AGC sizes or type')
        entries.append({'index': i, 'bundle_stage': stage, 'flags': flags, 'binary_type': binary_type,
                        'code_offset': begin, 'code_bytes': code_bytes,
                        'header_offset': header, 'header_bytes': header_bytes})
        cursor = end
    if cursor != len(data) - 12:
        raise ValueError('unrecognized trailing bundle data')
    return entries


def extract(game, out):
    out.mkdir(parents=True, exist_ok=False)
    payloads = out / 'programs'
    payloads.mkdir()
    entries, rejected = [], []
    unique_code, unique_header = set(), set()
    types = Counter()
    files = sorted(game.rglob('*.csdr'))
    for file in files:
        relative = str(file.relative_to(game))
        try:
            if not file.is_file() or file.stat().st_size > 128 * 1024 * 1024:
                raise ValueError('unsupported file size or type')
            data = file.read_bytes()
            parsed = parse_bundle(data)
        except (OSError, ValueError) as error:
            rejected.append({'file': relative, 'reason': str(error)})
            continue
        for entry in parsed:
            code = data[entry['code_offset']:entry['code_offset'] + entry['code_bytes']]
            header = data[entry['header_offset']:entry['header_offset'] + entry['header_bytes']]
            code_sha, header_sha = hashlib.sha256(code).hexdigest(), hashlib.sha256(header).hexdigest()
            if code_sha not in unique_code:
                (payloads / (code_sha + '.bin')).write_bytes(code)
                unique_code.add(code_sha)
            if header_sha not in unique_header:
                (payloads / (header_sha + '.agc')).write_bytes(header)
                unique_header.add(header_sha)
            types[entry['binary_type']] += 1
            entries.append(entry | {'file': relative, 'code_sha256': code_sha, 'header_sha256': header_sha})
    result = {'game': str(game), 'bundle_files': len(files), 'parsed_files': len(files) - len(rejected),
              'rejected_files': rejected, 'program_entries': len(entries), 'unique_code': len(unique_code),
              'unique_agc_headers': len(unique_header), 'binary_types': dict(types),
              'limits': __doc__.strip(), 'entries': entries}
    (out / 'index.json').write_text(json.dumps(result, indent=2) + '\n')
    return {k: v for k, v in result.items() if k not in ('entries', 'rejected_files')} | {'rejected_files': len(rejected)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='new output directory')
    a = p.parse_args()
    if not a.game.is_dir():
        p.error('--game must be an existing game directory')
    if a.out.exists():
        p.error('--out must be a new directory')
    print(json.dumps(extract(a.game.resolve(), a.out.resolve()), indent=2))


if __name__ == '__main__':
    main()
