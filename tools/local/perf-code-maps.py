#!/usr/bin/env python3
"""Make a code-profiling copy of perf.data without Kyty data-page protection churn.

Keeps every sample and executable mapping, and every non-executable transition
overlapping an executable range. Only non-executable /memfd:KytyDirectMemory
mappings disjoint from all code ranges are omitted. Keep the original for data
address / memory-map analysis. Supports the local uncompressed PERFILE2 format.

Format: https://github.com/torvalds/linux/blob/master/tools/perf/Documentation/perf.data-file-format.txt
"""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import struct


def code_map_copy(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError('Destination must be a new file distinct from the raw profile')
    if source.stat().st_size > 1024**3:
        raise ValueError('Profile exceeds the bounded 1 GiB input size')
    raw = source.read_bytes()
    if len(raw) < 104 or raw[:8] != b'PERFILE2':
        raise ValueError('Expected a little-endian PERFILE2 regular file')
    header_size, attr_size, attr_offset, attr_bytes, start, size, event_offset, event_bytes = struct.unpack_from('<8Q', raw, 8)
    if header_size != 104 or start < header_size or start + size > len(raw):
        raise ValueError('Unsupported header or data section')
    if attr_offset + attr_bytes > start or event_offset + event_bytes > start:
        raise ValueError('Only metadata sections preceding data are supported')
    flags = int.from_bytes(raw[72:104], 'little')
    features = [i for i in range(256) if flags >> i & 1]
    # AUXTRACE, directory and compressed formats can contain additional file
    # offsets. Refuse them instead of silently relocating opaque references.
    supported = set(range(2, 18)) | {20, 21, 22, 23, 25, 26, 28, 29, 30, 31, 32}
    if not set(features) <= supported:
        raise ValueError('Unsupported perf feature sections: ' + str(set(features) - supported))
    end = start + size
    if end + 16 * len(features) > len(raw):
        raise ValueError('Truncated feature table')

    def records():
        pos = start
        while pos < end:
            if pos + 8 > end:
                raise ValueError('Truncated event header')
            kind, misc, length = struct.unpack_from('<IHH', raw, pos)
            if length < 8 or pos + length > end:
                raise ValueError('Invalid event length')
            if kind == 10 and length < 73:
                raise ValueError('Truncated MMAP2 record')
            yield pos, kind, length
            pos += length

    executable = []
    for pos, kind, length in records():
        if kind == 10 and struct.unpack_from('<I', raw, pos + 64)[0] & 4:
            address, length_bytes = struct.unpack_from('<QQ', raw, pos + 16)
            if address + length_bytes > 2**64:
                raise ValueError('Overflowing executable mapping')
            executable.append((address, address + length_bytes))
    merged = []
    for begin, finish in sorted(executable):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(finish, merged[-1][1]))
        else:
            merged.append((begin, finish))
    code_starts = [r[0] for r in merged]
    parts, removed, samples, total = [], 0, 0, 0
    sample_hash = hashlib.sha256()
    for pos, kind, length in records():
        total += 1
        if kind == 10 and not struct.unpack_from('<I', raw, pos + 64)[0] & 4:
            name = raw[pos + 72:pos + length].split(b'\0', 1)[0]
            if name == b'/memfd:KytyDirectMemory (deleted)':
                address, length_bytes = struct.unpack_from('<QQ', raw, pos + 16)
                if address + length_bytes > 2**64:
                    raise ValueError('Overflowing data mapping')
                index = bisect.bisect_left(code_starts, address + length_bytes)
                overlaps_code = index != 0 and merged[index - 1][1] > address
                if not overlaps_code:
                    removed += 1
                    continue
        event = raw[pos:pos + length]
        parts.append(event)
        if kind == 9:
            samples += 1
            sample_hash.update(event)
    data = b''.join(parts)
    delta = len(data) - size
    prefix, suffix = bytearray(raw[:start]), bytearray(raw[end:])
    struct.pack_into('<Q', prefix, 48, len(data))
    for index, feature in enumerate(features):
        offset, feature_size = struct.unpack_from('<QQ', suffix, index * 16)
        if offset < end + len(features) * 16 or offset + feature_size > len(raw):
            raise ValueError(f'Invalid perf feature {feature}')
        struct.pack_into('<Q', suffix, index * 16, offset + delta)
    with destination.open('xb') as out:
        out.write(prefix)
        out.write(data)
        out.write(suffix)
    return {'source': str(source), 'destination': str(destination), 'raw_bytes': len(raw),
            'analysis_bytes': len(raw) + delta, 'events': total, 'omitted_data_maps': removed,
            'samples': samples, 'sample_records_sha256': sample_hash.hexdigest(),
            'purpose': 'code profiling only; retain raw file for memory-map/data analysis'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(code_map_copy(args.source, args.destination), indent=2))
