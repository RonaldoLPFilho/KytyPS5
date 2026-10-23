#!/usr/bin/env python3
"""Attribute local synchronous GPU readbacks to fault PCs and memory windows."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    blocks = []
    current = None
    hex_fields = {'ip', 'window', 'address', 'rdi', 'rsi', 'rcx', 'rdx'}
    with args.log.open(errors='replace') as source:
        for line in source:
            if line.startswith('LOCAL_READBACK_SUMMARY '):
                current = {key: int(value) for key, value in
                           (field.split('=') for field in line.split()[1:])}
                current['items'] = []
                blocks.append(current)
            elif line.startswith('LOCAL_READBACK ') and current is not None:
                current['items'].append({key: int(value, 16 if key in hex_fields else 10)
                                        for key, value in
                                        (field.split('=') for field in line.split()[1:])})
    complete = [block for block in blocks if block['records'] == len(block['items'])]
    frames = sum(block['frames'] for block in complete)
    if frames <= 0:
        parser.error('No complete readback blocks')
    sums = ('count', 'requested', 'bytes', 'copies', 'elapsed_ns', 'wait_ns', 'waits')

    def aggregate(fields):
        result = {}
        for block in complete:
            for item in block['items']:
                key = tuple(item[field] for field in fields)
                if key not in result:
                    result[key] = {**dict(zip(fields, key)), **dict.fromkeys(sums, 0), 'examples': []}
                row = result[key]
                for field in sums:
                    row[field] += item[field]
                example = {field: hex(item[field]) if field in hex_fields else item[field]
                           for field in ('ip', 'window', 'address', 'size', 'rdi', 'rsi', 'rcx', 'rdx')}
                if example not in row['examples'] and len(row['examples']) < 8:
                    row['examples'].append(example)
        rows = sorted(result.values(), key=lambda row: -row['wait_ns'])
        for row in rows:
            row['wait_ms_per_frame'] = row['wait_ns'] / frames / 1e6
            row['elapsed_ms_per_frame'] = row['elapsed_ns'] / frames / 1e6
            row['calls_per_frame'] = row['count'] / frames
            row['bytes_per_frame'] = row['bytes'] / frames
            for field in fields:
                if field in hex_fields:
                    row[field] = hex(row[field])
        return rows

    result = dict(log=str(args.log.resolve()), frames=frames, complete_blocks=len(complete),
                  incomplete_blocks=len(blocks) - len(complete),
                  overflow=sum(block['overflow'] for block in complete),
                  nested=sum(block['nested'] for block in complete),
                  modes=sorted({block['mode'] for block in complete}),
                  note='Separate instrumented window; wall time includes preemption and tracing. '
                       'Requested bytes are fault/request sizes; bytes are actual downloaded spans. '
                       'Nested readback timings overlap if nested is nonzero.',
                  by_origin=aggregate(('write', 'gpu')),
                  by_pc=aggregate(('ip', 'write', 'gpu')),
                  by_window=aggregate(('ip', 'window', 'write', 'gpu')))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{frames} frames; overflow={result['overflow']}; nested={result['nested']}")
    print('PC                 write gpu    calls/f    KiB/f    wait ms/f   elapsed ms/f')
    for row in result['by_pc'][:24]:
        print(f"{row['ip']:18s} {row['write']:5d} {row['gpu']:3d} {row['calls_per_frame']:10.2f} "
              f"{row['bytes_per_frame'] / 1024:8.1f} {row['wait_ms_per_frame']:12.3f} "
              f"{row['elapsed_ms_per_frame']:14.3f}")


if __name__ == '__main__':
    main()
