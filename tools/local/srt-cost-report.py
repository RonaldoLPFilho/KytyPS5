#!/usr/bin/env python3
"""Attribute diagnostic SRT wall time by shader and execution path; never an FPS benchmark."""
import argparse
import json
from pathlib import Path

PATHS = {0: 'interpreter', 1: 'linear', 2: 'controlled-linear', 3: 'plan-mismatch'}


def fields(line):
    return {key: int(value, 16 if key in ('shader', 'plan') else 10)
            for key, value in (word.split('=') for word in line.split()[1:])}


def parse(lines):
    blocks, frame, block = [], None, None
    for line in lines:
        if line.startswith('LOCAL_RENDER_COST frames='):
            frame = fields(line)
            block = None
        elif line.startswith('LOCAL_SRT_SUMMARY '):
            block = fields(line) | {'rows': []}
            if frame is None or block['frames'] != frame['frames']:
                raise ValueError('SRT and renderer frame boundaries differ')
            block['elapsed_ns'] = frame['elapsed_ns']
            if block['frames'] <= 0 or block['records'] < 0:
                raise ValueError('Invalid block size')
            blocks.append(block)
        elif line.startswith('LOCAL_SRT ') and block is not None:
            row = fields(line)
            if (row['path'] not in PATHS or row['calls'] <= 0 or min(row.values()) < 0 or
                    row['exclusive_ns'] > row['inclusive_ns'] or
                    row['predicate_ns'] > row['inclusive_ns'] or
                    row['maximum_ns'] > row['inclusive_ns']):
                raise ValueError('Invalid SRT timing record')
            key = row['shader'], row['stage'], row['path']
            if any((other['shader'], other['stage'], other['path']) == key for other in block['rows']):
                raise ValueError('Duplicate record in one SRT block')
            block['rows'].append(row)
            if len(block['rows']) > block['records']:
                raise ValueError('More records than declared')
    complete = [item for item in blocks if len(item['rows']) == item['records']]
    frames = sum(item['frames'] for item in complete)
    if frames <= 0:
        raise ValueError('No complete SRT timing blocks')
    groups = {}
    for item in complete:
        for row in item['rows']:
            key = row['shader'], row['stage'], row['path']
            if key not in groups:
                groups[key] = dict(shader=f"{row['shader']:016x}", stage=row['stage'], path=PATHS[row['path']],
                    calls=0, inclusive_ns=0, exclusive_ns=0, predicate_ns=0, maximum_ns=0, plans=set(), shapes=set())
            group = groups[key]
            for name in ('calls', 'inclusive_ns', 'exclusive_ns', 'predicate_ns'):
                group[name] += row[name]
            group['maximum_ns'] = max(group['maximum_ns'], row['maximum_ns'])
            group['plans'].add(hex(row['plan']))
            group['shapes'].add(tuple(row[name] for name in ('nodes', 'sources', 'flat', 'blocks')))
    rows = sorted(groups.values(), key=lambda x: -x['exclusive_ns'])
    for row in rows:
        row['plans'] = sorted(row['plans'])
        row['shapes'] = [dict(zip(('nodes', 'sources', 'flat', 'blocks'), shape, strict=True))
                         for shape in sorted(row['shapes'])]
        row['calls_per_frame'] = row['calls'] / frames
        row['ns_per_call'] = row['inclusive_ns'] / row['calls']
        for name in ('inclusive', 'exclusive', 'predicate'):
            row[name + '_ms_per_frame'] = row[name + '_ns'] / frames / 1e6
    return dict(complete_blocks=len(complete), incomplete_blocks=len(blocks)-len(complete), frames=frames,
        instrumented_frame_ms=sum(item['elapsed_ns'] for item in complete) / frames / 1e6,
        overflow=sum(item['overflow'] for item in complete), nested=sum(item['nested'] for item in complete),
        srt_exclusive_ms_per_frame=sum(row['exclusive_ms_per_frame'] for row in rows), rows=rows,
        note='Diagnostic wall times include instrumentation, callbacks and preemption. Predicate time is a subset. '
             'Do not add these times to the parent renderer stages or treat instrumented FPS as a performance result.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with args.log.open(errors='replace') as source:
        result = parse(source)
    result['log'] = str(args.log.resolve())
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['frames']} diagnostic frames; SRT {result['srt_exclusive_ms_per_frame']:.3f} ms/frame; "
          f"overflow={result['overflow']}, nested={result['nested']}")
    for row in result['rows'][:20]:
        print(f"{row['shader']} stage={row['stage']} {row['path']:18s} "
              f"{row['calls_per_frame']:8.1f}/frame {row['exclusive_ms_per_frame']:7.3f} ms "
              f"predicate={row['predicate_ms_per_frame']:.3f} ms avg={row['ns_per_call']:.0f} ns")


if __name__ == '__main__':
    main()
