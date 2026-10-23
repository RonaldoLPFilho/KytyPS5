#!/usr/bin/env python3
"""Summarize complete local renderer timer blocks (diagnostics, not a clean FPS result).

The renderer resets its counters every 60 root scopes and prints one block, so a
whole log averages the measurement window together with menus, loading and the
title screen - which are cheap and drown the part you care about.  `--last-blocks`
restricts the summary to the end of the run, which for benchmark-entry.py is the
stationary measurement window.  Careful with the unit: the renderer's "frame" is a root scope exit, and there are
about 60 of those per presented frame here - which is exactly why a block (60 root
scopes) lines up with one presented frame.  The per-frame columns below are per
root scope; `--per-block` reports per timer block, which is the presented-frame
cost you can compare against a 33.3 ms budget.

`--groups` adds a functional roll-up (recording, resource binding, submission),
which is what separates "one hot function" from "per-draw overhead spread thin".
"""
import argparse
import json
from pathlib import Path


# Functional roll-up: which part of the renderer the exclusive time belongs to.
GROUPS = {
    'record commands': ['Draw', 'DrawRun', 'Compute', 'VulkanDraw', 'VulkanCompute',
                        'Targets', 'Pipeline', 'Specialize'],
    'resource binding': ['ObtainBuffer', 'RuntimeSources', 'FindImage', 'FindBuffers',
                         'PrepareBindings', 'RebindBuffers', 'RebindImages',
                         'CommitBindings', 'Materialize', 'Programs'],
    'submit / sync': ['Submit', 'QueueWait', 'GpuWait', 'FlipWait', 'Pm4Graphics',
                      'Pm4Compute', 'Pm4Flip', 'Pm4Boundary', 'Pm4Collect', 'Pm4Flush'],
    'memory traffic': ['CopyBuffer', 'UploadCopies', 'Readback'],
    'shaders': ['ShaderTranslate', 'ShaderCompile', 'DriverPipeline'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--last-blocks', type=int, metavar='N',
                        help='summarize only the last N complete blocks (skipping the '
                             'shutdown block), i.e. the stationary measurement window')
    parser.add_argument('--groups', action='store_true',
                        help='also print a functional roll-up of the exclusive time')
    parser.add_argument('--per-block', action='store_true',
                        help='report per timer block (~one presented frame) instead of '
                             'per root scope')
    args = parser.parse_args()
    kinds = ('Draw DrawRun Compute Targets Programs Materialize RuntimeSources Specialize '
             'PrepareBindings FindBuffers RebindBuffers RebindImages CommitBindings ObtainBuffer '
             'FindImage CopyBuffer UploadCopies Pipeline VulkanDraw VulkanCompute Submit GpuWait '
             'QueueWait FlipWait Pm4Graphics Pm4Compute Pm4Flip Pm4Boundary Pm4Collect '
             'Pm4Flush').split()
    optional = ['Readback', 'ShaderTranslate', 'ShaderCompile', 'DriverPipeline']
    blocks = []
    current = None
    with args.log.open(errors='replace') as source:
        for line in source:
            if line.startswith('LOCAL_RENDER_COST frames='):
                current = {'items': {}, **{k: int(v) for k, v in
                           (field.split('=') for field in line.split()[1:])}}
                blocks.append(current)
            elif line.startswith('LOCAL_RENDER_COST_ITEM ') and current is not None:
                _, name, count, inclusive, exclusive, maximum = line.split()
                if name not in kinds + optional or name in current['items']:
                    raise ValueError('Unknown or repeated stage in a timer block')
                item = dict(zip(('count', 'inclusive_ns', 'exclusive_ns', 'maximum_ns'),
                                map(int, (count, inclusive, exclusive, maximum))))
                if min(item.values()) < 0 or item['exclusive_ns'] > item['inclusive_ns']:
                    raise ValueError('Invalid timer counters')
                current['items'][name] = item
    kinds += [name for name in optional if any(name in block['items'] for block in blocks)]
    complete = [block for block in blocks if set(kinds).issubset(block['items'])]
    if any(block.get('mode', 1) >= 4 for block in complete):
        parser.error('Operation sampling modes are not full timer blocks; enable the timer scopes instead')
    if args.last_blocks:
        # The final block covers process shutdown, so it is never representative.
        complete = complete[:-1][-args.last_blocks:]
        if not complete:
            parser.error('Fewer complete blocks than --last-blocks asks for')
    frames = sum(block['frames'] for block in complete)
    elapsed = sum(block['elapsed_ns'] for block in complete)
    if frames <= 0 or elapsed <= 0:
        parser.error('No complete timer blocks')
    result = {'log': str(args.log.resolve()), 'complete_blocks': len(complete),
              'incomplete_blocks': len(blocks) - len(complete), 'frames': frames,
              'instrumented_frame_ms': elapsed / frames / 1e6,
              'instrumented_fps': frames * 1e9 / elapsed,
              'note': 'Scopes measure wall time including their own overhead and preemption. '
                      'Inclusive columns overlap; only exclusive columns can be added.',
              'items': {}}
    for name in kinds:
        values = [block['items'].get(name, dict(count=0, inclusive_ns=0, exclusive_ns=0, maximum_ns=0))
                  for block in complete]
        result['items'][name] = {
            'calls_per_frame': sum(item['count'] for item in values) / frames,
            'inclusive_ms_per_frame': sum(item['inclusive_ns'] for item in values) / frames / 1e6,
            'exclusive_ms_per_frame': sum(item['exclusive_ns'] for item in values) / frames / 1e6,
            'maximum_call_ms': max(item['maximum_ns'] for item in values) / 1e6}
    exclusive = sum(item['exclusive_ms_per_frame'] for item in result['items'].values())
    result['unattributed_ms_per_frame'] = result['instrumented_frame_ms'] - exclusive
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    # Per root scope by default; per timer block (~one presented frame) on request.
    scale = frames / len(complete) if args.per_block else 1.0
    unit = 'block' if args.per_block else 'frame'
    budget = result['instrumented_frame_ms'] * scale
    result['blocks_summarized'] = len(complete)
    result['instrumented_block_ms'] = result['instrumented_frame_ms'] * frames / len(complete)
    print(f"{len(complete)} blocks, {frames} root scopes, "
          f"{budget:.3f} ms/{unit} with instrumentation")
    print(f'Stage                  calls/{unit:<6s} inclusive ms  exclusive ms')
    for name, item in sorted(result['items'].items(),
                             key=lambda pair: -pair[1]['exclusive_ms_per_frame']):
        print(f"{name:22s} {item['calls_per_frame'] * scale:11.1f} "
              f"{item['inclusive_ms_per_frame'] * scale:13.3f} "
              f"{item['exclusive_ms_per_frame'] * scale:13.3f}")
    print(f"Unattributed: {result['unattributed_ms_per_frame'] * scale:.3f} ms/{unit}")
    if args.groups:
        print()
        print(f'Group                exclusive ms   share')
        for group, names in GROUPS.items():
            value = sum(result['items'][name]['exclusive_ms_per_frame']
                        for name in names if name in result['items']) * scale
            print(f'{group:20s} {value:12.2f} {100 * value / budget:6.1f}%')


if __name__ == '__main__':
    main()
