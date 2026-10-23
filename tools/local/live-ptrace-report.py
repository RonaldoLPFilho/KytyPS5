#!/usr/bin/env python3
"""Group sampled page-protection changes (live `ptrace`) by call chain.

    live-ptrace-report.py TRACE [--depth 6] [--top 25]

Records are [slot, pages, 14 return addresses]; slot = thread*8 + kind*2 with thread 0 =
render thread and kind 0 release write-watch, 1 release read-watch, 2 arm write, 3 arm read.
Each record stands for 8 protection changes.
"""
import argparse, bisect, collections, struct, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
spec = importlib.util.spec_from_file_location('prof', Path(__file__).parent / 'live-prof-report.py')
prof = importlib.util.module_from_spec(spec); spec.loader.exec_module(prof)
KIND = ['release-write', 'release-read', 'arm-write', 'arm-read']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace'); parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--top', type=int, default=25)
    args = parser.parse_args()
    raw = Path(args.trace).read_bytes()
    records = [struct.unpack_from('<16Q', raw, i) for i in range(0, len(raw), 128)]
    maps = prof.load_maps(args.trace + '.maps'); starts = [m[0] for m in maps]
    def name(pc):
        i = bisect.bisect_right(starts, pc) - 1
        if i < 0 or pc >= maps[i][1]: return '?'
        b, _e, o, module = maps[i]; table = prof.symbols(module)
        j = bisect.bisect_right(table, (pc - b + o, chr(0x10ffff))) - 1
        n = table[j][1] if j >= 0 else '?'
        return n.split('(')[0].replace('Libs::Graphics::', '')[-70:]
    chains, pages = collections.Counter(), collections.Counter()
    skip = ('PageManager', 'UpdateRegionWatchers', 'UpdatePageWatchers', 'RegionManager', 'ChangeState',
            'UpdateCpuProtection', 'UpdateGpuProtection', 'MemoryTracker::Iterate', '?')
    for r in records:
        thread = 'render' if r[0] < 8 else 'other'
        kind = KIND[(r[0] % 8) // 2]
        frames = [name(pc) for pc in r[2:] if pc]
        frames = [f for f in frames if not any(s in f for s in skip)][:args.depth]
        key = f'{thread:6} {kind:13} ' + ' <- '.join(frames)
        chains[key] += 8; pages[key] += 8 * r[1]
    total = sum(chains.values())
    print(f'{len(records)} records (~{total} protection changes)')
    for key, count in chains.most_common(args.top):
        print(f'{count:6d} calls {pages[key]:8d} pages  {key}')

if __name__ == '__main__':
    main()
