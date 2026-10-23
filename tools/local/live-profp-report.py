#!/usr/bin/env python3
"""Whole-process profile from a live `profp <seconds> <path>` capture: per thread, top code.

    live-profp-report.py SAMPLES [--threads 8] [--top 12]

Samples are (pc, tid | 1<<63).  Threads are named from /proc/<pid>/task/<tid>/comm when
the session is still alive (pass --pid), else shown by tid.  Guest code (the PS5 module,
0x900000000+) has no symbols; it is shown as guest+offset in 256-byte buckets.
"""
import argparse, bisect, collections, struct
from pathlib import Path
import importlib.util
spec = importlib.util.spec_from_file_location('prof', Path(__file__).parent / 'live-prof-report.py')
prof = importlib.util.module_from_spec(spec); spec.loader.exec_module(prof)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples'); parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--top', type=int, default=12); parser.add_argument('--pid', type=int)
    args = parser.parse_args()
    raw = Path(args.samples).read_bytes()
    words = struct.unpack(f'<{len(raw) // 8}Q', raw)
    pairs = [(words[i], words[i + 1] & 0xffffffff) for i in range(0, len(words) - 4, 5)]
    maps = prof.load_maps(args.samples + '.maps'); starts = [m[0] for m in maps]
    def name(pc):
        if 0x900000000 <= pc < 0x904000000:
            return f'guest+{(pc - 0x900000000) & ~0xff:#x}'
        i = bisect.bisect_right(starts, pc) - 1
        if i < 0 or pc >= maps[i][1]: return '[unmapped]'
        b, _e, o, module = maps[i]; table = prof.symbols(module)
        j = bisect.bisect_right(table, (pc - b + o, chr(0x10ffff))) - 1
        n = table[j][1] if j >= 0 else '?'
        return f'{n.split("(")[0][-80:]} [{Path(module).name}]'
    by_thread = collections.defaultdict(collections.Counter)
    for pc, tid in pairs:
        by_thread[tid][name(pc)] += 1
    total = len(pairs)
    print(f'{total} samples')
    for tid, counter in sorted(by_thread.items(), key=lambda kv: -sum(kv[1].values()))[:args.threads]:
        comm = '?'
        if args.pid:
            try: comm = Path(f'/proc/{args.pid}/task/{tid}/comm').read_text().strip()
            except OSError: pass
        n = sum(counter.values())
        print(f'\n== tid {tid} ({comm}) {100 * n / total:.1f}% of samples')
        for fn, c in counter.most_common(args.top):
            print(f'  {100 * c / n:5.1f}%  {fn}')

if __name__ == '__main__':
    main()
