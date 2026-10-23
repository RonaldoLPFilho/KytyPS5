#!/usr/bin/env python3
"""Call-chain profile of a live-session `prof`/`profw` capture (16-word samples).

    live-callprof.py SAMPLES                     self and inclusive top lists
    live-callprof.py SAMPLES --callers FUNC      who calls FUNC (two levels), inclusive
    live-callprof.py SAMPLES --callees FUNC      what FUNC spends its inclusive time in
    live-callprof.py SAMPLES --ms-per-frame 33.3 print times as ms/frame as well

A sample is: pc, the word at rsp (caller of a leaf routine), then frame-pointer return
addresses.  Frames resolve through SAMPLES.maps and `nm` of each module (see
live-prof-report.py).  Percentages are of all samples.
"""
import argparse
import bisect
import collections
import importlib.util
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('live_prof_report', HERE / 'live-prof-report.py')
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)

WORDS = 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument('samples')
    p.add_argument('--top', type=int, default=40)
    p.add_argument('--callers')
    p.add_argument('--callees')
    p.add_argument('--ms-per-frame', type=float, default=0)
    p.add_argument('--width', type=int, default=110)
    args = p.parse_args()

    maps = report.load_maps(args.samples + '.maps')
    data = Path(args.samples).read_bytes()
    n = len(data) // (8 * WORDS)
    cache = {}

    def resolve(address):
        if address in cache:
            return cache[address]
        name = None
        for begin, end, offset, module in maps:
            if begin <= address < end:
                symbols = report.symbols(module)
                file_offset = address - begin + offset
                i = bisect.bisect_right(symbols, (file_offset, '￿')) - 1
                name = symbols[i][1] if i >= 0 else f'?{Path(module).name}'
                break
        cache[address] = name
        return name

    def fmt(count):
        text = f'{100.0 * count / n:6.2f}%'
        if args.ms_per_frame:
            text += f' {args.ms_per_frame * count / n:6.2f} ms'
        return text

    selfc = collections.Counter()
    incl = collections.Counter()
    callers = collections.Counter()
    callees = collections.Counter()
    for i in range(n):
        words = struct.unpack_from(f'<{WORDS}Q', data, i * 8 * WORDS)
        leaf = resolve(words[0]) or '[unmapped]'
        # Frame chain: pc, then (for leaves without a frame) the rsp word when it is code,
        # then the frame-pointer chain.
        chain = [leaf]
        rsp_word = resolve(words[1])
        if rsp_word and rsp_word != leaf:
            chain.append(rsp_word)
        for w in words[2:]:
            if w == 0:
                continue
            name = resolve(w)
            if name and name != chain[-1]:
                chain.append(name)
        selfc[leaf] += 1
        for name in set(chain):
            incl[name] += 1
        if args.callers:
            for j, name in enumerate(chain):
                if args.callers in name:
                    callers[' <- '.join(x[:55] for x in chain[j + 1:j + 3]) or '(top)'] += 1
                    break
        if args.callees:
            for j, name in enumerate(chain):
                if args.callees in name:
                    callees[chain[j - 1][:args.width] if j > 0 else '(self)'] += 1
                    break

    print(f'{n} samples')
    if args.callers:
        total = sum(callers.values())
        print(f'callers of {args.callers}: {fmt(total)}')
        for key, count in callers.most_common(args.top):
            print(f'  {fmt(count)}  {key}')
        return
    if args.callees:
        total = sum(callees.values())
        print(f'callees of {args.callees}: {fmt(total)}')
        for key, count in callees.most_common(args.top):
            print(f'  {fmt(count)}  {key}')
        return
    print('self:')
    for key, count in selfc.most_common(args.top):
        print(f'  {fmt(count)}  {key[:args.width]}')
    print('inclusive:')
    for key, count in incl.most_common(args.top):
        print(f'  {fmt(count)}  {key[:args.width]}')


if __name__ == '__main__':
    main()
