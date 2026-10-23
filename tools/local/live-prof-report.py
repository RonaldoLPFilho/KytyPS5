#!/usr/bin/env python3
"""Flat profile of the render thread from a live-session `prof` capture.

    live-prof-report.py SAMPLES [--top 50] [--modules] [--callers libc.so.6]

SAMPLES is written by the emulator's live `prof <seconds> <path>` command: pairs of
(instruction pointer, word at rsp).  SAMPLES.maps is its /proc/self/maps.  Each sample
is attributed to a module through the maps and to a function through `nm` of that
module (exported symbols only for stripped libraries such as libc or the NVIDIA
driver, so e.g. glibc's internal memmove shows under a neighbouring export).  For
samples inside --callers (default libc) the word at rsp is resolved too: for leaf
routines such as memmove or the mprotect wrapper it is the caller's return address.
"""
import argparse
import bisect
import collections
import re
import struct
import subprocess
from pathlib import Path


def load_maps(path):
    maps = []
    for line in Path(path).read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6 or 'x' not in parts[1]:
            continue
        begin, end = (int(x, 16) for x in parts[0].split('-'))
        maps.append((begin, end, int(parts[2], 16), parts[5]))
    maps.sort()
    return maps


_symbols = {}


def symbols(module):
    """Sorted (file offset, name) for a module; file offsets match maps offsets for text."""
    if module in _symbols:
        return _symbols[module]
    table = []
    if Path(module).is_file():
        headers = subprocess.run(['readelf', '-lW', module], capture_output=True, text=True).stdout
        loads = [(int(m[1], 16), int(m[2], 16)) for m in
                 re.finditer(r'LOAD\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)', headers)]
        for flag in ([], ['-D']):
            out = subprocess.run(['nm', '-C', '--defined-only', *flag, module], capture_output=True,
                                 text=True).stdout
            for line in out.splitlines():
                match = re.match(r'([0-9a-f]+) [tTwWiI] (.+)', line)
                if not match:
                    continue
                vaddr = int(match[1], 16)
                offset = vaddr
                for file_offset, segment_vaddr in loads:
                    if segment_vaddr <= vaddr:
                        offset = vaddr - segment_vaddr + file_offset
                table.append((offset, match[2]))
            if table:
                break
    table.sort()
    _symbols[module] = table
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('samples')
    parser.add_argument('--top', type=int, default=50)
    parser.add_argument('--modules', action='store_true', help='also print the per-module split')
    parser.add_argument('--callers', default='libc.so.6')
    args = parser.parse_args()
    raw = Path(args.samples).read_bytes()
    words = struct.unpack(f'<{len(raw) // 8}Q', raw)
    records = [words[i:i + 5] for i in range(0, len(words) - 4, 5)]
    maps = load_maps(args.samples + '.maps')
    starts = [m[0] for m in maps]

    def resolve(pc):
        index = bisect.bisect_right(starts, pc) - 1
        if index < 0 or pc >= maps[index][1]:
            return '[unmapped/JIT]', '[unmapped/JIT]'
        begin, _end, offset, module = maps[index]
        table = symbols(module)
        position = bisect.bisect_right(table, (pc - begin + offset, chr(0x10ffff))) - 1
        return (table[position][1] if position >= 0 else '?'), Path(module).name

    by_function, by_module, by_caller = collections.Counter(), collections.Counter(), collections.Counter()
    for record in records:
        pc = record[0]
        name, short = resolve(pc)
        by_module[short] += 1
        by_function[f'{name}  [{short}]'] += 1
        if short == args.callers:
            label = '[no code address in 4 stack words]'
            for word in record[1:]:
                caller, caller_module = resolve(word)
                if caller_module not in ('[unmapped/JIT]',) and caller_module != args.callers:
                    label = f'{caller}  [{caller_module}]'
                    break
            by_caller[label] += 1
    total = len(records)
    print(f'{total} samples')
    if args.modules:
        for module, count in by_module.most_common(12):
            print(f'{100 * count / total:6.2f}%  {module}')
        print()
    cumulative = 0
    for name, count in by_function.most_common(args.top):
        cumulative += count
        print(f'{100 * count / total:6.2f}%  {100 * cumulative / total:6.2f}%  {name[:150]}')
    if by_caller:
        print(f'\ncallers of {args.callers} (word at rsp; exact for leaf routines like memmove/mprotect):')
        for name, count in by_caller.most_common(25):
            print(f'{100 * count / total:6.2f}%  {name[:150]}')


if __name__ == '__main__':
    main()
