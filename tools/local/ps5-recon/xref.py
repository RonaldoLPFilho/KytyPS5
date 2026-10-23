#!/usr/bin/env python3
"""Look up where a string (convar name, etc.) is referenced from code.

Reuses the RIP-relative scan the earlier recon pass stored in ref_*.npy.
va = file offset - 0x4000 across the string region.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = '/home/chen-xiao/nvme1n1/game/PPSA01341-app0/decrypted/eboot.bin'
RECON = Path('/tmp/ps5recon')

ref_tgt = np.load(RECON / 'ref_tgt_str.npy')
ref_pos = np.load(RECON / 'ref_pos_str.npy')

name2va = {}
for line in (RECON / 'strings4.txt').read_text(errors='replace').splitlines():
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    offset, text = int(parts[0], 16), parts[1]
    if 0x2000000 <= offset < 0x2900000:
        name2va.setdefault(text, offset - 0x4000)

va2name = {}
for text, va in name2va.items():
    va2name.setdefault(va, text)

index = {}
for pos, tgt in zip(ref_pos, ref_tgt):
    index.setdefault(int(tgt), []).append(int(pos))


def sites(pattern):
    rx = re.compile(pattern)
    return sorted({va for text, va in name2va.items() if rx.fullmatch(text)})


if __name__ == '__main__':
    pattern = sys.argv[1]
    data = Path(SRC).read_bytes()
    for va in sites(pattern):
        callers = sorted(index.get(va, []))
        print(f'{va:#010x}  {va2name.get(va)!r}  refs={len(callers)}')
        for caller in callers:
            print(f'    code {caller:#010x}')
