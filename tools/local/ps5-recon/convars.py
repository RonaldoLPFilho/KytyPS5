#!/usr/bin/env python3
"""Extract, for each convar, its registration code site and static convar object.

The registration pattern in this binary is:

    lea  rbx,[rip+...]      # <convar object>      <- comes first
    lea  r14,[rbp-0x40]     # descriptor on stack
    lea  rax,[rip+...]      # <name string>        <- what the string xref finds
    ...
    mov  [rbp-0x40],rax     # descriptor.name
    mov  [rbp-0x38],<len>   # descriptor.length
    call <construct>
    call <register>

so the object is the last RIP-relative load into a register before the name load,
whose target sits in the engine's convar object region.
"""
import re
import subprocess
import sys

import numpy as np

SRC = '/home/chen-xiao/nvme1n1/game/PPSA01341-app0/decrypted/eboot.bin'
RECON = '/tmp/ps5recon/'

ref_tgt = np.load(RECON + 'ref_tgt_str.npy')
ref_pos = np.load(RECON + 'ref_pos_str.npy')
name2va = {}
for line in open(RECON + 'strings4.txt', errors='replace').read().splitlines():
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    offset, text = int(parts[0], 16), parts[1]
    if 0x2000000 <= offset < 0x2900000:
        name2va.setdefault(text, offset - 0x4000)
index = {}
for pos, tgt in zip(ref_pos, ref_tgt):
    index.setdefault(int(tgt), []).append(int(pos))

OBJECT_REGION = range(0x2E00000, 0x3400000)


def disassemble(start, size):
    with open(SRC, 'rb') as f:
        f.seek(start + 0x4000)
        blob = f.read(size)
    import os
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as handle:
        handle.write(blob)
        path = handle.name
    try:
        out = subprocess.run(['objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64', '-M', 'intel',
                              f'--adjust-vma={start:#x}', path],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(path)
    return out


def lookup(name):
    va = name2va.get(name)
    if va is None:
        return None
    callers = sorted(index.get(va, []))
    if not callers:
        return None
    site = callers[0]
    text = disassemble(max(0, site - 160), 384)
    obj = None
    ctor = None
    default = None
    seen_name = False
    for line in text.splitlines():
        address = re.match(r'\s*([0-9a-f]+):', line)
        if not address:
            continue
        address = int(address.group(1), 16)
        lea = re.search(r'lea\s+\w+,\[rip[^\]]*\]\s*#\s*(0x[0-9a-f]+)', line)
        if lea and address <= site:
            target = int(lea.group(1), 16)
            if target in OBJECT_REGION:
                obj = target
        if address >= site:
            seen_name = True
        if seen_name:
            imm = re.search(r'mov\s+DWORD PTR \[rbp-0x[0-9a-f]+\],(0x[0-9a-f]+)', line)
            if imm:
                default = int(imm.group(1), 16)
            call = re.search(r'call\s+(0x[0-9a-f]+)', line)
            if call and ctor is None:
                ctor = int(call.group(1), 16)
    return va, site, obj, ctor, default


CONVAR_CTORS = {0x450a20: 'bool', 0x9260b0: 'float/numeric'}


if __name__ == '__main__':
    for name in sys.argv[1:]:
        found = lookup(name)
        if not found:
            print(f'{name:34} NOT FOUND')
            continue
        va, site, obj, ctor, default = found
        kind = CONVAR_CTORS.get(ctor, f'ctor {ctor and hex(ctor)}')
        extra = f' default={default:#x}' if default is not None else ''
        print(f'{name:34} reg={site:#010x} object={obj and hex(obj)} type={kind}{extra}')
