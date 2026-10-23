#!/usr/bin/env python3
"""Recover a cp11 flow's state table and the edges between its states.

A cp11 flow is a state machine whose handler for state N lives at
`flow + N * 0x10`.  The flow's constructor fills that table with a long run of

    lea REG,[rip+...]           ; the handler
    mov QWORD PTR [rdi+OFF],REG ; OFF = N * 0x10

and every handler advances the state itself, with `inc dword ptr [rbx+0xc]` for
"the next one" or `mov dword ptr [rbx+0xc],imm` for a jump.  That is what makes
whole stretches of the boot flow removable by rewriting one immediate instead of
NOPing each screen's transition; see
`docs/compatibility/demons-souls-boot-patch.md`.

    flowtable.py <ctor-start> <ctor-end> [handler-low] [handler-high]

`ctor-start`/`ctor-end` bracket the constructor.  The handler bounds keep
unrelated pointers out of the table and default to the constructor's own 64 KiB
neighbourhood.  Addresses are hex, as virtual addresses.

The front-end flow of PPSA01341 01.007.000, for example:

    flowtable.py 7ca730 7cade0 7c6000 7cb000
"""
import re
import subprocess
import sys
from pathlib import Path

SRC = Path('/home/chen-xiao/nvme1n1/game/PPSA01341-app0/decrypted/eboot.bin')
WORK = Path('/tmp/ps5recon')
FILE_DELTA = 0x4000        # va = file offset - 0x4000 for this PT_LOAD layout

STEP_CALLS = {0xde27d0: 'PUSH', 0xde14a0: 'FIND', 0xde3620: 'RELEASE'}


def disassemble(data, start, end):
    WORK.mkdir(parents=True, exist_ok=True)
    blob = WORK / 'flowtable.bin'
    blob.write_bytes(data[start + FILE_DELTA:end + FILE_DELTA])
    listing = subprocess.run(
        ['objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64', '-M', 'intel',
         f'--adjust-vma={start:#x}', str(blob)],
        capture_output=True, text=True, check=True).stdout
    for line in listing.splitlines():
        match = re.match(r'\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)', line)
        if match:
            yield int(match.group(1), 16), match.group(3).strip()


def strings(data):
    """Map the string pool by virtual address, for annotating handlers."""
    pool = {}
    for match in re.finditer(rb'[\x20-\x7e]{3,}', data):
        offset = match.start()
        if 0x2000000 <= offset < 0x2900000:
            pool.setdefault(offset - FILE_DELTA, match.group().decode('latin1'))
    return pool


def read_table(data, start, end, low, high):
    """The constructor's `mov [rdi+N*0x10],handler` stores, as {state: handler}."""
    registers, table = {}, {}
    for _, text in disassemble(data, start, end):
        match = re.match(r'lea\s+(\w+),\[rip\+[^\]]+\]\s+# 0x([0-9a-f]+)', text)
        if match:
            registers[match.group(1)] = int(match.group(2), 16)
            continue
        match = re.match(r'mov\s+QWORD PTR \[rdi(?:\+(0x[0-9a-f]+))?\],(\w+)$', text)
        if match:
            offset = int(match.group(1), 16) if match.group(1) else 0
            target = registers.get(match.group(2))
            if target is not None and low <= target < high:
                table[offset // 0x10] = target
    return table


def describe(data, pool, state, handler):
    """One line per state: what it touches and where it can go next."""
    notes = []
    consecutive_int3 = 0
    for _, text in disassemble(data, handler, handler + 0x600):
        if text.startswith('int3'):
            consecutive_int3 += 1
            if consecutive_int3 >= 4:
                break
            continue
        consecutive_int3 = 0

        target = re.search(r'# 0x([0-9a-f]+)', text)
        if target and int(target.group(1), 16) in pool:
            notes.append(f'STR {pool[int(target.group(1), 16)]!r}')
        call = re.match(r'call\s+0x([0-9a-f]+)', text)
        if call and int(call.group(1), 16) in STEP_CALLS:
            notes.append(STEP_CALLS[int(call.group(1), 16)])
        jump = re.search(r'mov\s+DWORD PTR \[rbx\+0xc\],(0x[0-9a-f]+)', text)
        if jump:
            notes.append(f'-> {int(jump.group(1), 16)}')
        if re.search(r'inc\s+DWORD PTR \[rbx\+0xc\]', text):
            notes.append(f'-> {state + 1} (inc)')

    unique = list(dict.fromkeys(notes))
    return f'[{state:3d}] {handler:#x}  ' + ' | '.join(unique)


def main(argv):
    if not 2 <= len(argv) <= 4:
        sys.exit(__doc__)
    start, end = (int(value, 16) for value in argv[:2])
    low = int(argv[2], 16) if len(argv) > 2 else start & ~0xffff
    high = int(argv[3], 16) if len(argv) > 3 else (start & ~0xffff) + 0x10000

    data = SRC.read_bytes()
    pool = strings(data)
    table = read_table(data, start, end, low, high)
    if not table:
        sys.exit('no handler stores found; check the constructor bounds')
    for state in sorted(table):
        print(describe(data, pool, state, table[state]))


if __name__ == '__main__':
    main(sys.argv[1:])
