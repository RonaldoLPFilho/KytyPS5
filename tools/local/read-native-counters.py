#!/usr/bin/env python3
"""Read native verification counters from an owned run without a debugger or writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess

from debug_cpu_policy import start_ticks

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('receipt', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
r = json.loads(a.receipt.read_text())
pid, ticks, expected = r['pid'], str(r['pid_start_ticks']), r['binary_sha256']
if type(pid) is not int or pid <= 1 or not ticks.isdigit() or not re.fullmatch('[0-9a-f]{64}', expected):
    p.error('Invalid owned process identity')
proc = Path('/proc') / str(pid)
binary = (proc/'exe').resolve()
if start_ticks(pid) != ticks or hashlib.sha256(binary.read_bytes()).hexdigest() != expected:
    p.error('Process identity changed')
header = binary.read_bytes()[:64]
if header[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', header, 16)[0] != 3:
    p.error('Expected the fingerprinted little-endian ELF64 PIE build')
bases = [int(fields[0].split('-')[0], 16) for line in (proc/'maps').read_text().splitlines()
         if len(fields := line.split(maxsplit=5)) == 6 and fields[5] == str(binary)
         and int(fields[2], 16) == 0]
if len(bases) != 1:
    p.error('Ambiguous executable mapping')
phoff = struct.unpack_from('<Q', header, 32)[0]
phsize, phnum = struct.unpack_from('<HH', header, 54)
with binary.open('rb') as f:
    f.seek(phoff); table = f.read(phsize * phnum)
loads = [struct.unpack_from('<IIQQQQQQ', table, i*phsize) for i in range(phnum)]
origins = [ph[3] for ph in loads if ph[0] == 1 and ph[2] == 0]
if origins != [0]:
    p.error('Unexpected ELF load origin')
names = {'kyty_local_srt_native_stats': (6, 8), 'kyty_local_srt_predicate_stats': (3, 8),
         'kyty_local_srt_native_mode': (1, 4), 'kyty_local_srt_predicate_mode': (1, 4),
         'kyty_local_srt_native_diagnostics': (1, 4)}
symbols = {}
for line in subprocess.check_output(['nm', '-n', str(binary)], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[2] in names:
        symbols[fields[2]] = int(fields[0], 16)
if symbols.keys() != names.keys():
    p.error('Missing native diagnostic symbols')
result = {'pid': pid, 'pid_start_ticks': ticks, 'binary_sha256': expected, 'counters': {}}
with (proc/'mem').open('rb', buffering=0) as memory:
    for name, (count, size) in names.items():
        raw = os.pread(memory.fileno(), count*size, bases[0]+symbols[name])
        result['counters'][name] = list(struct.unpack('<' + ('Q' if size == 8 else 'I') * count, raw))
if start_ticks(pid) != ticks:
    p.error('Process changed during snapshot')
a.out.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
