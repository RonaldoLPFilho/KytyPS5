#!/usr/bin/env python3
"""Read bounded SRT plan dumps from an identity-checked owned diagnostic process."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, choices=range(1, 9), default=3)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    pid, identity, digest = receipt['pid'], str(receipt['pid_start_ticks']), receipt['binary_sha256']
    if (not isinstance(pid, int) or pid <= 1 or not identity.isdigit() or
            not re.fullmatch('[0-9a-f]{64}', digest) or
            receipt.get('status') not in ('running', 'measured', 'diagnostics-only')):
        raise ValueError('Not an active owned diagnostic receipt')
    proc = Path('/proc') / str(pid)
    if proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] != identity:
        raise ValueError('Owned process identity changed')
    rows = json.loads(args.report.read_text())['rows']
    selected = [(int(row['shader'], 16), int(row['plans'][0], 16)) for row in rows[:args.limit]]
    if not selected or any(not 0 <= shader < 1 << 64 or not 0 < address < 1 << 64
                           for shader, address in selected):
        raise ValueError('Invalid plan addresses')
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    script = f'''set pagination off
set may-call-functions off
set print elements 512
set print repeats 0
python
import gdb,pathlib,hashlib,json
p=pathlib.Path('/proc/{pid}')
assert p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=={identity!r}
assert hashlib.sha256(p.joinpath('exe').read_bytes()).hexdigest()=={digest!r}
records=[]
for shader,address in {selected!r}:
    value=gdb.Value(address).cast(gdb.lookup_type('Libs::Graphics::ShaderRecompiler::IR::ResourcePlan').pointer()).dereference()
    assert int(value['shader_hash'])==shader
    dump=gdb.execute('p *(Libs::Graphics::ShaderRecompiler::IR::ResourcePlan*)'+str(address),to_string=True)
    name=format(shader,'016x')+'-plan.txt'
    pathlib.Path({str(output)!r},name).write_text(dump)
    records.append(dict(shader=format(shader,'016x'),address=hex(address),file=name,print_elements_limit=512))
print('SRT_PLANS='+json.dumps(records))
end
detach
'''
    path = output / 'capture.gdb'
    path.write_text(script)
    with (ROOT / '_Build/automation/input.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['sudo', '-n', 'gdb', '-q', '-nx', '-batch', '-iex',
            'set print thread-events off', '-p', str(pid), '-x', str(path)],
            capture_output=True, text=True, timeout=25)
    (output / 'capture.log').write_text(result.stdout + result.stderr)
    records = re.findall(r'^SRT_PLANS=(.+)$', result.stdout, re.M)
    if result.returncode or len(records) != 1:
        raise RuntimeError(f'Plan capture failed; see {output}')
    report = dict(pid=pid, pid_start_ticks=identity, binary_sha256=digest,
                  plans=json.loads(records[0]), bounded_dump=True)
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
