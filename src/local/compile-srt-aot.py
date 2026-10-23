#!/usr/bin/env python3
"""Compile exported SRT plans into an immutable local library, with source receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, default=ROOT / '_Build/srt-aot/libraries')
    p.add_argument('--compiler', default='clang++-18')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--march', choices=['x86-64-v3', 'native'], default='x86-64-v3')
    a = p.parse_args()
    if not 1 <= a.jobs <= 16:
        p.error('--jobs must be 1..16')
    compiler = shutil.which(a.compiler)
    if not compiler:
        p.error('C++ compiler was not found')
    compiler = str(Path(compiler).resolve())
    version = subprocess.check_output([compiler, '--version'], text=True)
    header = ROOT / 'src/local/SrtAotAbi.h'
    sources = sorted(a.source.resolve().glob('kyty_srt_aot_*.cpp'))
    if not sources:
        p.error('No exported SRT plans found')
    for source in sources:
        if not re.fullmatch(r'kyty_srt_aot_[0-9a-f]{16}\.cpp', source.name):
            p.error('Unexpected export filename: ' + source.name)
    contents = [source.read_bytes() for source in sources]
    flags = ['-O3', '-std=c++20', '-fPIC', '-fvisibility=hidden', '-fno-exceptions',
             '-fno-rtti', '-fno-strict-aliasing', '-march=' + a.march]
    inputs = dict(compiler=compiler, compiler_sha256=digest(Path(compiler).read_bytes()),
                  compiler_version=version, flags=flags, header_sha256=digest(header.read_bytes()),
                  sources={path.name: digest(data) for path, data in zip(sources, contents)})
    if a.march == 'native':
        inputs['cpuinfo'] = Path('/proc/cpuinfo').read_text()
    build_key = digest(json.dumps(inputs, sort_keys=True).encode())
    out = a.output.resolve() / build_key
    out.mkdir(parents=True, exist_ok=True)
    # Serialize builders of identical inputs. Neither a running library nor a
    # completed receipt is overwritten; changed source gets a different path.
    import fcntl
    with (out / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = out / 'receipt.json'
        library = out / 'srt-aot.so'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['library_sha256'] != digest(library.read_bytes()):
                raise RuntimeError('Existing immutable library changed: ' + str(library))
            print(json.dumps(saved, indent=2))
            return
        (out / 'SrtAotAbi.h').write_bytes(header.read_bytes())
        shutil.copyfile(__file__, out / 'compile-tool.py')
        units = []
        for start in range(0, len(contents), 24):
            unit = out / f'unit-{start // 24:04}.cpp'
            unit.write_bytes(b'\n'.join(contents[start:start + 24]))
            units.append(unit)

        def compile_one(unit):
            obj = unit.with_suffix('.o')
            command = [compiler, *flags, '-I', str(out), '-c', str(unit), '-o', str(obj)]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            unit.with_suffix('.log').write_bytes(result.stdout)
            if result.returncode:
                raise RuntimeError('Compilation failed: ' + str(unit.with_suffix('.log')))
            return obj, command

        commands = []
        with ThreadPoolExecutor(max_workers=a.jobs) as pool:
            futures = [pool.submit(compile_one, unit) for unit in units]
            for index, future in enumerate(as_completed(futures), 1):
                obj, command = future.result()
                commands.append(command)
                print(f'compiled {index}/{len(units)} units', flush=True)
        temporary = out / 'srt-aot.building.so'
        command = [compiler, '-shared', '-Wl,-z,defs', '-Wl,-z,relro,-z,now',
                   *(str(unit.with_suffix('.o')) for unit in units), '-o', str(temporary)]
        subprocess.run(command, check=True)
        os.replace(temporary, library)
        result = dict(library=str(library), library_sha256=digest(library.read_bytes()),
                      plan_count=len(sources), input_key=build_key, inputs=inputs,
                      compile_commands=commands, link_command=command)
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ['library', 'library_sha256', 'plan_count', 'input_key']}, indent=2))


if __name__ == '__main__':
    main()
