#!/usr/bin/env python3
"""Build the SRT AOT library as a Windows DLL (srt-aot.dll) next to its Linux build.

    python tools/local/compile-srt-aot-windows.py _Build/srt-aot/walk-libraries/<key>

The directory is a Linux library directory made by compile-srt-aot.py: its unit-*.cpp hold
the exported plans and SrtAotAbi.h the ABI they were written for. The plans are plain C++
behind a C ABI, so the same sources serve both systems; the emulator matches every function
by the full source signature, never by address or file. Exports come from a .def file
because older plan sources only carry the ELF visibility attribute.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLAGS = ['/nologo', '/c', '/std:c++20', '/O2', '/clang:-O3', '/clang:-march=x86-64-v3', '/GR-', '/EHs-c-',
         '/clang:-fno-strict-aliasing', '/MD', '-Wno-ignored-attributes', '-Wno-unknown-attributes']


def vs_environment():
    vswhere = Path(r'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe')
    path = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-property', 'installationPath'],
                                   text=True).strip()
    output = subprocess.check_output(f'"{path}\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul && set', shell=True,
                                     text=True)
    environment = {key.upper(): value for key, value in
                   (line.split('=', 1) for line in output.splitlines() if '=' in line)}
    llvm = Path(r'C:\Program Files\LLVM\bin')
    if llvm.exists():
        environment['PATH'] = str(llvm) + ';' + environment.get('PATH', '')
    return environment


def tool(name, environment):
    # CreateProcess searches this process's PATH, not the child's: resolve it first.
    found = shutil.which(name, path=environment['PATH'])
    if not found:
        sys.exit(f'{name} not found')
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('directory', type=Path)
    p.add_argument('--jobs', type=int, default=16)
    a = p.parse_args()
    directory = a.directory.resolve()
    units = sorted(directory.glob('unit-*.cpp'))
    if not units or not (directory / 'SrtAotAbi.h').exists():
        sys.exit(f'{directory}: no unit-*.cpp with SrtAotAbi.h')
    names = sorted({name for unit in units
                    for name in re.findall(r'\b(kyty_srt_aot_[0-9a-f]{16}(?:_signature|_materialize)?)\(',
                                           unit.read_text(errors='replace'))})
    environment = vs_environment()
    clang_cl, lld_link = tool('clang-cl', environment), tool('lld-link', environment)
    build = directory / 'windows'
    build.mkdir(exist_ok=True)

    def compile_one(unit):
        obj = build / (unit.stem + '.obj')
        # A unit compiled by an earlier, interrupted run is kept.
        if obj.exists() and obj.stat().st_mtime >= unit.stat().st_mtime:
            return obj
        result = subprocess.run([clang_cl, *FLAGS, f'/I{directory}', str(unit), f'/Fo{obj}'],
                                env=environment, capture_output=True, text=True)
        if result.returncode:
            sys.exit(f'{unit.name}: compilation failed\n{result.stdout}{result.stderr}')
        return obj

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        objects = list(pool.map(compile_one, units))
    definitions = build / 'srt-aot.def'
    definitions.write_text('EXPORTS\n' + ''.join(f'    {name}\n' for name in names))
    library = directory / 'srt-aot.dll'
    # Thousands of plans make hundreds of objects: past the command-line length limit.
    response = build / 'objects.rsp'
    response.write_text(''.join(f'"{obj}"\n' for obj in objects))
    subprocess.run([lld_link, '/nologo', '/DLL', f'/DEF:{definitions}', f'/OUT:{library}', '/OPT:REF',
                    f'@{response}'], env=environment, check=True)
    receipt = dict(library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                   exports=len(names), units=len(units), flags=FLAGS)
    (directory / 'receipt-windows.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
