#!/usr/bin/env python3
"""Disassemble guest x86-64 code out of a running emulator process.

Guest modules are decrypted at load time into anonymous executable mappings, so
the on-disk SELF cannot be disassembled directly. Profile samples land on guest
addresses, and reading the instructions behind a hot address is the only way to
tell a spin from real work.

Only mappings the process itself marks executable are readable here, and the
bytes are handed to objdump in memory rather than written anywhere.
"""
import argparse
import os
import subprocess
import sys


def executable_ranges(pid):
    ranges = []
    with open(f'/proc/{pid}/maps', 'r') as maps:
        for line in maps:
            bounds, permissions = line.split()[:2]
            if 'x' not in permissions:
                continue
            low, high = (int(part, 16) for part in bounds.split('-'))
            ranges.append((low, high))
    return ranges


def read_code(pid, address, size):
    for low, high in executable_ranges(pid):
        if low <= address and address + size <= high:
            break
    else:
        raise ValueError(f'{address:#x}+{size:#x} is not inside one executable mapping')
    with open(f'/proc/{pid}/mem', 'rb', buffering=0) as memory:
        memory.seek(address)
        code = memory.read(size)
    if len(code) != size:
        raise ValueError(f'Read {len(code)} of {size} requested bytes')
    return code


def disassemble(code, address, marks):
    # objdump seeks its input, which a pipe does not support. An anonymous
    # in-memory file is seekable and leaves the bytes off the filesystem.
    descriptor = os.memfd_create('guest-code')
    os.write(descriptor, code)
    os.lseek(descriptor, 0, os.SEEK_SET)
    result = subprocess.run(
        ['objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64', '-M', 'intel',
         f'--adjust-vma={address:#x}', f'/proc/self/fd/{descriptor}'],
        pass_fds=(descriptor,), capture_output=True, check=True)
    os.close(descriptor)
    lines = []
    for line in result.stdout.decode().splitlines():
        head = line.split(':', 1)[0].strip()
        try:
            marked = int(head, 16) in marks
        except ValueError:
            marked = False
        lines.append(('>>> ' if marked else '    ') + line)
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--address', required=True, help='Guest virtual address, e.g. 0x90083b680')
    parser.add_argument('--size', default='0x200', help='Byte count to disassemble')
    parser.add_argument('--mark', action='append', default=[],
                        help='Address to flag in the output; repeatable')
    arguments = parser.parse_args(argv)
    address, size = int(arguments.address, 0), int(arguments.size, 0)
    if not 0 < size <= 0x10000:
        parser.error('--size must be within 1..0x10000')
    marks = {int(mark, 0) for mark in arguments.mark}
    print(disassemble(read_code(arguments.pid, address, size), address, marks))
    return 0


if __name__ == '__main__':
    sys.exit(main())
