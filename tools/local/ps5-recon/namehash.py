#!/usr/bin/env python3
"""The cp11/bpe engine name hash, recovered from 0xdd45d0 in the decrypted eboot.

    h = 0x811c9dc5
    for c in name:
        if 'a' <= c <= 'z': c -= 0x20        # ASCII uppercase fold
        h = ((h ^ c) * 0x01000193) & 0xffffffff

Verified: namehash('SkipIntro') == 0xefd56e6c, the constant an independent scan
found at the sound-name lookup site (code 0x10015D6).
"""
import sys


def name_hash(name):
    h = 0x811c9dc5
    for ch in name.encode('latin1'):
        if 0x61 <= ch <= 0x7a:
            ch -= 0x20
        h = ((h ^ ch) * 0x01000193) & 0xffffffff
    return h


if __name__ == '__main__':
    for argument in sys.argv[1:]:
        print(f'{name_hash(argument):#010x}  {argument}')
