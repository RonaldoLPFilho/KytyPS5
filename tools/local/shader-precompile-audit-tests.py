#!/usr/bin/env python3
"""Exercise corrupt-input rejection, AGC decoding and stall attribution."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from shader_precompile_audit import compute_input, correlate, read_cache, runtime_hash, union_ns, xxh3


class AuditTests(unittest.TestCase):
    def cache(self, path, tail=(), invalid_ref=False):
        # Minimal CS, zero resources. This fixture checks the audit framing;
        # the production compiler additionally validates the stage static key.
        r = [4, 1, 0, 2, 0, 1, 0xbf810000, 0, 0] + [0] * 17 + [0, 0]
        words = [1, len(r), *r, 1, 3, 0xffffffff, 0xffffffff, int(invalid_ref), *tail]
        payload = struct.pack(f'<{len(words)}I', *words)
        data = b'KytyShaderWarmup3:TEST:device\n' + struct.pack('<Q', xxh3(payload)) + payload
        path.write_bytes(data)
        return data

    def test_cache_checks_checksum_and_shader_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache'
            data = self.cache(path)
            _, shaders, pipelines = read_cache(path)
            self.assertEqual(len(shaders), 1)
            self.assertEqual(pipelines[0][2], 0)
            path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_cache(path)
            self.cache(path, invalid_ref=True)
            with self.assertRaisesRegex(ValueError, 'recipe'):
                read_cache(path)

    def test_valid_checksum_does_not_accept_trailing_or_truncated_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cache'
            self.cache(path, tail=(12,))
            with self.assertRaisesRegex(ValueError, 'trailing'):
                read_cache(path)
            data = self.cache(path)
            prefix = data.index(b'\n') + 1
            payload = data[prefix + 8:-4]
            path.write_bytes(data[:prefix] + struct.pack('<Q', xxh3(payload)) + payload)
            with self.assertRaisesRegex(ValueError, 'truncated'):
                read_cache(path)

    def header(self):
        h = bytearray(160)
        h[:8] = b'1234\x18\x00\x00\x00'
        struct.pack_into('<Q', h, 32, 96 - 32)
        struct.pack_into('<Q', h, 40, 128 - 40)
        struct.pack_into('<I', h, 64, len(h))
        struct.pack_into('<H', h, 84, 7)
        h[92] = 4
        for i, reg in enumerate(((0x207, 64), (0x208, 2), (0x209, 1),
                                 (0x213, (3 << 1) | (5 << 7) | (2 << 11) | (4 << 15) | (1 << 10)))):
            struct.pack_into('<II', h, 96 + 8 * i, *reg)
        struct.pack_into('<I', h, 144, 1 << 15)
        return h

    def test_agc_relative_pointers_register_bits_and_wave_size(self):
        h = self.header()
        result = compute_input(h)
        self.assertEqual((result['x'], result['y'], result['z']), (64, 2, 1))
        self.assertEqual((result['workgroup_register'], result['wave_size']), (3, 32))
        self.assertEqual((result['lds_dwords'], result['scratch_dwords']), (512, 7))
        self.assertEqual((result['group_x'], result['group_y'], result['group_z']), (1, 0, 1))
        self.assertEqual((result['thread_ids_num'], result['tg_size_en']), (3, 1))
        self.assertNotIn('host_subgroup_size', result)
        struct.pack_into('<Q', h, 32, 999999)
        with self.assertRaisesRegex(ValueError, 'pointer'):
            compute_input(h)

    def test_declared_hash_and_content_hash_are_distinguished(self):
        data = bytearray(40)
        struct.pack_into('<II', data, 0, 0xbeeb03ff, 0)
        struct.pack_into('<Q', data, 24, 0x1122334455667788)
        self.assertEqual(runtime_hash(data), '1122334455667788')
        data[:4] = b'RDNA'
        self.assertEqual(runtime_hash(bytes(data)), f'{xxh3(bytes(data)):016x}')

    def test_overlapping_concurrent_intervals_are_not_added(self):
        self.assertEqual(union_ns([(2, 8), (4, 9), (10, 12)]), 9)
        self.assertEqual(union_ns([]), 0)

    def test_graphics_handle_is_not_mistaken_for_code_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'playtest.log'
            path.write_text('PLAYTEST_LONG_FRAME frame=1 start_ns=10 end_ns=100 elapsed_ns=90\n'
                'PLAYTEST_FRAMES frames=1 frame=1 start_ns=10 end_ns=100 elapsed_ns=90 max_frame_ns=90\n'
                'PLAYTEST_OPERATION op=1 kind=ComputePipeline start_ns=0 end_ns=70 elapsed_ns=70 id=1\n'
                'PLAYTEST_OPERATION op=2 kind=ComputePipeline start_ns=30 end_ns=110 elapsed_ns=80 id=1\n'
                'PLAYTEST_OPERATION op=3 kind=GraphicsPipeline start_ns=20 end_ns=90 elapsed_ns=70 id=1\n'
                'PLAYTEST_COMPILE_BEGIN op=4 kind=ComputePipeline start_ns=110 id=1\n')
            report = correlate(path, {'0000000000000001': [{'stage': 'CS'}]})
            self.assertEqual(report['longest'][0]['compile_overlap_ms'], 90 / 1e6)
            graphics = next(r for r in report['slowest_compiles_after_first_present']
                            if r['kind'] == 'GraphicsPipeline')
            self.assertNotIn('asset_candidates', graphics)
            self.assertEqual(report['unfinished_compiles'][0]['asset_candidates'][0]['stage'], 'CS')
            self.assertEqual(report['malformed'], 0)


if __name__ == '__main__':
    unittest.main()
