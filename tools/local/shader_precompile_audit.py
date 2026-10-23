#!/usr/bin/env python3
"""Join static shader assets, validated warmup inputs and actual compile stalls.

Reads ds_decompile's catalogs/extracted payloads and Kyty's v2 cache. Produces a
report and AGC-derived compute inputs, NOT a cache of compiled shaders/pipelines.
Never modifies the game, the warmup cache, or the selected playable build.
"""
import argparse
from collections import Counter, defaultdict
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import re
import struct

from playtest_log import event

NO_SHADER = 0xffffffff
STAGES = {1: 'VS', 2: 'PS', 4: 'CS', 5: 'Mesh'}
COMPUTE_FIELDS = ('x', 'y', 'z', 'lds_dwords', 'scratch_dwords',
                  'host_subgroup_size', 'wave_size', 'dispatch_x', 'dispatch_y',
                  'dispatch_z', 'group_x', 'group_y', 'group_z',
                  'dispatch_thread_dimensions', 'thread_ids_num',
                  'workgroup_register', 'tg_size_en')
AGC_FIELDS = tuple(x for x in COMPUTE_FIELDS if x not in (
    'host_subgroup_size', 'dispatch_x', 'dispatch_y', 'dispatch_z',
    'dispatch_thread_dimensions'))
_xxh_library = None


def xxh3(data):
    global _xxh_library
    if _xxh_library is None:
        name = ctypes.util.find_library('xxhash')
        if name is None:
            raise ValueError('libxxhash is required to verify the production cache')
        _xxh_library = ctypes.CDLL(name)
        _xxh_library.XXH3_64bits.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        _xxh_library.XXH3_64bits.restype = ctypes.c_uint64
    return _xxh_library.XXH3_64bits(data, len(data))


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Words:
    def __init__(self, words):
        self.words, self.pos = words, 0

    def take(self, n):
        if n < 0 or n > len(self.words) - self.pos:
            raise ValueError('truncated word stream')
        out = self.words[self.pos:self.pos + n]
        self.pos += n
        return out

    def one(self):
        return self.take(1)[0]

    def vector(self, bound):
        n = self.one()
        if n > bound:
            raise ValueError('vector exceeds schema bound')
        return self.take(n)

    def finish(self):
        if self.pos != len(self.words):
            raise ValueError('unexpected trailing words')


def unpack_words(data):
    if len(data) % 4:
        raise ValueError('unaligned word stream')
    return struct.unpack(f'<{len(data) // 4}I', data)


def decode_record(words):
    r = Words(words)
    stage, lo, hi, user_count, push_cursor = r.take(5)
    if stage not in STAGES or user_count > 108 or push_cursor > 65536:
        raise ValueError('invalid shader record header')
    code = r.vector(256 * 1024)
    back = r.vector(256 * 1024)
    key = r.vector(1024)
    if not code:
        raise ValueError('empty shader code')
    if stage in (1, 5):
        inputs = r.take(28)
        if inputs[0] > 32:
            raise ValueError('invalid vertex resource count')
        r.take(inputs[0] * 8)
    else:
        inputs = r.take(17 if stage == 4 else 67)
    nb = r.one()
    buffers = [r.take(4) for _ in range(nb)] if nb <= 65536 else None
    if buffers is None:
        raise ValueError('invalid buffer count')
    ni = r.one()
    images = [r.take(10) for _ in range(ni)] if ni <= 65536 else None
    if images is None:
        raise ValueError('invalid image count')
    r.finish()
    code_bytes = struct.pack(f'<{len(code)}I', *code)
    return {'stage': STAGES[stage], 'hash': f'{lo | (hi << 32):016x}',
            'code_sha256': sha(code_bytes), 'code_bytes': len(code_bytes),
            'back_code_words': len(back), 'user_data_count': user_count,
            'push_cursor': push_cursor, 'static_key': key, 'inputs': inputs,
            'buffers': buffers, 'images': images}


def read_cache(path):
    data = path.read_bytes()
    newline = data.find(b'\n', 0, 1024)
    if newline < 0 or not data.startswith(b'KytyShaderWarmup3:'):
        raise ValueError('unsupported cache signature')
    prefix = newline + 1
    if len(data) < prefix + 16 or len(data) > 256 * 1024 * 1024:
        raise ValueError('invalid cache size')
    payload = data[prefix + 8:]
    if xxh3(payload) != struct.unpack_from('<Q', data, prefix)[0]:
        raise ValueError('warmup checksum mismatch')
    r = Words(unpack_words(payload))
    count = r.one()
    if count > 16384:
        raise ValueError('cache exceeds current production record limit')
    records = [decode_record(r.vector(1024 * 1024)) for _ in range(count)]
    count = r.one()
    if count > 65536:
        raise ValueError('cache exceeds current production pipeline limit')
    pipelines = []
    for _ in range(count):
        p = r.vector(4096)
        if len(p) < 3:
            raise ValueError('truncated pipeline')
        vs, ps, cs = p[:3]
        def valid(i, stages):
            return i < len(records) and records[i]['stage'] in stages
        if cs != NO_SHADER:
            if len(p) != 3 or vs != NO_SHADER or ps != NO_SHADER or not valid(cs, ('CS',)):
                raise ValueError('invalid compute recipe')
        elif (len(p) != 247 or not valid(vs, ('VS', 'Mesh')) or
              (ps != NO_SHADER and not valid(ps, ('PS',)))):
            raise ValueError('invalid graphics recipe')
        pipelines.append(p)
    r.finish()
    return data[:prefix].decode('ascii').rstrip(), records, pipelines


def runtime_hash(code):
    if len(code) >= 8 and struct.unpack_from('<I', code)[0] == 0xbeeb03ff:
        offset = (struct.unpack_from('<I', code, 4)[0] + 1) * 8
        if offset + 28 > len(code):
            raise ValueError('truncated declared shader hash')
        declared = struct.unpack_from('<Q', code, offset + 16)[0]
        if declared:
            return f'{declared:016x}'
    return f'{xxh3(code):016x}'


def relative(header, field, size):
    rel = struct.unpack_from('<q', header, field)[0]
    ptr = field + rel
    if not rel or ptr < 96 or ptr + size > len(header):
        raise ValueError('AGC relative pointer out of bounds')
    return ptr


def compute_input(header):
    if len(header) < 96 or header[:8] != b'1234\x18\x00\x00\x00' or header[90] != 0:
        raise ValueError('unsupported compute AGC header')
    if struct.unpack_from('<I', header, 64)[0] != len(header):
        raise ValueError('AGC header size mismatch')
    ptr = relative(header, 32, header[92] * 8)
    registers = dict(struct.unpack_from('<II', header, ptr + i * 8)
                     for i in range(header[92]))
    if not {0x207, 0x208, 0x209, 0x213} <= registers.keys():
        raise ValueError('missing compute static registers')
    special = relative(header, 40, 20)
    dispatch = struct.unpack_from('<I', header, special + 16)[0]
    rsrc2 = registers[0x213]
    return dict(zip(AGC_FIELDS, (
        registers[0x207], registers[0x208], registers[0x209],
        ((rsrc2 >> 15) & 511) * 128, struct.unpack_from('<H', header, 84)[0],
        32 if dispatch & (1 << 15) else 64,
        (rsrc2 >> 7) & 1, (rsrc2 >> 8) & 1, (rsrc2 >> 9) & 1,
        ((rsrc2 >> 11) & 3) + 1, (rsrc2 >> 1) & 31, (rsrc2 >> 10) & 1)))


def union_ns(spans):
    end, total = -1, 0
    for a, b in sorted(spans):
        if b > max(a, end):
            total += b - max(a, end)
        end = max(end, b)
    return total


def agc_imports(elf_path):
    """Resolve the documented PLT indices through PT_DYNAMIC, not a catalog."""
    b = elf_path.read_bytes()
    if b[:6] != b'\x7fELF\x02\x01':
        raise ValueError('expected a little-endian ELF64')
    phoff = struct.unpack_from('<Q', b, 32)[0]
    stride, count = struct.unpack_from('<HH', b, 54)
    if stride != 56 or phoff + stride * count > len(b):
        raise ValueError('invalid ELF program headers')
    segments = [struct.unpack_from('<IIQQQQQQ', b, phoff + i * stride) for i in range(count)]
    def offset(addr, size):
        for t, _, start, va, _, length, _, _ in segments:
            if t == 1 and va <= addr and addr + size <= va + length and start + length <= len(b):
                return start + addr - va
        raise ValueError('ELF address not fully file-backed')
    dynamic = next(s for s in segments if s[0] == 2)
    start, size = dynamic[2], dynamic[5]
    if start + size > len(b) or size % 16:
        raise ValueError('invalid ELF dynamic segment')
    tags = dict(struct.unpack_from('<qQ', b, i) for i in range(start, start + size, 16))
    if tags[20] != 7 or tags[11] != 24 or tags[2] < 0x315 * 24:
        raise ValueError('unexpected PLT relocation format')
    strings = b[offset(tags[5], tags[10]):offset(tags[5], tags[10]) + tags[10]]
    source_path = Path(__file__).resolve().parents[2] / 'src/libs/libAgcDriver.cpp'
    names = dict(re.findall(r'LIB_FUNC\("([^"]+)",\s*([^\)]+)\)', source_path.read_text()))
    results = []
    for index in range(0x30a, 0x315):
        got, info, _ = struct.unpack_from('<QQq', b, offset(tags[23] + 24 * index, 24))
        name = struct.unpack_from('<I', b, offset(tags[6] + 24 * (info >> 32), 24))[0]
        symbol = strings[name:strings.index(b'\0', name)].decode('ascii')
        # Check the code itself agrees with the relocation index and GOT address.
        stub = 0x1e73bf0 + index * 16
        pos = offset(stub, 16)
        if (b[pos:pos + 2] != b'\xff\x25' or b[pos + 6] != 0x68 or
                struct.unpack_from('<I', b, pos + 7)[0] != index or
                stub + 6 + struct.unpack_from('<i', b, pos + 2)[0] != got):
            raise ValueError('PLT stub does not match the documented game build')
        results.append({'index': hex(index), 'stub': hex(0x900000000 + stub),
                        'got': hex(0x900000000 + got), 'symbol': symbol,
                        'host_function': names.get(symbol.split('#')[0])})
    return {'elf': str(elf_path.resolve()), 'elf_sha256': sha(b),
            'host_mapping_source_sha256': sha(source_path.read_bytes()), 'imports': results}


def correlate(path, by_hash):
    frames, ops, pending, first_present = [], [], {}, None
    malformed = 0
    for line in path.read_text().splitlines(keepends=True):
        try:
            kind, row = event(line)
        except ValueError:
            malformed += 1
            continue
        if kind == 'PLAYTEST_FRAMES':
            first_present = first_present or row['start_ns']
        elif kind == 'PLAYTEST_LONG_FRAME':
            frames.append(row)
        elif kind == 'PLAYTEST_COMPILE_BEGIN':
            pending[row['op']] = row
        elif kind == 'PLAYTEST_OPERATION':
            pending.pop(row['op'], None)
            ops.append(row)
    def describe(row):
        result = dict(row)
        # GraphicsPipeline.id is a process-local shader handle, not a code hash.
        if row['kind'] in ('ShaderTranslate', 'ShaderCompile', 'ComputePipeline'):
            result['asset_candidates'] = by_hash.get(f"{row['id']:016x}", [])
        return result
    long = []
    for frame in sorted(frames, key=lambda f: f['elapsed_ns'], reverse=True)[:30]:
        groups = defaultdict(list)
        for op in ops:
            a, b = max(frame['start_ns'], op['start_ns']), min(frame['end_ns'], op['end_ns'])
            if a < b:
                groups[op['kind']].append((a, b))
        compile_spans = [s for k, spans in groups.items() for s in spans
                         if k in ('ShaderTranslate', 'ShaderCompile', 'ComputePipeline', 'GraphicsPipeline')]
        long.append(frame | {'overlap_ms_by_kind': {k: union_ns(v) / 1e6 for k, v in groups.items()},
                             'compile_overlap_ms': union_ns(compile_spans) / 1e6})
    after_present = [r for r in ops if first_present is not None and r['start_ns'] >= first_present
                     and r['kind'] in ('ShaderTranslate', 'ShaderCompile', 'ComputePipeline', 'GraphicsPipeline')]
    return {'log': str(path.resolve()), 'sha256': sha(path.read_bytes()),
            'first_present_ns': first_present, 'long_frames': len(frames),
            'over_1s': sum(f['elapsed_ns'] >= 1_000_000_000 for f in frames),
            'longest': long, 'compile_after_first_present_counts': dict(Counter(r['kind'] for r in after_present)),
            'slowest_compiles_after_first_present': [describe(r) for r in sorted(
                after_present, key=lambda r: r['elapsed_ns'], reverse=True)[:30]],
            'unfinished_compiles': [describe(r) for r in pending.values()], 'malformed': malformed,
            'limits': 'First presentation is not gameplay start. Overlaps are unions, not additive thread CPU time; concurrent IO/GPU waits do not prove frame causation. Graphics IDs are not shader hashes.'}


def audit(catalog, extracted, cache, logs):
    inventory_path = catalog / 'programs_base_ps5.json'
    inventory = json.loads(inventory_path.read_text())['programs']
    manifest_path = extracted / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['manifest']['include_trinity'] or manifest['manifest']['rejected']:
        raise ValueError('expected a fully extracted base inventory')
    payloads = {p['code_sha256']: p for p in manifest['programs']}
    if len(payloads) != len(inventory) or set(payloads) != {p['code_sha256'] for p in inventory}:
        raise ValueError('catalog and extraction disagree')
    by_hash, compute, by_sha = defaultdict(list), {}, {}
    code_bytes, header_hashes = 0, set()
    for item in inventory:
        digest = item['code_sha256']
        p = payloads[digest]
        hd = p['agc_header_sha256']
        if not all(re.fullmatch('[0-9a-f]{64}', h) for h in (digest, hd)):
            raise ValueError('invalid payload digest')
        code = (extracted / 'programs' / (digest + '.bin')).read_bytes()
        header = (extracted / 'programs' / (hd + '.agc')).read_bytes()
        if sha(code) != digest or sha(header) != hd or len(code) != p['code_bytes']:
            raise ValueError(f'payload integrity error: {digest}')
        code_bytes += len(code)
        header_hashes.add(hd)
        rh = runtime_hash(code)
        asset = {k: item[k] for k in ('code_sha256', 'stage', 'kinds', 'groups')}
        asset.update(runtime_hash=rh, first_file=p['first_file'])
        by_hash[rh].append(asset)
        by_sha[digest] = asset
        if item['stage'] == 'CS':
            static = compute_input(header)
            compute[digest] = asset | {'agc_header_sha256': hd, 'static_input': static,
                                       'user_data_count': static['workgroup_register']}
    identity, records, pipelines = read_cache(cache)
    observed = {r['code_sha256'] for r in records}
    base = set(payloads)
    counts = {}
    for stage in sorted({p['stage'] for p in inventory} | {r['stage'] for r in records}):
        full = {p['code_sha256'] for p in inventory if p['stage'] == stage}
        seen = {r['code_sha256'] for r in records if r['stage'] == stage}
        counts[stage] = {'inventory': len(full), 'observed': len(seen), 'in_inventory': len(seen & full)}
    checked, mismatches = 0, []
    runtime_axes = Counter()
    for i, r in enumerate(records):
        if r['stage'] != 'CS' or r['code_sha256'] not in compute:
            continue
        predicted = compute[r['code_sha256']]
        actual = dict(zip(COMPUTE_FIELDS, r['inputs']))
        diff = {k: [v, actual[k]] for k, v in predicted['static_input'].items() if v != actual[k]}
        if r['user_data_count'] != predicted['user_data_count']:
            diff['user_data_count'] = [predicted['user_data_count'], r['user_data_count']]
        if diff:
            mismatches.append({'record': i, 'hash': r['hash'], 'fields': diff})
        checked += 1
        runtime_axes[(actual['host_subgroup_size'], actual['dispatch_thread_dimensions'])] += 1
    compute_refs = {p[2] for p in pipelines if p[2] != NO_SHADER}
    graphics = [p for p in pipelines if p[2] == NO_SHADER]
    report = {'provenance': {str(p.resolve()): sha(p.read_bytes()) for p in (inventory_path, manifest_path, cache)},
              'cache_identity': identity, 'shader_records': len(records), 'pipeline_recipes': len(pipelines),
              'inventory_programs': len(base), 'raw_code_bytes': code_bytes,
              'primary_headers_verified': len(header_hashes), 'stage_coverage': counts,
              'observed_game_programs': len(observed & base),
              'observed_by_kind_and_stage': [
                  {'kind': kind, 'stage': stage, 'programs': n}
                  for (kind, stage), n in sorted(Counter(
                      (kind, item['stage']) for item in inventory if item['code_sha256'] in observed
                      for kind in item['kinds']).items())],
              'observed_outside_inventory': sorted(observed - base),
              'base_coverage_percent': 100 * len(observed & base) / len(base),
              'compute_static_inputs': len(compute), 'compute_observed_records_checked': checked,
              'compute_static_mismatches': mismatches,
              'compute_host_subgroup_and_dispatch_mode': [{'values': k, 'records': v} for k, v in runtime_axes.items()],
              'graphics_recipes': len(graphics),
              'graphics_empty_vertex_layouts': sum(p[14] == 0 and p[15] == 0 for p in graphics),
              'compute_records_without_recipe': [{'index': i, 'hash': r['hash']} for i, r in enumerate(records)
                                                 if r['stage'] == 'CS' and i not in compute_refs],
              'warmup_format_limits': {'records': 16384, 'bytes': 256 * 1024 * 1024,
                                       'base_programs_exceed_record_limit': len(base) > 16384},
              'logs': [correlate(p, by_hash) for p in logs],
              'limits': ['Catalog membership does not prove shader/pipeline warmup coverage.',
                         'Compute static input export omits runtime resource specializations, push cursor, host subgroup and dispatch mode.',
                         'Primary AGC header per program checked; multiple headers for one code require preserving all variants before runtime integration.',
                         'Zero observed vertex layouts does not prove all scenes use empty layouts. Native scratch reset does not disable metadata decoding.']}
    return report, list(compute.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--decompile', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--playtest-log', type=Path, action='append', default=[])
    p.add_argument('--elf', type=Path, help='also verify the documented AGC PLT stubs against this ELF')
    p.add_argument('--out', type=Path, required=True, help='new output directory')
    a = p.parse_args()
    if a.out.exists():
        p.error('--out must be new')
    report, compute = audit(a.decompile / 'catalog/shaders', a.decompile / 'extracted-shaders-base',
                            a.cache, a.playtest_log)
    if a.elf:
        report['agc_imports'] = agc_imports(a.elf)
    a.out.mkdir(parents=True)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (a.out / 'compute-static-inputs.json').write_text(json.dumps({
        'schema': 1, 'compiled': False, 'runtime_integrated': False,
        'required_runtime_fields': ['specialization', 'push_cursor', 'host_subgroup_size', 'dispatch_thread_dimensions'],
        'programs': compute}, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('inventory_programs', 'observed_game_programs',
        'base_coverage_percent', 'compute_static_inputs', 'compute_observed_records_checked',
        'compute_static_mismatches', 'graphics_empty_vertex_layouts')}, indent=2))


if __name__ == '__main__':
    main()
