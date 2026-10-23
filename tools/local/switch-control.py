#!/usr/bin/env python3
"""Read and write the emulator's diagnostic switches without a debugger.

The switches are plain 32-bit atomics with static symbol offsets, so resolving the
running image's load base is enough. Verifying the pid start time and the on-disk
image hash keeps this bound to the owned run, exactly as the debugger path did.
GDB is avoided here because attaching to this process crashes it on this host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
COUNTER_WORDS = {'kyty_local_guest_epoch': 1, 'kyty_local_render_epoch': 1, 'kyty_local_feedback_prefetch_stats': 4, 'kyty_local_pbr_full_stats': 36, 'kyty_local_lod_stats_counts': 3, 'kyty_local_copy_feedback_stats': 16,
                 'kyty_local_srt_native_stats': 6, 'kyty_local_render_pass_counts': 3,
                 'kyty_local_render_pass_sites': 128,
                 'kyty_local_image_barrier_counts': 5,
                 'kyty_local_descriptor_repeat_counts': 3,
                 'kyty_local_bda_ranges_stats': 8,
                 'kyty_local_stream_batch_stats': 9,
                 'kyty_local_buffer_state_stats': 7,
                 'kyty_local_resident_bindings_stats': 10,
                 'kyty_local_buffer_footprint_stats': 8,
                 'kyty_local_resource_events_stats': 12,
                 'kyty_local_buffer_table_stats': 10,
                 'kyty_local_upload_repeat_counts': 4,
                 'kyty_local_async_lod_stats_counts': 2,
                 'kyty_local_gpu_wait_spin_counts': 3,
                 'kyty_local_guest_readback_stats': 25,
                 'kyty_local_frame_pipeline_stats': 11,
                 'kyty_local_draw_run_stats': 11,
                 'kyty_local_draw_run_ranges_stats': 9,
                 'kyty_local_direct_multi_draw_stats': 5,
                 'kyty_local_image_query_stats': 4,
                 'kyty_local_backing_read_stats': 5,
                 'kyty_local_specialization_memo_stats': 5,
                 'kyty_local_specialization_guard_stats': 9,
                 'kyty_local_specialization_pipeline_stats': 3,
                 'kyty_local_srt_memo_probe_stats': 6,
                 'kyty_local_draw_run_precheck_stats': 2,
                 'kyty_local_compute_mix_stats': 3,
                 'kyty_local_parallel_window_stats': 6,
                 'kyty_local_graphics_state_stats': 36,
                 'kyty_local_route_a_ceiling_stats': 4,
                 'kyty_local_buffer_memo_stats': 24,
                 'kyty_local_present_stats': 16,
                 'kyty_local_present_timestamps': 2048,
                 'kyty_local_frame_attr_ring': 12288,
                 'kyty_local_frame_attr_stats': 8,
                 'kyty_local_frame_attr_render_busy_ns': 1,
                 'kyty_local_frame_attr_render_submissions': 1,
                 'kyty_local_frame_attr_render_gpu_wait_ns': 1,
                 'kyty_local_frame_attr_render_queue_wait_ns': 1,
                 'kyty_local_frame_attr_render_queue_wait_count': 1,
                 'kyty_local_frame_attr_render_command_ns': 1,
                 'kyty_local_frame_attr_render_command_count': 1,
                 'kyty_local_frame_attr_guest_wait_total_ns': 1,
                 'kyty_local_frame_attr_guest_wait_count': 1,
                 'kyty_local_frame_attr_done_count': 1,
                 'kyty_local_write_mask_stats': 8,
                 'kyty_local_write_mask_hist': 16,
                 'kyty_local_stream_batch_stats': 9,
                 'kyty_local_backing_read_stats': 5}


def symbol_offsets(binary, names):
    listing = subprocess.run(['nm', '-S', '--defined-only', str(binary)],
                             capture_output=True, text=True, check=True).stdout
    found = {}
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[3] in names:
            name, size = fields[3], int(fields[1], 16)
            if name in found:
                raise RuntimeError(f'Ambiguous symbol {name}')
            expected = COUNTER_WORDS[name] * 8 if name in COUNTER_WORDS else 4
            if size != expected:
                raise RuntimeError(f'Symbol {name} has {size} bytes; expected '
                                   f'{expected}. '
                                   'Use the counter layout for this binary.')
            found[name] = int(fields[0], 16)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f'Missing symbols: {sorted(missing)}')
    return found


def load_base(pid, binary):
    target = str(Path(binary).resolve())
    # A relinked image keeps running from its old inode; the kernel then labels
    # the mapping "... (deleted)". The label is whitespace-separated from the
    # path, so take the path as the maps pathname field and strip the label.
    suffix = ' (deleted)'
    bases = []
    for line in (Path('/proc') / str(pid) / 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        pathname = fields[5]
        if pathname.endswith(suffix):
            pathname = pathname[:-len(suffix)]
        if pathname == target:
            bases.append(int(fields[0].split('-', 1)[0], 16))
    if not bases:
        raise RuntimeError('Running image is not mapped from the expected path')
    return min(bases)


def running_binary(pid):
    """Resolve the binary a live pid runs, tolerating a replaced/unlinked image."""
    target = Path('/proc') .joinpath(str(pid), 'exe').readlink()
    suffix = ' (deleted)'
    if str(target).endswith(suffix):
        target = Path(str(target)[:-len(suffix)])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path, help='result.json of a retained owned run')
    parser.add_argument('--set', action='append', default=[], metavar='SYMBOL=VALUE')
    parser.add_argument('--get', action='append', default=[], metavar='SYMBOL')
    parser.add_argument('--reset', action='append', default=[], metavar='COUNTER_SYMBOL')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    pid, identity, sha = receipt['pid'], str(receipt['pid_start_ticks']), receipt['binary_sha256']
    if type(pid) is not int or pid <= 1 or not identity.isdigit() or not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('Invalid owned-run identity')
    if receipt.get('status') not in ('running', 'measured', 'diagnostics-only'):
        raise ValueError('Owned run has ended')
    proc = Path('/proc') / str(pid)
    if proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19] != identity:
        raise RuntimeError('Process identity changed')
    binary = running_binary(pid)
    if hashlib.sha256(binary.read_bytes()).hexdigest() != sha:
        raise RuntimeError('Running image does not match the receipt')
    assignments = [item.split('=', 1) for item in args.set]
    names = {name for name, _ in assignments} | set(args.get) | set(args.reset)
    for name in args.reset:
        if name not in COUNTER_WORDS:
            raise ValueError(f'Unknown counter array {name}')
    offsets = symbol_offsets(binary, names)
    base = load_base(pid, binary)
    report = {'pid': pid, 'binary_sha256': sha, 'load_base': hex(base),
              'monotonic_ns': time.monotonic_ns(), 'before': {}, 'after': {}}
    with open(proc / 'mem', 'r+b', buffering=0) as memory:
        def read(name, words=1):
            memory.seek(base + offsets[name])
            return list(struct.unpack(f'<{words}I', memory.read(4 * words)))

        def write(name, value, words=1):
            memory.seek(base + offsets[name])
            memory.write(struct.pack(f'<{words}I', *([value] * words)))

        for name in names:
            report['before'][name] = read(name, COUNTER_WORDS.get(name, 1) * (2 if name in COUNTER_WORDS else 1))
        for name, value in assignments:
            write(name, int(value))
        for name in args.reset:
            write(name, 0, COUNTER_WORDS[name] * 2)
        for name in names:
            report['after'][name] = read(name, COUNTER_WORDS.get(name, 1) * (2 if name in COUNTER_WORDS else 1))
    for name, value in assignments:
        if report['after'][name] != [int(value)]:
            raise RuntimeError(f'Switch {name} did not take the requested value')
    for name in COUNTER_WORDS:
        for phase in ('before', 'after'):
            if name in report[phase]:
                raw = report[phase][name]
                report[phase][name] = [low | (high << 32) for low, high in zip(raw[::2], raw[1::2])]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
